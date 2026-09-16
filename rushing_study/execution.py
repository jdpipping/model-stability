"""Scheduling, resume, and checkpoint integration for study cells."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import StudyData, load_study_data
from .design import (
    HYBRID_EXECUTION_PROFILE,
    TABULAR_MODEL_IDS,
    declared_execution_settings,
    execution_profile,
    execution_queues,
    manifest_seed,
    verify_runtime_provenance,
)
from .metrics import (
    NUM_CLASSES,
    _locality_features_from_proba,
    crps_contributions,
    local_conformal_padding,
)
from .models import (
    MODEL_IDS,
    canonical_model_id,
    frozen_model_config,
    sensitivity_candidates,
)
from .runner import run_cell
from .storage import (
    CellKey,
    CellStatus,
    RunStorage,
    StatusCounts,
    ValidationResult,
    validate_checksum,
)


SENSITIVITY_STAGE1_RECEIPT = Path(
    "final/sensitivity/stage1/sensitivity_stage1_decision.json"
)


def study_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("config", manifest)


def _verify_runtime_if_frozen(
    manifest: dict[str, Any],
    *,
    require_environment: bool,
    verify_tensorflow: bool = True,
) -> None:
    """Enforce frozen provenance while permitting small synthetic test manifests."""

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return
    code = provenance.get("code")
    repo_root = code.get("repo_root", ".") if isinstance(code, dict) else "."
    kwargs: dict[str, Any] = {"require_environment": require_environment}
    if not verify_tensorflow:
        kwargs["verify_tensorflow"] = False
        # Hybrid manifests are planned on the GPU queue. Their CPU-only queue
        # necessarily has different hardware, while sharing the exact frozen
        # container, code, data, packages, OS, and deterministic settings.
        kwargs["verify_hardware"] = False
    verify_runtime_provenance(manifest, repo_root, **kwargs)


def configured_models(manifest: dict[str, Any]) -> list[str]:
    config = study_config(manifest)
    values = [canonical_model_id(value) for value in config["models"]]
    if set(values) != set(MODEL_IDS) or len(values) != len(MODEL_IDS):
        raise ValueError(f"The definitive study requires exactly the four frozen models: {MODEL_IDS}.")
    return list(MODEL_IDS)


def configured_execution_queues(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the manifest-bound model/device queues."""

    config = study_config(manifest)
    if "provenance" not in config:
        execution = config.get("execution", {})
        if execution.get("profile") == HYBRID_EXECUTION_PROFILE:
            queues = {
                name: dict(spec)
                for name, spec in execution.get("queues", {}).items()
            }
        else:
            queues = {
                "sequential": {
                    "device": "gpu_preferred",
                    "workers": 1,
                    "models": list(MODEL_IDS),
                }
            }
    else:
        queues = execution_queues(config)
    observed: set[str] = set()
    for queue_name, spec in queues.items():
        models = [canonical_model_id(value) for value in spec["models"]]
        overlap = observed & set(models)
        if overlap:
            raise ValueError(
                f"Execution queues duplicate models {sorted(overlap)}."
            )
        observed.update(models)
        spec["models"] = models
        if int(spec["workers"]) <= 0:
            raise ValueError(f"Execution queue {queue_name!r} has no workers.")
    if observed != set(MODEL_IDS):
        raise ValueError(
            "Execution queues must cover every model exactly once; got "
            f"{sorted(observed)}."
        )
    return queues


def configured_execution_profile(manifest: dict[str, Any]) -> str:
    """Return the execution profile, including for lightweight test manifests."""

    config = study_config(manifest)
    if "provenance" not in config:
        return str(
            config.get("execution", {}).get("profile", "sequential_gpu1")
        )
    return execution_profile(config)


def execution_queue_for_model(manifest: dict[str, Any], model: str) -> str:
    """Return the unique locked execution queue for one model."""

    canonical = canonical_model_id(model)
    for queue_name, spec in configured_execution_queues(manifest).items():
        if canonical in spec["models"]:
            return queue_name
    raise ValueError(f"No execution queue owns model {canonical!r}.")


def resolve_execution_queue(
    manifest: dict[str, Any],
    queue: str | None,
    models: Iterable[str] | None,
) -> tuple[str, list[str]]:
    """Validate a worker queue and return its canonical selected models."""

    queues = configured_execution_queues(manifest)
    if queue is None:
        if len(queues) != 1:
            raise ValueError(
                "Hybrid workers must name --queue cpu_tabular or gpu_neural."
            )
        queue = next(iter(queues))
    if queue not in queues:
        raise ValueError(
            f"Unknown execution queue {queue!r}; expected one of {sorted(queues)}."
        )
    allowed = set(queues[queue]["models"])
    selected = (
        allowed
        if models is None
        else {canonical_model_id(value) for value in models}
    )
    if not selected or not selected <= allowed:
        raise ValueError(
            f"Queue {queue!r} permits models {sorted(allowed)}, got "
            f"{sorted(selected)}."
        )
    return queue, [model for model in MODEL_IDS if model in selected]


def configure_worker_device(manifest: dict[str, Any], queue: str) -> dict[str, Any]:
    """Enforce the queue's device/model policy after runtime verification."""

    queues = configured_execution_queues(manifest)
    if queue not in queues:
        raise ValueError(f"Unknown execution queue {queue!r}.")
    device = str(queues[queue]["device"])
    if device == "gpu_preferred":
        return {"queue": queue, "device": device, "visible_gpu_count": None}

    if device == "cpu":
        unexpected = set(queues[queue]["models"]) - set(TABULAR_MODEL_IDS)
        if unexpected:
            raise RuntimeError(
                f"CPU queue {queue!r} contains non-tabular models {sorted(unexpected)}."
            )
        # Both locked tabular implementations are CPU-only. Avoid importing a
        # roughly 500 MB TensorFlow runtime in every one of the 12 workers.
        return {
            "queue": queue,
            "device": device,
            "visible_gpu_count": None,
            "isolation": "cpu_only_model_routing",
        }
    elif device == "gpu":
        import tensorflow as tf

        visible = tf.config.get_visible_devices("GPU")
        if not visible:
            raise RuntimeError(
                f"GPU queue {queue!r} requires a TensorFlow-visible GPU."
            )
    else:
        raise ValueError(f"Unsupported execution device policy {device!r}.")
    return {
        "queue": queue,
        "device": device,
        "visible_gpu_count": len(visible),
    }


def configured_anchors(manifest: dict[str, Any], branch: str) -> list[int]:
    config = study_config(manifest)
    if branch == "main":
        raw = config.get("study", {}).get(
            "anchors", config.get("splits", {}).get("nested_train_anchors", [])
        )
    elif branch == "sensitivity":
        sensitivity = config.get("sensitivity", {})
        raw = sensitivity.get("anchors", sensitivity.get("stage1", {}).get("anchors", [20, 160, 360]))
    else:
        raise ValueError("branch must be 'main' or 'sensitivity'.")
    anchors = [int(value) for value in raw]
    if branch == "main" and anchors != [20, 40, 80, 160, 240, 360]:
        raise ValueError(f"Unexpected main anchors: {anchors}")
    if branch == "sensitivity" and anchors != [20, 160, 360]:
        raise ValueError(f"Unexpected sensitivity anchors: {anchors}")
    return anchors


def configured_repeat_count(
    manifest: dict[str, Any], branch: str, *, sensitivity_extended: bool = False
) -> int:
    config = study_config(manifest)
    if branch == "main":
        return int(
            config.get("study", {}).get(
                "repeats", config.get("execution", {}).get("confirmatory_repeats", 50)
            )
        )
    sensitivity = config.get("sensitivity", {})
    if sensitivity_extended:
        return int(
            sensitivity.get("extended_repeats", sensitivity.get("extension", {}).get("target_total_repeats", 50))
        )
    return int(sensitivity.get("initial_repeats", sensitivity.get("stage1", {}).get("repeats", 20)))


def expected_cell_keys(
    manifest: dict[str, Any],
    branch: str,
    *,
    sensitivity_extended: bool = False,
    repeat_ids: Iterable[int] | None = None,
    anchors: Iterable[int] | None = None,
    models: Iterable[str] | None = None,
) -> list[CellKey]:
    allowed_repeats = set(
        range(1, configured_repeat_count(manifest, branch, sensitivity_extended=sensitivity_extended) + 1)
    )
    selected_repeats = allowed_repeats if repeat_ids is None else {int(value) for value in repeat_ids}
    if not selected_repeats <= allowed_repeats:
        raise ValueError(f"Requested repeat IDs outside the locked branch: {sorted(selected_repeats - allowed_repeats)}")
    allowed_anchors = set(configured_anchors(manifest, branch))
    selected_anchors = allowed_anchors if anchors is None else {int(value) for value in anchors}
    if not selected_anchors <= allowed_anchors:
        raise ValueError(f"Requested anchors outside the locked branch: {sorted(selected_anchors - allowed_anchors)}")
    allowed_models = set(configured_models(manifest))
    selected_models = allowed_models if models is None else {canonical_model_id(value) for value in models}
    if not selected_models <= allowed_models:
        raise ValueError(f"Requested models outside the locked branch: {sorted(selected_models - allowed_models)}")
    return [
        CellKey(branch=branch, repeat=repeat, n_train=n_train, model=model)
        for repeat in sorted(selected_repeats)
        for n_train in sorted(selected_anchors)
        for model in MODEL_IDS
        if model in selected_models
    ]


def _repeat_split(manifest: dict[str, Any], repeat_id: int) -> dict[str, Any] | None:
    """Return a frozen repeat split when the full design manifest is available."""

    records = manifest.get("split_manifests")
    if records is None:
        return None
    if not isinstance(records, list):
        raise ScientificCellValidationError("manifest split_manifests is malformed")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("repeat_id") == int(repeat_id)
    ]
    if len(matches) != 1:
        raise ScientificCellValidationError(
            f"manifest must contain exactly one split for repeat {repeat_id}"
        )
    return matches[0]


def _partition_game_seasons(
    split: dict[str, Any], partition: str
) -> dict[str, int]:
    field = f"{partition}_games"
    games = split.get(field)
    by_season = split.get("by_season")
    if not isinstance(games, list) or not isinstance(by_season, dict):
        raise ScientificCellValidationError(
            f"repeat split lacks a complete {partition} game definition"
        )
    expected = {str(game) for game in games}
    if len(expected) != len(games):
        raise ScientificCellValidationError(
            f"repeat split has duplicate {partition} game IDs"
        )
    seasons: dict[str, int] = {}
    for raw_season, record in by_season.items():
        if not isinstance(record, dict) or not isinstance(record.get(field), list):
            raise ScientificCellValidationError(
                f"repeat split season {raw_season!r} lacks {field}"
            )
        try:
            season = int(raw_season)
        except (TypeError, ValueError) as exc:
            raise ScientificCellValidationError(
                f"repeat split has invalid season {raw_season!r}"
            ) from exc
        for game in record[field]:
            game_id = str(game)
            if game_id in seasons:
                raise ScientificCellValidationError(
                    f"repeat split assigns {partition} game {game_id!r} more than once"
                )
            seasons[game_id] = season
    if set(seasons) != expected:
        raise ScientificCellValidationError(
            f"repeat split {partition} games disagree with its season records"
        )
    return seasons


def validate_sensitivity_stage1_receipt(
    storage: RunStorage, *, require_trigger: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Validate the immutable stage-one decision that authorizes repeats 21--50."""

    path = storage.run_dir / SENSITIVITY_STAGE1_RECEIPT
    if not validate_checksum(path):
        raise RuntimeError(
            "Sensitivity extension requires a checksummed stage-one decision receipt. "
            "Aggregate the complete 20-repeat sensitivity stage first."
        )
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Sensitivity stage-one receipt is unreadable: {path}") from exc
    expected_keys = expected_cell_keys(storage.manifest, "sensitivity")
    expected_paths = {str(key.relative_dir): key for key in expected_keys}
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("manifest_hash") != storage.manifest_hash
        or receipt.get("branch") != "sensitivity"
        or receipt.get("analysis_stage") != "stage1"
        or receipt.get("repeat_ids") != list(range(1, 21))
        or receipt.get("anchors") != configured_anchors(storage.manifest, "sensitivity")
        or receipt.get("models") != configured_models(storage.manifest)
        or not isinstance(receipt.get("extension_trigger"), bool)
    ):
        raise RuntimeError("Sensitivity stage-one decision receipt does not match the manifest.")
    if require_trigger and not receipt["extension_trigger"]:
        raise RuntimeError(
            "Sensitivity repeats 21-50 are not authorized: the recorded stage-one rule did not trigger."
        )

    cells = receipt.get("cell_markers")
    if not isinstance(cells, list) or len(cells) != len(expected_paths):
        raise RuntimeError("Sensitivity stage-one receipt has an incomplete cell index.")
    recorded: dict[str, str] = {}
    for entry in cells:
        if not isinstance(entry, dict):
            raise RuntimeError("Sensitivity stage-one receipt has a malformed cell index.")
        relative = entry.get("path")
        digest = entry.get("marker_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str) or relative in recorded:
            raise RuntimeError("Sensitivity stage-one receipt has a malformed cell index.")
        recorded[relative] = digest
    if set(recorded) != set(expected_paths):
        raise RuntimeError("Sensitivity stage-one receipt cell grid does not match the manifest.")
    for relative, key in expected_paths.items():
        marker = storage.cell_dir(key) / "_SUCCESS"
        if (
            storage.validate_cell(key).status is not CellStatus.COMPLETE
            or not marker.is_file()
            or hashlib.sha256(marker.read_bytes()).hexdigest() != recorded[relative]
        ):
            raise RuntimeError(
                f"Sensitivity stage-one receipt no longer matches committed cell {relative}."
            )

    artifacts = receipt.get("analysis_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Sensitivity stage-one receipt has no analysis-artifact index.")
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise RuntimeError("Sensitivity stage-one receipt has a malformed artifact index.")
        relative = entry.get("file")
        digest = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Sensitivity stage-one receipt has a malformed artifact index.")
        artifact = storage.run_dir / relative
        try:
            artifact.resolve().relative_to(storage.run_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("Sensitivity stage-one receipt references an external artifact.") from exc
        if not validate_checksum(artifact, digest):
            raise RuntimeError(
                f"Sensitivity stage-one analysis artifact failed validation: {relative}"
            )
    return path, receipt


_CLASSICAL_MODELS = {"ridge_sgd_l2", "lightgbm_multiclass"}
_DISTRIBUTION_COLUMNS = (
    "predictive_mean_class",
    "predictive_mean_yards",
    "predictive_std_class",
    "predictive_entropy",
)
_RECOMPUTED_FLOAT_METRICS = (
    "crps",
    "coverage",
    "mean_width",
    "mean_width_inclusive",
    "std_width",
    "mean_q",
    "std_q",
    "crps_game_equal",
    "coverage_game_equal",
    "mean_width_game_equal",
    "mean_width_inclusive_game_equal",
)
_PROBABILITY_ATOL = 2e-6
_VALUE_ATOL = 3e-6
_VALUE_RTOL = 3e-6


class ScientificCellValidationError(ValueError):
    """Internal detail for a checksummed but scientifically invalid cell."""


def _scientific_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificCellValidationError(message)


def _numeric_column(frame: pd.DataFrame, name: str, partition: str) -> np.ndarray:
    _scientific_assert(name in frame.columns, f"{partition} predictions lack column {name!r}")
    try:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScientificCellValidationError(
            f"{partition} prediction column {name!r} is not numeric"
        ) from exc
    _scientific_assert(
        bool(np.all(np.isfinite(values))),
        f"{partition} prediction column {name!r} contains nonfinite values",
    )
    return values


def _integer_column(frame: pd.DataFrame, name: str, partition: str) -> np.ndarray:
    values = _numeric_column(frame, name, partition)
    rounded = np.rint(values)
    _scientific_assert(
        bool(np.allclose(values, rounded, rtol=0.0, atol=1e-10)),
        f"{partition} prediction column {name!r} is not integer-valued",
    )
    return rounded.astype(np.int64)


def _compare_values(
    label: str,
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    exact: bool = False,
) -> None:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    _scientific_assert(
        observed.shape == expected.shape,
        f"{label} shape {observed.shape} does not match recomputed {expected.shape}",
    )
    matches = (
        np.array_equal(observed, expected)
        if exact
        else np.allclose(
            observed.astype(np.float64),
            expected.astype(np.float64),
            rtol=_VALUE_RTOL,
            atol=_VALUE_ATOL,
            equal_nan=False,
        )
    )
    if not matches:
        difference = np.max(
            np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        )
        raise ScientificCellValidationError(
            f"{label} does not match recomputation (max abs difference {difference:.6g})"
        )


def _probability_array(arrays: dict[str, np.ndarray], name: str, rows: int) -> np.ndarray:
    _scientific_assert(name in arrays, f"probability artifact lacks {name!r}")
    raw = np.asarray(arrays[name])
    _scientific_assert(
        np.issubdtype(raw.dtype, np.number), f"{name} must have a numeric dtype"
    )
    _scientific_assert(
        raw.ndim == 2 and raw.shape == (rows, NUM_CLASSES),
        f"{name} shape must be ({rows}, {NUM_CLASSES}), got {raw.shape}",
    )
    values = raw.astype(np.float64)
    _scientific_assert(bool(np.all(np.isfinite(values))), f"{name} contains nonfinite values")
    _scientific_assert(
        bool(np.all(values >= -_PROBABILITY_ATOL)), f"{name} contains negative probabilities"
    )
    _scientific_assert(
        bool(np.all(values <= 1.0 + _PROBABILITY_ATOL)),
        f"{name} contains probabilities above one",
    )
    row_sums = values.sum(axis=1)
    _scientific_assert(
        bool(np.allclose(row_sums, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL)),
        f"{name} rows are not normalized to one",
    )
    return values / row_sums[:, None]


def _central_intervals(proba: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(proba, axis=1)
    lower = np.argmax(cdf >= alpha / 2.0, axis=1).astype(np.int64)
    upper = np.argmax(cdf >= 1.0 - alpha / 2.0, axis=1).astype(np.int64)
    return lower, upper


def _distribution_values(proba: np.ndarray) -> dict[str, np.ndarray]:
    classes = np.arange(NUM_CLASSES, dtype=np.float64)
    mean_class = proba @ classes
    second = proba @ (classes**2)
    return {
        "predictive_mean_class": mean_class,
        "predictive_mean_yards": mean_class - 28.0,
        "predictive_std_class": np.sqrt(np.maximum(second - mean_class**2, 0.0)),
        "predictive_entropy": -np.sum(
            proba * np.log(np.clip(proba, 1e-12, None)), axis=1
        ),
    }


def _expected_seed(manifest: dict[str, Any], key: CellKey, stage: str) -> int:
    return int(
        manifest_seed(
            manifest,
            "cell",
            key.repeat,
            stage,
            key.model,
            key.n_train,
        )
    )


def _model_config_hash(config: Any) -> str:
    try:
        encoded = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificCellValidationError("model configuration is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_model_contract(
    manifest: dict[str, Any],
    key: CellKey,
    metrics: dict[str, Any],
    history: dict[str, Any],
) -> None:
    config = study_config(manifest)
    model_config = history.get("model_config")
    _scientific_assert(isinstance(model_config, dict), "history lacks model_config")
    recomputed_hash = _model_config_hash(model_config)
    _scientific_assert(
        history.get("model_config_hash") == recomputed_hash,
        "history model_config_hash does not match model_config",
    )
    _scientific_assert(
        metrics.get("model_config_hash") == recomputed_hash,
        "metric model_config_hash does not match model_config",
    )

    if key.branch == "main":
        expected_config = frozen_model_config(config, key.model)
        _scientific_assert(
            model_config == expected_config,
            "main model_config differs from the frozen manifest configuration",
        )
        _scientific_assert(
            "selected_config" not in metrics,
            "main cell unexpectedly records a sensitivity selected_config",
        )
    else:
        candidates = sensitivity_candidates(config, key.model)
        selected_index = history.get("selected_candidate_index")
        _scientific_assert(
            isinstance(selected_index, Integral)
            and not isinstance(selected_index, bool)
            and 0 <= int(selected_index) < len(candidates),
            "sensitivity selected_candidate_index is invalid",
        )
        selected_index = int(selected_index)
        _scientific_assert(
            model_config == candidates[selected_index],
            "sensitivity model_config does not match its locked candidate index",
        )
        selected_config = metrics.get("selected_config")
        _scientific_assert(
            isinstance(selected_config, str), "sensitivity metric selected_config is missing"
        )
        try:
            decoded_selected = json.loads(selected_config)
        except json.JSONDecodeError as exc:
            raise ScientificCellValidationError(
                "sensitivity metric selected_config is invalid JSON"
            ) from exc
        _scientific_assert(
            decoded_selected == model_config,
            "sensitivity metric selected_config differs from model_config",
        )
        candidate_scores = history.get("candidate_scores")
        _scientific_assert(
            isinstance(candidate_scores, list) and len(candidate_scores) == len(candidates),
            "sensitivity candidate records do not cover the locked grid",
        )
        for index, (record, expected_config) in enumerate(zip(candidate_scores, candidates)):
            _scientific_assert(
                isinstance(record, dict), f"sensitivity candidate {index} is not an object"
            )
            _scientific_assert(
                record.get("config") == expected_config,
                f"sensitivity candidate {index} config differs from the locked grid",
            )
            _scientific_assert(
                record.get("config_hash") == _model_config_hash(expected_config),
                f"sensitivity candidate {index} config_hash is invalid",
            )

    parameter_count = metrics.get("parameter_count")
    _scientific_assert(
        isinstance(parameter_count, Integral)
        and not isinstance(parameter_count, bool)
        and int(parameter_count) > 0,
        "parameter_count must be a positive integer",
    )
    if key.model not in _CLASSICAL_MODELS:
        expected_parameters = model_config.get("expected_parameters")
        if expected_parameters is not None:
            _scientific_assert(
                isinstance(expected_parameters, Integral)
                and not isinstance(expected_parameters, bool)
                and int(parameter_count) == int(expected_parameters),
                f"neural parameter_count {parameter_count!r} does not match expected "
                f"{expected_parameters!r}",
            )


def _validate_seed_contract(
    manifest: dict[str, Any], key: CellKey, history: dict[str, Any]
) -> None:
    expected: dict[str, int]
    if key.model in _CLASSICAL_MODELS:
        expected = {"fit_seed": _expected_seed(manifest, key, "fit")}
    elif key.branch == "main":
        expected = {
            "selection_seed": _expected_seed(manifest, key, "epoch_selection"),
            "refit_seed": _expected_seed(manifest, key, "refit"),
        }
    else:
        expected = {
            "selection_seed": _expected_seed(manifest, key, "epoch_selection"),
            "refit_seed": _expected_seed(manifest, key, "refit"),
            "prediction_rebuild_seed": _expected_seed(
                manifest, key, "prediction_rebuild"
            ),
        }
    for field, seed in expected.items():
        actual = history.get(field)
        _scientific_assert(
            isinstance(actual, Integral) and not isinstance(actual, bool) and int(actual) == seed,
            f"actual {field} {actual!r} does not match manifest seed {seed}",
        )

    if key.branch != "sensitivity":
        return
    candidate_scores = history.get("candidate_scores")
    _scientific_assert(
        isinstance(candidate_scores, list) and bool(candidate_scores),
        "sensitivity history lacks candidate seed records",
    )
    selection_seed = _expected_seed(manifest, key, "epoch_selection")
    refit_seed = _expected_seed(manifest, key, "refit")
    fit_seed = _expected_seed(manifest, key, "fit")
    for index, record in enumerate(candidate_scores):
        _scientific_assert(
            isinstance(record, dict), f"sensitivity candidate {index} is not an object"
        )
        _scientific_assert(
            record.get("candidate_index") == index,
            f"sensitivity candidate index {record.get('candidate_index')!r} is not {index}",
        )
        expected_fit = fit_seed if key.model in _CLASSICAL_MODELS else selection_seed
        _scientific_assert(
            record.get("fit_seed") == expected_fit,
            f"sensitivity candidate {index} fit seed is not manifest-derived",
        )
        expected_refit = None if key.model in _CLASSICAL_MODELS else refit_seed
        _scientific_assert(
            record.get("refit_seed") == expected_refit,
            f"sensitivity candidate {index} refit seed is not manifest-derived",
        )


def _validate_partition_identity(
    manifest: dict[str, Any],
    key: CellKey,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any] | None:
    """Bind saved evaluation rows to the repeat's frozen game/season split."""

    split = _repeat_split(manifest, key.repeat)
    if split is None:
        return None
    for partition, frame in (("calibration", calibration), ("test", test)):
        expected = _partition_game_seasons(split, partition)
        observed_games = set(frame["game_id"].astype(str))
        _scientific_assert(
            observed_games == set(expected),
            f"{partition} prediction games do not match the frozen repeat split",
        )
        observed_seasons = _integer_column(frame, "season", partition)
        expected_seasons = np.array(
            [expected[str(game)] for game in frame["game_id"].astype(str)],
            dtype=np.int64,
        )
        _compare_values(
            f"{partition} game seasons",
            observed_seasons,
            expected_seasons,
            exact=True,
        )
    return split


def _validate_rows_against_data(
    data: StudyData,
    split: dict[str, Any],
    key: CellKey,
    metrics: dict[str, Any],
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    calibration_y: np.ndarray,
    test_y: np.ndarray,
) -> None:
    """Bind play IDs, outcomes, row order, and training-play count to frozen data."""

    for partition, frame, observed_y in (
        ("calibration", calibration, calibration_y),
        ("test", test, test_y),
    ):
        games = split.get(f"{partition}_games")
        _scientific_assert(
            isinstance(games, list), f"repeat split lacks {partition}_games"
        )
        indices = data.indices_for_games(games)
        expected = data.metadata.iloc[indices]
        _scientific_assert(
            len(frame) == len(expected),
            f"{partition} prediction row count does not match frozen study data",
        )
        _compare_values(
            f"{partition} row seasons in study data",
            _integer_column(frame, "season", partition),
            expected["season"].to_numpy(dtype=np.int64),
            exact=True,
        )
        _scientific_assert(
            frame["game_id"].astype(str).tolist()
            == expected["game_id"].astype(str).tolist(),
            f"{partition} game row order does not match frozen study data",
        )
        _scientific_assert(
            frame["play_id"].astype(str).tolist()
            == expected["play_id"].astype(str).tolist(),
            f"{partition} play IDs/order do not match frozen study data",
        )
        _compare_values(
            f"{partition} outcomes in study data",
            observed_y,
            data.y[indices],
            exact=True,
        )

    anchors = split.get("anchors")
    anchor = anchors.get(str(key.n_train)) if isinstance(anchors, dict) else None
    _scientific_assert(isinstance(anchor, dict), "repeat split lacks the cell training anchor")
    train_games = anchor.get("train_games")
    _scientific_assert(isinstance(train_games, list), "training anchor lacks train_games")
    expected_train_plays = int(len(data.indices_for_games(train_games)))
    actual_train_plays = metrics.get("n_train_plays")
    _scientific_assert(
        isinstance(actual_train_plays, Integral)
        and not isinstance(actual_train_plays, bool)
        and int(actual_train_plays) == expected_train_plays,
        f"metric n_train_plays {actual_train_plays!r} does not match frozen study data "
        f"{expected_train_plays}",
    )


def _validate_scientific_payload(
    storage: RunStorage,
    key: CellKey,
    *,
    recompute_conformal: bool,
    study_data: StudyData | None,
) -> None:
    manifest = storage.manifest
    _scientific_assert(key.branch in {"main", "sensitivity"}, "unknown cell branch")
    max_repeat = configured_repeat_count(
        manifest, key.branch, sensitivity_extended=key.branch == "sensitivity"
    )
    _scientific_assert(
        1 <= key.repeat <= max_repeat, f"repeat {key.repeat} is outside the manifest grid"
    )
    _scientific_assert(
        key.n_train in configured_anchors(manifest, key.branch),
        f"training size {key.n_train} is outside the manifest grid",
    )
    _scientific_assert(
        key.model in configured_models(manifest), f"model {key.model!r} is outside the manifest grid"
    )

    metrics = storage.load_metrics(key)
    observed_identity = (
        metrics.get("branch"),
        metrics.get("repeat"),
        metrics.get("n_train"),
        metrics.get("model"),
    )
    expected_identity = (key.branch, key.repeat, key.n_train, key.model)
    _scientific_assert(
        observed_identity == expected_identity,
        f"metric identity {observed_identity!r} does not match {expected_identity!r}",
    )
    _scientific_assert(
        isinstance(metrics.get("repeat"), Integral)
        and not isinstance(metrics.get("repeat"), bool)
        and isinstance(metrics.get("n_train"), Integral)
        and not isinstance(metrics.get("n_train"), bool),
        "metric repeat and n_train identities must be integers",
    )
    _scientific_assert(
        metrics.get("manifest_hash") == storage.manifest_hash,
        "metric manifest hash does not match the run manifest",
    )
    expected_queue = execution_queue_for_model(manifest, key.model)
    observed_queue = metrics.get("execution_queue")
    if configured_execution_profile(manifest) == HYBRID_EXECUTION_PROFILE:
        _scientific_assert(
            observed_queue == expected_queue,
            f"metric execution queue {observed_queue!r} does not match "
            f"the locked queue {expected_queue!r}",
        )
    elif observed_queue is not None:
        _scientific_assert(
            observed_queue == expected_queue,
            f"metric execution queue {observed_queue!r} does not match "
            f"the locked queue {expected_queue!r}",
        )
    for name, value in metrics.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            _scientific_assert(
                math.isfinite(float(value)), f"metric {name!r} is nonfinite"
            )

    history = storage.load_history(key)
    _scientific_assert(isinstance(history, dict), "cell history is missing or malformed")
    _validate_seed_contract(manifest, key, history)
    _validate_model_contract(manifest, key, metrics, history)

    predictions = storage.load_predictions(key)
    _scientific_assert(
        isinstance(predictions, pd.DataFrame),
        "scientific predictions must be a tabular calibration/test artifact",
    )
    _scientific_assert("partition" in predictions.columns, "predictions lack partition labels")
    _scientific_assert(
        not predictions["partition"].isna().any(), "predictions contain null partition labels"
    )
    observed_partitions = set(predictions["partition"].astype(str))
    _scientific_assert(
        observed_partitions == {"calibration", "test"},
        f"prediction partitions are {sorted(observed_partitions)!r}, expected calibration/test",
    )
    for identifier in ("game_id", "play_id", "season"):
        _scientific_assert(
            identifier in predictions.columns and not predictions[identifier].isna().any(),
            f"predictions lack complete {identifier!r} identity",
        )
    _scientific_assert(
        not predictions.duplicated(["partition", "game_id", "play_id"]).any(),
        "predictions contain duplicate partition/game/play rows",
    )
    calibration = predictions.loc[
        predictions["partition"].astype(str) == "calibration"
    ].reset_index(drop=True)
    test = predictions.loc[predictions["partition"].astype(str) == "test"].reset_index(
        drop=True
    )
    _scientific_assert(len(calibration) > 0 and len(test) > 0, "prediction partitions are empty")
    split = _validate_partition_identity(manifest, key, calibration, test)

    arrays = storage.load_arrays(key)
    _scientific_assert(isinstance(arrays, dict), "cell probability arrays are missing")
    calibration_proba = _probability_array(
        arrays, "calibration_proba", len(calibration)
    )
    test_proba = _probability_array(arrays, "test_proba", len(test))
    calibration_y = _integer_column(calibration, "y_class", "calibration")
    test_y = _integer_column(test, "y_class", "test")
    _scientific_assert(
        bool(np.all((calibration_y >= 0) & (calibration_y < NUM_CLASSES))),
        "calibration outcomes fall outside the 80 classes",
    )
    _scientific_assert(
        bool(np.all((test_y >= 0) & (test_y < NUM_CLASSES))),
        "test outcomes fall outside the 80 classes",
    )
    if study_data is not None:
        _scientific_assert(
            split is not None,
            "study-data validation requires frozen repeat split manifests",
        )
        _validate_rows_against_data(
            study_data,
            split,
            key,
            metrics,
            calibration,
            test,
            calibration_y,
            test_y,
        )

    config = study_config(manifest)
    alpha = float(config.get("uncertainty", {}).get("alpha", float("nan")))
    _scientific_assert(math.isfinite(alpha) and 0.0 < alpha < 1.0, "manifest alpha is invalid")
    calibration_lo, calibration_hi = _central_intervals(calibration_proba, alpha)
    test_lo, test_hi = _central_intervals(test_proba, alpha)
    _compare_values(
        "calibration central_lo",
        _integer_column(calibration, "central_lo", "calibration"),
        calibration_lo,
        exact=True,
    )
    _compare_values(
        "calibration central_hi",
        _integer_column(calibration, "central_hi", "calibration"),
        calibration_hi,
        exact=True,
    )
    _compare_values(
        "test central_lo",
        _integer_column(test, "central_lo", "test"),
        test_lo,
        exact=True,
    )
    _compare_values(
        "test central_hi",
        _integer_column(test, "central_hi", "test"),
        test_hi,
        exact=True,
    )
    _scientific_assert(
        bool(np.all(calibration_lo <= calibration_hi) and np.all(test_lo <= test_hi)),
        "central interval lower bound exceeds upper bound",
    )
    for partition_name, frame, proba in (
        ("calibration", calibration, calibration_proba),
        ("test", test, test_proba),
    ):
        for name, expected in _distribution_values(proba).items():
            _compare_values(
                f"{partition_name} {name}",
                _numeric_column(frame, name, partition_name),
                expected,
            )

    conformal_q = _integer_column(test, "conformal_q", "test")
    conformal_lo = _integer_column(test, "conformal_lo", "test")
    conformal_hi = _integer_column(test, "conformal_hi", "test")
    _scientific_assert(bool(np.all(conformal_q >= 0)), "conformal padding is negative")
    if recompute_conformal:
        local_k = int(config.get("uncertainty", {}).get("local_k", 0))
        _scientific_assert(local_k > 0, "manifest local_k is invalid")
        recomputed_q = local_conformal_padding(
            calibration_y,
            calibration_lo,
            calibration_hi,
            _locality_features_from_proba(calibration_proba),
            _locality_features_from_proba(test_proba),
            alpha,
            local_k=local_k,
        )
        _compare_values("test conformal_q", conformal_q, recomputed_q, exact=True)
    _scientific_assert(
        bool(
            np.all((0 <= conformal_lo) & (conformal_lo <= test_lo))
            and np.all((test_hi <= conformal_hi) & (conformal_hi < NUM_CLASSES))
            and np.all(conformal_lo <= conformal_hi)
        ),
        "conformal interval ordering or bounds are invalid",
    )
    expected_conformal_lo = np.maximum(0, test_lo - conformal_q)
    expected_conformal_hi = np.minimum(NUM_CLASSES - 1, test_hi + conformal_q)
    _compare_values("test conformal_lo", conformal_lo, expected_conformal_lo, exact=True)
    _compare_values("test conformal_hi", conformal_hi, expected_conformal_hi, exact=True)
    width = expected_conformal_hi - expected_conformal_lo
    width_inclusive = width + 1
    covered = ((test_y >= expected_conformal_lo) & (test_y <= expected_conformal_hi)).astype(
        np.int64
    )
    crps = crps_contributions(test_y, test_proba)
    _compare_values("test width", _numeric_column(test, "width", "test"), width)
    _compare_values(
        "test width_inclusive",
        _numeric_column(test, "width_inclusive", "test"),
        width_inclusive,
    )
    _compare_values(
        "test covered", _integer_column(test, "covered", "test"), covered, exact=True
    )
    _compare_values(
        "test crps_contribution",
        _numeric_column(test, "crps_contribution", "test"),
        crps,
    )

    game_rows = pd.DataFrame(
        {
            "game_id": test["game_id"].to_numpy(),
            "crps": crps,
            "coverage": covered.astype(np.float64),
            "mean_width": width.astype(np.float64),
            "mean_width_inclusive": width_inclusive.astype(np.float64),
        }
    )
    by_game = game_rows.groupby("game_id", sort=False).mean(numeric_only=True)
    recomputed: dict[str, float | int] = {
        "n_calibration": len(calibration),
        "n_test": len(test),
        "n_test_games": int(test["game_id"].nunique()),
        "crps": float(np.mean(crps)),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "mean_width_inclusive": float(np.mean(width_inclusive)),
        "std_width": float(np.std(width, ddof=1)) if len(width) > 1 else 0.0,
        "mean_q": float(np.mean(conformal_q)),
        "std_q": float(np.std(conformal_q, ddof=1)) if len(conformal_q) > 1 else 0.0,
        "crps_game_equal": float(by_game["crps"].mean()),
        "coverage_game_equal": float(by_game["coverage"].mean()),
        "mean_width_game_equal": float(by_game["mean_width"].mean()),
        "mean_width_inclusive_game_equal": float(
            by_game["mean_width_inclusive"].mean()
        ),
    }
    for name in ("n_calibration", "n_test", "n_test_games"):
        actual = metrics.get(name)
        _scientific_assert(
            isinstance(actual, Integral)
            and not isinstance(actual, bool)
            and int(actual) == recomputed[name],
            f"metric {name!r} {actual!r} does not match recomputed {recomputed[name]!r}",
        )
    for name in _RECOMPUTED_FLOAT_METRICS:
        actual = metrics.get(name)
        _scientific_assert(
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isfinite(float(actual)),
            f"metric {name!r} is missing or nonfinite",
        )
        _scientific_assert(
            bool(
                np.isclose(
                    float(actual),
                    float(recomputed[name]),
                    rtol=_VALUE_RTOL,
                    atol=_VALUE_ATOL,
                )
            ),
            f"metric {name!r} {actual!r} does not match recomputed {recomputed[name]!r}",
        )


def validate_scientific_cell(
    storage: RunStorage,
    key: CellKey,
    *,
    recompute_conformal: bool = True,
    study_data: StudyData | None = None,
) -> ValidationResult:
    """Validate checksums, design identity, seeds, predictions, and metrics."""

    base = storage.validate_cell(key)
    if not base.is_complete:
        return base
    try:
        _validate_scientific_payload(
            storage,
            key,
            recompute_conformal=recompute_conformal,
            study_data=study_data,
        )
    except Exception as exc:
        return ValidationResult(
            CellStatus.CORRUPT,
            f"scientific validation failed: {exc}",
        )
    return ValidationResult(CellStatus.COMPLETE)


def classify_scientific_cells(
    storage: RunStorage,
    keys: Iterable[CellKey],
    *,
    recompute_conformal: bool = True,
    study_data: StudyData | None = None,
) -> dict[CellKey, ValidationResult]:
    return {
        key: validate_scientific_cell(
            storage,
            key,
            recompute_conformal=recompute_conformal,
            study_data=study_data,
        )
        for key in keys
    }


def classification_counts(
    classified: dict[CellKey, ValidationResult],
) -> StatusCounts:
    counts = {status: 0 for status in CellStatus}
    for result in classified.values():
        counts[result.status] += 1
    return StatusCounts(
        complete=counts[CellStatus.COMPLETE],
        running=counts[CellStatus.RUNNING],
        corrupt=counts[CellStatus.CORRUPT],
        missing=counts[CellStatus.MISSING],
    )


def execute_repeat(
    storage: RunStorage,
    branch: str,
    repeat_id: int,
    *,
    sensitivity_extended: bool = False,
    anchors: Iterable[int] | None = None,
    models: Iterable[str] | None = None,
    queue: str | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Run the missing cells for one repeat, loading the large tensors once."""
    if sensitivity_extended and branch != "sensitivity":
        raise ValueError("sensitivity_extended is valid only for the sensitivity branch.")
    if branch == "sensitivity" and sensitivity_extended and int(repeat_id) > 20:
        validate_sensitivity_stage1_receipt(storage, require_trigger=True)
    queue_name, queue_models = resolve_execution_queue(
        storage.manifest, queue, models
    )
    queue_device = configured_execution_queues(storage.manifest)[queue_name][
        "device"
    ]
    _verify_runtime_if_frozen(
        storage.manifest,
        require_environment=True,
        verify_tensorflow=queue_device != "cpu",
    )
    # GPU children repeat the full device check. CPU-only children verify the
    # shared frozen software/data contract without comparing against the DGX
    # hardware on which the hybrid manifest was planned.
    configure_worker_device(storage.manifest, queue_name)
    keys = expected_cell_keys(
        storage.manifest,
        branch,
        sensitivity_extended=sensitivity_extended,
        repeat_ids=[repeat_id],
        anchors=anchors,
        models=queue_models,
    )
    classified = classify_scientific_cells(storage, keys)
    if (storage.run_dir / "_SUCCESS").exists():
        if all(result.is_complete for result in classified.values()):
            return {"completed": len(keys), "executed": 0, "skipped": len(keys)}
        raise RuntimeError(
            "Finalized run contains a requested cell that fails scientific validation; "
            "the immutable run will not be modified."
        )
    if not resume and any(result.status is not CellStatus.MISSING for result in classified.values()):
        raise RuntimeError("Study cells already exist; pass --resume to validate and continue them.")
    pending = [
        key
        for key, result in classified.items()
        if not result.is_complete and result.status is not CellStatus.RUNNING
    ]
    if not pending:
        return {"completed": len(keys), "executed": 0, "skipped": len(keys)}

    config = study_config(storage.manifest)
    data = load_study_data(config)
    executed = 0
    skipped = len(keys) - len(pending)
    for key in pending:
        storage.mark_cell_running(key, overwrite=True)
        try:
            payload = run_cell(
                data,
                storage.manifest,
                key.branch,
                key.repeat,
                key.n_train,
                key.model,
            )
            predictions = pd.concat(
                [
                    payload.calibration_predictions.assign(partition="calibration"),
                    payload.test_predictions.assign(partition="test"),
                ],
                ignore_index=True,
                sort=False,
            )
            metrics = dict(payload.metrics)
            metrics["manifest_hash"] = storage.manifest_hash
            metrics["execution_queue"] = queue_name
            history = dict(payload.history)
            history["execution_queue"] = queue_name
            if payload.candidate_scores:
                history["candidate_scores"] = payload.candidate_scores
            storage.write_cell_artifacts(
                key,
                metrics=metrics,
                predictions=predictions,
                history=history,
                arrays=payload.arrays,
                overwrite=True,
            )
            validation = validate_scientific_cell(
                storage,
                key,
                study_data=(
                    data
                    if isinstance(data, StudyData)
                    and isinstance(storage.manifest.get("split_manifests"), list)
                    else None
                ),
            )
            if not validation.is_complete:
                raise RuntimeError(
                    f"Committed cell failed scientific validation: {key}: {validation.reason}"
                )
            executed += 1
            print(
                f"[{branch} repeat {repeat_id:03d}] n={key.n_train} {key.model} complete "
                f"CRPS={metrics['crps']:.6f}",
                flush=True,
            )
        finally:
            storage.clear_cell_running(key)
    return {"completed": len(keys), "executed": executed, "skipped": skipped}


def schedule_subprocesses(
    run_dir: str | Path,
    branch: str,
    *,
    sensitivity_extended: bool = False,
    resume: bool = True,
    repeat_ids: Iterable[int] | None = None,
    anchors: Iterable[int] | None = None,
    models: Iterable[str] | None = None,
    workers: int | None = None,
    queue: str | None = None,
) -> None:
    """Dispatch manifest-routed repeat workers with deterministic settings."""

    storage = RunStorage.open(run_dir)
    if sensitivity_extended and branch != "sensitivity":
        raise ValueError("sensitivity_extended is valid only for the sensitivity branch.")
    if branch == "sensitivity" and sensitivity_extended:
        validate_sensitivity_stage1_receipt(storage, require_trigger=True)
    settings = declared_execution_settings(study_config(storage.manifest))
    queues = configured_execution_queues(storage.manifest)
    if queue is not None and queue not in queues:
        raise ValueError(
            f"Unknown execution queue {queue!r}; expected one of {sorted(queues)}."
        )
    _verify_runtime_if_frozen(
        storage.manifest,
        require_environment=False,
        verify_tensorflow=(
            queue is None or str(queues[queue].get("device")) != "cpu"
        ),
    )
    requested_models = (
        set(queues[queue]["models"])
        if models is None and queue is not None
        else (
            set(MODEL_IDS)
            if models is None
            else {canonical_model_id(value) for value in models}
        )
    )
    active_queues = {
        name: spec
        for name, spec in queues.items()
        if (queue is None or name == queue)
        and requested_models.intersection(spec["models"])
    }
    routed_models = {
        model
        for spec in active_queues.values()
        for model in spec["models"]
        if model in requested_models
    }
    if not active_queues or routed_models != requested_models:
        raise ValueError(
            f"Requested models {sorted(requested_models)} are not covered by queue "
            f"selection {queue!r}."
        )
    if workers is not None:
        if len(active_queues) != 1:
            raise ValueError(
                "--workers may override/check only one explicitly selected queue; "
                "the complete hybrid run uses its locked 12+1 topology."
            )
        queue_name, spec = next(iter(active_queues.items()))
        if int(workers) != int(spec["workers"]):
            raise ValueError(
                f"Queue {queue_name!r} requires --workers {spec['workers']}; "
                f"got {workers}."
            )
    keys = expected_cell_keys(
        storage.manifest,
        branch,
        sensitivity_extended=sensitivity_extended,
        repeat_ids=repeat_ids,
        anchors=anchors,
        models=routed_models,
    )
    classified = classify_scientific_cells(storage, keys)
    if (storage.run_dir / "_SUCCESS").exists():
        if all(result.is_complete for result in classified.values()):
            return
        raise RuntimeError(
            "Finalized run contains a requested cell that fails scientific validation; "
            "no worker was launched."
        )
    if not resume:
        if any(result.status is not CellStatus.MISSING for result in classified.values()):
            raise RuntimeError("Study cells already exist; pass --resume to continue safely.")
    provenance_code = storage.manifest.get("provenance", {}).get("code", {})
    frozen_repo_root = (
        Path(provenance_code["repo_root"]).resolve()
        if isinstance(provenance_code, dict) and provenance_code.get("repo_root")
        else None
    )

    queue_jobs: list[tuple[str, int, list[list[str]], dict[str, str]]] = []
    for queue_name, spec in active_queues.items():
        queue_models = [
            model for model in MODEL_IDS if model in routed_models and model in spec["models"]
        ]
        queue_repeats = sorted(
            {
                key.repeat
                for key, result in classified.items()
                if key.model in queue_models
                and not result.is_complete
                and result.status is not CellStatus.RUNNING
            }
        )
        commands: list[list[str]] = []
        for repeat_id in queue_repeats:
            command = [
                sys.executable,
                "-m",
                "rushing_study",
                "run-repeat",
                "--run-dir",
                str(run_dir),
                "--branch",
                branch,
                "--repeat-id",
                str(repeat_id),
                "--queue",
                queue_name,
                "--models",
                ",".join(queue_models),
            ]
            if resume:
                command.append("--resume")
            if sensitivity_extended:
                command.append("--extend-sensitivity")
            if anchors is not None:
                command.extend(
                    ["--anchors", ",".join(str(value) for value in anchors)]
                )
            commands.append(command)
        environment = os.environ.copy()
        environment.update(settings["environment_variables"])
        environment["RUSHING_STUDY_QUEUE"] = queue_name
        environment["TF_CPP_MIN_LOG_LEVEL"] = environment.get(
            "TF_CPP_MIN_LOG_LEVEL", "2"
        )
        if frozen_repo_root is not None:
            prior_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(frozen_repo_root)
                if not prior_pythonpath
                else str(frozen_repo_root) + os.pathsep + prior_pythonpath
            )
        if commands:
            queue_jobs.append(
                (queue_name, int(spec["workers"]), commands, environment)
            )

    def launch_queue(
        queue_name: str,
        worker_count: int,
        commands: list[list[str]],
        environment: dict[str, str],
    ) -> None:
        def launch(command: list[str]) -> None:
            subprocess.run(
                command,
                check=True,
                env=environment,
                cwd=None if frozen_repo_root is None else str(frozen_repo_root),
            )

        if worker_count == 1:
            for command in commands:
                launch(command)
            return
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"rushing-{queue_name}",
        ) as executor:
            list(executor.map(launch, commands))

    if len(queue_jobs) == 1:
        launch_queue(*queue_jobs[0])
    elif queue_jobs:
        with ThreadPoolExecutor(
            max_workers=len(queue_jobs), thread_name_prefix="rushing-queues"
        ) as executor:
            futures = [executor.submit(launch_queue, *job) for job in queue_jobs]
            for future in futures:
                future.result()


def load_metrics_frame(
    storage: RunStorage,
    keys: Iterable[CellKey],
    *,
    study_data: StudyData | None = None,
) -> pd.DataFrame:
    rows = []
    for key in keys:
        result = validate_scientific_cell(storage, key, study_data=study_data)
        if not result.is_complete:
            raise RuntimeError(f"Cannot aggregate {key}: {result.status.value} ({result.reason})")
        rows.append(storage.load_metrics(key))
    frame = pd.DataFrame(rows)
    key_columns = ["branch", "repeat", "n_train", "model"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("Duplicate metric cells were loaded.")
    return frame.sort_values(key_columns).reset_index(drop=True)
