"""Scientific cell revalidation and within-task final aggregation.

Aggregation is deliberately more than a checksum pass. A committed cell is
accepted only when its identities, split membership, seeds, model receipt,
training history, predictions, and every reported score can be reconstructed
from the frozen manifest and prepared task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    DEFAULT_MODELS,
    DEFAULT_TARGET_COVERAGE,
    LEGACY_STRUCTURE_MODELS,
    STRUCTURE_MODELS,
    pilot_task_analysis_tables,
    task_analysis_receipt_fields,
    task_analysis_tables,
    validate_task_grid,
)
from .design import validate_task_design
from .execution import open_run
from .metrics import (
    bernoulli_entropy,
    binary_log_loss_contributions,
    brier_contributions,
    calibration_bias,
    crps_contributions,
    empirical_distribution_null,
    fit_venn_abers,
    game_equal_mean,
    hierarchical_distribution_intervals,
    label_conditional_prediction_sets,
    pathwise_conformal_tube,
    prevalence_null,
    rms_reliability_error,
    skill_score,
    trajectory_path_rmse,
    trajectory_rmse,
    trajectory_squared_error,
    validate_binary_probability,
    validate_probability_matrix,
)
from .models import (
    CLASSICAL_PARAMETER_COUNT_DEFINITION,
    NEURAL_FAMILIES,
    NEURAL_PARAMETER_COUNT_DEFINITION,
    PUNT_CDF_RESIDUAL_CONTRACT,
)
from .prepared import load_prepared_task
from .runner import (
    CLASSICAL_PREPROCESSING_SCOPE,
    NEURAL_FINAL_PREPROCESSING_SCOPE,
    NEURAL_SELECTOR_PREPROCESSING_SCOPE,
)
from .storage import (
    CellKey,
    atomic_write_csv,
    atomic_write_json,
    cell_keys_from_design,
    validate_checksum,
)


class ScientificValidationError(RuntimeError):
    pass


_TOLERANCE = 2e-7


def _close(actual: float, expected: float, label: str, tolerance: float = _TOLERANCE) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(
        actual, expected, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ScientificValidationError(f"{label}: saved={actual}, recomputed={expected}")


def _metric_close(metrics: Mapping[str, Any], name: str, expected: float) -> None:
    if name not in metrics:
        raise ScientificValidationError(f"metric {name!r} is missing")
    try:
        actual = float(metrics[name])
    except (TypeError, ValueError) as exc:
        raise ScientificValidationError(f"metric {name!r} is not numeric") from exc
    _close(actual, float(expected), name)


def _uncertainty_subsample_seed(
    metrics: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> int:
    seeds = metrics.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ScientificValidationError("cell metrics lack their frozen seed registry")
    value = seeds.get("uncertainty_subsample")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScientificValidationError("frozen uncertainty-subsample seed is invalid")
    if metrics.get("uncertainty_subsample_seed") != value:
        raise ScientificValidationError(
            "reported uncertainty-subsample seed differs from the frozen cell seed"
        )
    saved = np.asarray(arrays.get("uncertainty_subsample_seed"))
    if saved.shape != (1,) or not np.issubdtype(saved.dtype, np.integer):
        raise ScientificValidationError(
            "uncertainty-subsample seed array is missing or malformed"
        )
    if int(saved[0]) != value:
        raise ScientificValidationError(
            "uncertainty-subsample seed artifact differs from the frozen cell seed"
        )
    return int(value)


def _array_close(actual: Any, expected: Any, label: str, *, equal_nan: bool = False) -> None:
    observed = np.asarray(actual)
    wanted = np.asarray(expected)
    if observed.shape != wanted.shape or not np.allclose(
        observed,
        wanted,
        atol=_TOLERANCE,
        rtol=_TOLERANCE,
        equal_nan=equal_nan,
    ):
        raise ScientificValidationError(
            f"{label} differs (saved shape {observed.shape}, expected {wanted.shape})"
        )


def _json_equal(actual: Any, expected: Any) -> bool:
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def _validate_hashed_signature(
    value: Any,
    *,
    schema_version: str,
    label: str,
) -> Mapping[str, Any]:
    """Validate a canonical JSON signature before using it as an equality gate."""

    if not isinstance(value, Mapping):
        raise ScientificValidationError(f"{label} is missing or is not an object")
    if value.get("schema_version") != schema_version:
        raise ScientificValidationError(f"{label} schema is invalid")
    digest = value.get("signature_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ScientificValidationError(f"{label} hash is invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "signature_sha256"
    }
    recomputed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if digest != recomputed:
        raise ScientificValidationError(f"{label} hash does not match its payload")
    return value


def _partition_index(task: Any, game_ids: Sequence[Any]) -> np.ndarray:
    requested = {str(value) for value in game_ids}
    observed = task.examples["game_id"].astype(str).to_numpy()
    index = np.flatnonzero(np.isin(observed, list(requested)))
    if set(observed[index]) != requested:
        missing = sorted(requested - set(observed[index]))
        raise ScientificValidationError(f"prepared task lacks frozen games {missing[:10]}")
    return index


def _numeric_targets(frame: pd.DataFrame, label: str) -> np.ndarray:
    values = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=float)
    original_missing = frame["target"].isna().to_numpy()
    if np.any(np.isnan(values) & ~original_missing):
        raise ScientificValidationError(f"{label} contains a non-numeric target")
    return values


def _validate_prediction_identity(
    predictions: pd.DataFrame,
    split: Mapping[str, Any],
    task: Any,
    key: CellKey,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the complete ordered calibration/test identity registry."""

    required = {"partition", "example_id", "game_id", "stratum", "target"}
    missing = required - set(predictions.columns)
    if missing:
        raise ScientificValidationError(f"prediction table lacks {sorted(missing)}")
    if predictions.duplicated(["partition", "example_id"]).any():
        raise ScientificValidationError("prediction identities are duplicated")
    calibration_index = _partition_index(task, split["calibration_game_ids"])
    test_index = _partition_index(task, split["test_game_ids"])
    expected_index = np.concatenate([calibration_index, test_index])
    expected_partitions = np.asarray(
        ["calibration"] * len(calibration_index) + ["test"] * len(test_index), dtype=str
    )
    actual_partitions = predictions["partition"].astype(str).to_numpy()
    if not np.array_equal(actual_partitions, expected_partitions):
        raise ScientificValidationError(
            f"{key.relative_dir} prediction partitions or row order differ from frozen split"
        )
    expected_ids = task.examples.iloc[expected_index]["example_id"].astype(str).to_numpy()
    actual_ids = predictions["example_id"].astype(str).to_numpy()
    if not np.array_equal(actual_ids, expected_ids):
        raise ScientificValidationError(
            f"{key.relative_dir} example IDs/order differ from prepared split rows"
        )
    expected_games = task.examples.iloc[expected_index]["game_id"].astype(str).to_numpy()
    actual_games = predictions["game_id"].astype(str).to_numpy()
    if not np.array_equal(actual_games, expected_games):
        raise ScientificValidationError(
            f"{key.relative_dir} saved game IDs/order differ from prepared split rows"
        )
    expected_strata = task.examples.iloc[expected_index]["stratum"].astype(str).to_numpy()
    actual_strata = predictions["stratum"].astype(str).to_numpy()
    if not np.array_equal(actual_strata, expected_strata):
        raise ScientificValidationError(
            f"{key.relative_dir} saved strata/order differ from prepared split rows"
        )
    expected_targets = pd.to_numeric(
        task.examples.iloc[expected_index]["target"], errors="coerce"
    ).to_numpy(dtype=float)
    actual_targets = _numeric_targets(predictions, "prediction table")
    _array_close(actual_targets, expected_targets, "saved targets", equal_nan=True)
    return calibration_index, test_index


def _required_cell(design: Mapping[str, Any], key: CellKey) -> Mapping[str, Any]:
    matches = [
        record
        for record in design.get("required_cells", [])
        if record.get("branch") == key.branch
        and record.get("repeat") == key.repeat
        and record.get("n_train") == key.n_train
        and record.get("model") == key.model
    ]
    if len(matches) != 1:
        raise ScientificValidationError(f"required-cell lookup is not unique for {key}")
    return matches[0]


def _validate_cell_binding(
    storage: Any,
    key: CellKey,
    metrics: Mapping[str, Any],
    history: Any,
    task: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray]:
    manifest = storage.manifest
    design = manifest["task_design"]
    profile = str(manifest.get("execution", {}).get("profile", ""))
    pilot_mode = profile == "pilot10"
    if pilot_mode and manifest.get("execution", {}).get("mode") != "pilot":
        raise ScientificValidationError("pilot10 aggregation requires pilot execution mode")
    cell = _required_cell(design, key)
    try:
        split = design["split_manifests"][key.repeat - 1]
    except (IndexError, TypeError) as exc:
        raise ScientificValidationError(f"no split manifest for repeat {key.repeat}") from exc
    if split.get("repeat") != key.repeat:
        raise ScientificValidationError("split repeat differs from cell key")
    for field in ("outer_split_hash", "nested_split_hash"):
        expected = split.get(field)
        if cell.get(field) != expected or metrics.get(field) != expected:
            raise ScientificValidationError(f"cell/metrics {field} differs from frozen split")
    spec = manifest["task_spec"]
    model_entry = spec.get("models", {}).get(key.model)
    if not isinstance(model_entry, Mapping):
        raise ScientificValidationError("cell model is absent from frozen TaskSpec")
    family = model_entry.get("family")
    config = model_entry.get("selected_config")
    seeds = cell.get("seeds")
    if not isinstance(config, Mapping) or not isinstance(seeds, Mapping):
        raise ScientificValidationError("cell lacks a frozen model configuration or seeds")
    expected_identity = {
        "task_id": task.task_id,
        "branch": key.branch,
        "ablation_id": cell.get("ablation_id"),
        "sensitivity_id": cell.get("sensitivity_id"),
        "repeat": key.repeat,
        "n_train": key.n_train,
        "model": key.model,
        "family": family,
    }
    for name, expected in expected_identity.items():
        if metrics.get(name) != expected:
            raise ScientificValidationError(f"metric identity {name!r} differs from cell key")
    if not _json_equal(metrics.get("model_config"), config):
        raise ScientificValidationError("saved metric model_config differs from frozen configuration")
    if not _json_equal(metrics.get("seeds"), seeds):
        raise ScientificValidationError("saved metric seeds differ from required-cell seeds")
    if not isinstance(history, Mapping):
        raise ScientificValidationError("cell training history is missing or is not an object")
    for name in ("branch", "ablation_id", "sensitivity_id"):
        if history.get(name) != expected_identity[name]:
            raise ScientificValidationError(
                f"history identity {name!r} differs from required cell"
            )
    if history.get("family") != family:
        raise ScientificValidationError("history family differs from frozen model family")
    if not _json_equal(history.get("model_config"), config):
        raise ScientificValidationError("history model_config differs from frozen configuration")
    if not _json_equal(history.get("seeds"), seeds):
        raise ScientificValidationError("history seeds differ from required-cell seeds")
    if history.get("uncertainty_subsample_seed") != seeds.get(
        "uncertainty_subsample"
    ):
        raise ScientificValidationError(
            "history uncertainty-subsample seed differs from required-cell seeds"
        )
    for field, expected in (
        ("train_game_ids", split["nested_train_game_ids"][str(key.n_train)]),
        ("calibration_game_ids", split["calibration_game_ids"]),
        ("test_game_ids", split["test_game_ids"]),
    ):
        if history.get(field) != expected:
            raise ScientificValidationError(f"history {field} differs from frozen split order")
    parameter_count = history.get("parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ScientificValidationError("history parameter_count is invalid")
    expected_parameter_definition = (
        NEURAL_PARAMETER_COUNT_DEFINITION
        if family in NEURAL_FAMILIES
        else CLASSICAL_PARAMETER_COUNT_DEFINITION
    )
    if history.get("parameter_count_definition") != expected_parameter_definition:
        raise ScientificValidationError("history parameter_count definition is invalid")
    preprocessing = history.get("preprocessing")
    expected_preprocessing_scope = (
        NEURAL_FINAL_PREPROCESSING_SCOPE
        if family in NEURAL_FAMILIES
        else CLASSICAL_PREPROCESSING_SCOPE
    )
    if (
        not isinstance(preprocessing, Mapping)
        or preprocessing.get("fit_scope") != expected_preprocessing_scope
    ):
        raise ScientificValidationError("final preprocessing is not training-subset bound")
    outcome = "binary" if task.outcome_type == "frame_event" else task.outcome_type
    effective_config = dict(config)
    if outcome == "distribution":
        effective_config.setdefault("output_contract", PUNT_CDF_RESIDUAL_CONTRACT)
    if family in {"relnet", "attn_relnet", "set_transformer"} and outcome == "binary":
        effective_config.setdefault("binary_loss", "log_loss")
    if family in {"relnet", "attn_relnet", "set_transformer"}:
        effective_config.setdefault(
            "representative_time_steps", int(task.player_tokens.shape[1])
        )
    if outcome == "trajectory":
        effective_config.setdefault("trajectory_target", "residual")
        effective_config.setdefault(
            "trajectory_decoder", "shared_horizon_conditioned_v1"
        )
    include_team_identity = (
        task.task_id == "bdb2025_man_zone"
        and cell.get("sensitivity_id") == "include_team_identity"
    )
    if include_team_identity:
        effective_config["include_team_identity"] = True
    if not _json_equal(history.get("effective_model_config"), effective_config):
        raise ScientificValidationError(
            "history effective_model_config differs from the branch intervention"
        )
    expected_team_identity = (
        "included_sensitivity" if include_team_identity else "excluded_primary"
    )
    if preprocessing.get("team_identity") != expected_team_identity:
        raise ScientificValidationError(
            "final preprocessing team-identity policy differs from the branch"
        )
    if family in NEURAL_FAMILIES:
        selector_preprocessing = history.get("selector_preprocessing")
        if not isinstance(selector_preprocessing, Mapping):
            raise ScientificValidationError("selector preprocessing is missing")
        for label, audit in (
            ("final", preprocessing),
            ("selector", selector_preprocessing),
        ):
            if audit.get("team_identity") != expected_team_identity:
                raise ScientificValidationError(
                    f"{label} preprocessing team-identity policy differs"
                )
            if audit.get("ablation_id") != cell.get("ablation_id"):
                raise ScientificValidationError(
                    f"{label} preprocessing ablation identity differs"
                )
            if audit.get("ablation_mask_applied_to_scaler") is not (
                cell.get("ablation_id") is not None
            ):
                raise ScientificValidationError(
                    f"{label} preprocessing did not apply the frozen ablation mask"
                )
    train_index = _partition_index(task, split["nested_train_game_ids"][str(key.n_train)])
    expected_train_examples = len(train_index)
    for name, expected in (
        ("n_train_examples", expected_train_examples),
        ("n_calibration_examples", len(_partition_index(task, split["calibration_game_ids"]))),
        ("n_test_examples", len(_partition_index(task, split["test_game_ids"]))),
    ):
        if isinstance(metrics.get(name), bool) or metrics.get(name) != int(expected):
            raise ScientificValidationError(f"{name} differs from frozen games")
    if "elapsed_seconds" in metrics:
        elapsed = metrics["elapsed_seconds"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ScientificValidationError("elapsed_seconds is invalid")
    if family in NEURAL_FAMILIES:
        _validate_neural_history(history, config, seeds, task, train_index)
    return split, cell, train_index


def _validate_neural_history(
    history: Mapping[str, Any],
    config: Mapping[str, Any],
    seeds: Mapping[str, Any],
    task: Any,
    train_index: np.ndarray,
) -> None:
    # Importing this helper does not initialize TensorFlow.
    from .models import grouped_validation_indices

    split_seed = seeds.get("validation_split")
    if history.get("validation_split_seed") != split_seed:
        raise ScientificValidationError("history validation split seed differs from cell seed")
    local_fit, local_validation = grouped_validation_indices(
        task.game_ids[train_index], int(split_seed), strata=task.strata[train_index]
    )
    selector_train = train_index[local_fit]
    selector_validation = train_index[local_validation]
    expected_validation_games = sorted(
        str(value) for value in np.unique(task.game_ids[selector_validation])
    )
    if history.get("validation_game_ids") != expected_validation_games:
        raise ScientificValidationError("history validation games differ from deterministic grouped split")
    expected_counts = {
        "selector_fit_games": len(np.unique(task.game_ids[selector_train])),
        "final_fit_games": len(np.unique(task.game_ids[train_index])),
        "final_fit_examples": len(train_index),
    }
    for field, expected in expected_counts.items():
        if history.get(field) != int(expected):
            raise ScientificValidationError(f"history {field} is incorrect")
    selector_preprocessing = history.get("selector_preprocessing")
    if (
        not isinstance(selector_preprocessing, Mapping)
        or selector_preprocessing.get("fit_scope")
        != NEURAL_SELECTOR_PREPROCESSING_SCOPE
    ):
        raise ScientificValidationError("selector preprocessing is not selector-training bound")
    selector_history = history.get("selector_history")
    refit_history = history.get("refit_history")
    if not isinstance(selector_history, Mapping) or not isinstance(refit_history, Mapping):
        raise ScientificValidationError("neural selector/refit histories are missing")
    try:
        validation_loss = np.asarray(selector_history["val_loss"], dtype=float)
        refit_loss = np.asarray(refit_history["loss"], dtype=float)
        best_epoch = int(history["best_epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScientificValidationError("neural epoch history is malformed") from exc
    maximum = int(config.get("max_epochs", 50))
    if (
        best_epoch < 1
        or len(validation_loss) < best_epoch
        or len(validation_loss) > maximum
        or len(refit_loss) != best_epoch
        or not np.all(np.isfinite(validation_loss))
        or not np.all(np.isfinite(refit_loss))
        or int(np.argmin(validation_loss)) + 1 != best_epoch
    ):
        raise ScientificValidationError("neural best epoch/refit history does not recompute")
    _validate_hashed_signature(
        history.get("representative_neural_input"),
        schema_version="bdb-representative-neural-input-v1",
        label="representative neural input signature",
    )
    _validate_hashed_signature(
        history.get("output_head_loss_signature"),
        schema_version="bdb-neural-head-loss-v1",
        label="neural output-head/loss signature",
    )


def _validate_matched_neural_history_pairs(
    storage: Any,
    keys: Sequence[CellKey],
) -> int:
    """Reassert the shared-neural and relational-pair gates for saved cells.

    RelNet and AttnRelNet must share their complete typed-edge input receipt and
    remain parameter/FLOP matched.  Whenever the prospective Global Set
    Transformer is present, all three neural roles must additionally share the
    token/history/mask/time/context receipt and the exact output-head/loss
    receipt.  The two-role path remains valid for immutable ``full50`` cells
    and for the prespecified relational-only structural ablations.
    """

    models = storage.manifest.get("task_spec", {}).get("models", {})
    if not isinstance(models, Mapping):
        raise ScientificValidationError("frozen model panel is missing")
    model_by_family: dict[str, str] = {}
    for model_id, entry in models.items():
        if not isinstance(entry, Mapping):
            continue
        family = str(entry.get("family"))
        if family in {"relnet", "attn_relnet", "set_transformer"}:
            if family in model_by_family:
                raise ScientificValidationError(
                    f"frozen model panel has more than one {family} role"
                )
            model_by_family[family] = str(model_id)
    if not {"relnet", "attn_relnet"}.issubset(model_by_family):
        raise ScientificValidationError("frozen model panel lacks its matched neural pair")

    neural_ids = set(model_by_family.values())
    grouped: dict[tuple[str, int, int], dict[str, CellKey]] = {}
    for key in keys:
        if key.model not in neural_ids:
            continue
        family = str(models[key.model]["family"])
        grouped.setdefault((key.branch, key.repeat, key.n_train), {})[family] = key

    profile = str(
        storage.manifest.get("execution", {}).get(
            "profile",
            storage.manifest.get("task_design", {}).get("profile", ""),
        )
    )
    prospective_triplet = "set_transformer" in model_by_family and profile != "full50"
    validated = 0
    for identity, pair in sorted(grouped.items()):
        expected_families = {"relnet", "attn_relnet"}
        if prospective_triplet and identity[0] != "structural_ablation":
            expected_families.add("set_transformer")
        if set(pair) != expected_families:
            raise ScientificValidationError(
                f"neural history group {identity} has families {sorted(pair)}; "
                f"expected {sorted(expected_families)}"
            )
        relnet_history = storage.load_history(pair["relnet"])
        attention_history = storage.load_history(pair["attn_relnet"])
        if not isinstance(relnet_history, Mapping) or not isinstance(
            attention_history, Mapping
        ):
            raise ScientificValidationError(
                f"matched neural histories are missing for {identity}"
            )
        for field in ("representative_neural_input", "output_head_loss_signature"):
            if not _json_equal(relnet_history.get(field), attention_history.get(field)):
                raise ScientificValidationError(
                    f"RelNet and AttnRelNet {field} differ for {identity}"
                )
        histories: dict[str, Mapping[str, Any]] = {
            "relnet": relnet_history,
            "attn_relnet": attention_history,
        }
        if "set_transformer" in pair:
            set_history = storage.load_history(pair["set_transformer"])
            if not isinstance(set_history, Mapping):
                raise ScientificValidationError(
                    f"Global Set Transformer history is missing for {identity}"
                )
            histories["set_transformer"] = set_history
            for history in histories.values():
                _validate_hashed_signature(
                    history.get("representative_shared_neural_input"),
                    schema_version="bdb-representative-neural-input-v1",
                    label="shared neural input signature",
                )
                _validate_hashed_signature(
                    history.get("output_head_loss_signature"),
                    schema_version="bdb-neural-head-loss-v1",
                    label="shared neural output-head/loss signature",
                )
            for field in (
                "representative_shared_neural_input",
                "output_head_loss_signature",
            ):
                baseline = relnet_history.get(field)
                if baseline is None:
                    raise ScientificValidationError(
                        f"shared neural {field} is missing for {identity}"
                    )
                if any(
                    not _json_equal(baseline, history.get(field))
                    for history in histories.values()
                ):
                    raise ScientificValidationError(
                        f"shared neural {field} differs for {identity}"
                    )
            architecture = set_history.get("global_set_architecture")
            expected_architecture = {
                "architecture_id": "bdb_global_set_transformer_v1",
                "inputs": "task_tokens_player_frame_masks_time_context",
                "attention_contract": "global_set_time_attention_v1",
                "relation_scope": "global_masked_all_player_self_attention",
                "typed_graph_edges_consumed": False,
                "temporal_encoder": "factorized_masked_temporal_attention",
                "protocol_note": (
                    "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot"
                ),
                "parameter_cap": 350_000,
            }
            if not isinstance(architecture, Mapping) or any(
                architecture.get(field) != expected
                for field, expected in expected_architecture.items()
            ):
                raise ScientificValidationError(
                    f"Global Set Transformer architecture receipt drifted for {identity}"
                )
            if set(architecture) != {
                *expected_architecture,
                "parameter_count",
                "representative_forward_flops",
            }:
                raise ScientificValidationError(
                    f"Global Set Transformer architecture receipt fields differ for {identity}"
                )
            for field in ("parameter_count", "representative_forward_flops"):
                value = architecture.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ScientificValidationError(
                        f"Global Set Transformer architecture {field} is invalid"
                    )
            if architecture.get("parameter_count") != set_history.get(
                "parameter_count"
            ) or architecture.get("representative_forward_flops") != set_history.get(
                "representative_forward_flops"
            ):
                raise ScientificValidationError(
                    f"Global Set Transformer complexity receipt is internally inconsistent for {identity}"
                )
        counts: dict[str, tuple[int, int]] = {}
        for family, history in histories.items():
            parameter_count = history.get("parameter_count")
            forward_flops = history.get("representative_forward_flops")
            if (
                isinstance(parameter_count, bool)
                or not isinstance(parameter_count, int)
                or parameter_count <= 0
                or isinstance(forward_flops, bool)
                or not isinstance(forward_flops, int)
                or forward_flops <= 0
            ):
                raise ScientificValidationError(
                    f"{family} matched-neural complexity receipt is invalid"
                )
            if parameter_count > 350_000:
                raise ScientificValidationError(
                    f"{family} exceeds 350000 parameters for {identity}"
                )
            counts[family] = (parameter_count, forward_flops)
        relnet_parameters, relnet_flops = counts["relnet"]
        attention_parameters, attention_flops = counts["attn_relnet"]
        parameter_gap = abs(relnet_parameters - attention_parameters) / max(
            relnet_parameters, attention_parameters
        )
        flop_gap = abs(relnet_flops - attention_flops) / max(
            relnet_flops, attention_flops
        )
        if parameter_gap > 0.05 or flop_gap > 0.15:
            raise ScientificValidationError(
                f"matched neural complexity tolerances fail for {identity}"
            )
        validated += 1
    return validated


def _test_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.loc[predictions["partition"].eq("test")].reset_index(drop=True)


def _calibration_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.loc[predictions["partition"].eq("calibration")].reset_index(drop=True)


def _game_equal_quantile(
    values: np.ndarray, game_ids: np.ndarray, quantile: float
) -> float:
    observed = np.asarray(values, dtype=float).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    if len(observed) != len(games) or not 0.0 <= float(quantile) <= 1.0:
        raise ScientificValidationError("game-equal quantile inputs are invalid")
    return float(
        np.mean(
            [
                np.quantile(observed[games == game], quantile)
                for game in np.unique(games)
            ]
        )
    )


def _game_equal_label_mean(
    values: np.ndarray,
    labels: np.ndarray,
    game_ids: np.ndarray,
    *,
    label: int,
) -> float:
    observed = np.asarray(values, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    if not (len(observed) == len(y) == len(games)):
        raise ScientificValidationError("labelwise game-equal inputs are misaligned")
    per_game = [
        float(np.mean(observed[(games == game) & (y == int(label))]))
        for game in np.unique(games)
        if np.any((games == game) & (y == int(label)))
    ]
    if not per_game:
        raise ScientificValidationError(f"no test game contains binary label {label}")
    return float(np.mean(per_game))


def _game_equal_reliability(
    probability: np.ndarray,
    labels: np.ndarray,
    game_ids: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    bins = np.searchsorted(np.asarray(boundaries, dtype=float), probability, side="right")
    if not (len(probability) == len(labels) == len(games)):
        raise ScientificValidationError("game-equal reliability inputs are misaligned")
    per_game: list[float] = []
    for game in np.unique(games):
        selected_game = games == game
        squared = 0.0
        count = 0
        for value in np.unique(bins[selected_game]):
            selected = selected_game & (bins == value)
            n_bin = int(np.sum(selected))
            difference = float(
                np.mean(labels[selected]) - np.mean(probability[selected])
            )
            squared += n_bin * difference**2
            count += n_bin
        if count <= 0:
            raise ScientificValidationError("a test game has no reliability rows")
        per_game.append(math.sqrt(squared / count))
    return float(np.mean(per_game))


def _validate_binary(
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    task: Any,
    train_index: np.ndarray,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    required = {
        "raw_probability", "calibrated_probability", "venn_abers_probability",
        "venn_abers_p0", "venn_abers_p1", "venn_abers_imprecision",
        "set_includes_0", "set_includes_1", "conformal_set_size",
        "loss_contribution",
    }
    if task.task_id == "bdb2024_tackle":
        required.add("raw_case_control_score")
    missing = required - set(predictions.columns)
    if missing:
        raise ScientificValidationError(f"binary prediction artifacts lack {sorted(missing)}")
    calibration = _calibration_frame(predictions)
    test = _test_frame(predictions)
    calibration_probability = validate_binary_probability(
        calibration["raw_probability"].to_numpy(dtype=float), "calibration raw probability"
    )
    raw_probability = validate_binary_probability(
        test["raw_probability"].to_numpy(dtype=float), "test raw probability"
    )
    train_y = np.asarray(task.y[train_index], dtype=int)
    calibration_y = np.asarray(task.y[calibration_index], dtype=int)
    test_y = np.asarray(task.y[test_index], dtype=int)
    _array_close(_numeric_targets(calibration, "calibration predictions"), calibration_y, "binary calibration labels")
    _array_close(_numeric_targets(test, "test predictions"), test_y, "binary test labels")
    recomputed_joint = fit_venn_abers(
        calibration_probability,
        calibration_y,
        np.concatenate([calibration_probability, raw_probability]),
    )
    calibration_count = len(calibration_probability)
    calibration_calibrated_probability = (
        recomputed_joint.calibrated_probability[:calibration_count]
    )
    recomputed_calibrated_probability = (
        recomputed_joint.calibrated_probability[calibration_count:]
    )
    recomputed_p0 = recomputed_joint.p0[calibration_count:]
    recomputed_p1 = recomputed_joint.p1[calibration_count:]
    recomputed_imprecision = recomputed_joint.imprecision[calibration_count:]
    uncertainty_seed = _uncertainty_subsample_seed(metrics, arrays)
    recomputed_sets = label_conditional_prediction_sets(
        calibration_probability,
        calibration_y,
        task.game_ids[calibration_index],
        raw_probability,
        seed=uncertainty_seed,
    )
    expected_va = {
        "calibrated_probability": recomputed_calibrated_probability,
        "venn_abers_probability": recomputed_calibrated_probability,
        "venn_abers_p0": recomputed_p0,
        "venn_abers_p1": recomputed_p1,
        "venn_abers_imprecision": recomputed_imprecision,
    }
    for column, expected in expected_va.items():
        values = validate_binary_probability(test[column].to_numpy(dtype=float), column)
        _array_close(values, expected, column)
    for column in expected_va:
        if not calibration[column].isna().all():
            raise ScientificValidationError(f"calibration rows must not contain post-fit {column}")
    included = np.asarray(recomputed_sets["included"], dtype=bool)
    set_size = np.asarray(recomputed_sets["set_size"], dtype=int)
    if not np.array_equal(_boolean_vector(test["set_includes_0"], "set_includes_0"), included[:, 0]):
        raise ScientificValidationError("label-0 conformal set membership differs")
    if not np.array_equal(_boolean_vector(test["set_includes_1"], "set_includes_1"), included[:, 1]):
        raise ScientificValidationError("label-1 conformal set membership differs")
    _array_close(test["conformal_set_size"].to_numpy(dtype=int), set_size, "binary conformal set size")
    for column in ("set_includes_0", "set_includes_1", "conformal_set_size"):
        if not calibration[column].isna().all():
            raise ScientificValidationError(f"calibration rows must not contain post-fit {column}")
    expected_arrays = {
        "binary_calibration_calibrated_probability": np.asarray(
            calibration_calibrated_probability, dtype=float
        ),
        "binary_set_quantiles": np.asarray(recomputed_sets["quantiles"], dtype=float),
        "binary_set_ranks": np.asarray(recomputed_sets["ranks"], dtype=int),
        "binary_set_label0_calibration_indices": np.asarray(
            recomputed_sets["selected_label0_indices"], dtype=int
        ),
        "binary_set_label1_calibration_indices": np.asarray(
            recomputed_sets["selected_label1_indices"], dtype=int
        ),
    }
    for name, expected in expected_arrays.items():
        if name not in arrays:
            raise ScientificValidationError(f"binary arrays lack {name}")
        _array_close(arrays[name], expected, name)
    if task.task_id == "bdb2024_tackle":
        _array_close(test["raw_case_control_score"].to_numpy(dtype=float), raw_probability, "raw case-control score")
        if not calibration["raw_case_control_score"].isna().all():
            raise ScientificValidationError("calibration rows must not contain a case-control test score")
    raw_contribution = brier_contributions(test_y, raw_probability)
    calibrated_contribution = brier_contributions(
        test_y, recomputed_calibrated_probability
    )
    raw_log_loss = binary_log_loss_contributions(test_y, raw_probability)
    calibrated_log_loss = binary_log_loss_contributions(
        test_y, recomputed_calibrated_probability
    )
    reliability, reliability_boundaries = rms_reliability_error(
        calibration_calibrated_probability,
        recomputed_calibrated_probability,
        test_y,
    )
    if "binary_reliability_boundaries" not in arrays:
        raise ScientificValidationError("binary arrays lack reliability boundaries")
    _array_close(
        arrays["binary_reliability_boundaries"],
        reliability_boundaries,
        "binary reliability boundaries",
    )
    null_probability = prevalence_null(train_y)
    null_contribution = brier_contributions(
        test_y, np.full(len(test_y), null_probability, dtype=float)
    )
    _array_close(test["loss_contribution"].to_numpy(dtype=float), raw_contribution, "binary per-example Brier")
    if not calibration["loss_contribution"].isna().all():
        raise ScientificValidationError("calibration rows must not contain test loss contributions")
    primary = float(raw_contribution.mean())
    null_loss = float(null_contribution.mean())
    game_equal = game_equal_mean(raw_contribution, task.game_ids[test_index])
    test_games = task.game_ids[test_index]
    entropy = bernoulli_entropy(recomputed_calibrated_probability)
    label0 = test_y == 0
    label1 = test_y == 1
    if not np.any(label0) or not np.any(label1):
        raise ScientificValidationError("binary test partition lacks one class")
    expected_metrics = {
        "primary_loss": primary,
        "null_loss": null_loss,
        "game_equal_loss": game_equal,
        "game_equal_brier": game_equal,
        "raw_brier": primary,
        "calibrated_brier": float(calibrated_contribution.mean()),
        "game_equal_calibrated_brier": game_equal_mean(
            calibrated_contribution, task.game_ids[test_index]
        ),
        "raw_log_loss": float(raw_log_loss.mean()),
        "calibrated_log_loss": float(calibrated_log_loss.mean()),
        "game_equal_raw_log_loss": game_equal_mean(
            raw_log_loss, test_games
        ),
        "game_equal_calibrated_log_loss": game_equal_mean(
            calibrated_log_loss, test_games
        ),
        "calibration_bias": calibration_bias(
            test_y, recomputed_calibrated_probability
        ),
        "game_equal_calibration_bias": game_equal_mean(
            test_y - recomputed_calibrated_probability, test_games
        ),
        "rms_reliability": float(reliability),
        "game_equal_rms_reliability": _game_equal_reliability(
            recomputed_calibrated_probability,
            test_y,
            test_games,
            reliability_boundaries,
        ),
        "mean_entropy": float(entropy.mean()),
        "game_equal_mean_entropy": game_equal_mean(entropy, test_games),
        "null_probability": null_probability,
        "mean_va_imprecision": float(recomputed_imprecision.mean()),
        "game_equal_mean_va_imprecision": game_equal_mean(
            recomputed_imprecision, test_games
        ),
        "p90_va_imprecision": float(np.quantile(recomputed_imprecision, 0.90)),
        "game_equal_p90_va_imprecision": _game_equal_quantile(
            recomputed_imprecision, test_games, 0.90
        ),
        "label0_set_coverage": float(included[label0, 0].mean()),
        "label1_set_coverage": float(included[label1, 1].mean()),
        "game_equal_label0_set_coverage": _game_equal_label_mean(
            included[:, 0], test_y, test_games, label=0
        ),
        "game_equal_label1_set_coverage": _game_equal_label_mean(
            included[:, 1], test_y, test_games, label=1
        ),
        "set_singleton_rate": float(np.mean(set_size == 1)),
        "set_doubleton_rate": float(np.mean(set_size == 2)),
        "set_empty_rate": float(np.mean(set_size == 0)),
        "game_equal_set_singleton_rate": game_equal_mean(
            (set_size == 1).astype(float), test_games
        ),
        "game_equal_set_doubleton_rate": game_equal_mean(
            (set_size == 2).astype(float), test_games
        ),
        "game_equal_set_empty_rate": game_equal_mean(
            (set_size == 0).astype(float), test_games
        ),
        "uncertainty_subsample_seed": float(uncertainty_seed),
        "skill": skill_score(primary, null_loss),
    }
    for name, expected in expected_metrics.items():
        _metric_close(metrics, name, expected)


def _boolean_vector(values: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(numeric)) or np.any(~np.isin(numeric, [0.0, 1.0])):
        raise ScientificValidationError(f"{label} is not boolean")
    return numeric.astype(bool)


def _validate_distribution(
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    task: Any,
    train_index: np.ndarray,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    required_columns = {
        "loss_contribution", "interval_lower_index", "interval_upper_index",
        "interval_lower", "interval_upper", "conformal_padding", "covered",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ScientificValidationError(
            f"distribution prediction artifacts lack {sorted(missing_columns)}"
        )
    required_arrays = {
        "calibration_probabilities",
        "test_probabilities",
        "null_probability",
        "conformal_selected_calibration_indices",
        "conformal_quantile",
        "conformal_rank",
    }
    missing_arrays = required_arrays - set(arrays)
    if missing_arrays:
        raise ScientificValidationError(f"distribution arrays lack {sorted(missing_arrays)}")
    support = np.asarray(task.support, dtype=float)
    if support.ndim != 1 or len(support) < 2:
        raise ScientificValidationError("prepared distribution support is invalid")
    calibration_probability = validate_probability_matrix(
        arrays["calibration_probabilities"], len(support)
    )
    test_probability = validate_probability_matrix(arrays["test_probabilities"], len(support))
    if len(calibration_probability) != len(calibration_index) or len(test_probability) != len(test_index):
        raise ScientificValidationError("distribution probability arrays are misaligned")
    train_y = np.asarray(task.y[train_index], dtype=int)
    calibration_y = np.asarray(task.y[calibration_index], dtype=int)
    test_y = np.asarray(task.y[test_index], dtype=int)
    calibration = _calibration_frame(predictions)
    test = _test_frame(predictions)
    _array_close(_numeric_targets(calibration, "calibration predictions"), support[calibration_y], "distribution calibration targets")
    _array_close(_numeric_targets(test, "test predictions"), support[test_y], "distribution test targets")
    contribution = crps_contributions(test_y, test_probability)
    null_probability = empirical_distribution_null(train_y, len(support))
    _array_close(arrays["null_probability"], null_probability, "empirical null distribution")
    null_contribution = crps_contributions(
        test_y, np.repeat(null_probability[None, :], len(test_y), axis=0)
    )
    uncertainty_seed = _uncertainty_subsample_seed(metrics, arrays)
    intervals = hierarchical_distribution_intervals(
        calibration_probability,
        calibration_y,
        task.game_ids[calibration_index],
        test_probability,
        alpha=0.10,
        seed=uncertainty_seed,
    )
    _array_close(
        arrays["conformal_selected_calibration_indices"],
        intervals["selected_calibration_indices"],
        "ordered conformal calibration registry",
    )
    _array_close(
        arrays["conformal_quantile"],
        np.asarray([intervals["quantile"]]),
        "ordered conformal quantile",
    )
    _array_close(
        arrays["conformal_rank"],
        np.asarray([intervals["rank"]]),
        "ordered conformal rank",
    )
    covered = (test_y >= intervals["lower"]) & (test_y <= intervals["upper"])
    expected_columns = {
        "loss_contribution": contribution,
        "interval_lower_index": intervals["lower"],
        "interval_upper_index": intervals["upper"],
        "interval_lower": support[intervals["lower"]],
        "interval_upper": support[intervals["upper"]],
        "conformal_padding": intervals["padding"],
    }
    for column, expected in expected_columns.items():
        _array_close(test[column].to_numpy(dtype=float), expected, column)
    if not np.array_equal(_boolean_vector(test["covered"], "covered"), covered):
        raise ScientificValidationError("saved conformal coverage indicators differ")
    for column in required_columns:
        if not calibration[column].isna().all():
            raise ScientificValidationError(f"calibration rows must not contain test-only {column}")
    width = intervals["width"].astype(float)
    primary = float(contribution.mean())
    null_loss = float(null_contribution.mean())
    game_equal = game_equal_mean(contribution, task.game_ids[test_index])
    coverage_game_equal = game_equal_mean(
        covered.astype(float), task.game_ids[test_index]
    )
    width_game_equal = game_equal_mean(width, task.game_ids[test_index])
    expected_metrics = {
        "primary_loss": primary,
        "null_loss": null_loss,
        "game_equal_loss": game_equal,
        "game_equal_crps": game_equal,
        "crps": primary,
        "raw_crps": primary,
        "coverage": coverage_game_equal,
        "coverage_game_equal": coverage_game_equal,
        "coverage_example_weighted": float(covered.mean()),
        "interval_width": width_game_equal,
        "width_game_equal": width_game_equal,
        "interval_width_example_weighted": float(width.mean()),
        "inclusive_class_count": float(np.mean(width + 1.0)),
        "interval_width_sd": float(width.std(ddof=1)) if len(width) > 1 else 0.0,
        "uncertainty_subsample_seed": float(uncertainty_seed),
        "skill": skill_score(primary, null_loss),
    }
    for name, expected in expected_metrics.items():
        _metric_close(metrics, name, expected)


def _game_equal_trajectory_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    game_ids: np.ndarray,
) -> float:
    values = []
    for game_id in np.unique(game_ids):
        selected = game_ids == game_id
        values.append(trajectory_rmse(truth[selected], prediction[selected], mask[selected]))
    return float(np.mean(values))


def _validate_trajectory(
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    task: Any,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    required_arrays = {
        "calibration_prediction", "calibration_target", "calibration_mask",
        "test_prediction", "test_predicted_residual", "test_target", "test_baseline",
        "test_mask", "horizon_rmse", "path_tube_horizon_scale", "path_tube_radius",
        "path_tube_field_bounds", "path_tube_quantile", "path_tube_rank",
        "path_tube_selected_calibration_indices",
    }
    missing = required_arrays - set(arrays)
    if missing:
        raise ScientificValidationError(f"trajectory arrays lack {sorted(missing)}")
    required_columns = {
        "loss_contribution", "path_rmse", "path_covered", "mean_tube_diameter"
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ScientificValidationError(
            f"trajectory prediction table lacks {sorted(missing_columns)}"
        )
    calibration = _calibration_frame(predictions)
    test = _test_frame(predictions)
    if not calibration["loss_contribution"].isna().all():
        raise ScientificValidationError("calibration rows must not contain test loss contributions")
    calibration_target = np.asarray(task.target_values[calibration_index])
    calibration_mask = np.asarray(task.target_mask[calibration_index], dtype=bool)
    test_target = np.asarray(task.target_values[test_index])
    test_baseline = np.asarray(task.target_baseline[test_index])
    test_mask = np.asarray(task.target_mask[test_index], dtype=bool)
    _array_close(arrays["calibration_target"], calibration_target, "trajectory calibration targets", equal_nan=True)
    if not np.array_equal(np.asarray(arrays["calibration_mask"], dtype=bool), calibration_mask):
        raise ScientificValidationError("trajectory calibration mask differs from prepared data")
    if np.asarray(arrays["calibration_prediction"]).shape != calibration_target.shape:
        raise ScientificValidationError("trajectory calibration predictions are misaligned")
    _array_close(arrays["test_target"], test_target, "trajectory test targets", equal_nan=True)
    _array_close(arrays["test_baseline"], test_baseline, "trajectory constant-velocity baseline", equal_nan=True)
    if not np.array_equal(np.asarray(arrays["test_mask"], dtype=bool), test_mask):
        raise ScientificValidationError("trajectory test mask differs from prepared data")
    absolute = np.asarray(arrays["test_prediction"], dtype=float)
    residual = np.asarray(arrays["test_predicted_residual"], dtype=float)
    if absolute.shape != test_target.shape or residual.shape != test_target.shape:
        raise ScientificValidationError("trajectory predictions/residuals are misaligned")
    _array_close(absolute, test_baseline + residual, "trajectory residual reconstruction", equal_nan=True)
    contribution = trajectory_squared_error(test_target, absolute, test_mask)
    _array_close(test["loss_contribution"].to_numpy(dtype=float), contribution, "trajectory per-example squared error")
    path_rmse = trajectory_path_rmse(test_target, absolute, test_mask)
    _array_close(test["path_rmse"].to_numpy(dtype=float), path_rmse, "trajectory path RMSE")
    uncertainty_seed = _uncertainty_subsample_seed(metrics, arrays)
    tube = pathwise_conformal_tube(
        calibration_target,
        np.asarray(arrays["calibration_prediction"], dtype=float),
        calibration_mask,
        task.game_ids[calibration_index],
        test_target,
        absolute,
        test_mask,
        np.asarray(arrays["path_tube_horizon_scale"], dtype=float),
        field_bounds=tuple(
            np.asarray(arrays["path_tube_field_bounds"], dtype=float).tolist()
        ),
        seed=uncertainty_seed,
    )
    _array_close(
        arrays["path_tube_field_bounds"],
        tube["field_bounds"],
        "trajectory tube legal field bounds",
    )
    if not np.array_equal(
        _boolean_vector(test["path_covered"], "path_covered"),
        np.asarray(tube["path_covered"], dtype=bool),
    ):
        raise ScientificValidationError("trajectory path coverage differs")
    _array_close(
        test["mean_tube_diameter"].to_numpy(dtype=float),
        tube["per_path_mean_diameter"],
        "trajectory tube diameter",
    )
    _array_close(arrays["path_tube_radius"], tube["radius"], "trajectory tube radius")
    _array_close(
        arrays["path_tube_quantile"],
        np.asarray([tube["quantile"]]),
        "trajectory tube quantile",
    )
    _array_close(
        arrays["path_tube_rank"], np.asarray([tube["rank"]]), "trajectory tube rank"
    )
    _array_close(
        arrays["path_tube_selected_calibration_indices"],
        tube["selected_calibration_indices"],
        "trajectory tube calibration registry",
    )
    primary = trajectory_rmse(test_target, absolute, test_mask)
    null_loss = trajectory_rmse(test_target, test_baseline, test_mask)
    game_equal = _game_equal_trajectory_rmse(
        test_target, absolute, test_mask, task.game_ids[test_index]
    )
    horizon = []
    for step in range(test_target.shape[1]):
        valid = test_mask[:, step]
        if not np.any(valid):
            horizon.append(np.nan)
        else:
            error = absolute[valid, step] - test_target[valid, step]
            horizon.append(float(np.sqrt(np.mean(np.square(error)))))
    _array_close(arrays["horizon_rmse"], np.asarray(horizon), "trajectory horizon RMSE", equal_nan=True)
    expected_metrics = {
        "primary_loss": primary,
        "null_loss": null_loss,
        "game_equal_loss": game_equal,
        "rmse": primary,
        "official_pooled_rmse": primary,
        "game_equal_rmse": game_equal,
        "path_equal_rmse": float(path_rmse.mean()),
        "path_coverage": float(np.asarray(tube["path_covered"], dtype=bool).mean()),
        "game_equal_path_coverage": game_equal_mean(
            np.asarray(tube["path_covered"], dtype=float), task.game_ids[test_index]
        ),
        "mean_tube_diameter": float(
            np.asarray(tube["per_path_mean_diameter"], dtype=float).mean()
        ),
        "game_equal_tube_diameter": game_equal_mean(
            np.asarray(tube["per_path_mean_diameter"], dtype=float),
            task.game_ids[test_index],
        ),
        "uncertainty_subsample_seed": float(uncertainty_seed),
        "skill": skill_score(primary, null_loss),
    }
    for name, expected in expected_metrics.items():
        _metric_close(metrics, name, expected)


def _declared_names(artifacts: Mapping[str, Any], section: str) -> tuple[str, ...]:
    value = artifacts.get(section)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScientificValidationError(f"TaskSpec artifacts.{section} must be a field-name array")
    names = tuple(value)
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ScientificValidationError(f"TaskSpec artifacts.{section} contains invalid fields")
    return names


def _validate_declared_artifact_schema(
    spec: Mapping[str, Any],
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    outcome: str,
) -> None:
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScientificValidationError("TaskSpec has no artifact schema")
    declared_metrics = _declared_names(artifacts, "metrics")
    declared_predictions = _declared_names(artifacts, "predictions")
    metric_aliases = {"horizon_rmse": "array:horizon_rmse"}
    prediction_aliases = {
        "class_probabilities": "array:test_probabilities",
        "per_example_crps": "column:loss_contribution",
        "constant_velocity_baseline": "array:test_baseline",
        "predicted_residual": "array:test_predicted_residual",
        "absolute_xy": "array:test_prediction",
        "target_mask": "array:test_mask",
        "path_tube_radius": "array:path_tube_radius",
        "path_tube_horizon_scale": "array:path_tube_horizon_scale",
        "path_tube_field_bounds": "array:path_tube_field_bounds",
    }
    for name in declared_metrics:
        location = metric_aliases.get(name, f"metric:{name}")
        kind, physical = location.split(":", 1)
        available = physical in (arrays if kind == "array" else metrics)
        if not available:
            raise ScientificValidationError(f"declared metric artifact {name!r} is missing")
    for name in declared_predictions:
        location = prediction_aliases.get(name, f"column:{name}")
        kind, physical = location.split(":", 1)
        available = physical in (arrays if kind == "array" else predictions.columns)
        if not available:
            raise ScientificValidationError(f"declared prediction artifact {name!r} is missing")
    allowed_by_outcome = {
        "binary": {
            "raw_probability", "raw_case_control_score", "calibrated_probability",
            "venn_abers_probability", "venn_abers_p0", "venn_abers_p1",
            "venn_abers_imprecision", "set_includes_0", "set_includes_1",
            "conformal_set_size", "loss_contribution",
        },
        "distribution": {
            "class_probabilities", "interval_lower", "interval_upper",
            "per_example_crps", "loss_contribution", "conformal_padding", "covered",
        },
        "trajectory": {
            "constant_velocity_baseline", "predicted_residual", "absolute_xy", "target_mask",
            "path_tube_radius", "path_tube_horizon_scale", "path_rmse",
            "path_tube_field_bounds",
            "path_covered", "mean_tube_diameter", "loss_contribution",
        },
    }
    unknown = set(declared_predictions) - allowed_by_outcome[outcome]
    if unknown:
        raise ScientificValidationError(
            f"declared prediction fields do not belong to {outcome}: {sorted(unknown)}"
        )


def validate_cell_scientifically(storage: Any, key: CellKey, task: Any) -> dict[str, Any]:
    metrics = storage.load_metrics(key)
    predictions = storage.load_predictions(key)
    history = storage.load_history(key)
    arrays = storage.load_arrays(key) or {}
    if not isinstance(metrics, Mapping):
        raise ScientificValidationError("cell metrics must be an object")
    if not isinstance(predictions, pd.DataFrame):
        raise ScientificValidationError("cell identity predictions must be a table")
    if not isinstance(arrays, Mapping):
        raise ScientificValidationError("cell arrays artifact must be an object")
    split, _cell, train_index = _validate_cell_binding(
        storage, key, metrics, history, task
    )
    calibration_index, test_index = _validate_prediction_identity(
        predictions, split, task, key
    )
    outcome = "binary" if task.outcome_type == "frame_event" else task.outcome_type
    if outcome == "binary":
        _validate_binary(
            metrics, predictions, arrays, task, train_index, calibration_index, test_index
        )
    elif outcome == "distribution":
        _validate_distribution(
            metrics, predictions, arrays, task, train_index, calibration_index, test_index
        )
    elif outcome == "trajectory":
        _validate_trajectory(
            metrics, predictions, arrays, task, calibration_index, test_index
        )
    else:
        raise ScientificValidationError(f"unsupported outcome {outcome}")
    _validate_declared_artifact_schema(
        storage.manifest["task_spec"], metrics, predictions, arrays, outcome
    )
    return dict(metrics)


def _reject_unexpected_cell_directories(storage: Any, keys: Sequence[CellKey]) -> None:
    cells_root = Path(storage.run_dir) / "cells"
    if not cells_root.exists():
        return
    expected = {key.relative_dir for key in keys}
    discovered = {
        path.relative_to(storage.run_dir)
        for path in cells_root.glob("*/*/*/*")
        if path.is_dir()
    }
    unexpected = sorted(discovered - expected, key=lambda path: path.as_posix())
    if unexpected:
        raise ScientificValidationError(
            f"run contains {len(unexpected)} unexpected cell directories: "
            f"{[path.as_posix() for path in unexpected[:10]]}"
        )
    for marker in cells_root.rglob("_SUCCESS"):
        relative = marker.parent.relative_to(storage.run_dir)
        if relative not in expected:
            raise ScientificValidationError(f"unexpected committed cell marker: {relative}")


_SECONDARY_METRICS = (
    "primary_loss",
    "game_equal_loss",
    "skill",
    "coverage",
    "interval_width",
    "calibrated_brier",
    "calibrated_log_loss",
    "game_equal_calibrated_brier",
    "game_equal_calibrated_log_loss",
    "calibration_bias",
    "game_equal_calibration_bias",
    "rms_reliability",
    "game_equal_rms_reliability",
    "mean_entropy",
    "game_equal_mean_entropy",
    "mean_va_imprecision",
    "game_equal_mean_va_imprecision",
    "p90_va_imprecision",
    "game_equal_p90_va_imprecision",
    "label0_set_coverage",
    "label1_set_coverage",
    "game_equal_label0_set_coverage",
    "game_equal_label1_set_coverage",
    "set_singleton_rate",
    "set_doubleton_rate",
    "set_empty_rate",
    "game_equal_set_singleton_rate",
    "game_equal_set_doubleton_rate",
    "game_equal_set_empty_rate",
    "path_coverage",
    "game_equal_path_coverage",
    "mean_tube_diameter",
    "game_equal_tube_diameter",
)


def _secondary_summary(
    frame: pd.DataFrame,
    *,
    identity_column: str,
) -> pd.DataFrame:
    """Describe a secondary branch without entering the primary max-t family."""

    required = {"task_id", "branch", identity_column, "repeat", "model", "n_train"}
    missing = required - set(frame.columns)
    if missing:
        raise ScientificValidationError(
            f"secondary branch lacks identity columns {sorted(missing)}"
        )
    metrics = [name for name in _SECONDARY_METRICS if name in frame.columns]
    if not metrics:
        raise ScientificValidationError("secondary branch has no analyzable metrics")
    rows: list[dict[str, Any]] = []
    group_columns = ["task_id", "branch", identity_column, "model", "n_train"]
    for identity, group in frame.groupby(group_columns, sort=True, dropna=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            if len(values) == 0 or not np.all(np.isfinite(values)):
                raise ScientificValidationError(
                    f"secondary metric {metric!r} is empty or non-finite"
                )
            rows.append(
                {
                    **dict(zip(group_columns, identity, strict=True)),
                    "metric": metric,
                    "repeats": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "q10": float(np.quantile(values, 0.10)),
                    "q90": float(np.quantile(values, 0.90)),
                    "inferential_family": "secondary_descriptive_only",
                }
            )
    return pd.DataFrame(rows).sort_values(
        [identity_column, "metric", "model", "n_train"]
    ).reset_index(drop=True)


def _paired_secondary_effects(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    identity_column: str,
) -> pd.DataFrame:
    """Pair a secondary intervention with its fixed-main counterpart."""

    metrics = [
        name
        for name in _SECONDARY_METRICS
        if name in primary.columns and name in secondary.columns
    ]
    rows: list[dict[str, Any]] = []
    for identity_value, identity_frame in secondary.groupby(
        identity_column, sort=True, dropna=False
    ):
        for (model, anchor), group in identity_frame.groupby(
            ["model", "n_train"], sort=True
        ):
            baseline = primary.loc[
                (primary["model"].astype(str) == str(model))
                & (primary["n_train"].astype(int) == int(anchor))
                & primary["repeat"].astype(int).isin(group["repeat"].astype(int))
            ]
            for metric in metrics:
                paired = group[["repeat", metric]].merge(
                    baseline[["repeat", metric]],
                    on="repeat",
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_secondary", "_primary"),
                )
                if len(paired) != len(group):
                    raise ScientificValidationError(
                        "secondary branch lacks a paired fixed-main cell"
                    )
                difference = (
                    pd.to_numeric(paired[f"{metric}_secondary"], errors="coerce")
                    - pd.to_numeric(paired[f"{metric}_primary"], errors="coerce")
                ).to_numpy(dtype=float)
                if not np.all(np.isfinite(difference)):
                    raise ScientificValidationError(
                        f"paired secondary difference for {metric!r} is non-finite"
                    )
                rows.append(
                    {
                        "task_id": str(group["task_id"].iloc[0]),
                        identity_column: identity_value,
                        "model": str(model),
                        "n_train": int(anchor),
                        "metric": metric,
                        "definition": "secondary_minus_fixed_main_within_repeat",
                        "repeats": int(len(difference)),
                        "mean_difference": float(difference.mean()),
                        "difference_sd": (
                            float(difference.std(ddof=1))
                            if len(difference) > 1
                            else 0.0
                        ),
                        "difference_q10": float(np.quantile(difference, 0.10)),
                        "difference_q90": float(np.quantile(difference, 0.90)),
                        "inferential_family": "secondary_descriptive_only",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [identity_column, "metric", "model", "n_train"]
    ).reset_index(drop=True)


def _secondary_model_pairwise(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    expected_models: Sequence[str] = ("relnet", "attn_relnet"),
    all_pairs: bool = False,
) -> pd.DataFrame:
    """Compute paired model contrasts inside one secondary intervention.

    Structural ablations use the default matched-neural contrast.  Frozen
    sensitivities include all five roles, so their descriptive table contains
    all ten model pairs without entering the primary multiplicity family.
    """

    models = tuple(sorted(set(frame["model"].astype(str))))
    expected = tuple(str(value) for value in expected_models)
    if len(expected) < 2 or len(expected) != len(set(expected)):
        raise ScientificValidationError("secondary model contract is invalid")
    if set(models) != set(expected):
        raise ScientificValidationError(
            "secondary branch model panel differs from its frozen contract"
        )
    if all_pairs:
        pairs = [
            (left, right)
            for left_index, left in enumerate(expected)
            for right in expected[left_index + 1 :]
        ]
    else:
        if set(expected) != {"relnet", "attn_relnet"}:
            raise ScientificValidationError(
                "a single secondary contrast requires RelNet and AttnRelNet"
            )
        pairs = [("relnet", "attn_relnet")]
    metrics = [name for name in _SECONDARY_METRICS if name in frame.columns]
    rows: list[dict[str, Any]] = []
    for (identity_value, anchor), group in frame.groupby(
        [identity_column, "n_train"], sort=True, dropna=False
    ):
        for metric in metrics:
            pivot = group.pivot(index="repeat", columns="model", values=metric)
            if set(pivot.columns.astype(str)) != set(expected) or pivot.isna().any().any():
                raise ScientificValidationError(
                    "secondary-branch model pairing is incomplete"
                )
            for left, right in pairs:
                difference = (
                    pd.to_numeric(pivot[left], errors="coerce")
                    - pd.to_numeric(pivot[right], errors="coerce")
                ).to_numpy(dtype=float)
                if not np.all(np.isfinite(difference)):
                    raise ScientificValidationError(
                        f"secondary pairwise {metric!r} is non-finite"
                    )
                rows.append(
                    {
                        "task_id": str(group["task_id"].iloc[0]),
                        identity_column: identity_value,
                        "n_train": int(anchor),
                        "metric": metric,
                        "model_left": left,
                        "model_right": right,
                        "definition": f"{left}_minus_{right}_within_repeat",
                        "repeats": int(len(difference)),
                        "mean_difference": float(difference.mean()),
                        "difference_sd": (
                            float(difference.std(ddof=1))
                            if len(difference) > 1
                            else 0.0
                        ),
                        "difference_q10": float(np.quantile(difference, 0.10)),
                        "difference_q90": float(np.quantile(difference, 0.90)),
                        "inferential_family": "secondary_descriptive_only",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [identity_column, "metric", "model_left", "model_right", "n_train"]
    ).reset_index(drop=True)


def _load_sensitivity_primary_reference(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> pd.DataFrame:
    """Load the immutable full100 panel paired to a sensitivity20 run.

    Manifest verification rebuilds the complete primary-reference binding.  We
    still verify the exact aggregate file here before allowing it to enter a
    paired secondary comparison.
    """

    binding = manifest.get("primary_reference")
    if not isinstance(binding, Mapping):
        raise ScientificValidationError(
            "a frozen-sensitivity run lacks its full100 primary reference"
        )
    root = Path(repo_root).resolve()
    run_dir = (root / str(binding.get("run_dir", ""))).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ScientificValidationError(
            "sensitivity primary-reference run escapes repo_root"
        ) from exc
    metrics_path = run_dir / "final" / "primary_metrics.csv"
    expected_sha256 = binding.get("primary_metrics_sha256")
    if (
        not isinstance(expected_sha256, str)
        or not validate_checksum(metrics_path, expected_sha256)
    ):
        raise ScientificValidationError(
            "sensitivity primary-reference metrics failed their pinned checksum"
        )
    primary = pd.read_csv(metrics_path)
    required = {
        "task_id", "branch", "repeat", "n_train", "model",
        "outer_split_hash", "nested_split_hash",
    }
    missing = required - set(primary.columns)
    if missing:
        raise ScientificValidationError(
            f"sensitivity primary-reference metrics lack {sorted(missing)}"
        )
    if len(primary) != 3000 or set(primary["branch"].astype(str)) != {"fixed_main"}:
        raise ScientificValidationError(
            "sensitivity primary reference is not the exact 3,000-cell main panel"
        )
    if primary.duplicated(["repeat", "n_train", "model"]).any():
        raise ScientificValidationError(
            "sensitivity primary-reference metrics contain duplicate cells"
        )
    if set(primary["task_id"].astype(str)) != {str(manifest["task_id"])}:
        raise ScientificValidationError(
            "sensitivity primary-reference task identity differs"
        )
    return primary


def aggregate_task_run(
    run_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    require_complete: bool = True,
) -> dict[str, Any]:
    # Final replay is deliberately hardware-independent. GPU suitability was
    # admitted by the exact preflight receipt; aggregation recomputes saved
    # scientific artifacts on a CPU node and must not require a visible GPU.
    storage = open_run(
        run_dir, repo_root=repo_root, verify_tensorflow_runtime=False
    )
    manifest = storage.manifest
    design = manifest["task_design"]
    profile = str(manifest.get("execution", {}).get("profile", ""))
    pilot_mode = profile == "pilot10"
    if pilot_mode and manifest.get("execution", {}).get("mode") != "pilot":
        raise ScientificValidationError(
            "pilot10 aggregation requires pilot execution mode"
        )
    try:
        validate_task_design(design, manifest["task_spec"])
    except Exception as exc:
        raise ScientificValidationError(f"frozen task design failed validation: {exc}") from exc
    keys = cell_keys_from_design(design)
    _reject_unexpected_cell_directories(storage, keys)
    incomplete = [key for key in keys if not storage.validate_cell(key).is_complete]
    if require_complete and incomplete:
        raise ScientificValidationError(f"run has {len(incomplete)} incomplete cells")
    if incomplete:
        incomplete_set = set(incomplete)
        keys = [key for key in keys if key not in incomplete_set]
    task = load_prepared_task(
        Path(repo_root).resolve() / manifest["prepared"]["path"], mmap_mode="r", verify_files=True
    )
    rows = [validate_cell_scientifically(storage, key, task) for key in keys]
    matched_neural_pair_groups = _validate_matched_neural_history_pairs(
        storage, keys
    )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ScientificValidationError("run contains no scientifically valid cells")
    if "branch" not in metrics:
        raise ScientificValidationError("cell metrics lack their analysis branch")
    primary = metrics.loc[metrics["branch"].eq("fixed_main")].copy()
    ablation = metrics.loc[metrics["branch"].eq("structural_ablation")].copy()
    sensitivity = metrics.loc[metrics["branch"].eq("frozen_sensitivity")].copy()
    unknown_branches = sorted(
        set(metrics["branch"].astype(str))
        - {"fixed_main", "structural_ablation", "frozen_sensitivity"}
    )
    if unknown_branches:
        raise ScientificValidationError(
            f"run contains unknown analysis branches: {unknown_branches}"
        )
    declared_counts = design.get("cell_counts", {})
    expected_counts = {
        "primary": int(declared_counts.get("primary", -1)),
        "structural_ablation": int(
            declared_counts.get("structural_ablation", -1)
        ),
        "frozen_sensitivity": int(
            declared_counts.get("frozen_sensitivity", -1)
        ),
    }
    observed_counts = {
        "primary": int(len(primary)),
        "structural_ablation": int(len(ablation)),
        "frozen_sensitivity": int(len(sensitivity)),
    }
    if require_complete and observed_counts != expected_counts:
        raise ScientificValidationError(
            f"analysis branch counts differ: observed={observed_counts}, "
            f"expected={expected_counts}"
        )

    primary_complete = len(primary) == expected_counts["primary"] and len(primary) > 0
    if primary_complete:
        observed_primary_models = set(primary["model"].astype(str))
        primary_models = (
            STRUCTURE_MODELS
            if observed_primary_models == set(STRUCTURE_MODELS)
            else LEGACY_STRUCTURE_MODELS
            if observed_primary_models == set(LEGACY_STRUCTURE_MODELS)
            else DEFAULT_MODELS
            if observed_primary_models == set(DEFAULT_MODELS)
            else tuple(sorted(observed_primary_models))
        )
        primary = validate_task_grid(
            primary,
            task_id=manifest["task_id"],
            anchors=design["anchors"],
            repeats=int(design["repeats"]),
            models=primary_models,
        )
    elif require_complete and expected_counts["primary"]:
        raise ScientificValidationError("required primary grid is incomplete")
    final_dir = storage.run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    outcome_type = str(manifest["task_spec"]["outcome_type"])
    seed = int(manifest["analysis"]["bootstrap_seed"])
    draws = int(manifest["analysis"]["bootstrap_draws"])
    tables: dict[str, pd.DataFrame] = {}
    if primary_complete:
        if pilot_mode:
            if int(design["repeats"]) != 10 or len(primary) != 300:
                raise ScientificValidationError(
                    "pilot10 analysis requires the exact ten-repeat 300-cell panel"
                )
            tables = pilot_task_analysis_tables(
                primary,
                outcome_type=outcome_type,
            )
        else:
            tables = task_analysis_tables(
                primary,
                bootstrap_seed=seed,
                bootstrap_draws=draws,
                outcome_type=outcome_type,
                target_coverage=DEFAULT_TARGET_COVERAGE,
            )
        summary = tables["summary"]
        expected_summary_groups = (
            primary["model"].astype(str).nunique()
            * primary["n_train"].astype(int).nunique()
        )
        if require_complete and (
            len(summary) != expected_summary_groups
            or set(summary["repeats"]) != {int(design["repeats"])}
        ):
            raise ScientificValidationError(
                "primary summary must contain the complete model-anchor groups"
            )
    else:
        summary = pd.DataFrame()

    ablation_tables: dict[str, pd.DataFrame] = {}
    if not ablation.empty:
        ablation_tables["structural_ablation_summary"] = _secondary_summary(
            ablation, identity_column="ablation_id"
        )
        if (
            len(ablation) == expected_counts["structural_ablation"]
            and primary_complete
        ):
            ablation_tables["structural_ablation_effects"] = (
                _paired_secondary_effects(
                    primary, ablation, identity_column="ablation_id"
                )
            )
            ablation_tables["structural_ablation_pairwise"] = (
                _secondary_model_pairwise(
                    ablation, identity_column="ablation_id"
                )
            )
    sensitivity_tables: dict[str, pd.DataFrame] = {}
    if not sensitivity.empty:
        sensitivity_tables["frozen_sensitivity_summary"] = _secondary_summary(
            sensitivity, identity_column="sensitivity_id"
        )
        primary_reference = _load_sensitivity_primary_reference(
            manifest, repo_root=repo_root
        )
        sensitivity_tables["frozen_sensitivity_effects"] = (
            _paired_secondary_effects(
                primary_reference,
                sensitivity,
                identity_column="sensitivity_id",
            )
        )
        sensitivity_tables["frozen_sensitivity_pairwise"] = (
            _secondary_model_pairwise(
                sensitivity,
                identity_column="sensitivity_id",
                expected_models=tuple(manifest["task_spec"]["models"]),
                all_pairs=True,
            )
        )
    records = {
        "metrics": atomic_write_csv(final_dir / "metrics.csv", metrics),
    }
    if not primary.empty:
        records["primary_metrics"] = atomic_write_csv(
            final_dir / "primary_metrics.csv", primary
        )
    if not ablation.empty:
        records["structural_ablation_metrics"] = atomic_write_csv(
            final_dir / "structural_ablation_metrics.csv", ablation
        )
    if not sensitivity.empty:
        records["frozen_sensitivity_metrics"] = atomic_write_csv(
            final_dir / "frozen_sensitivity_metrics.csv", sensitivity
        )
    records.update(
        {
            name: atomic_write_csv(final_dir / f"{name}.csv", table)
            for name, table in {
                **tables,
                **ablation_tables,
                **sensitivity_tables,
            }.items()
        }
    )
    receipt = {
        "task_id": manifest["task_id"],
        "manifest_hash": storage.manifest_hash,
        "cells": len(metrics),
        "primary_cells": len(primary),
        "ablation_cells": len(ablation),
        "sensitivity_cells": len(sensitivity),
        "primary_summary_groups": len(summary),
        # Backward-compatible alias for callers that predate branch-aware
        # aggregation. It always means the primary fixed-main summary.
        "summary_groups": len(summary),
        "ablation_summary_groups": len(
            ablation_tables.get("structural_ablation_summary", ())
        ),
        "sensitivity_summary_groups": len(
            sensitivity_tables.get("frozen_sensitivity_summary", ())
        ),
        "sensitivity_effect_groups": len(
            sensitivity_tables.get("frozen_sensitivity_effects", ())
        ),
        "sensitivity_pairwise_groups": len(
            sensitivity_tables.get("frozen_sensitivity_pairwise", ())
        ),
        "matched_neural_pair_groups": matched_neural_pair_groups,
        "matched_neural_assertions": {
            "identical_representative_input": True,
            "identical_output_head_and_loss": True,
            "shared_neural_families": [
                "relnet",
                "attn_relnet",
                "set_transformer",
            ],
            "identical_shared_tokens_history_masks_time_context": True,
            "global_set_transformer_typed_edges_consumed": False,
            "parameter_tolerance_fraction": 0.05,
            "forward_flop_tolerance_fraction": 0.15,
            "parameter_cap": 350_000,
        },
        **(
            {
                "evidence_status": "exploratory_provisional",
                "analysis_scope": "pilot10_descriptive_only_no_confirmatory_inference",
                "bootstrap_seed": seed,
                "bootstrap_draws": 0,
                "bootstrap_unit": "none",
                "max_t_family": "none",
                "cross_task_inference": "none",
                "stability_ribbon": "mean_with_between_repeat_q10_q90",
                "pilot_repeat_quantiles": (
                    "descriptive_rerun_quantiles_not_confidence_intervals"
                ),
            }
            if pilot_mode and tables
            else
            task_analysis_receipt_fields(
                tables,
                bootstrap_seed=seed,
                bootstrap_draws=draws,
                outcome_type=outcome_type,
                target_coverage=DEFAULT_TARGET_COVERAGE,
            )
            if tables
            else {
                "analysis_scope": "secondary_profile_without_primary_max_t",
                "bootstrap_seed": seed,
                "bootstrap_draws": draws,
            }
        ),
        "scientific_revalidation": (
            "ordered_identities_splits_seeds_config_history_predictions_all_metrics"
        ),
        "multiplicity_scope": (
            "none_pilot_descriptive" if pilot_mode else "fixed_main_only"
        ),
        "secondary_branches": {
            "structural_ablation": "descriptive_paired_not_in_primary_max_t",
            "frozen_sensitivity": (
                "descriptive_paired_to_pinned_full100_not_in_primary_max_t"
            ),
        },
        **(
            {"primary_reference": dict(manifest["primary_reference"])}
            if not sensitivity.empty
            else {}
        ),
        "artifacts": {name: record.as_dict() for name, record in records.items()},
    }
    receipt_record = atomic_write_json(final_dir / "receipt.json", receipt)
    if require_complete:
        storage.finalize_run(
            keys,
            final_artifacts=[final_dir / record.file for record in records.values()]
            + [final_dir / receipt_record.file],
        )
    return receipt
