"""Immutable no-refit, game-clustered interval companion for BDB2020.

This module is deliberately downstream of the completed rushing study.  It
never imports a fitting entry point and accepts only marker-validated saved
calibration/test predictions and probability arrays from the locked ``main``
branch.  The locally weighted conformal interval in the source run remains the
primary interval; this artifact is a separate cluster-aware sensitivity.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .design import FULL_CONFIRMATORY_REPEATS, stable_seed
from .execution import (
    configured_anchors,
    configured_models,
    expected_cell_keys,
    validate_scientific_cell,
)
from .models import MODEL_IDS
from .storage import (
    CellKey,
    RunStorage,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    validate_checksum,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "bdb2020_game_clustered_interval_no_refit_v1"
SELECTION_NAMESPACE = "uncertainty_subsample/one_calibration_play_per_game"
EVIDENCE_LABEL = "completed_sensitivity_no_refit"
EXPECTED_ANCHORS = (20, 40, 80, 160, 240, 360)
NUM_CLASSES = 80
ALPHA = 0.10
TARGET_COVERAGE = 0.90


class ClusterIntervalError(RuntimeError):
    """Base class for cluster-interval artifact failures."""


class SourceValidationError(ClusterIntervalError):
    """Raised when the completed source run is missing, incomplete, or changed."""


class ImmutableArtifactError(ClusterIntervalError):
    """Raised when an output path is already occupied or an artifact is corrupt."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutableArtifactError(f"{description} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ImmutableArtifactError(f"{description} must be a JSON object: {path}")
    return value


def _source_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("config", manifest)
    if not isinstance(value, Mapping):
        raise SourceValidationError("source manifest has no valid study configuration")
    return value


def _assert_nonoverlapping_paths(source_run_dir: Path, output_dir: Path) -> None:
    source = source_run_dir.resolve()
    output = output_dir.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ImmutableArtifactError(
            "source and output directories must be disjoint; the no-refit sensitivity "
            "may not write anywhere inside (or above) the immutable source run"
        )


def _full100_source_contract(
    storage: RunStorage,
) -> tuple[list[CellKey], dict[str, Any], str]:
    """Validate the immutable full100 source envelope before reading any cell."""

    config = _source_config(storage.manifest)
    study_id = config.get("study_id", storage.manifest.get("study_id"))
    if study_id != "rushing_confirmatory_v1":
        raise SourceValidationError(
            f"source study_id must be 'rushing_confirmatory_v1', got {study_id!r}"
        )
    repeats = config.get("execution", {}).get("confirmatory_repeats")
    if repeats != FULL_CONFIRMATORY_REPEATS:
        raise SourceValidationError(
            f"source must be the immutable full100 profile, got {repeats!r} repeats"
        )
    if tuple(configured_anchors(storage.manifest, "main")) != EXPECTED_ANCHORS:
        raise SourceValidationError("source main anchors do not match the locked six-anchor grid")
    if tuple(configured_models(storage.manifest)) != tuple(MODEL_IDS):
        raise SourceValidationError("source models do not match the locked four-model grid")
    uncertainty = config.get("uncertainty", {})
    if (
        not isinstance(uncertainty, Mapping)
        or not math.isclose(float(uncertainty.get("alpha", float("nan"))), ALPHA)
        or uncertainty.get("central_interval") != "alpha_over_2_equal_tail"
    ):
        raise SourceValidationError("source does not declare the locked equal-tail 90% interval")

    keys = expected_cell_keys(storage.manifest, "main")
    expected_count = FULL_CONFIRMATORY_REPEATS * len(EXPECTED_ANCHORS) * len(MODEL_IDS)
    if len(keys) != expected_count or len(set(keys)) != expected_count:
        raise SourceValidationError(
            f"source main grid has {len(keys)} cells; expected exactly {expected_count}"
        )
    final_path = storage.run_dir / "_SUCCESS"
    if not final_path.is_file() or not storage.validate_final():
        raise SourceValidationError(
            "source run-level marker is absent or invalid; only a finalized full100 run is accepted"
        )
    final_marker = _read_json(final_path, description="source run-level marker")
    recorded_main: set[CellKey] = set()
    try:
        for entry in final_marker["cells"]:
            key = CellKey(**entry["key"])
            if key.branch == "main":
                recorded_main.add(key)
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceValidationError("source run-level marker has malformed cell identities") from exc
    if recorded_main != set(keys):
        raise SourceValidationError(
            "source run-level marker does not bind exactly the 2,400 full100 main cells"
        )
    return keys, final_marker, _sha256_file(final_path)


def _derive_repeat_seed(
    manifest: Mapping[str, Any], repeat: int
) -> tuple[int, dict[str, Any]]:
    config = _source_config(manifest)
    derivation = manifest.get("seed_derivation", {})
    base_seed = config.get("base_seed", derivation.get("base_seed"))
    algorithm = config.get("seed_algorithm", derivation.get("algorithm"))
    study_id = config.get("study_id", manifest.get("study_id"))
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, Integral)
        or int(base_seed) <= 0
    ):
        raise SourceValidationError("source has no valid positive base seed")
    if algorithm != "hmac-sha256-31bit-v1":
        raise SourceValidationError(
            f"source seed algorithm is not the locked HMAC contract: {algorithm!r}"
        )
    semantic_key = [
        str(study_id),
        PROTOCOL_ID,
        SELECTION_NAMESPACE,
        "repeat",
        int(repeat),
    ]
    seed = int(stable_seed(int(base_seed), *semantic_key))
    return seed, {
        "algorithm": algorithm,
        "base_seed": int(base_seed),
        "semantic_key": semantic_key,
        "seed": seed,
    }


def _identifier(value: Any, *, name: str) -> str:
    if pd.isna(value):
        raise SourceValidationError(f"{name} contains a missing identifier")
    if isinstance(value, (np.integer, Integral)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise SourceValidationError(f"{name} contains a nonintegral numeric identifier")
        return str(int(number))
    text = str(value)
    if not text:
        raise SourceValidationError(f"{name} contains an empty identifier")
    return text


def _selection_digest(seed: int, game_id: str, play_id: str) -> str:
    payload = {
        "namespace": SELECTION_NAMESPACE,
        "seed": int(seed),
        "game_id": game_id,
        "play_id": play_id,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _selected_registry(
    calibration: pd.DataFrame, *, repeat: int, seed: int
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    required = {"game_id", "play_id", "season", "y_class"}
    missing = required - set(calibration.columns)
    if missing:
        raise SourceValidationError(
            f"calibration predictions lack selection fields: {sorted(missing)}"
        )
    rows: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for position, row in calibration.reset_index(drop=True).iterrows():
        game_id = _identifier(row["game_id"], name="game_id")
        play_id = _identifier(row["play_id"], name="play_id")
        identity = (game_id, play_id)
        if identity in positions:
            raise SourceValidationError(f"duplicate calibration play identity: {identity!r}")
        y_class = row["y_class"]
        if (
            isinstance(y_class, bool)
            or not isinstance(y_class, (Integral, np.integer, Real, np.floating))
            or not float(y_class).is_integer()
        ):
            raise SourceValidationError("calibration y_class must be integer-valued")
        season = row["season"]
        if not isinstance(season, (Integral, np.integer, Real, np.floating)) or not float(
            season
        ).is_integer():
            raise SourceValidationError("calibration season must be integer-valued")
        positions[identity] = int(position)
        rows.append(
            {
                "repeat": int(repeat),
                "uncertainty_subsample_seed": int(seed),
                "game_id": game_id,
                "play_id": play_id,
                "season": int(season),
                "y_class": int(y_class),
                "selection_digest": _selection_digest(seed, game_id, play_id),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SourceValidationError("calibration partition is empty")
    selected = (
        frame.sort_values(
            ["game_id", "selection_digest", "play_id"], kind="mergesort"
        )
        .groupby("game_id", sort=True, as_index=False)
        .first()
        .sort_values("game_id", kind="mergesort")
        .reset_index(drop=True)
    )
    selected = selected[
        [
            "repeat",
            "uncertainty_subsample_seed",
            "game_id",
            "play_id",
            "season",
            "y_class",
            "selection_digest",
        ]
    ]
    if len(selected) != frame["game_id"].nunique():
        raise SourceValidationError("selection did not choose exactly one play per calibration game")
    return selected, positions


def _registry_semantic_hash(registry: pd.DataFrame) -> str:
    return _sha256_json(registry.to_dict(orient="records"))


def _central_intervals(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise SourceValidationError(
            f"probability array must have shape [n,{NUM_CLASSES}], got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values < -1e-8):
        raise SourceValidationError("probability array contains invalid values")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
        raise SourceValidationError("probability rows are not normalized")
    normalized = values / row_sums[:, None]
    cdf = np.cumsum(normalized, axis=1)
    lower = np.argmax(cdf >= ALPHA / 2.0, axis=1).astype(np.int64)
    upper = np.argmax(cdf >= 1.0 - ALPHA / 2.0, axis=1).astype(np.int64)
    return lower, upper


def _finite_sample_padding(scores: np.ndarray) -> tuple[int, int, bool]:
    values = np.asarray(scores, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise SourceValidationError("calibration nonconformity scores must be nonempty/nonnegative")
    rank = int(math.ceil((len(values) + 1) * TARGET_COVERAGE))
    if rank > len(values):
        # A padding of K-1 expands every valid central interval to [0,K-1].
        return NUM_CLASSES - 1, rank, True
    q = int(np.partition(values, rank - 1)[rank - 1])
    return q, rank, False


def _source_cell_binding(storage: RunStorage, key: CellKey) -> dict[str, Any]:
    marker_path = storage.cell_dir(key) / "_SUCCESS"
    marker = _read_json(marker_path, description="source cell marker")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or not {
        "metrics",
        "predictions",
        "history",
        "arrays",
    }.issubset(artifacts):
        raise SourceValidationError(
            f"source cell {key.relative_dir} lacks metrics/predictions/history/arrays"
        )
    return {
        "key": asdict(key),
        "path": str(key.relative_dir),
        "marker_sha256": _sha256_file(marker_path),
        "artifacts": {
            name: {
                "file": artifacts[name]["file"],
                "sha256": artifacts[name]["sha256"],
            }
            for name in ("metrics", "predictions", "history", "arrays")
        },
    }


def _partition_payloads(
    storage: RunStorage, key: CellKey
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    result = validate_scientific_cell(storage, key, recompute_conformal=False)
    if not result.is_complete:
        raise SourceValidationError(
            f"source cell {key.relative_dir} failed marker/scientific validation: {result.reason}"
        )
    predictions = storage.load_predictions(key)
    arrays = storage.load_arrays(key)
    if not isinstance(predictions, pd.DataFrame) or not isinstance(arrays, dict):
        raise SourceValidationError(f"source cell {key.relative_dir} lacks tabular saved inputs")
    calibration = predictions.loc[
        predictions["partition"].astype(str) == "calibration"
    ].reset_index(drop=True)
    test = predictions.loc[predictions["partition"].astype(str) == "test"].reset_index(
        drop=True
    )
    try:
        calibration_proba = np.asarray(arrays["calibration_proba"], dtype=np.float64)
        test_proba = np.asarray(arrays["test_proba"], dtype=np.float64)
    except KeyError as exc:
        raise SourceValidationError(
            f"source cell {key.relative_dir} lacks saved calibration/test probabilities"
        ) from exc
    if len(calibration_proba) != len(calibration) or len(test_proba) != len(test):
        raise SourceValidationError(f"source cell {key.relative_dir} arrays are row-misaligned")
    return calibration, test, calibration_proba, test_proba


def _cell_metrics(
    storage: RunStorage,
    key: CellKey,
    *,
    repeat_seed: int,
    expected_registry: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calibration, test, calibration_proba, test_proba = _partition_payloads(storage, key)
    registry, positions = _selected_registry(
        calibration, repeat=key.repeat, seed=repeat_seed
    )
    if expected_registry is not None and registry.to_dict(orient="records") != expected_registry.to_dict(
        orient="records"
    ):
        raise SourceValidationError(
            f"calibration registry changed across model/anchor cells in repeat {key.repeat}"
        )

    calibration_lo, calibration_hi = _central_intervals(calibration_proba)
    test_lo, test_hi = _central_intervals(test_proba)
    selected_positions = np.asarray(
        [positions[(row.game_id, row.play_id)] for row in registry.itertuples(index=False)],
        dtype=np.int64,
    )
    selected_y = calibration.iloc[selected_positions]["y_class"].to_numpy(dtype=np.int64)
    scores = np.maximum.reduce(
        [
            calibration_lo[selected_positions] - selected_y,
            selected_y - calibration_hi[selected_positions],
            np.zeros(len(selected_positions), dtype=np.int64),
        ]
    ).astype(np.int64)
    q, rank, fallback = _finite_sample_padding(scores)
    lo = np.maximum(0, test_lo - q)
    hi = np.minimum(NUM_CLASSES - 1, test_hi + q)
    test_y = test["y_class"].to_numpy(dtype=np.int64)
    covered = ((test_y >= lo) & (test_y <= hi)).astype(np.float64)
    width = (hi - lo).astype(np.float64)
    width_inclusive = width + 1.0
    game_rows = pd.DataFrame(
        {
            "game_id": [
                _identifier(value, name="test game_id") for value in test["game_id"]
            ],
            "coverage": covered,
            "width": width,
            "width_inclusive": width_inclusive,
        }
    )
    by_game = game_rows.groupby("game_id", sort=True).mean(numeric_only=True)
    source_metrics = storage.load_metrics(key)
    example_coverage = float(covered.mean())
    example_width = float(width.mean())
    example_width_inclusive = float(width_inclusive.mean())
    game_coverage = float(by_game["coverage"].mean())
    game_width = float(by_game["width"].mean())
    game_width_inclusive = float(by_game["width_inclusive"].mean())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "evidence_label": EVIDENCE_LABEL,
        "source_interval_status": "locked_local_interval_remains_primary",
        "branch": key.branch,
        "repeat": int(key.repeat),
        "n_train": int(key.n_train),
        "model": key.model,
        "source_manifest_hash": storage.manifest_hash,
        "source_cell_marker_sha256": _sha256_file(storage.cell_dir(key) / "_SUCCESS"),
        "uncertainty_subsample_seed": int(repeat_seed),
        "selected_registry_semantic_sha256": _registry_semantic_hash(registry),
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "central_interval": "alpha_over_2_equal_tail",
        "calibration_sampling": "one_semantically_hashed_play_per_game",
        "nonconformity": "max(central_lo_minus_y,y_minus_central_hi,0)",
        "finite_sample_quantile": "ceil((m+1)*0.90)_order_statistic",
        "full_support_fallback": bool(fallback),
        "n_calibration_plays": int(len(calibration)),
        "n_calibration_games": int(calibration["game_id"].nunique()),
        "n_selected_calibration_plays": int(len(registry)),
        "quantile_rank": int(rank),
        "q": int(q),
        "selected_score_min": int(scores.min()),
        "selected_score_mean": float(scores.mean()),
        "selected_score_max": int(scores.max()),
        "n_test_plays": int(len(test)),
        "n_test_games": int(test["game_id"].nunique()),
        # Explicit names make the weighting visible. Conventional rushing-study
        # aliases are retained beside them so downstream plotting is trivial.
        "example_equal_coverage": example_coverage,
        "example_equal_mean_width": example_width,
        "example_equal_mean_width_inclusive": example_width_inclusive,
        "game_equal_coverage": game_coverage,
        "game_equal_mean_width": game_width,
        "game_equal_mean_width_inclusive": game_width_inclusive,
        "coverage": example_coverage,
        "mean_width": example_width,
        "mean_width_inclusive": example_width_inclusive,
        "coverage_game_equal": game_coverage,
        "mean_width_game_equal": game_width,
        "mean_width_inclusive_game_equal": game_width_inclusive,
        "source_primary_local_coverage": float(source_metrics["coverage"]),
        "source_primary_local_mean_width": float(source_metrics["mean_width"]),
        "source_primary_local_coverage_game_equal": float(
            source_metrics["coverage_game_equal"]
        ),
        "source_primary_local_mean_width_game_equal": float(
            source_metrics["mean_width_game_equal"]
        ),
    }, registry


def _summary_frame(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "example_equal_coverage",
        "example_equal_mean_width",
        "example_equal_mean_width_inclusive",
        "game_equal_coverage",
        "game_equal_mean_width",
        "game_equal_mean_width_inclusive",
        "q",
    )
    rows: list[dict[str, Any]] = []
    grouped = cell_metrics.groupby(["model", "n_train"], sort=True, observed=True)
    for (model, n_train), frame in grouped:
        row: dict[str, Any] = {
            "model": str(model),
            "n_train": int(n_train),
            "n_repeats": int(frame["repeat"].nunique()),
            "cell_count": int(len(frame)),
            "full_support_fallback_count": int(frame["full_support_fallback"].sum()),
        }
        for metric in metric_names:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_p10"] = float(np.quantile(values, 0.10))
            row[f"{metric}_p90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "n_train"], kind="mergesort").reset_index(
        drop=True
    )


def _compute_cells(
    storage: RunStorage, keys: Sequence[CellKey]
) -> tuple[
    list[dict[str, Any]],
    dict[int, pd.DataFrame],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics: list[dict[str, Any]] = []
    registries: dict[int, pd.DataFrame] = {}
    seed_records: dict[int, dict[str, Any]] = {}
    source_cells: list[dict[str, Any]] = []
    seen_seeds: dict[int, int] = {}
    for key in sorted(keys):
        if key.branch != "main":
            raise SourceValidationError("cluster interval sensitivity accepts main cells only")
        if key.repeat not in seed_records:
            seed, record = _derive_repeat_seed(storage.manifest, key.repeat)
            prior_repeat = seen_seeds.get(seed)
            if prior_repeat is not None and prior_repeat != key.repeat:
                raise SourceValidationError(
                    f"semantic uncertainty seed collision between repeats {prior_repeat} and {key.repeat}"
                )
            seen_seeds[seed] = key.repeat
            seed_records[key.repeat] = record
        repeat_seed = int(seed_records[key.repeat]["seed"])
        cell, registry = _cell_metrics(
            storage,
            key,
            repeat_seed=repeat_seed,
            expected_registry=registries.get(key.repeat),
        )
        if key.repeat not in registries:
            registries[key.repeat] = registry
        metrics.append(cell)
        source_cells.append(_source_cell_binding(storage, key))
    if len(metrics) != len(keys):
        raise SourceValidationError("not every requested source cell was evaluated")
    return metrics, registries, seed_records, source_cells


def _artifact_entry(record: Any, relative_path: str) -> dict[str, str]:
    return {
        "file": relative_path,
        "sha256": str(record.sha256),
        "format": str(record.format),
    }


def _write_artifact(
    output_dir: Path,
    storage: RunStorage,
    keys: Sequence[CellKey],
    metrics: list[dict[str, Any]],
    registries: Mapping[int, pd.DataFrame],
    seed_records: Mapping[int, Mapping[str, Any]],
    source_cells: list[dict[str, Any]],
    *,
    source_final_marker_sha256: str | None,
) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutableArtifactError(
            f"refusing to overwrite an existing cluster-interval artifact: {output_dir}"
        ) from exc

    source_binding = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "source_study_id": _source_config(storage.manifest).get(
            "study_id", storage.manifest.get("study_id")
        ),
        "source_manifest_hash": storage.manifest_hash,
        "source_run_hash": storage.manifest.get("run_hash"),
        "source_manifest_file_sha256": _sha256_file(storage.run_dir / "manifest.json"),
        "source_final_marker_sha256": source_final_marker_sha256,
        "source_main_cell_count": len(keys),
        "source_cells": source_cells,
        "binding_statement": (
            "Read-only binding to saved marker-validated calibration/test probabilities and "
            "histories; no model fitting, refitting, or source-run mutation is permitted."
        ),
    }
    source_record = atomic_write_json(output_dir / "source_binding.json", source_binding)
    seed_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "sharing_contract": "one seed per repeat shared across every model and anchor",
        "repeat_seeds": {
            f"repeat_{repeat:03d}": dict(seed_records[repeat])
            for repeat in sorted(seed_records)
        },
    }
    seed_record = atomic_write_json(output_dir / "repeat_seed_registry.json", seed_payload)
    combined_registry = pd.concat(
        [registries[repeat] for repeat in sorted(registries)], ignore_index=True
    )
    registry_record = atomic_write_csv(
        output_dir / "selected_calibration_registry.csv", combined_registry
    )

    by_identity = {
        (int(item["repeat"]), int(item["n_train"]), str(item["model"])): item
        for item in metrics
    }
    cell_markers: list[dict[str, Any]] = []
    for key in sorted(keys):
        metric = by_identity[(key.repeat, key.n_train, key.model)]
        directory = output_dir / key.relative_dir
        registry_artifact = atomic_write_csv(
            directory / "selected_calibration_registry.csv", registries[key.repeat]
        )
        metric["selected_registry_artifact_sha256"] = registry_artifact.sha256
        metrics_artifact = atomic_write_json(directory / "metrics.json", metric)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_ID,
            "source_manifest_hash": storage.manifest_hash,
            "source_cell_marker_sha256": metric["source_cell_marker_sha256"],
            "key": asdict(key),
            "artifacts": {
                "metrics": metrics_artifact.as_dict(),
                "selected_calibration_registry": registry_artifact.as_dict(),
            },
        }
        marker_artifact = atomic_write_json(directory / "_SUCCESS", marker)
        cell_markers.append(
            {
                "key": asdict(key),
                "path": str(key.relative_dir),
                "marker_file": str(key.relative_dir / "_SUCCESS"),
                "marker_sha256": marker_artifact.sha256,
            }
        )

    metrics_frame = pd.DataFrame(metrics).sort_values(
        ["repeat", "n_train", "model"], kind="mergesort"
    )
    cell_metrics_record = atomic_write_csv(output_dir / "cell_metrics.csv", metrics_frame)
    summary = _summary_frame(metrics_frame)
    summary_record = atomic_write_csv(output_dir / "summary.csv", summary)
    root_artifacts = {
        "source_binding": _artifact_entry(source_record, "source_binding.json"),
        "repeat_seed_registry": _artifact_entry(
            seed_record, "repeat_seed_registry.json"
        ),
        "selected_calibration_registry": _artifact_entry(
            registry_record, "selected_calibration_registry.csv"
        ),
        "cell_metrics": _artifact_entry(cell_metrics_record, "cell_metrics.csv"),
        "summary": _artifact_entry(summary_record, "summary.csv"),
    }
    implementation_path = Path(__file__)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "evidence_label": EVIDENCE_LABEL,
        "source_manifest_hash": storage.manifest_hash,
        "source_interval_status": "locked_local_interval_remains_primary",
        "no_fit_attestation": True,
        "immutable": True,
        "cell_count": len(keys),
        "repeat_count": len(seed_records),
        "anchors": sorted({int(key.n_train) for key in keys}),
        "models": [model for model in MODEL_IDS if any(key.model == model for key in keys)],
        "algorithm": {
            "central_interval": "equal_tail_90_percent",
            "calibration_sampling": "one_semantically_hashed_play_per_calibration_game",
            "nonconformity": "max(lo-y,y-hi,0)",
            "finite_sample_rank": "ceil((m+1)*0.90)",
            "rank_overflow": "q=79_full_support_fallback",
            "test_application": "single_cell_q_padded_and_clipped_to_[0,79]",
            "reported_weighting": ["example_equal", "game_equal"],
        },
        "summary_contract": "mean, sample SD, p10, and p90 across reruns by model/anchor",
        "implementation_file": "rushing_study/cluster_intervals.py",
        "implementation_sha256": _sha256_file(implementation_path),
        "artifacts": root_artifacts,
        "cell_markers": cell_markers,
    }
    receipt_record = atomic_write_json(output_dir / "receipt.json", receipt)
    success = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "source_manifest_hash": storage.manifest_hash,
        "cell_count": len(keys),
        "receipt_file": "receipt.json",
        "receipt_sha256": receipt_record.sha256,
    }
    atomic_write_json(output_dir / "_SUCCESS", success)
    validate_cluster_interval_artifact(output_dir)
    return {
        "output_dir": str(output_dir),
        "protocol": PROTOCOL_ID,
        "source_manifest_hash": storage.manifest_hash,
        "cell_count": len(keys),
        "summary_rows": int(len(summary)),
        "receipt_sha256": receipt_record.sha256,
        "changed": True,
    }


def _generate_for_keys(
    storage: RunStorage,
    keys: Sequence[CellKey],
    output_dir: str | Path,
    *,
    source_final_marker_sha256: str | None,
) -> dict[str, Any]:
    """Internal small-grid seam used by focused tests; public CLI is full100-only."""

    output = Path(output_dir)
    if output.exists():
        raise ImmutableArtifactError(
            f"refusing to overwrite an existing cluster-interval artifact: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics, registries, seeds, source_cells = _compute_cells(storage, list(keys))
    return _write_artifact(
        output,
        storage,
        list(keys),
        metrics,
        registries,
        seeds,
        source_cells,
        source_final_marker_sha256=source_final_marker_sha256,
    )


def generate_cluster_interval_sensitivity(
    source_run_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create the immutable full100 no-refit cluster-aware interval artifact."""

    source = Path(source_run_dir)
    output = Path(output_dir)
    _assert_nonoverlapping_paths(source, output)
    if output.exists():
        raise ImmutableArtifactError(
            f"refusing to overwrite an existing cluster-interval artifact: {output}"
        )
    storage = RunStorage.open(source)
    keys, _final_marker, final_marker_sha256 = _full100_source_contract(storage)
    return _generate_for_keys(
        storage,
        keys,
        output,
        source_final_marker_sha256=final_marker_sha256,
    )


def _validate_indexed_artifact(root: Path, entry: Mapping[str, Any]) -> None:
    filename = entry.get("file")
    digest = entry.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        raise ImmutableArtifactError("artifact index has a malformed file/hash entry")
    path = root / filename
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ImmutableArtifactError(f"artifact escapes output directory: {filename}") from exc
    if not validate_checksum(path, digest):
        raise ImmutableArtifactError(f"artifact is missing or failed checksum: {filename}")


def _read_registry_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            dtype={"game_id": str, "play_id": str, "selection_digest": str},
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ImmutableArtifactError(f"registry CSV is unreadable: {path}") from exc


def _assert_replayed_frame(
    actual: pd.DataFrame, expected: pd.DataFrame, *, description: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise SourceValidationError(
            f"{description} does not match independent no-refit replay"
        ) from exc


def _replay_against_source(
    root: Path,
    receipt: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    source: RunStorage,
    keys: Sequence[CellKey],
) -> None:
    """Independently rebuild every downstream value from the bound source arrays."""

    replay_metrics, replay_registries, replay_seeds, replay_source_cells = _compute_cells(
        source, list(keys)
    )
    if replay_source_cells != source_binding.get("source_cells"):
        raise SourceValidationError("source cell binding does not match independent replay")

    artifacts = receipt["artifacts"]
    seed_payload = _read_json(
        root / artifacts["repeat_seed_registry"]["file"],
        description="repeat seed registry",
    )
    expected_seed_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "sharing_contract": "one seed per repeat shared across every model and anchor",
        "repeat_seeds": {
            f"repeat_{repeat:03d}": dict(replay_seeds[repeat])
            for repeat in sorted(replay_seeds)
        },
    }
    if seed_payload != expected_seed_payload:
        raise SourceValidationError("repeat seed registry does not match independent replay")

    expected_registry = pd.concat(
        [replay_registries[repeat] for repeat in sorted(replay_registries)],
        ignore_index=True,
    )
    actual_registry = _read_registry_csv(
        root / artifacts["selected_calibration_registry"]["file"]
    )
    _assert_replayed_frame(
        actual_registry,
        expected_registry,
        description="combined selected calibration registry",
    )

    replay_by_key = {
        (int(item["repeat"]), int(item["n_train"]), str(item["model"])): item
        for item in replay_metrics
    }
    marker_entries = {
        CellKey(**entry["key"]): entry for entry in receipt["cell_markers"]
    }
    for key in sorted(keys):
        marker_path = root / marker_entries[key]["marker_file"]
        marker = _read_json(marker_path, description="cluster-interval cell marker")
        registry_entry = marker["artifacts"]["selected_calibration_registry"]
        actual_cell_registry = _read_registry_csv(marker_path.parent / registry_entry["file"])
        _assert_replayed_frame(
            actual_cell_registry,
            replay_registries[key.repeat],
            description=f"selected calibration registry for {key.relative_dir}",
        )
        metrics_entry = marker["artifacts"]["metrics"]
        actual_metric = _read_json(
            marker_path.parent / metrics_entry["file"],
            description="cluster-interval cell metrics",
        )
        expected_metric = dict(replay_by_key[(key.repeat, key.n_train, key.model)])
        expected_metric["selected_registry_artifact_sha256"] = registry_entry["sha256"]
        if actual_metric != expected_metric:
            raise SourceValidationError(
                f"cell metrics do not match independent no-refit replay: {key.relative_dir}"
            )
        replay_by_key[(key.repeat, key.n_train, key.model)] = expected_metric

    expected_metrics = pd.DataFrame(list(replay_by_key.values())).sort_values(
        ["repeat", "n_train", "model"], kind="mergesort"
    )
    try:
        actual_metrics = pd.read_csv(root / artifacts["cell_metrics"]["file"])
        actual_summary = pd.read_csv(root / artifacts["summary"]["file"])
    except (OSError, UnicodeError, ValueError) as exc:
        raise ImmutableArtifactError("cell metrics or summary CSV is unreadable") from exc
    _assert_replayed_frame(
        actual_metrics,
        expected_metrics,
        description="combined cell metrics",
    )
    _assert_replayed_frame(
        actual_summary,
        _summary_frame(expected_metrics),
        description="cluster interval summary",
    )


def validate_cluster_interval_artifact(
    output_dir: str | Path, *, source_run_dir: str | Path | None = None
) -> dict[str, Any]:
    """Validate a committed sensitivity artifact and optionally its source binding."""

    root = Path(output_dir)
    success_path = root / "_SUCCESS"
    if not validate_checksum(success_path):
        raise ImmutableArtifactError("cluster-interval root marker is missing or corrupt")
    success = _read_json(success_path, description="cluster-interval root marker")
    if success.get("schema_version") != SCHEMA_VERSION or success.get("protocol") != PROTOCOL_ID:
        raise ImmutableArtifactError("cluster-interval root marker has the wrong protocol")
    receipt_file = success.get("receipt_file")
    if not isinstance(receipt_file, str) or Path(receipt_file).name != receipt_file:
        raise ImmutableArtifactError("cluster-interval receipt path is malformed")
    receipt_path = root / receipt_file
    receipt_sha = success.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or not validate_checksum(receipt_path, receipt_sha):
        raise ImmutableArtifactError("cluster-interval receipt is missing or corrupt")
    receipt = _read_json(receipt_path, description="cluster-interval receipt")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("protocol") != PROTOCOL_ID
        or receipt.get("immutable") is not True
        or receipt.get("no_fit_attestation") is not True
        or receipt.get("source_manifest_hash") != success.get("source_manifest_hash")
        or receipt.get("cell_count") != success.get("cell_count")
    ):
        raise ImmutableArtifactError("cluster-interval receipt contract is invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "source_binding",
        "repeat_seed_registry",
        "selected_calibration_registry",
        "cell_metrics",
        "summary",
    }:
        raise ImmutableArtifactError("cluster-interval root artifact index is incomplete")
    for entry in artifacts.values():
        if not isinstance(entry, Mapping):
            raise ImmutableArtifactError("cluster-interval root artifact entry is malformed")
        _validate_indexed_artifact(root, entry)

    markers = receipt.get("cell_markers")
    if not isinstance(markers, list) or len(markers) != receipt.get("cell_count"):
        raise ImmutableArtifactError("cluster-interval cell marker index is incomplete")
    seen: set[CellKey] = set()
    for entry in markers:
        if not isinstance(entry, Mapping):
            raise ImmutableArtifactError("cluster-interval cell marker entry is malformed")
        try:
            key = CellKey(**entry["key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ImmutableArtifactError("cluster-interval cell marker key is malformed") from exc
        if key in seen or entry.get("path") != str(key.relative_dir):
            raise ImmutableArtifactError("cluster-interval cell marker identity is duplicated/misaligned")
        seen.add(key)
        marker_file = entry.get("marker_file")
        marker_sha = entry.get("marker_sha256")
        if marker_file != str(key.relative_dir / "_SUCCESS"):
            raise ImmutableArtifactError("cluster-interval cell marker path is misbound")
        marker_path = root / marker_file
        if not isinstance(marker_sha, str) or not validate_checksum(marker_path, marker_sha):
            raise ImmutableArtifactError(f"cluster-interval cell marker is corrupt: {marker_file}")
        marker = _read_json(marker_path, description="cluster-interval cell marker")
        if (
            marker.get("protocol") != PROTOCOL_ID
            or marker.get("key") != asdict(key)
            or marker.get("source_manifest_hash") != receipt.get("source_manifest_hash")
        ):
            raise ImmutableArtifactError("cluster-interval cell marker contract is invalid")
        indexed = marker.get("artifacts")
        if not isinstance(indexed, dict) or set(indexed) != {
            "metrics",
            "selected_calibration_registry",
        }:
            raise ImmutableArtifactError("cluster-interval cell artifact index is incomplete")
        for artifact in indexed.values():
            filename = artifact.get("file") if isinstance(artifact, Mapping) else None
            digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(digest, str)
                or not validate_checksum(marker_path.parent / filename, digest)
            ):
                raise ImmutableArtifactError(
                    f"cluster-interval cell payload is corrupt: {key.relative_dir}"
                )

    source_binding = _read_json(
        root / artifacts["source_binding"]["file"], description="source binding"
    )
    try:
        bound_keys = {CellKey(**item["key"]) for item in source_binding["source_cells"]}
    except (KeyError, TypeError, ValueError) as exc:
        raise ImmutableArtifactError("source cell binding identities are malformed") from exc
    if (
        source_binding.get("source_manifest_hash") != receipt.get("source_manifest_hash")
        or source_binding.get("source_main_cell_count") != receipt.get("cell_count")
        or not isinstance(source_binding.get("source_cells"), list)
        or len(source_binding["source_cells"]) != receipt.get("cell_count")
        or bound_keys != seen
    ):
        raise ImmutableArtifactError("source binding is inconsistent with the receipt")

    finalized_full100 = isinstance(source_binding.get("source_final_marker_sha256"), str)
    if finalized_full100:
        expected_full_grid = {
            CellKey("main", repeat, n_train, model)
            for repeat in range(1, FULL_CONFIRMATORY_REPEATS + 1)
            for n_train in EXPECTED_ANCHORS
            for model in MODEL_IDS
        }
        if (
            seen != expected_full_grid
            or receipt.get("repeat_count") != FULL_CONFIRMATORY_REPEATS
            or receipt.get("anchors") != list(EXPECTED_ANCHORS)
            or receipt.get("models") != list(MODEL_IDS)
        ):
            raise ImmutableArtifactError(
                "finalized companion does not contain the exact 100x6x4 main grid"
            )

    if source_run_dir is not None:
        source = RunStorage.open(source_run_dir)
        if source.manifest_hash != receipt.get("source_manifest_hash"):
            raise SourceValidationError("bound source manifest hash no longer matches")
        if finalized_full100:
            replay_keys, _, final_sha = _full100_source_contract(source)
            if final_sha != source_binding.get("source_final_marker_sha256"):
                raise SourceValidationError("bound source run-level marker no longer matches")
        else:
            replay_keys = sorted(seen)
        for item in source_binding["source_cells"]:
            try:
                key = CellKey(**item["key"])
                expected_sha = item["marker_sha256"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ImmutableArtifactError("source cell binding is malformed") from exc
            marker_path = source.cell_dir(key) / "_SUCCESS"
            if _sha256_file(marker_path) != expected_sha or not source.validate_cell(key).is_complete:
                raise SourceValidationError(
                    f"bound source cell no longer validates: {key.relative_dir}"
                )
        _replay_against_source(
            root,
            receipt,
            source_binding,
            source,
            replay_keys,
        )
    return receipt


__all__ = [
    "ALPHA",
    "ClusterIntervalError",
    "EVIDENCE_LABEL",
    "ImmutableArtifactError",
    "PROTOCOL_ID",
    "SourceValidationError",
    "generate_cluster_interval_sensitivity",
    "validate_cluster_interval_artifact",
]
