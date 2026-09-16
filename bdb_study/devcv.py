"""Sequestered five-fold development tuning for the BDB task suite.

Development games are selected by :mod:`bdb_study.design` before model scores
are inspected and never occur in a confirmatory partition.  This module keeps
candidate ordering and exact-tie behavior explicit and receipt-bound.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import numpy as np

from .adapters.common import PreparedTask
from .contracts import (
    GLM_ALPHA_GRID,
    HORIZON_SCALE_SUPPORT_REQUIREMENT,
    HORIZON_SCALE_TAIL_POLICY,
    NEURAL_DROPOUT_GRID,
    NEURAL_LEARNING_RATE_GRID,
    TRAJECTORY_GLM_ALPHA_GRID,
    TRANSFORMER_CAPACITY,
    sha256_json,
)
from .design import design_seed
from .models import implementation_family


DEVCV_SCHEMA_VERSION = "bdb-development-cv-v3"
DEVCV_CHECKPOINT_SCHEMA_VERSION = "bdb-development-cv-checkpoint-v2"
DEVCV_SHARD_SCHEMA_VERSION = "bdb-development-cv-shard-v1"
JOINT_NEURAL_TARGET_SCHEMA_VERSION = "bdb-joint-neural-trajectory-target-v1"
JOINT_NEURAL_TARGET_PROTOCOL = (
    "pooled_equal_model_weight_grouped_oof_mean_rmse_v1"
)
JOINT_NEURAL_FAMILIES = ("relnet", "attn_relnet", "set_transformer")
JOINT_TRAJECTORY_TARGET_ORDER = ("residual", "absolute")
JOINT_NEURAL_TARGET_DEFAULT_PATH = "joint_neural_trajectory_target.json"
MODEL_FAMILIES = (
    "glm",
    "lightgbm",
    "cnn",
    "transformer",
    "set_transformer",
    "relnet",
    "attn_relnet",
    "linear_structure",
    "boosted_structure",
)


def candidate_grid(outcome_type: str, family: str) -> tuple[dict[str, Any], ...]:
    """Return the locked, strongest-regularization-first candidate registry."""

    modeled_type = "binary" if outcome_type == "frame_event" else outcome_type
    family = implementation_family(family)
    if family == "glm":
        alphas = TRAJECTORY_GLM_ALPHA_GRID if modeled_type == "trajectory" else GLM_ALPHA_GRID
        base = tuple(
            {
                "alpha": alpha,
                "epochs": 50,
                "batch_size": 64,
                "learning_rate": "optimal",
                "eta0": 0.01,
            }
            for alpha in alphas
        )
        if modeled_type == "trajectory":
            return tuple(
                {**config, "trajectory_target": target}
                for config in base
                for target in ("residual", "absolute")
            )
        return base
    if family == "lightgbm":
        common = {"n_estimators": 200}
        base = (
            # Strongest L1/L2 and child-size regularization is first so the
            # stable exact-tie rule implements the protocol.
            {**common, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 0.5},
            {**common, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {**common, "learning_rate": 0.05, "max_depth": 7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {**common, "learning_rate": 0.1, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
        )
        if modeled_type == "trajectory":
            return tuple(
                {**config, "trajectory_target": target}
                for config in base
                for target in ("residual", "absolute")
            )
        return base
    if family in {"cnn", "transformer", "set_transformer", "relnet", "attn_relnet"}:
        common: dict[str, Any] = {
            "max_epochs": 50,
            "patience": 10,
            "batch_size": 64,
            "deterministic": True,
        }
        if family in {"transformer", "set_transformer"}:
            common.update(TRANSFORMER_CAPACITY)
        elif family in {"relnet", "attn_relnet"}:
            common.update({"d_model": 48, "edge_dim": 8, "edge_hidden": 64})
        # Dropout .3 is the stronger regularizer and is listed first.  Within
        # a dropout level, the long-standing 1e-3 configuration is first.
        base = tuple(
            {**common, "learning_rate": learning_rate, "dropout": dropout}
            for dropout in NEURAL_DROPOUT_GRID
            for learning_rate in NEURAL_LEARNING_RATE_GRID
        )
        if modeled_type == "trajectory":
            return tuple(
                {**config, "trajectory_target": target}
                for config in base
                for target in ("residual", "absolute")
            )
        return base
    raise ValueError(f"unknown model family {family!r}")


def fit_development_horizon_scale(
    horizon_residual_median: np.ndarray,
    observed_count: np.ndarray,
    *,
    floor: float = 0.25,
) -> np.ndarray:
    """Pool grouped-OOF errors into a frozen nondecreasing horizon scale.

    Variable-length official paths need not give the sequestered development
    games target support at every confirmatory horizon.  Positive-count
    horizons must therefore form one nonempty prefix.  We fit the locked PAVA
    rule on that prefix and use its final level as the unique minimal
    nondecreasing (right-constant) completion of an unobserved trailing suffix.
    """

    medians = np.asarray(horizon_residual_median, dtype=np.float64).reshape(-1)
    raw_counts = np.asarray(observed_count)
    if raw_counts.dtype.kind not in {"i", "u"}:
        raise ValueError("grouped-OOF horizon counts must be integers")
    counts = raw_counts.astype(np.int64, copy=False).reshape(-1)
    if (
        len(medians) == 0
        or medians.shape != counts.shape
        or np.any(counts < 0)
        or np.any(np.diff(counts) > 0)
        or not math.isfinite(float(floor))
        or floor <= 0.0
    ):
        raise ValueError("grouped-OOF horizon scale statistics are invalid")
    observed = counts > 0
    if (
        not bool(observed[0])
        or np.any(observed[1:] & ~observed[:-1])
        or np.any(~np.isfinite(medians[observed]))
        or np.any(medians[observed] < 0.0)
        or np.any(np.isfinite(medians[~observed]))
    ):
        raise ValueError(
            "grouped-OOF horizon support must be one positive-count prefix"
        )
    observed_horizons = int(np.sum(observed))
    raw = np.maximum(medians[:observed_horizons], float(floor))
    observed_counts = counts[:observed_horizons]
    # Weighted pool-adjacent-violators, preserving the greater information at
    # early horizons without allowing a scientifically implausible decrease.
    levels: list[float] = []
    weights: list[int] = []
    spans: list[int] = []
    for value, count in zip(raw, observed_counts):
        levels.append(float(value))
        weights.append(int(count))
        spans.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            pooled = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / weight
            span = spans[-2] + spans[-1]
            levels[-2:] = [pooled]
            weights[-2:] = [weight]
            spans[-2:] = [span]
    fitted = np.repeat(np.asarray(levels, dtype=np.float64), spans)
    if observed_horizons == len(medians):
        return fitted
    return np.pad(
        fitted,
        (0, len(medians) - observed_horizons),
        mode="constant",
        constant_values=float(fitted[-1]),
    )


def _trajectory_horizon_audit_arrays(
    audit: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one fold/candidate's receipt-only horizon statistics."""

    medians = np.asarray(
        audit.get("horizon_scale_median"), dtype=np.float64
    ).reshape(-1)
    raw_count = np.asarray(audit.get("horizon_scale_count"))
    if raw_count.dtype.kind not in {"i", "u"}:
        raise ValueError("BDB2026 grouped-OOF horizon counts must be integers")
    counts = raw_count.astype(np.int64, copy=False).reshape(-1)
    if medians.shape != (94,) or counts.shape != (94,):
        raise ValueError(
            "BDB2026 development folds must emit 94-horizon OOF scale statistics"
        )
    observed = counts > 0
    if (
        np.any(counts < 0)
        or np.any(np.diff(counts) > 0)
        or not bool(observed[0])
        or np.any(observed[1:] & ~observed[:-1])
        or np.any(~np.isfinite(medians[observed]))
        or np.any(medians[observed] < 0.0)
        or np.any(np.isfinite(medians[~observed]))
    ):
        raise ValueError("BDB2026 grouped-OOF horizon statistics are invalid")
    return medians, counts


def _augment_trajectory_selection(
    selection: dict[str, Any], rows: list[dict[str, Any]], task: PreparedTask
) -> dict[str, Any]:
    if task.task_id != "bdb2026_trajectory":
        return selection
    selected = dict(selection["selected"])
    chosen = [
        row
        for row in rows
        if int(row["candidate_index"]) == int(selected["candidate_index"])
    ]
    medians = []
    counts = []
    for row in chosen:
        audit = row.get("audit", {})
        fold_medians, fold_counts = _trajectory_horizon_audit_arrays(audit)
        medians.append(fold_medians)
        counts.append(fold_counts)
    if len(chosen) != 5:
        raise ValueError(
            "BDB2026 development folds must emit 94-horizon OOF scale statistics"
        )
    # Each grouped fold contributes its horizon median with its number of OOF
    # paths as the PAVA weight. This keeps game folds sequestered and avoids
    # treating the tuning receipt as a raw-prediction data dump.
    median_matrix = np.stack(medians)
    count_matrix = np.stack(counts)
    pooled_median = np.full(median_matrix.shape[1], np.nan, dtype=np.float64)
    pooled_count = np.sum(count_matrix, axis=0)
    for horizon in range(median_matrix.shape[1]):
        observed = count_matrix[:, horizon] > 0
        if not np.any(observed):
            continue
        values = median_matrix[observed, horizon]
        weights = count_matrix[observed, horizon]
        order = np.argsort(values, kind="stable")
        ordered_weight = weights[order]
        cutoff = (ordered_weight.sum() + 1) // 2
        position = int(np.searchsorted(np.cumsum(ordered_weight), cutoff))
        pooled_median[horizon] = values[order[position]]
    scale = fit_development_horizon_scale(
        pooled_median, pooled_count, floor=0.25
    )
    observed_horizons = int(np.sum(pooled_count > 0))
    config = dict(selected["config"])
    config["dev_horizon_scale"] = [float(value) for value in scale]
    config["dev_horizon_scale_sha256"] = sha256_json(config["dev_horizon_scale"])
    config["dev_horizon_scale_observed_through"] = observed_horizons
    config["dev_horizon_scale_support_requirement"] = (
        HORIZON_SCALE_SUPPORT_REQUIREMENT
    )
    config["dev_horizon_scale_tail_policy"] = HORIZON_SCALE_TAIL_POLICY
    selected["config"] = config
    selected["config_hash"] = config_hash(config)
    return {**selection, "selected": selected}


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_hash"}
    return hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _with_receipt_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("receipt_hash", None)
    result["receipt_hash"] = _receipt_hash(result)
    return result


def select_candidate(rows: list[dict[str, Any]], *, expected_folds: int = 5) -> dict[str, Any]:
    """Select the smallest mean loss; exact ties retain manifest order."""

    if not rows:
        raise ValueError("development CV contains no candidate rows")
    indices = sorted({int(row["candidate_index"]) for row in rows})
    if indices != list(range(len(indices))):
        raise ValueError("candidate indices must be consecutive from zero")
    summaries: list[dict[str, Any]] = []
    for index in indices:
        candidate_rows = [row for row in rows if int(row["candidate_index"]) == index]
        folds = sorted(int(row["fold"]) for row in candidate_rows)
        if folds != list(range(1, expected_folds + 1)):
            raise ValueError(f"candidate {index} does not contain exactly five folds")
        hashes = {str(row["config_hash"]) for row in candidate_rows}
        configs = {json.dumps(row["config"], sort_keys=True) for row in candidate_rows}
        losses = np.asarray([row["validation_loss"] for row in candidate_rows], dtype=float)
        if len(hashes) != 1 or len(configs) != 1 or not np.all(np.isfinite(losses)):
            raise ValueError(f"candidate {index} rows are inconsistent or non-finite")
        summaries.append(
            {
                "candidate_index": index,
                "config": candidate_rows[0]["config"],
                "config_hash": candidate_rows[0]["config_hash"],
                "mean_validation_loss": float(losses.mean()),
                "sd_validation_loss": float(losses.std(ddof=1)),
                "folds": expected_folds,
            }
        )
    # Python's min is stable. Candidate order therefore implements the exact
    # tie rule without a floating tolerance that could relabel near ties.
    selected = min(summaries, key=lambda row: row["mean_validation_loss"])
    return {"selected": selected, "candidates": summaries}


def _target_candidate_selection(
    rows: list[dict[str, Any]], target: str
) -> dict[str, Any]:
    """Select one family's candidate within a frozen trajectory target."""

    if target not in JOINT_TRAJECTORY_TARGET_ORDER:
        raise ValueError(f"unknown trajectory target {target!r}")
    all_candidates = select_candidate(rows)
    eligible = [
        dict(candidate)
        for candidate in all_candidates["candidates"]
        if candidate.get("config", {}).get("trajectory_target") == target
    ]
    if len(eligible) != len(NEURAL_DROPOUT_GRID) * len(NEURAL_LEARNING_RATE_GRID):
        raise ValueError(
            f"trajectory target {target!r} does not contain the locked neural LR/dropout grid"
        )
    # Candidate summaries are in original manifest order; stable min retains
    # the existing stronger-regularization/manifest-order tie rule.
    selected = min(eligible, key=lambda row: row["mean_validation_loss"])
    selected_index = int(selected["candidate_index"])
    matrix = []
    for candidate in eligible:
        candidate_index = int(candidate["candidate_index"])
        fold_rows = sorted(
            (
                row
                for row in rows
                if int(row["candidate_index"]) == candidate_index
            ),
            key=lambda row: int(row["fold"]),
        )
        if [int(row["fold"]) for row in fold_rows] != list(range(1, 6)):
            raise ValueError(
                f"trajectory candidate {candidate_index} lacks five ordered folds"
            )
        matrix.append(
            {
                "candidate_index": candidate_index,
                "config": dict(candidate["config"]),
                "config_hash": str(candidate["config_hash"]),
                "fold_rmse": [float(row["validation_loss"]) for row in fold_rows],
                "mean_rmse": float(candidate["mean_validation_loss"]),
            }
        )
    return {
        "candidate_indices": [int(row["candidate_index"]) for row in eligible],
        "candidate_matrix": matrix,
        "selected_candidate_index": selected_index,
        "selected_config": dict(selected["config"]),
        "selected_config_hash": str(selected["config_hash"]),
        "selected_mean_rmse": float(selected["mean_validation_loss"]),
    }


def _raw_receipt_from_finalized(
    receipt: Mapping[str, Any], task: PreparedTask
) -> dict[str, Any]:
    """Recover the immutable pre-joint receipt bound by a finalized receipt."""

    if "joint_neural_target_selection" not in receipt:
        return dict(receipt)
    binding = receipt.get("joint_neural_target_selection")
    if not isinstance(binding, Mapping):
        raise ValueError("joint neural target binding is malformed")
    rows = receipt.get("fold_scores")
    if not isinstance(rows, list):
        raise ValueError("finalized development receipt has no fold scores")
    independent = _augment_trajectory_selection(select_candidate(rows), rows, task)
    raw = {
        key: value
        for key, value in receipt.items()
        if key not in {"joint_neural_target_selection", "receipt_hash"}
    }
    raw["selected"] = independent["selected"]
    raw["candidates"] = independent["candidates"]
    raw = _with_receipt_hash(raw)
    if raw["receipt_hash"] != binding.get("source_development_receipt_hash"):
        raise ValueError("finalized receipt does not reconstruct its bound raw receipt")
    return raw


def _joint_receipt_model_id(receipt: Mapping[str, Any], family: str) -> str:
    rows = receipt.get("fold_scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{family} development receipt has no fold-score matrix")
    model_ids = {str(row.get("model_id")) for row in rows if isinstance(row, Mapping)}
    if len(model_ids) != 1:
        raise ValueError(f"{family} development receipt has inconsistent model IDs")
    model_id = next(iter(model_ids))
    if model_id != family:
        raise ValueError(
            f"joint neural target selection requires canonical model ID {family!r}, got {model_id!r}"
        )
    return model_id


def _validate_joint_task(task: PreparedTask, design: Mapping[str, Any]) -> None:
    if (
        task.task_id != "bdb2026_trajectory"
        or task.outcome_type != "trajectory"
        or task.primary_metric != "rmse"
        or design.get("task_id") != task.task_id
    ):
        raise ValueError(
            "joint neural trajectory-target selection is restricted to BDB2026 RMSE"
        )


def _validated_raw_joint_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    task: PreparedTask,
    design: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _validate_joint_task(task, design)
    if set(receipts) != set(JOINT_NEURAL_FAMILIES):
        raise ValueError(
            f"joint neural target selection requires exactly {JOINT_NEURAL_FAMILIES!r}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for family in JOINT_NEURAL_FAMILIES:
        receipt = dict(receipts[family])
        model_id = _joint_receipt_model_id(receipt, family)
        validate_development_receipt(
            receipt,
            task,
            design,
            family,
            model_id=model_id,
            expected_provenance=expected_provenance,
        )
        raw = _raw_receipt_from_finalized(receipt, task)
        if "joint_neural_target_selection" in receipt:
            validate_development_receipt(
                raw,
                task,
                design,
                family,
                model_id=model_id,
                expected_provenance=expected_provenance,
            )
        validated[family] = raw
    return validated


def _joint_neural_target_payload(
    raw_receipts: Mapping[str, Mapping[str, Any]],
    task: PreparedTask,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    source_receipts: dict[str, Any] = {}
    per_target: dict[str, Any] = {}
    for family in JOINT_NEURAL_FAMILIES:
        receipt = raw_receipts[family]
        source_receipts[family] = {
            "model_id": _joint_receipt_model_id(receipt, family),
            "development_receipt_hash": str(receipt["receipt_hash"]),
            "candidate_matrix_sha256": sha256_json(receipt["fold_scores"]),
            "candidate_summaries_sha256": sha256_json(receipt["candidates"]),
            "provenance_sha256": sha256_json(receipt["provenance"]),
        }
    for target in JOINT_TRAJECTORY_TARGET_ORDER:
        family_selections = {
            family: _target_candidate_selection(
                list(raw_receipts[family]["fold_scores"]), target
            )
            for family in JOINT_NEURAL_FAMILIES
        }
        pooled = float(
            sum(
                family_selections[family]["selected_mean_rmse"]
                for family in JOINT_NEURAL_FAMILIES
            )
            / len(JOINT_NEURAL_FAMILIES)
        )
        per_target[target] = {
            "family_selections": family_selections,
            "pooled_equal_weight_mean_rmse": pooled,
        }
    # Iteration follows the frozen target order, so an exact pooled-score tie
    # deterministically selects residual without a floating tolerance.
    selected_target = min(
        JOINT_TRAJECTORY_TARGET_ORDER,
        key=lambda target: per_target[target]["pooled_equal_weight_mean_rmse"],
    )
    payload = {
        "schema_version": JOINT_NEURAL_TARGET_SCHEMA_VERSION,
        "protocol": JOINT_NEURAL_TARGET_PROTOCOL,
        "task_id": task.task_id,
        "design_hash": design.get("design_hash"),
        "development_registry_hash": design.get("game_registry", {}).get(
            "registry_hash"
        ),
        "native_metric": "rmse",
        "families": list(JOINT_NEURAL_FAMILIES),
        "model_ids": {family: family for family in JOINT_NEURAL_FAMILIES},
        "target_order": list(JOINT_TRAJECTORY_TARGET_ORDER),
        "tie_rule": "first_target_in_frozen_order",
        "source_development_receipts": source_receipts,
        "per_target": per_target,
        "selected_target": selected_target,
        "selected_pooled_equal_weight_mean_rmse": per_target[selected_target][
            "pooled_equal_weight_mean_rmse"
        ],
    }
    return _with_receipt_hash(payload)


def build_joint_neural_target_receipt(
    receipts: Mapping[str, Mapping[str, Any]],
    task: PreparedTask,
    design: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the three-family BDB2026 decoder-target selection receipt."""

    raw = _validated_raw_joint_receipts(
        receipts,
        task,
        design,
        expected_provenance=expected_provenance,
    )
    return _joint_neural_target_payload(raw, task, design)


def validate_joint_neural_target_receipt(
    joint_receipt: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    task: PreparedTask,
    design: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay a joint target decision from both candidate matrices."""

    expected_fields = {
        "schema_version",
        "protocol",
        "task_id",
        "design_hash",
        "development_registry_hash",
        "native_metric",
        "families",
        "model_ids",
        "target_order",
        "tie_rule",
        "source_development_receipts",
        "per_target",
        "selected_target",
        "selected_pooled_equal_weight_mean_rmse",
        "receipt_hash",
    }
    if set(joint_receipt) != expected_fields:
        raise ValueError("joint neural target receipt fields differ from the locked schema")
    if (
        joint_receipt.get("schema_version") != JOINT_NEURAL_TARGET_SCHEMA_VERSION
        or joint_receipt.get("protocol") != JOINT_NEURAL_TARGET_PROTOCOL
        or joint_receipt.get("target_order") != list(JOINT_TRAJECTORY_TARGET_ORDER)
        or joint_receipt.get("tie_rule") != "first_target_in_frozen_order"
    ):
        raise ValueError("joint neural target receipt protocol differs")
    raw = _validated_raw_joint_receipts(
        receipts,
        task,
        design,
        expected_provenance=expected_provenance,
    )
    replayed = _joint_neural_target_payload(raw, task, design)
    if dict(joint_receipt) != replayed:
        raise ValueError("joint neural target receipt does not replay")
    finalized_bindings = {
        family: receipt.get("joint_neural_target_selection")
        for family, receipt in receipts.items()
        if "joint_neural_target_selection" in receipt
    }
    if finalized_bindings and set(finalized_bindings) != set(JOINT_NEURAL_FAMILIES):
        raise ValueError(
            "joint neural target replay cannot mix raw and finalized family receipts"
        )
    if finalized_bindings:
        paths = set()
        for family in JOINT_NEURAL_FAMILIES:
            binding = finalized_bindings[family]
            if not isinstance(binding, Mapping):
                raise ValueError(f"{family} joint neural target binding is malformed")
            if (
                binding.get("joint_receipt_hash") != joint_receipt.get("receipt_hash")
                or binding.get("selected_target")
                != joint_receipt.get("selected_target")
            ):
                raise ValueError(
                    f"{family} finalized receipt does not bind the supplied joint receipt"
                )
            paths.add(str(binding.get("joint_receipt_path")))
        if len(paths) != 1:
            raise ValueError(
                "finalized neural receipts bind different joint receipt paths"
            )
    return dict(joint_receipt)


def _joint_receipt_path(value: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ValueError("joint neural target receipt path must be a safe relative path")
    return path.as_posix()


def finalize_joint_neural_development_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    task: PreparedTask,
    design: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
    joint_receipt_path: str = JOINT_NEURAL_TARGET_DEFAULT_PATH,
) -> dict[str, Any]:
    """Freeze all neural configs to one jointly selected trajectory target.

    The input receipts remain immutable/raw. Returned finalized receipts bind
    the joint decision while preserving each family's own chosen LR/dropout and
    grouped-OOF 94-horizon scale curve.
    """

    raw = _validated_raw_joint_receipts(
        receipts,
        task,
        design,
        expected_provenance=expected_provenance,
    )
    joint = _joint_neural_target_payload(raw, task, design)
    target = str(joint["selected_target"])
    receipt_path = _joint_receipt_path(joint_receipt_path)
    finalized: dict[str, dict[str, Any]] = {}
    for family in JOINT_NEURAL_FAMILIES:
        source = raw[family]
        rows = list(source["fold_scores"])
        target_selection = _target_candidate_selection(rows, target)
        selected_summary = next(
            candidate
            for candidate in source["candidates"]
            if int(candidate["candidate_index"])
            == int(target_selection["selected_candidate_index"])
        )
        augmented = _augment_trajectory_selection(
            {
                "selected": dict(selected_summary),
                "candidates": list(source["candidates"]),
            },
            rows,
            task,
        )
        binding = {
            "schema_version": JOINT_NEURAL_TARGET_SCHEMA_VERSION,
            "protocol": JOINT_NEURAL_TARGET_PROTOCOL,
            "joint_receipt_path": receipt_path,
            "joint_receipt_hash": joint["receipt_hash"],
            "selected_target": target,
            "source_development_receipt_hash": source["receipt_hash"],
            "candidate_matrix_sha256": sha256_json(source["fold_scores"]),
        }
        payload = {
            key: value for key, value in source.items() if key != "receipt_hash"
        }
        payload["selected"] = augmented["selected"]
        payload["candidates"] = augmented["candidates"]
        payload["joint_neural_target_selection"] = binding
        finalized[family] = _with_receipt_hash(payload)

    # Validate every local finalized selection and the joint replay
    # before returning anything to the freeze layer.
    for family in JOINT_NEURAL_FAMILIES:
        validate_development_receipt(
            finalized[family],
            task,
            design,
            family,
            model_id=family,
            expected_provenance=expected_provenance,
        )
    validate_joint_neural_target_receipt(
        joint,
        finalized,
        task,
        design,
        expected_provenance=expected_provenance,
    )
    return {
        "joint_receipt": joint,
        "finalized_development_receipts": finalized,
    }


Evaluator = Callable[
    [str, Mapping[str, Any], np.ndarray, np.ndarray, int, int],
    float | tuple[float, Mapping[str, Any]],
]


def _development_queue(family: str) -> str:
    normalized = implementation_family(family)
    return "cpu_tabular" if normalized in {"glm", "lightgbm"} else "gpu_neural"


def _development_grid_context(
    task: PreparedTask,
    design: Mapping[str, Any],
    family: str,
    *,
    model_id: str | None,
) -> dict[str, Any]:
    """Build the one canonical fold/candidate registry used by every dev path."""

    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported development family {family!r}")
    if design.get("task_id") != task.task_id:
        raise ValueError("prepared task and design task IDs differ")
    registry = design.get("game_registry", {})
    development = {str(value) for value in registry.get("development_game_ids", [])}
    if not development:
        raise ValueError("design has no development games")
    game_ids = task.examples["game_id"].astype(str).to_numpy()
    missing = sorted(development - set(game_ids))
    if missing:
        raise ValueError(f"prepared task lacks development games: {missing[:10]}")
    candidates = candidate_grid(task.outcome_type, family)
    resolved_model_id = family if model_id is None else str(model_id)
    seed_model_id = (
        "shared_neural"
        if implementation_family(family) in set(JOINT_NEURAL_FAMILIES)
        else resolved_model_id
    )
    folds: list[dict[str, Any]] = []
    for fold_entry in registry.get("development_folds", []):
        fold = int(fold_entry["fold"])
        validation_games = {
            str(value) for value in fold_entry["validation_game_ids"]
        }
        training_games = development - validation_games
        if not training_games or validation_games & training_games:
            raise ValueError(f"invalid development fold {fold}")
        train_index = np.flatnonzero(np.isin(game_ids, list(training_games)))
        validation_index = np.flatnonzero(
            np.isin(game_ids, list(validation_games))
        )
        if len(train_index) == 0 or len(validation_index) == 0:
            raise ValueError(f"development fold {fold} has no examples")
        folds.append(
            {
                "fold": fold,
                "training_games": training_games,
                "validation_games": validation_games,
                "train_index": train_index,
                "validation_index": validation_index,
                "fit_seed": design_seed(
                    design,
                    "task",
                    task.task_id,
                    "development_cv",
                    fold,
                    "fit",
                    seed_model_id,
                ),
                "prediction_seed": design_seed(
                    design,
                    "task",
                    task.task_id,
                    "development_cv",
                    fold,
                    "prediction",
                    seed_model_id,
                ),
            }
        )
    if [entry["fold"] for entry in folds] != list(range(1, 6)):
        raise ValueError("development design must contain exactly ordered folds 1--5")
    return {
        "task": task,
        "design": design,
        "registry": registry,
        "family": family,
        "model_id": resolved_model_id,
        "queue": _development_queue(family),
        "candidates": candidates,
        "folds": folds,
    }


def _development_cell_identity(
    context: Mapping[str, Any],
    fold_entry: Mapping[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    task = context["task"]
    config = dict(context["candidates"][candidate_index])
    train_index = np.asarray(fold_entry["train_index"])
    validation_index = np.asarray(fold_entry["validation_index"])
    return {
        "task_id": task.task_id,
        "queue": str(context["queue"]),
        "family": str(context["family"]),
        "model_id": str(context["model_id"]),
        "fold": int(fold_entry["fold"]),
        "candidate_index": int(candidate_index),
        "config": config,
        "config_hash": config_hash(config),
        "fit_seed": int(fold_entry["fit_seed"]),
        "prediction_seed": int(fold_entry["prediction_seed"]),
        "train_games": len(fold_entry["training_games"]),
        "validation_games": len(fold_entry["validation_games"]),
        "train_examples": int(len(train_index)),
        "validation_examples": int(len(validation_index)),
    }


def _development_checkpoint_path(
    checkpoint_root: Path, fold: int, candidate_index: int
) -> Path:
    return (
        checkpoint_root
        / f"fold_{int(fold):02d}"
        / f"candidate_{int(candidate_index):02d}.json"
    )


def _normalize_development_filter(
    values: Sequence[int] | None,
    allowed: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    frozen = tuple(int(value) for value in allowed)
    if values is None:
        return frozen
    requested: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"development {label} filters must be integers")
        requested.append(int(value))
    if not requested:
        raise ValueError(f"development {label} filter cannot be empty")
    if len(requested) != len(set(requested)):
        raise ValueError(f"development {label} filter contains duplicates")
    unknown = sorted(set(requested) - set(frozen))
    if unknown:
        raise ValueError(
            f"development {label} filter is outside the frozen grid: {unknown}"
        )
    requested_set = set(requested)
    return tuple(value for value in frozen if value in requested_set)


def _run_development_cells(
    context: Mapping[str, Any],
    evaluator: Evaluator,
    *,
    provenance: Mapping[str, Any],
    checkpoint_root: Path | None,
    resume: bool,
    folds: Sequence[int],
    candidate_indices: Sequence[int],
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if resume and checkpoint_root is None:
        raise ValueError("resume requires a development checkpoint directory")
    requested_folds = set(int(value) for value in folds)
    requested_candidates = set(int(value) for value in candidate_indices)
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for fold_entry in context["folds"]:
        fold = int(fold_entry["fold"])
        if fold not in requested_folds:
            continue
        train_index = np.asarray(fold_entry["train_index"])
        validation_index = np.asarray(fold_entry["validation_index"])
        for candidate_index in range(len(context["candidates"])):
            if candidate_index not in requested_candidates:
                continue
            identity = _development_cell_identity(
                context, fold_entry, candidate_index
            )
            checkpoint_path = (
                None
                if checkpoint_root is None
                else _development_checkpoint_path(
                    checkpoint_root, fold, candidate_index
                )
            )
            if checkpoint_path is not None and checkpoint_path.exists():
                if not resume:
                    raise FileExistsError(
                        "development checkpoint already exists; pass resume=True: "
                        f"{checkpoint_path}"
                    )
                row = _load_development_checkpoint(
                    checkpoint_path,
                    expected_identity=identity,
                    expected_provenance=provenance,
                )
                status = "resumed"
            else:
                evaluated = evaluator(
                    str(context["family"]),
                    identity["config"],
                    train_index,
                    validation_index,
                    int(identity["fit_seed"]),
                    int(identity["prediction_seed"]),
                )
                if isinstance(evaluated, tuple):
                    loss, audit = evaluated
                else:
                    loss, audit = evaluated, {}
                if not math.isfinite(float(loss)) or float(loss) < 0.0:
                    raise ValueError("development evaluator returned an invalid loss")
                audit_value = dict(audit)
                try:
                    json.dumps(audit_value, sort_keys=True, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "development evaluator audit must be finite JSON data"
                    ) from exc
                row = {
                    **identity,
                    "validation_loss": float(loss),
                    "audit": audit_value,
                }
                if checkpoint_path is not None:
                    _write_development_checkpoint(
                        checkpoint_path,
                        row=row,
                        provenance=provenance,
                    )
                status = "complete"
            rows.append(row)
            status_record = {
                "queue": identity["queue"],
                "family": identity["family"],
                "fold": fold,
                "candidate_index": candidate_index,
                "config_hash": identity["config_hash"],
                "status": status,
            }
            statuses.append(status_record)
            if progress is not None:
                progress({**identity, "status": status})
    expected_count = len(requested_folds) * len(requested_candidates)
    if len(rows) != expected_count:
        raise ValueError("development filter did not resolve its exact frozen cell grid")
    return rows, statuses


def _development_receipt(
    context: Mapping[str, Any],
    rows: list[dict[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    task = context["task"]
    selection = _augment_trajectory_selection(select_candidate(rows), rows, task)
    payload: dict[str, Any] = {
        "schema_version": DEVCV_SCHEMA_VERSION,
        "task_id": task.task_id,
        "design_hash": context["design"].get("design_hash"),
        "development_registry_hash": context["registry"].get("registry_hash"),
        "family": str(context["family"]),
        "native_metric": task.primary_metric,
        "provenance": dict(provenance),
        "tie_rule": "exact_tie_stronger_regularization_then_manifest_order",
        "fold_scores": rows,
        **selection,
    }
    unsigned = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["receipt_hash"] = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    return payload


def run_development_cv(
    task: PreparedTask,
    design: Mapping[str, Any],
    family: str,
    evaluator: Evaluator,
    *,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one family on the design's permanently sequestered games.

    The evaluator receives ``(family, config, train_indices, validation_indices,
    fit_seed, prediction_seed)``.  It must return the task's native loss and
    may additionally return a JSON-compatible audit mapping.
    """

    context = _development_grid_context(
        task, design, family, model_id=model_id
    )
    provenance_value = {} if provenance is None else dict(provenance)
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
    rows, _ = _run_development_cells(
        context,
        evaluator,
        provenance=provenance_value,
        checkpoint_root=checkpoint_root,
        resume=resume,
        folds=[int(entry["fold"]) for entry in context["folds"]],
        candidate_indices=list(range(len(context["candidates"]))),
        progress=progress,
    )
    return _development_receipt(context, rows, provenance_value)


def run_development_cv_shard(
    task: PreparedTask,
    design: Mapping[str, Any],
    family: str,
    evaluator: Evaluator,
    *,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path,
    folds: Sequence[int] | None = None,
    candidate_indices: Sequence[int] | None = None,
    resume: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run only requested fits and emit checkpoints, never a family receipt."""

    if folds is None and candidate_indices is None:
        raise ValueError(
            "a development shard requires at least one fold or candidate filter"
        )
    context = _development_grid_context(
        task, design, family, model_id=model_id
    )
    selected_folds = _normalize_development_filter(
        folds,
        [int(entry["fold"]) for entry in context["folds"]],
        "fold",
    )
    selected_candidates = _normalize_development_filter(
        candidate_indices,
        list(range(len(context["candidates"]))),
        "candidate",
    )
    provenance_value = {} if provenance is None else dict(provenance)
    checkpoint_root = Path(checkpoint_dir)
    _, statuses = _run_development_cells(
        context,
        evaluator,
        provenance=provenance_value,
        checkpoint_root=checkpoint_root,
        resume=resume,
        folds=selected_folds,
        candidate_indices=selected_candidates,
        progress=progress,
    )
    payload: dict[str, Any] = {
        "schema_version": DEVCV_SHARD_SCHEMA_VERSION,
        "task_id": task.task_id,
        "design_hash": design.get("design_hash"),
        "development_registry_hash": context["registry"].get("registry_hash"),
        "queue": context["queue"],
        "family": family,
        "model_id": context["model_id"],
        "folds": list(selected_folds),
        "candidate_indices": list(selected_candidates),
        "cells": statuses,
        "completed": sum(item["status"] == "complete" for item in statuses),
        "resumed": sum(item["status"] == "resumed" for item in statuses),
        "final_receipt_emitted": False,
    }
    payload["shard_hash"] = _receipt_hash(payload)
    return payload


def reduce_development_checkpoints(
    task: PreparedTask,
    design: Mapping[str, Any],
    family: str,
    *,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Build one receipt only from the exact complete immutable checkpoint grid."""

    context = _development_grid_context(
        task, design, family, model_id=model_id
    )
    provenance_value = {} if provenance is None else dict(provenance)
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.is_dir():
        raise ValueError(
            f"development checkpoint directory is missing: {checkpoint_root}"
        )
    expected_paths: list[Path] = [
        _development_checkpoint_path(
            checkpoint_root, int(fold_entry["fold"]), candidate_index
        )
        for fold_entry in context["folds"]
        for candidate_index in range(len(context["candidates"]))
    ]
    expected_relative = {
        path.relative_to(checkpoint_root).as_posix() for path in expected_paths
    }
    observed_paths = [
        path
        for path in checkpoint_root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    observed_relative = {
        path.relative_to(checkpoint_root).as_posix() for path in observed_paths
    }
    missing = sorted(expected_relative - observed_relative)
    unexpected = sorted(observed_relative - expected_relative)
    if missing or unexpected:
        raise ValueError(
            "development checkpoint grid is incomplete or noncanonical: "
            f"missing={missing}, unexpected={unexpected}"
        )
    rows: list[dict[str, Any]] = []
    for fold_entry in context["folds"]:
        for candidate_index in range(len(context["candidates"])):
            path = _development_checkpoint_path(
                checkpoint_root, int(fold_entry["fold"]), candidate_index
            )
            if path.is_symlink():
                raise ValueError(
                    f"development checkpoint must not be a symlink: {path}"
                )
            identity = _development_cell_identity(
                context, fold_entry, candidate_index
            )
            rows.append(
                _load_development_checkpoint(
                    path,
                    expected_identity=identity,
                    expected_provenance=provenance_value,
                )
            )
    receipt = _development_receipt(context, rows, provenance_value)
    return validate_development_receipt(
        receipt,
        task,
        design,
        family,
        model_id=str(context["model_id"]),
        expected_provenance=provenance_value,
    )


def validate_development_receipt(
    receipt: Mapping[str, Any],
    task: PreparedTask,
    design: Mapping[str, Any],
    family: str,
    *,
    model_id: str,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and validate every scientific field of a tuning receipt."""

    expected_top_level = {
        "schema_version",
        "task_id",
        "design_hash",
        "development_registry_hash",
        "family",
        "native_metric",
        "provenance",
        "tie_rule",
        "fold_scores",
        "selected",
        "candidates",
        "receipt_hash",
    }
    has_joint_binding = "joint_neural_target_selection" in receipt
    if has_joint_binding:
        expected_top_level.add("joint_neural_target_selection")
    if set(receipt) != expected_top_level:
        raise ValueError("development receipt fields differ from the locked schema")
    if receipt.get("schema_version") != DEVCV_SCHEMA_VERSION:
        raise ValueError("development receipt schema is unsupported")
    if (
        receipt.get("task_id") != task.task_id
        or design.get("task_id") != task.task_id
        or receipt.get("family") != family
        or receipt.get("native_metric") != task.primary_metric
        or receipt.get("design_hash") != design.get("design_hash")
        or receipt.get("development_registry_hash")
        != design.get("game_registry", {}).get("registry_hash")
    ):
        raise ValueError("development receipt task/design identity differs")
    if receipt.get("provenance") != dict(expected_provenance):
        raise ValueError("development receipt provenance differs")
    if receipt.get("tie_rule") != "exact_tie_stronger_regularization_then_manifest_order":
        raise ValueError("development receipt tie rule differs")

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    expected_receipt_hash = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if receipt.get("receipt_hash") != expected_receipt_hash:
        raise ValueError("development receipt hash is invalid")

    rows = receipt.get("fold_scores")
    if not isinstance(rows, list):
        raise ValueError("development receipt fold scores are invalid")
    candidates = candidate_grid(task.outcome_type, family)
    seed_model_id = (
        "shared_neural"
        if implementation_family(family) in set(JOINT_NEURAL_FAMILIES)
        else str(model_id)
    )
    registry = design.get("game_registry", {})
    development = {str(value) for value in registry.get("development_game_ids", [])}
    game_ids = task.examples["game_id"].astype(str).to_numpy()
    expected_row_fields = {
        "task_id",
        "queue",
        "family",
        "model_id",
        "fold",
        "candidate_index",
        "config",
        "config_hash",
        "fit_seed",
        "prediction_seed",
        "train_games",
        "validation_games",
        "train_examples",
        "validation_examples",
        "validation_loss",
        "audit",
    }
    expected_rows = len(candidates) * 5
    if len(rows) != expected_rows:
        raise ValueError("development receipt has the wrong fold/candidate grid")
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ValueError("development receipt score row fields differ")
        fold = position // len(candidates) + 1
        candidate_index = position % len(candidates)
        config = dict(candidates[candidate_index])
        fold_entries = [
            entry
            for entry in registry.get("development_folds", [])
            if int(entry.get("fold", -1)) == fold
        ]
        if len(fold_entries) != 1:
            raise ValueError("development receipt design folds are invalid")
        validation_games = {
            str(value) for value in fold_entries[0].get("validation_game_ids", [])
        }
        training_games = development - validation_games
        train_examples = int(np.isin(game_ids, list(training_games)).sum())
        validation_index = np.flatnonzero(
            np.isin(game_ids, list(validation_games))
        )
        validation_examples = int(len(validation_index))
        expected_identity = {
            "task_id": task.task_id,
            "queue": _development_queue(family),
            "family": family,
            "model_id": str(model_id),
            "fold": fold,
            "candidate_index": candidate_index,
            "config": config,
            "config_hash": config_hash(config),
            "fit_seed": design_seed(
                design, "task", task.task_id, "development_cv", fold, "fit", seed_model_id
            ),
            "prediction_seed": design_seed(
                design,
                "task",
                task.task_id,
                "development_cv",
                fold,
                "prediction",
                seed_model_id,
            ),
            "train_games": len(training_games),
            "validation_games": len(validation_games),
            "train_examples": train_examples,
            "validation_examples": validation_examples,
        }
        if {key: row.get(key) for key in expected_identity} != expected_identity:
            raise ValueError("development receipt score identity differs")
        loss = row.get("validation_loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or not isinstance(row.get("audit"), dict)
        ):
            raise ValueError("development receipt score is invalid")
        if task.task_id == "bdb2026_trajectory":
            _, observed_counts = _trajectory_horizon_audit_arrays(row["audit"])
            expected_counts = np.asarray(
                task.target_mask[validation_index], dtype=bool
            ).sum(axis=0, dtype=np.int64)
            if not np.array_equal(observed_counts, expected_counts):
                raise ValueError(
                    "BDB2026 grouped-OOF horizon counts differ from the prepared fold mask"
                )

    independent = select_candidate(rows)
    if has_joint_binding:
        if task.task_id != "bdb2026_trajectory" or family not in JOINT_NEURAL_FAMILIES:
            raise ValueError(
                "joint neural target binding is only valid for BDB2026 primary neural families"
            )
        binding = receipt.get("joint_neural_target_selection")
        expected_binding_fields = {
            "schema_version",
            "protocol",
            "joint_receipt_path",
            "joint_receipt_hash",
            "selected_target",
            "source_development_receipt_hash",
            "candidate_matrix_sha256",
        }
        if not isinstance(binding, Mapping) or set(binding) != expected_binding_fields:
            raise ValueError("joint neural target binding fields differ")
        selected_target = binding.get("selected_target")
        receipt_hash = binding.get("joint_receipt_hash")
        source_hash = binding.get("source_development_receipt_hash")
        if (
            binding.get("schema_version") != JOINT_NEURAL_TARGET_SCHEMA_VERSION
            or binding.get("protocol") != JOINT_NEURAL_TARGET_PROTOCOL
            or selected_target not in JOINT_TRAJECTORY_TARGET_ORDER
            or not isinstance(receipt_hash, str)
            or len(receipt_hash) != 64
            or any(character not in "0123456789abcdef" for character in receipt_hash)
            or not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
            or binding.get("candidate_matrix_sha256") != sha256_json(rows)
        ):
            raise ValueError("joint neural target binding is invalid")
        _joint_receipt_path(str(binding.get("joint_receipt_path", "")))
        target_selection = _target_candidate_selection(rows, str(selected_target))
        selected_summary = next(
            candidate
            for candidate in independent["candidates"]
            if int(candidate["candidate_index"])
            == int(target_selection["selected_candidate_index"])
        )
        replayed = _augment_trajectory_selection(
            {
                "selected": dict(selected_summary),
                "candidates": independent["candidates"],
            },
            rows,
            task,
        )
        _raw_receipt_from_finalized(receipt, task)
    else:
        replayed = _augment_trajectory_selection(independent, rows, task)
    if (
        receipt.get("selected") != replayed["selected"]
        or receipt.get("candidates") != replayed["candidates"]
    ):
        raise ValueError("development receipt selection does not replay")
    return dict(receipt)


def _canonical_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _immutable_json(value: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_payload(value)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable receipt {target}")
        return target
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _write_development_checkpoint(
    path: str | Path,
    *,
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": DEVCV_CHECKPOINT_SCHEMA_VERSION,
        "row": dict(row),
        "provenance": dict(provenance),
    }
    payload["checkpoint_hash"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return _immutable_json(payload, path)


def _load_development_checkpoint(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"development checkpoint is unreadable: {target}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema_version", "row", "provenance", "checkpoint_hash"}
        or payload.get("schema_version") != DEVCV_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(f"development checkpoint schema is invalid: {target}")
    unsigned = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    expected_hash = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("checkpoint_hash") != expected_hash:
        raise ValueError(f"development checkpoint hash is invalid: {target}")
    if payload.get("provenance") != dict(expected_provenance):
        raise ValueError(f"development checkpoint provenance drifted: {target}")
    row = payload.get("row")
    if (
        not isinstance(row, dict)
        or set(row) != set(expected_identity) | {"validation_loss", "audit"}
    ):
        raise ValueError(f"development checkpoint row is invalid: {target}")
    observed_identity = {key: row.get(key) for key in expected_identity}
    if observed_identity != dict(expected_identity):
        raise ValueError(f"development checkpoint identity drifted: {target}")
    loss = row.get("validation_loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0.0
        or not isinstance(row.get("audit"), dict)
    ):
        raise ValueError(f"development checkpoint result is invalid: {target}")
    return row


def write_development_receipt(receipt: Mapping[str, Any], path: str | Path) -> Path:
    return _immutable_json(receipt, path)


def write_joint_neural_target_receipt(
    receipt: Mapping[str, Any], path: str | Path
) -> Path:
    """Write an already replayed joint neural target receipt immutably."""

    if (
        receipt.get("schema_version") != JOINT_NEURAL_TARGET_SCHEMA_VERSION
        or receipt.get("protocol") != JOINT_NEURAL_TARGET_PROTOCOL
        or receipt.get("receipt_hash") != _receipt_hash(receipt)
    ):
        raise ValueError("joint neural target receipt is invalid")
    return _immutable_json(receipt, path)
