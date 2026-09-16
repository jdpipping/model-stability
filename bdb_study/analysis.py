"""Within-task repeated-study inference and descriptive suite synthesis.

The complete rerun is the sampling unit throughout this module. In
particular, the max-t bootstrap resamples a repeat's complete model-by-
six-anchor result vector; it never resamples cells or examples separately.
Cross-task output is deliberately descriptive because repeat IDs have no
paired interpretation across Big Data Bowl releases.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import skill_score


DEFAULT_MODELS = ("glm", "lightgbm", "cnn", "transformer")
LEGACY_STRUCTURE_MODELS = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
)
STRUCTURE_MODELS = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
STRUCTURE_FAMILIES = (
    "glm",
    "lightgbm",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_TARGET_COVERAGE = 0.90
CANONICAL_MODEL_COUNT = 5
CANONICAL_ANCHOR_COUNT = 6
CANONICAL_CONTRAST_COUNT = math.comb(CANONICAL_MODEL_COUNT, 2) * CANONICAL_ANCHOR_COUNT
CANONICAL_MODEL_ANCHOR_COUNT = CANONICAL_MODEL_COUNT * CANONICAL_ANCHOR_COUNT
_STANDARD_NORMAL = NormalDist()

# Ordered deliberately: the first four are required scientific summaries and
# the rest are task-native aliases or prespecified secondary outcomes.
SUMMARY_METRIC_ORDER = (
    "primary_loss",
    "game_equal_loss",
    "null_loss",
    "skill",
    "raw_brier",
    "game_equal_brier",
    "calibrated_brier",
    "raw_log_loss",
    "calibrated_log_loss",
    "game_equal_calibrated_brier",
    "game_equal_raw_log_loss",
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
    "crps",
    "raw_crps",
    "game_equal_crps",
    "coverage",
    "coverage_game_equal",
    "coverage_example_weighted",
    "interval_width",
    "width_game_equal",
    "interval_width_example_weighted",
    "inclusive_class_count",
    "interval_width_sd",
    "rmse",
    "official_pooled_rmse",
    "game_equal_rmse",
    "path_equal_rmse",
    "path_coverage",
    "game_equal_path_coverage",
    "mean_tube_diameter",
    "game_equal_tube_diameter",
)
REQUIRED_SUMMARY_METRICS = SUMMARY_METRIC_ORDER[:4]
DEFAULT_RIBBON_METRICS = (
    "primary_loss",
    "game_equal_loss",
    "null_loss",
    "skill",
    "coverage",
    "interval_width",
)
DEFAULT_CHANGE_METRICS = (
    "primary_loss",
    "game_equal_loss",
    "skill",
    "coverage",
    "interval_width",
)


def _single_task_id(metrics: pd.DataFrame) -> str:
    if "task_id" not in metrics:
        raise ValueError("task metrics lack task_id")
    task_ids = tuple(sorted(set(metrics["task_id"].astype(str))))
    if len(task_ids) != 1:
        raise ValueError("within-task analysis requires exactly one task ID")
    return task_ids[0]


def _model_order(values: Iterable[Any]) -> tuple[str, ...]:
    observed = {str(value) for value in values}
    if observed == set(DEFAULT_MODELS):
        return DEFAULT_MODELS
    if observed == set(LEGACY_STRUCTURE_MODELS):
        return LEGACY_STRUCTURE_MODELS
    if observed == set(STRUCTURE_MODELS):
        return STRUCTURE_MODELS
    return tuple(sorted(observed))


def _validate_model_panel(models: Sequence[str], *, label: str) -> None:
    """Admit the current five-role panel or the read-only legacy four-role panel."""

    observed = tuple(str(value) for value in models)
    if observed not in {DEFAULT_MODELS, LEGACY_STRUCTURE_MODELS, STRUCTURE_MODELS}:
        raise ValueError(
            f"{label} requires the five-role prospective panel or the "
            "read-only four-role legacy panel"
        )


def _family_size(models: Sequence[str], anchors: Sequence[int]) -> tuple[int, int]:
    return len(models) * len(anchors), math.comb(len(models), 2) * len(anchors)


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"task metrics lack {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{column} contains non-finite values")
    return values


def _with_skill(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    if "skill" not in frame:
        primary = _numeric_values(frame, "primary_loss")
        null = _numeric_values(frame, "null_loss")
        frame["skill"] = [skill_score(model, baseline) for model, baseline in zip(primary, null)]
    else:
        _numeric_values(frame, "skill")
    return frame


def _validate_identity_columns(metrics: pd.DataFrame) -> None:
    required = {"task_id", "repeat", "model", "n_train"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"task metrics lack columns {sorted(missing)}")
    _single_task_id(metrics)
    identities = metrics[["repeat", "model", "n_train"]].copy()
    identities["repeat"] = pd.to_numeric(identities["repeat"], errors="coerce")
    identities["n_train"] = pd.to_numeric(identities["n_train"], errors="coerce")
    numeric = identities[["repeat", "n_train"]].to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric <= 0)
        or not np.array_equal(numeric, np.floor(numeric))
    ):
        raise ValueError("repeat and n_train must be positive integers")
    identities["repeat"] = identities["repeat"].astype(int)
    identities["n_train"] = identities["n_train"].astype(int)
    identities["model"] = identities["model"].astype(str)
    if identities.duplicated().any():
        raise ValueError("task metrics contain duplicate cells")


def validate_task_grid(
    metrics: pd.DataFrame,
    *,
    task_id: str,
    anchors: Iterable[int],
    repeats: int = 50,
    models: Iterable[str] = DEFAULT_MODELS,
) -> pd.DataFrame:
    """Validate the exact balanced grid and recompute skill from raw losses."""

    frame = metrics.copy()
    required = {
        "task_id",
        "repeat",
        "model",
        "n_train",
        "primary_loss",
        "game_equal_loss",
        "null_loss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"task metrics lack columns {sorted(missing)}")
    if set(frame["task_id"].astype(str)) != {str(task_id)}:
        raise ValueError("metrics contain the wrong task ID")
    model_values = tuple(str(value) for value in models)
    anchor_values = tuple(int(value) for value in anchors)
    if len(set(model_values)) != len(model_values):
        raise ValueError("the task grid models must be unique")
    _validate_model_panel(model_values, label="the task grid")
    if (
        len(anchor_values) != CANONICAL_ANCHOR_COUNT
        or tuple(sorted(set(anchor_values))) != anchor_values
    ):
        raise ValueError("the task grid must contain six strictly increasing anchors")
    if isinstance(repeats, bool) or int(repeats) < 2:
        raise ValueError("the task grid requires at least two complete reruns")
    expected = {
        (repeat, model, anchor)
        for repeat in range(1, int(repeats) + 1)
        for model in model_values
        for anchor in anchor_values
    }
    observed_rows = list(
        zip(
            frame["repeat"].astype(int),
            frame["model"].astype(str),
            frame["n_train"].astype(int),
        )
    )
    observed = set(observed_rows)
    if len(observed_rows) != len(observed):
        raise ValueError("task metrics contain duplicate cells")
    if observed != expected:
        missing_cells = sorted(expected - observed)[:10]
        extra_cells = sorted(observed - expected)[:10]
        raise ValueError(f"task grid mismatch; missing={missing_cells}, extra={extra_cells}")
    for column in ("primary_loss", "game_equal_loss", "null_loss"):
        values = _numeric_values(frame, column)
        if np.any(values < 0.0):
            raise ValueError(f"{column} contains invalid values")
    if np.any(frame["null_loss"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("null losses must be positive")
    frame["skill"] = [
        skill_score(model, null)
        for model, null in zip(frame["primary_loss"], frame["null_loss"])
    ]
    return frame.sort_values(["repeat", "n_train", "model"]).reset_index(drop=True)


def _summary_row(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("summary values must contain at least two finite observations")
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    standard_deviation = float(np.std(values, ddof=1))
    return {
        "mean": float(np.mean(values)),
        "sd": standard_deviation,
        "mcse": standard_deviation / math.sqrt(len(values)),
        "median": float(median),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": float(q75 - q25),
    }


def _available_metrics(metrics: pd.DataFrame, requested: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in requested if column in metrics.columns)


def summarize_task(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize between-rerun variability for each model and anchor.

    ``primary_loss_sd`` (also exposed as ``raw_loss_sd``) is the prespecified
    raw-loss instability estimate. Quantiles are repeat quantiles, not
    uncertainty intervals around the mean.
    """

    _validate_identity_columns(metrics)
    frame = _with_skill(metrics)
    missing = set(REQUIRED_SUMMARY_METRICS) - set(frame.columns)
    if missing:
        raise ValueError(f"task metrics lack required summary columns {sorted(missing)}")
    metric_columns = _available_metrics(frame, SUMMARY_METRIC_ORDER)
    rows: list[dict[str, Any]] = []
    for (task_id, model, n_train), group in frame.groupby(
        ["task_id", "model", "n_train"], sort=True
    ):
        if group["repeat"].duplicated().any():
            raise ValueError("summary group contains duplicate repeat IDs")
        row: dict[str, Any] = {
            "task_id": str(task_id),
            "model": str(model),
            "n_train": int(n_train),
            "repeats": int(len(group)),
        }
        for column in metric_columns:
            values = _numeric_values(group, column)
            for statistic, value in _summary_row(values).items():
                row[f"{column}_{statistic}"] = value
        row["raw_loss_sd"] = row["primary_loss_sd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_id", "model", "n_train"]).reset_index(drop=True)


def _canonical_repeat_tensor(
    metrics: pd.DataFrame,
    value: str,
) -> tuple[str, np.ndarray, tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    """Return repeat x model x anchor values after strict panel validation."""

    _validate_identity_columns(metrics)
    task_id = _single_task_id(metrics)
    values = _numeric_values(metrics, value)
    frame = metrics[["repeat", "model", "n_train"]].copy()
    frame["repeat"] = frame["repeat"].astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["n_train"] = frame["n_train"].astype(int)
    frame[value] = values
    models = _model_order(frame["model"])
    anchors = tuple(sorted(set(frame["n_train"].astype(int))))
    repeats = tuple(sorted(set(frame["repeat"].astype(int))))
    _validate_model_panel(models, label="within-task inference")
    if len(anchors) != CANONICAL_ANCHOR_COUNT:
        raise ValueError("within-task inference requires exactly six anchors")
    if len(repeats) < 2:
        raise ValueError("within-task inference requires at least two complete reruns")
    expected = {
        (repeat, model, anchor)
        for repeat in repeats
        for model in models
        for anchor in anchors
    }
    observed = set(
        zip(
            frame["repeat"].astype(int),
            frame["model"].astype(str),
            frame["n_train"].astype(int),
        )
    )
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(f"repeat blocks are incomplete; missing={missing}, extra={extra}")
    pivot = frame.pivot(index="repeat", columns=["model", "n_train"], values=value)
    columns = pd.MultiIndex.from_product([models, anchors], names=["model", "n_train"])
    pivot = pivot.reindex(index=repeats, columns=columns)
    if pivot.isna().any().any():
        raise ValueError("repeat matrix is incomplete")
    tensor = pivot.to_numpy(dtype=np.float64).reshape(len(repeats), len(models), len(anchors))
    return task_id, tensor, repeats, models, anchors


def _bootstrap_indices(
    repeat_count: int,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Materialize and identify one deterministic complete-repeat registry."""

    if (
        isinstance(draws, (bool, np.bool_))
        or not isinstance(draws, (int, np.integer))
        or int(draws) < 1
    ):
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("bootstrap seed must be an integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("bootstrap seed must be nonnegative")
    registry = np.random.default_rng(seed_value).integers(
        0,
        int(repeat_count),
        size=(int(draws), int(repeat_count)),
        dtype=np.int64,
    )
    # Bind both dimensions as well as the canonical little-endian values.  The
    # hash lets receipts prove that every taskwise statistic used the same
    # repeat-block draws without persisting a redundant index array.
    payload = json.dumps(
        {"draws": int(draws), "repeat_count": int(repeat_count)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + registry.astype("<i8", copy=False).tobytes(order="C")
    return registry, sha256(payload).hexdigest()


def _standard_error_floor(estimate: np.ndarray) -> np.ndarray:
    values = np.asarray(estimate, dtype=np.float64)
    return np.maximum(
        np.sqrt(np.finfo(np.float64).eps) * np.maximum(np.abs(values), 1.0),
        1e-15,
    )


def _t_critical(confidence_level: float, degrees_of_freedom: int, *, two_sided: bool) -> float:
    """Student-t quantile with a dependency-free high-order fallback."""

    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    if int(degrees_of_freedom) < 1:
        raise ValueError("degrees of freedom must be positive")
    probability = (1.0 + level) / 2.0 if two_sided else level
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(probability, df=int(degrees_of_freedom)))
    except (ImportError, ModuleNotFoundError):
        z = float(_STANDARD_NORMAL.inv_cdf(probability))
        nu = float(degrees_of_freedom)
        return float(
            z
            + (z**3 + z) / (4.0 * nu)
            + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * nu**2)
            + (
                3.0 * z**7
                + 19.0 * z**5
                + 17.0 * z**3
                - 15.0 * z
            )
            / (384.0 * nu**3)
        )


def _bca_interval(
    observed: float,
    bootstrap: np.ndarray,
    jackknife: np.ndarray,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float, float, float, float, float]:
    """Return a scalar BCa interval and its bias/acceleration diagnostics."""

    bootstrap_values = np.asarray(bootstrap, dtype=np.float64)
    jackknife_values = np.asarray(jackknife, dtype=np.float64)
    if (
        bootstrap_values.ndim != 1
        or jackknife_values.ndim != 1
        or len(bootstrap_values) < 1
        or len(jackknife_values) < 3
        or not math.isfinite(float(observed))
        or not np.all(np.isfinite(bootstrap_values))
        or not np.all(np.isfinite(jackknife_values))
    ):
        raise ValueError("BCa inputs must be finite scalar bootstrap and jackknife samples")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")

    less = float(np.count_nonzero(bootstrap_values < observed))
    proportion = less / float(len(bootstrap_values))
    clip = 0.5 / float(len(bootstrap_values))
    proportion = min(max(proportion, clip), 1.0 - clip)
    bias_correction = float(_STANDARD_NORMAL.inv_cdf(proportion))

    jackknife_mean = float(jackknife_values.mean())
    influence = jackknife_mean - jackknife_values
    sum_squares = float(np.sum(np.square(influence)))
    if 6.0 * math.pow(sum_squares, 1.5) <= np.finfo(np.float64).eps:
        acceleration = 0.0
    else:
        acceleration = float(
            np.sum(np.power(influence, 3.0))
            / (6.0 * math.pow(sum_squares, 1.5))
        )

    tail = (1.0 - level) / 2.0
    adjusted: list[float] = []
    for probability in (tail, 1.0 - tail):
        normal_quantile = float(_STANDARD_NORMAL.inv_cdf(probability))
        shifted = bias_correction + normal_quantile
        denominator = 1.0 - acceleration * shifted
        if abs(denominator) <= np.finfo(np.float64).eps:
            adjusted_probability = 0.0 if shifted < 0.0 else 1.0
        else:
            adjusted_probability = float(
                _STANDARD_NORMAL.cdf(
                    bias_correction + shifted / denominator
                )
            )
        adjusted.append(min(max(adjusted_probability, 0.0), 1.0))
    adjusted.sort()
    lower, upper = np.quantile(bootstrap_values, adjusted, method="linear")
    return (
        float(lower),
        float(upper),
        bias_correction,
        acceleration,
        float(adjusted[0]),
        float(adjusted[1]),
    )


def paired_max_t_intervals(
    metrics: pd.DataFrame,
    *,
    value: str = "primary_loss",
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
) -> pd.DataFrame:
    """Simultaneous 95% intervals for the complete pair-by-anchor family.

    A bootstrap index selects complete rerun rows from the model x 6 tensor.
    Every contrast in a draw is evaluated on exactly the same resampled repeat
    blocks: 60 for the five-role prospective panel and 36 for legacy panels.
    """

    task_id, tensor, _repeats, models, anchors = _canonical_repeat_tensor(metrics, value)
    contrast_columns: list[np.ndarray] = []
    labels: list[tuple[int, str, str]] = []
    for anchor_index, anchor in enumerate(anchors):
        for left_index, right_index in combinations(range(len(models)), 2):
            contrast_columns.append(
                tensor[:, left_index, anchor_index] - tensor[:, right_index, anchor_index]
            )
            labels.append((anchor, models[left_index], models[right_index]))
    _model_anchor_count, contrast_count = _family_size(models, anchors)
    if len(labels) != contrast_count:
        raise ValueError("max-t family is not the complete model-pair by anchor grid")
    contrast_matrix = np.column_stack(contrast_columns)
    repeat_count = contrast_matrix.shape[0]
    observed = contrast_matrix.mean(axis=0)
    standard_error = contrast_matrix.std(axis=0, ddof=1) / math.sqrt(repeat_count)
    floor = _standard_error_floor(observed)
    standard_error = np.maximum(standard_error, floor)
    bootstrap_indices, registry_hash = _bootstrap_indices(
        repeat_count, draws=draws, seed=seed
    )
    maxima = np.empty(int(draws), dtype=np.float64)
    for draw, index in enumerate(bootstrap_indices):
        sample = contrast_matrix[index]
        draw_mean = sample.mean(axis=0)
        draw_se = np.maximum(sample.std(axis=0, ddof=1) / math.sqrt(repeat_count), floor)
        maxima[draw] = np.max(np.abs((draw_mean - observed) / draw_se))
    critical = max(
        _t_critical(0.95, repeat_count - 1, two_sided=True),
        float(np.quantile(maxima, 0.95, method="higher")),
    )
    rows = []
    for index, (anchor, left, right) in enumerate(labels):
        rows.append(
            {
                "task_id": task_id,
                "metric": value,
                "n_train": int(anchor),
                "model_left": left,
                "model_right": right,
                "mean_difference": float(observed[index]),
                "standard_error": float(standard_error[index]),
                "simultaneous_lower": float(observed[index] - critical * standard_error[index]),
                "simultaneous_upper": float(observed[index] + critical * standard_error[index]),
                "confidence_level": 0.95,
                "max_t_critical": critical,
                "bootstrap_draws": int(draws),
                "bootstrap_unit": f"complete_repeat_{len(models)}x6_vector",
                "bootstrap_index_sha256": registry_hash,
                "family_contrasts": contrast_count,
            }
        )
    return pd.DataFrame(rows)


def bca_stability_intervals(
    metrics: pd.DataFrame,
    *,
    value: str = "primary_loss",
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    sd_floor: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BCa intervals for model SDs and all paired log-SD ratios.

    Every bootstrap row resamples a complete rerun's model-by-six-anchor
    vector.  The delete-one jackknife likewise removes a complete rerun.  SD
    ratios are analyzed on the log scale and also returned after exponentiation.
    """

    task_id, tensor, repeats, models, anchors = _canonical_repeat_tensor(metrics, value)
    repeat_count = len(repeats)
    if repeat_count < 3:
        raise ValueError("BCa stability inference requires at least three complete reruns")
    if isinstance(draws, bool) or int(draws) < 2:
        raise ValueError("BCa stability inference requires at least two bootstrap draws")
    floor_value = float(sd_floor)
    if not math.isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError("SD floor must be a positive finite number")
    bootstrap_indices, registry_hash = _bootstrap_indices(
        repeat_count, draws=draws, seed=seed
    )
    observed_sd = tensor.std(axis=0, ddof=1)
    bootstrap_sd = np.empty(
        (len(bootstrap_indices), len(models), len(anchors)), dtype=np.float64
    )
    for draw, index in enumerate(bootstrap_indices):
        bootstrap_sd[draw] = tensor[index].std(axis=0, ddof=1)
    jackknife_sd = np.empty(
        (repeat_count, len(models), len(anchors)), dtype=np.float64
    )
    for omitted in range(repeat_count):
        jackknife_sd[omitted] = np.delete(tensor, omitted, axis=0).std(axis=0, ddof=1)

    sd_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for anchor_index, anchor in enumerate(anchors):
            estimate = float(observed_sd[model_index, anchor_index])
            (
                lower,
                upper,
                bias,
                acceleration,
                adjusted_lower,
                adjusted_upper,
            ) = _bca_interval(
                estimate,
                bootstrap_sd[:, model_index, anchor_index],
                jackknife_sd[:, model_index, anchor_index],
                confidence_level=confidence_level,
            )
            sd_rows.append(
                {
                    "task_id": task_id,
                    "metric": value,
                    "n_train": int(anchor),
                    "model": model,
                    "sd": estimate,
                    "bca_lower": lower,
                    "bca_upper": upper,
                    "confidence_level": float(confidence_level),
                    "interval_method": "BCa",
                    "bias_correction": bias,
                    "acceleration": acceleration,
                    "adjusted_lower_quantile": adjusted_lower,
                    "adjusted_upper_quantile": adjusted_upper,
                    "repeats": repeat_count,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": f"complete_repeat_{len(models)}x6_vector",
                    "jackknife_unit": f"delete_one_complete_repeat_{len(models)}x6_vector",
                    "bootstrap_index_sha256": registry_hash,
                }
            )

    ratio_rows: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors):
        for left_index, right_index in combinations(range(len(models)), 2):
            left_sd = float(observed_sd[left_index, anchor_index])
            right_sd = float(observed_sd[right_index, anchor_index])
            estimate = float(
                math.log(max(left_sd, floor_value) / max(right_sd, floor_value))
            )
            raw_bootstrap_left = bootstrap_sd[:, left_index, anchor_index]
            raw_bootstrap_right = bootstrap_sd[:, right_index, anchor_index]
            raw_jackknife_left = jackknife_sd[:, left_index, anchor_index]
            raw_jackknife_right = jackknife_sd[:, right_index, anchor_index]
            bootstrap_log_ratio = np.log(
                np.maximum(raw_bootstrap_left, floor_value)
                / np.maximum(raw_bootstrap_right, floor_value)
            )
            jackknife_log_ratio = np.log(
                np.maximum(raw_jackknife_left, floor_value)
                / np.maximum(raw_jackknife_right, floor_value)
            )
            (
                lower,
                upper,
                bias,
                acceleration,
                adjusted_lower,
                adjusted_upper,
            ) = _bca_interval(
                estimate,
                bootstrap_log_ratio,
                jackknife_log_ratio,
                confidence_level=confidence_level,
            )
            ratio_rows.append(
                {
                    "task_id": task_id,
                    "metric": value,
                    "n_train": int(anchor),
                    "model_left": models[left_index],
                    "model_right": models[right_index],
                    "sd_left": left_sd,
                    "sd_right": right_sd,
                    "log_sd_ratio": estimate,
                    "sd_ratio": float(math.exp(estimate)),
                    "bca_log_lower": lower,
                    "bca_log_upper": upper,
                    "bca_ratio_lower": float(math.exp(lower)),
                    "bca_ratio_upper": float(math.exp(upper)),
                    "confidence_level": float(confidence_level),
                    "interval_method": "BCa",
                    "bias_correction": bias,
                    "acceleration": acceleration,
                    "adjusted_lower_quantile": adjusted_lower,
                    "adjusted_upper_quantile": adjusted_upper,
                    "bootstrap_zero_sd_draws": int(
                        np.count_nonzero(
                            (raw_bootstrap_left <= 0.0)
                            | (raw_bootstrap_right <= 0.0)
                        )
                    ),
                    "jackknife_zero_sd_values": int(
                        np.count_nonzero(
                            (raw_jackknife_left <= 0.0)
                            | (raw_jackknife_right <= 0.0)
                        )
                    ),
                    "regularized": (left_sd <= floor_value) or (right_sd <= floor_value),
                    "sd_regularization_floor": floor_value,
                    "repeats": repeat_count,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": f"complete_repeat_{len(models)}x6_vector",
                    "jackknife_unit": f"delete_one_complete_repeat_{len(models)}x6_vector",
                    "bootstrap_index_sha256": registry_hash,
                }
            )
    model_anchor_count, contrast_count = _family_size(models, anchors)
    if len(sd_rows) != model_anchor_count:
        raise ValueError("SD table is not the complete model-by-anchor family")
    if len(ratio_rows) != contrast_count:
        raise ValueError("log-SD-ratio table is not the complete contrast family")
    return pd.DataFrame(sd_rows), pd.DataFrame(ratio_rows)


def supported_best_conclusions(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Make a best-model claim only after every simultaneous comparison."""

    required = {
        "task_id",
        "metric",
        "n_train",
        "model_left",
        "model_right",
        "simultaneous_lower",
        "simultaneous_upper",
        "confidence_level",
        "bootstrap_draws",
        "bootstrap_unit",
        "bootstrap_index_sha256",
        "family_contrasts",
    }
    missing = required - set(contrasts.columns)
    if missing:
        raise ValueError(f"contrast table lacks columns {sorted(missing)}")
    task_ids = tuple(sorted(set(contrasts["task_id"].astype(str))))
    metrics = tuple(sorted(set(contrasts["metric"].astype(str))))
    anchors = tuple(sorted(set(contrasts["n_train"].astype(int))))
    models = _model_order(
        pd.concat([contrasts["model_left"], contrasts["model_right"]]).astype(str)
    )
    if len(task_ids) != 1 or len(metrics) != 1:
        raise ValueError("supported-best conclusions require one task and one metric")
    _validate_model_panel(models, label="supported-best conclusions")
    if len(anchors) != CANONICAL_ANCHOR_COUNT:
        raise ValueError("supported-best conclusions require six anchors")
    _model_anchor_count, contrast_count = _family_size(models, anchors)
    if len(contrasts) != contrast_count:
        raise ValueError("supported-best conclusions require the complete contrast family")
    endpoints = contrasts[["simultaneous_lower", "simultaneous_upper"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(endpoints))
        or np.any(endpoints[:, 0] > endpoints[:, 1])
    ):
        raise ValueError("simultaneous contrast confidence bounds are invalid")
    for column, expected in (
        ("confidence_level", 0.95),
        ("family_contrasts", contrast_count),
    ):
        values = pd.to_numeric(contrasts[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.allclose(values, expected):
            raise ValueError(f"contrast table has inconsistent {column}")
    for column in ("bootstrap_draws", "bootstrap_unit", "bootstrap_index_sha256"):
        if contrasts[column].nunique(dropna=False) != 1:
            raise ValueError(f"contrast table has inconsistent {column}")
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        subset = contrasts[contrasts["n_train"].astype(int).eq(anchor)]
        expected_pairs = {frozenset(pair) for pair in combinations(models, 2)}
        observed_pairs = {
            frozenset((str(row.model_left), str(row.model_right)))
            for row in subset.itertuples(index=False)
        }
        if len(subset) != math.comb(len(models), 2) or observed_pairs != expected_pairs:
            raise ValueError(f"anchor {anchor} lacks the complete model-pair contrasts")
        anchor_rows: list[dict[str, Any]] = []
        for candidate in models:
            supported_against: list[str] = []
            for comparator in models:
                if comparator == candidate:
                    continue
                pair = subset[
                    (
                        subset["model_left"].astype(str).eq(candidate)
                        & subset["model_right"].astype(str).eq(comparator)
                    )
                    | (
                        subset["model_left"].astype(str).eq(comparator)
                        & subset["model_right"].astype(str).eq(candidate)
                    )
                ]
                if len(pair) != 1:
                    raise ValueError("contrast lookup is not unique")
                record = pair.iloc[0]
                if str(record["model_left"]) == candidate:
                    beats = float(record["simultaneous_upper"]) < 0.0
                else:
                    beats = float(record["simultaneous_lower"]) > 0.0
                if beats:
                    supported_against.append(comparator)
            supported = len(supported_against) == len(models) - 1
            anchor_rows.append(
                {
                    "task_id": task_ids[0],
                    "metric": metrics[0],
                    "n_train": int(anchor),
                    "model": candidate,
                    "supported_against": json.dumps(supported_against),
                    "comparators_supported": len(supported_against),
                    "comparators_required": len(models) - 1,
                    "supported_best": supported,
                    "conclusion": "supported_best" if supported else "no_supported_best_claim",
                    "decision_rule": (
                        "simultaneous_95pct_mean_loss_ci_beats_every_comparator"
                    ),
                    "confidence_level": float(subset["confidence_level"].iloc[0]),
                    "bootstrap_draws": int(subset["bootstrap_draws"].iloc[0]),
                    "bootstrap_unit": str(subset["bootstrap_unit"].iloc[0]),
                    "bootstrap_index_sha256": str(
                        subset["bootstrap_index_sha256"].iloc[0]
                    ),
                    "family_contrasts": int(subset["family_contrasts"].iloc[0]),
                }
            )
        if sum(bool(row["supported_best"]) for row in anchor_rows) > 1:
            raise ValueError("simultaneous contrast conclusions imply multiple best models")
        rows.extend(anchor_rows)
    return pd.DataFrame(rows)


def simultaneous_coverage_lower_bounds(
    metrics: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """One-sided simultaneous mean-coverage lower bounds for the full panel."""

    target = float(target_coverage)
    level = float(confidence_level)
    if not 0.0 < target < 1.0:
        raise ValueError("target coverage must lie strictly between zero and one")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    task_id, tensor, repeats, models, anchors = _canonical_repeat_tensor(metrics, "coverage")
    if np.any((tensor < 0.0) | (tensor > 1.0)):
        raise ValueError("coverage values must lie in [0, 1]")
    repeat_count = len(repeats)
    observed = tensor.mean(axis=0)
    standard_error = tensor.std(axis=0, ddof=1) / math.sqrt(repeat_count)
    floor = _standard_error_floor(observed)
    standard_error = np.maximum(standard_error, floor)
    bootstrap_indices, registry_hash = _bootstrap_indices(
        repeat_count, draws=draws, seed=seed
    )
    maxima = np.empty(len(bootstrap_indices), dtype=np.float64)
    for draw, index in enumerate(bootstrap_indices):
        sample = tensor[index]
        draw_mean = sample.mean(axis=0)
        draw_se = np.maximum(
            sample.std(axis=0, ddof=1) / math.sqrt(repeat_count), floor
        )
        maxima[draw] = float(np.max((draw_mean - observed) / draw_se))
    critical = max(
        _t_critical(level, repeat_count - 1, two_sided=False),
        float(np.quantile(maxima, level, method="higher")),
    )
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for anchor_index, anchor in enumerate(anchors):
            mean = float(observed[model_index, anchor_index])
            lower = max(
                0.0,
                float(mean - critical * standard_error[model_index, anchor_index]),
            )
            rows.append(
                {
                    "task_id": task_id,
                    "metric": "coverage",
                    "n_train": int(anchor),
                    "model": model,
                    "mean_coverage": mean,
                    "standard_error": float(standard_error[model_index, anchor_index]),
                    "simultaneous_lower": lower,
                    "target_coverage": target,
                    "coverage_controlled": lower >= target,
                    "confidence_level": level,
                    "sidedness": "one_sided_lower",
                    "max_t_critical": critical,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": f"complete_repeat_{len(models)}x6_vector",
                    "bootstrap_index_sha256": registry_hash,
                    "family_members": len(models) * len(anchors),
                }
            )
    model_anchor_count, _contrast_count = _family_size(models, anchors)
    if len(rows) != model_anchor_count:
        raise ValueError("coverage family is not the complete model-by-anchor grid")
    return pd.DataFrame(rows)


def controlled_sharpness_conclusions(
    width_contrasts: pd.DataFrame,
    coverage_lower_bounds: pd.DataFrame,
    *,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> pd.DataFrame:
    """Gate pairwise width conclusions on both models' coverage bounds.

    Sharpness is meaningful only at a common controlled coverage level.  A
    pair therefore gets a directional conclusion only when both one-sided
    simultaneous coverage lower bounds meet the target and the simultaneous
    width contrast excludes zero.
    """

    width_required = {
        "task_id",
        "metric",
        "n_train",
        "model_left",
        "model_right",
        "mean_difference",
        "simultaneous_lower",
        "simultaneous_upper",
        "confidence_level",
        "bootstrap_draws",
        "bootstrap_unit",
        "bootstrap_index_sha256",
        "family_contrasts",
    }
    missing_width = width_required - set(width_contrasts.columns)
    if missing_width:
        raise ValueError(f"width-contrast table lacks columns {sorted(missing_width)}")
    if set(width_contrasts["metric"].astype(str)) != {"interval_width"}:
        raise ValueError("controlled sharpness requires interval-width contrasts")
    width_models = _model_order(
        pd.concat(
            [width_contrasts["model_left"], width_contrasts["model_right"]]
        ).astype(str)
    )
    width_anchors = tuple(sorted(set(width_contrasts["n_train"].astype(int))))
    _validate_model_panel(width_models, label="controlled sharpness")
    if len(width_anchors) != CANONICAL_ANCHOR_COUNT:
        raise ValueError("controlled sharpness requires six anchors")
    model_anchor_count, contrast_count = _family_size(width_models, width_anchors)
    if len(width_contrasts) != contrast_count:
        raise ValueError("controlled sharpness requires the complete width contrasts")
    required_coverage = {
        "task_id",
        "n_train",
        "model",
        "mean_coverage",
        "simultaneous_lower",
        "target_coverage",
        "coverage_controlled",
        "confidence_level",
        "bootstrap_draws",
        "bootstrap_unit",
        "bootstrap_index_sha256",
        "family_members",
    }
    missing_coverage = required_coverage - set(coverage_lower_bounds.columns)
    if missing_coverage:
        raise ValueError(f"coverage-bound table lacks columns {sorted(missing_coverage)}")
    if len(coverage_lower_bounds) != model_anchor_count:
        raise ValueError("coverage-bound table must contain the complete model-anchor grid")
    target = float(target_coverage)
    if not np.allclose(
        pd.to_numeric(coverage_lower_bounds["target_coverage"]).to_numpy(dtype=float),
        target,
    ):
        raise ValueError("coverage-bound target differs from the requested target")
    coverage_lookup = coverage_lower_bounds.set_index(["task_id", "n_train", "model"])
    if not coverage_lookup.index.is_unique:
        raise ValueError("coverage-bound model-anchor identities are duplicated")

    rows: list[dict[str, Any]] = []
    for contrast in width_contrasts.itertuples(index=False):
        left_key = (str(contrast.task_id), int(contrast.n_train), str(contrast.model_left))
        right_key = (str(contrast.task_id), int(contrast.n_train), str(contrast.model_right))
        if left_key not in coverage_lookup.index or right_key not in coverage_lookup.index:
            raise ValueError("width and coverage inference grids differ")
        coverage_left = coverage_lookup.loc[left_key]
        coverage_right = coverage_lookup.loc[right_key]
        width_registry = str(contrast.bootstrap_index_sha256)
        if (
            str(coverage_left["bootstrap_index_sha256"]) != width_registry
            or str(coverage_right["bootstrap_index_sha256"]) != width_registry
        ):
            raise ValueError("width and coverage conclusions use different bootstrap registries")
        for coverage_row in (coverage_left, coverage_right):
            if (
                not math.isclose(
                    float(coverage_row["confidence_level"]),
                    float(contrast.confidence_level),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or int(coverage_row["bootstrap_draws"])
                != int(contrast.bootstrap_draws)
                or str(coverage_row["bootstrap_unit"])
                != str(contrast.bootstrap_unit)
            ):
                raise ValueError("width and coverage conclusions use different inference settings")
        lower = float(contrast.simultaneous_lower)
        upper = float(contrast.simultaneous_upper)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError("width simultaneous confidence bounds are invalid")
        left_lcb = float(coverage_left["simultaneous_lower"])
        right_lcb = float(coverage_right["simultaneous_lower"])
        left_controlled = bool(coverage_left["coverage_controlled"]) and left_lcb >= target
        right_controlled = bool(coverage_right["coverage_controlled"]) and right_lcb >= target
        both_controlled = left_controlled and right_controlled
        sharper_model: str | None = None
        if both_controlled and upper < 0.0:
            sharper_model = str(contrast.model_left)
        elif both_controlled and lower > 0.0:
            sharper_model = str(contrast.model_right)
        if sharper_model is not None:
            conclusion = "sharper_at_controlled_coverage"
        elif both_controlled:
            conclusion = "inconclusive_at_controlled_coverage"
        else:
            conclusion = "coverage_not_controlled"
        rows.append(
            {
                "task_id": str(contrast.task_id),
                "metric": str(contrast.metric),
                "n_train": int(contrast.n_train),
                "model_left": str(contrast.model_left),
                "model_right": str(contrast.model_right),
                "width_mean_difference_left_minus_right": float(
                    contrast.mean_difference
                ),
                "width_simultaneous_lower": lower,
                "width_simultaneous_upper": upper,
                "target_coverage": target,
                "coverage_mean_left": float(coverage_left["mean_coverage"]),
                "coverage_mean_right": float(coverage_right["mean_coverage"]),
                "coverage_lcb_left": left_lcb,
                "coverage_lcb_right": right_lcb,
                "coverage_controlled_left": left_controlled,
                "coverage_controlled_right": right_controlled,
                "both_coverage_controlled": both_controlled,
                "sharper_model": sharper_model,
                "sharper_at_controlled_coverage": sharper_model is not None,
                "conclusion": conclusion,
                "decision_rule": (
                    "both_simultaneous_mean_coverage_lcbs_meet_target_and_"
                    "simultaneous_width_ci_excludes_zero"
                ),
                "confidence_level": float(contrast.confidence_level),
                "width_family_contrasts": int(contrast.family_contrasts),
                "coverage_family_members": int(coverage_left["family_members"]),
                "bootstrap_draws": int(contrast.bootstrap_draws),
                "bootstrap_unit": str(contrast.bootstrap_unit),
                "bootstrap_index_sha256_width": str(
                    width_registry
                ),
                "bootstrap_index_sha256_coverage": str(
                    coverage_left["bootstrap_index_sha256"]
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_difference_stability(
    metrics: pd.DataFrame,
    value: str = "primary_loss",
) -> pd.DataFrame:
    """SD of paired within-rerun model differences at every anchor."""

    task_id, tensor, repeats, models, anchors = _canonical_repeat_tensor(metrics, value)
    rows = []
    for anchor_index, anchor in enumerate(anchors):
        for left_index, right_index in combinations(range(len(models)), 2):
            difference = tensor[:, left_index, anchor_index] - tensor[:, right_index, anchor_index]
            standard_deviation = float(difference.std(ddof=1))
            rows.append(
                {
                    "task_id": task_id,
                    "metric": value,
                    "n_train": int(anchor),
                    "model_left": models[left_index],
                    "model_right": models[right_index],
                    "repeats": len(repeats),
                    "difference_mean": float(difference.mean()),
                    "difference_sd": standard_deviation,
                    "difference_q10": float(np.quantile(difference, 0.10)),
                    "difference_q90": float(np.quantile(difference, 0.90)),
                    "difference_mcse": standard_deviation / math.sqrt(len(difference)),
                }
            )
    _model_anchor_count, contrast_count = _family_size(models, anchors)
    if len(rows) != contrast_count:
        raise ValueError("paired stability table is not the complete contrast family")
    return pd.DataFrame(rows)


def stability_ribbons(
    metrics: pd.DataFrame,
    *,
    values: Sequence[str] = DEFAULT_RIBBON_METRICS,
) -> pd.DataFrame:
    """Mean lines and 10th--90th repeat-quantile bands for plotting."""

    _validate_identity_columns(metrics)
    frame = _with_skill(metrics)
    selected = _available_metrics(frame, values)
    if not selected:
        raise ValueError("no requested ribbon metric is available")
    rows: list[dict[str, Any]] = []
    for (task_id, model, n_train), group in frame.groupby(
        ["task_id", "model", "n_train"], sort=True
    ):
        for metric in selected:
            summary = _summary_row(_numeric_values(group, metric))
            rows.append(
                {
                    "task_id": str(task_id),
                    "model": str(model),
                    "n_train": int(n_train),
                    "metric": metric,
                    "repeats": int(len(group)),
                    "mean": summary["mean"],
                    "ribbon_lower": summary["q10"],
                    "ribbon_upper": summary["q90"],
                    "ribbon_definition": "between_repeat_q10_q90",
                    "inferential_interval": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "model", "n_train"]).reset_index(drop=True)


def training_size_changes(
    metrics: pd.DataFrame,
    *,
    start_anchor: int = 20,
    end_anchor: int | None = None,
    values: Sequence[str] = DEFAULT_CHANGE_METRICS,
) -> pd.DataFrame:
    """Paired repeat changes from 20 games to the task's largest anchor.

    Changes are defined as ``largest minus 20``. They remain within-task
    descriptive summaries and do not pair repeat IDs between releases.
    """

    _validate_identity_columns(metrics)
    frame = _with_skill(metrics)
    task_id = _single_task_id(frame)
    anchors = tuple(sorted(set(frame["n_train"].astype(int))))
    start = int(start_anchor)
    end = int(max(anchors) if end_anchor is None else end_anchor)
    if start not in anchors:
        raise ValueError(f"start anchor {start} is absent")
    if end not in anchors or end <= start:
        raise ValueError("end anchor must be a present anchor larger than the start")
    selected = _available_metrics(frame, values)
    if not selected:
        raise ValueError("no requested change metric is available")
    models = _model_order(frame["model"])
    expected_repeats = set(frame["repeat"].astype(int))
    rows: list[dict[str, Any]] = []
    for model in models:
        subset = frame[(frame["model"].astype(str) == model) & frame["n_train"].isin([start, end])]
        for metric in selected:
            pivot = subset.pivot(index="repeat", columns="n_train", values=metric)
            if (
                set(pivot.index.astype(int)) != expected_repeats
                or set(pivot.columns.astype(int)) != {start, end}
                or pivot.isna().any().any()
            ):
                raise ValueError(f"{model} lacks paired {start}-to-{end} values for {metric}")
            change = pivot[end].to_numpy(dtype=np.float64) - pivot[start].to_numpy(dtype=np.float64)
            summary = _summary_row(change)
            row: dict[str, Any] = {
                "task_id": task_id,
                "model": model,
                "metric": metric,
                "from_n_train": start,
                "to_n_train": end,
                "definition": "value_at_largest_minus_value_at_20",
                "repeats": int(len(change)),
            }
            row.update({f"change_{name}": value for name, value in summary.items()})
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["metric", "model"]).reset_index(drop=True)


def training_size_difference_in_changes(
    metrics: pd.DataFrame,
    *,
    start_anchor: int = 20,
    end_anchor: int | None = None,
    values: Sequence[str] = DEFAULT_CHANGE_METRICS,
) -> pd.DataFrame:
    """Describe paired model differences in their within-rerun size changes.

    For each complete rerun this is
    ``(left_end-left_start) - (right_end-right_start)``. These model-pair
    summaries are secondary descriptions and do not expand the prespecified
    anchorwise primary max-t family.
    """

    _validate_identity_columns(metrics)
    frame = _with_skill(metrics)
    task_id = _single_task_id(frame)
    anchors = tuple(sorted(set(frame["n_train"].astype(int))))
    start = int(start_anchor)
    end = int(max(anchors) if end_anchor is None else end_anchor)
    if start not in anchors:
        raise ValueError(f"start anchor {start} is absent")
    if end not in anchors or end <= start:
        raise ValueError("end anchor must be a present anchor larger than the start")
    selected = _available_metrics(frame, values)
    if not selected:
        raise ValueError("no requested difference-in-changes metric is available")
    models = _model_order(frame["model"])
    expected_repeats = set(frame["repeat"].astype(int))
    rows: list[dict[str, Any]] = []
    for metric in selected:
        changes: dict[str, np.ndarray] = {}
        for model in models:
            subset = frame[
                (frame["model"].astype(str) == model)
                & frame["n_train"].isin([start, end])
            ]
            pivot = subset.pivot(index="repeat", columns="n_train", values=metric)
            if (
                set(pivot.index.astype(int)) != expected_repeats
                or set(pivot.columns.astype(int)) != {start, end}
                or pivot.isna().any().any()
            ):
                raise ValueError(
                    f"{model} lacks paired {start}-to-{end} values for {metric}"
                )
            pivot = pivot.sort_index()
            changes[model] = (
                pivot[end].to_numpy(dtype=np.float64)
                - pivot[start].to_numpy(dtype=np.float64)
            )
        for left, right in combinations(models, 2):
            difference = changes[left] - changes[right]
            summary = _summary_row(difference)
            rows.append(
                {
                    "task_id": task_id,
                    "model_left": left,
                    "model_right": right,
                    "metric": metric,
                    "from_n_train": start,
                    "to_n_train": end,
                    "definition": (
                        "left_largest_minus_20_minus_right_largest_minus_20_"
                        "within_repeat"
                    ),
                    "repeats": int(len(difference)),
                    **{
                        f"difference_in_change_{name}": value
                        for name, value in summary.items()
                    },
                    "multiplicity_scope": (
                        "secondary_descriptive_not_in_primary_max_t_family"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "model_left", "model_right"]
    ).reset_index(drop=True)


def descriptive_suite_summary(
    task_metrics: dict[str, pd.DataFrame],
    *,
    require_complete_task_ids: Sequence[str] | None = None,
    partial_task_anchors: Mapping[str, Sequence[int]] | None = None,
    partial_task_families: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Task-equal descriptive synthesis; intentionally no cross-task CI."""

    task_rows = []
    for task_id, supplied in sorted(task_metrics.items()):
        metrics = _with_skill(supplied)
        if "task_id" in metrics and set(metrics["task_id"].astype(str)) != {str(task_id)}:
            raise ValueError(f"suite key {task_id!r} does not match its metric task ID")
        required = {"model", "n_train", "skill"}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"suite task {task_id!r} lacks columns {sorted(missing)}")
        family_column = "family" if "family" in metrics.columns else "model"
        _numeric_values(metrics, "skill")
        for (family, anchor), group in metrics.groupby([family_column, "n_train"], sort=True):
            task_rows.append(
                {
                    "task_id": str(task_id),
                    "family": str(family),
                    "n_train": int(anchor),
                    "task_mean_skill": float(group["skill"].mean()),
                    "task_repeats": int(group["repeat"].nunique()) if "repeat" in group else int(len(group)),
                }
            )
    columns = (
        "family",
        "n_train",
        "tasks",
        "task_ids",
        "task_equal_mean_skill",
        "task_median_skill",
        "task_min_skill",
        "task_max_skill",
        "task_equal_weighting",
        "repeat_pairing",
        "inference",
    )
    if not task_rows:
        return pd.DataFrame(columns=columns)
    task_frame = pd.DataFrame(task_rows)
    common = task_frame[task_frame["n_train"].isin([10, 20, 40, 60])]
    if require_complete_task_ids is not None:
        expected_tasks = {str(value) for value in require_complete_task_ids}
        if not expected_tasks or set(task_metrics) != expected_tasks:
            raise ValueError("suite task registry differs from its required task IDs")
        partial = {
            str(task_id): tuple(int(anchor) for anchor in anchors)
            for task_id, anchors in (partial_task_anchors or {}).items()
        }
        if not set(partial) <= expected_tasks:
            raise ValueError("partial task anchors name an unknown suite task")
        partial_families = {
            str(task_id): tuple(str(family) for family in families)
            for task_id, families in (partial_task_families or {}).items()
        }
        if not set(partial_families) <= expected_tasks:
            raise ValueError("partial task families name an unknown suite task")
        for task_id, families in partial_families.items():
            if not families or len(families) != len(set(families)):
                raise ValueError(
                    f"partial task families for {task_id!r} must be unique and nonempty"
                )
        default_task_ids = sorted(expected_tasks - set(partial_families))
        observed_default_panels = {
            frozenset(
                task_frame.loc[
                    task_frame["task_id"].eq(task_id), "family"
                ].astype(str)
            )
            for task_id in default_task_ids
        }
        if observed_default_panels == {frozenset(STRUCTURE_FAMILIES)}:
            default_families = STRUCTURE_FAMILIES
        elif observed_default_panels == {frozenset(DEFAULT_MODELS)}:
            default_families = DEFAULT_MODELS
        elif observed_default_panels == {frozenset(LEGACY_STRUCTURE_MODELS)}:
            default_families = LEGACY_STRUCTURE_MODELS
        else:
            raise ValueError(
                "suite tasks do not share one canonical default family registry"
            )
        expected_grid_by_task: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
        for task_id in sorted(expected_tasks):
            task_anchors = partial.get(task_id, (10, 20, 40, 60))
            if (
                not task_anchors
                or tuple(sorted(set(task_anchors))) != task_anchors
                or not set(task_anchors) <= {10, 20, 40, 60}
            ):
                raise ValueError("partial task anchors must be ordered common anchors")
            task_families = partial_families.get(task_id, default_families)
            expected_grid_by_task[task_id] = (task_anchors, task_families)
            expected_pairs = {
                (family, anchor)
                for family in task_families
                for anchor in task_anchors
            }
            task_common = common[common["task_id"].eq(task_id)]
            observed_pairs = set(
                zip(
                    task_common["family"].astype(str),
                    task_common["n_train"].astype(int),
                )
            )
            if observed_pairs != expected_pairs or len(task_common) != len(expected_pairs):
                raise ValueError(
                    f"suite task {task_id!r} lacks its exact declared family/anchor grid"
                )
    rows = []
    for (family, anchor), group in common.groupby(["family", "n_train"], sort=True):
        task_values = group["task_mean_skill"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "family": family,
                "n_train": int(anchor),
                "tasks": int(len(group)),
                "task_ids": json.dumps(sorted(group["task_id"].astype(str))),
                "task_equal_mean_skill": float(task_values.mean()),
                "task_median_skill": float(np.median(task_values)),
                "task_min_skill": float(task_values.min()),
                "task_max_skill": float(task_values.max()),
                "task_equal_weighting": "one_mean_per_task",
                "repeat_pairing": "none_across_releases",
                "inference": "descriptive_fixed_dependent_tasks_no_cross_task_ci",
            }
        )
    result = pd.DataFrame(rows, columns=columns).sort_values(
        ["family", "n_train"]
    ).reset_index(drop=True)
    if require_complete_task_ids is not None:
        expected_summary_pairs = {
            (family, anchor)
            for anchors, families in expected_grid_by_task.values()
            for family in families
            for anchor in anchors
        }
        observed_summary_pairs = set(
            zip(result["family"].astype(str), result["n_train"].astype(int))
        )
        if observed_summary_pairs != expected_summary_pairs:
            raise ValueError("suite summary lacks a declared family-by-anchor row")
        for row in result.itertuples(index=False):
            expected_at_anchor = {
                task_id
                for task_id in expected_tasks
                if int(row.n_train) in set(expected_grid_by_task[task_id][0])
                and str(row.family) in set(expected_grid_by_task[task_id][1])
            }
            observed_at_anchor = set(json.loads(str(row.task_ids)))
            if (
                int(row.tasks) != len(expected_at_anchor)
                or observed_at_anchor != expected_at_anchor
            ):
                raise ValueError(
                    "suite summary task membership differs at a common anchor"
                )
    return result


def pilot_task_analysis_tables(
    metrics: pd.DataFrame,
    *,
    outcome_type: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build descriptive-only tables for the immutable ten-rerun pilot.

    These products intentionally contain no bootstrap confidence interval,
    max-t decision, supported-best declaration, or controlled-sharpness
    claim. Quantiles and standard deviations describe only the ten pilot
    reruns and must not be promoted to confirmatory inference.
    """

    normalized_outcome = None if outcome_type is None else str(outcome_type)
    if normalized_outcome == "frame_event":
        normalized_outcome = "binary"
    if normalized_outcome not in {None, "binary", "distribution", "trajectory"}:
        raise ValueError(f"unsupported outcome type {outcome_type!r}")
    analysis_frame = _with_skill(metrics)
    if normalized_outcome in {"binary", "trajectory"}:
        analysis_frame = analysis_frame.drop(
            columns=["coverage", "interval_width", "interval_width_sd"],
            errors="ignore",
        )
    ribbon_metrics: tuple[str, ...] = (
        "primary_loss",
        "game_equal_loss",
        "null_loss",
        "skill",
    )
    change_metrics: tuple[str, ...] = (
        "primary_loss",
        "game_equal_loss",
        "skill",
    )
    stability_metrics = _available_metrics(
        analysis_frame, ("primary_loss", "game_equal_loss", "skill")
    )
    if normalized_outcome == "distribution":
        missing = {"coverage", "interval_width"} - set(analysis_frame.columns)
        if missing:
            raise ValueError(
                f"distribution task metrics lack descriptive columns {sorted(missing)}"
            )
        additional = ("coverage", "interval_width")
    elif normalized_outcome == "binary":
        additional = _available_metrics(
            analysis_frame,
            (
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
            ),
        )
    elif normalized_outcome == "trajectory":
        additional = _available_metrics(
            analysis_frame,
            (
                "path_equal_rmse",
                "path_coverage",
                "game_equal_path_coverage",
                "mean_tube_diameter",
                "game_equal_tube_diameter",
            ),
        )
    else:
        additional = ()
    ribbon_metrics += additional
    change_metrics += additional
    stability_metrics += additional
    tables = {
        "summary": summarize_task(analysis_frame),
        "paired_stability": pd.concat(
            [
                paired_difference_stability(analysis_frame, value=value)
                for value in stability_metrics
            ],
            ignore_index=True,
        ),
        "stability_ribbons": stability_ribbons(
            analysis_frame, values=ribbon_metrics
        ),
        "training_size_changes": training_size_changes(
            analysis_frame, values=change_metrics
        ),
        "training_size_difference_in_changes": training_size_difference_in_changes(
            analysis_frame, values=change_metrics
        ),
    }
    for table in tables.values():
        table["evidence_status"] = "exploratory_provisional"
        table["inference_scope"] = "descriptive_only_no_confirmatory_inference"
    return tables


def task_analysis_tables(
    metrics: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    outcome_type: str | None = None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> dict[str, pd.DataFrame]:
    """Build the complete within-task analysis product set.

    ``outcome_type`` must be explicitly ``"distribution"`` before predictive
    coverage/width inference is produced.  In particular, Venn--Abers binary
    multiprobability bounds are never described as prediction intervals and
    never enter a coverage analysis.
    """

    normalized_outcome = None if outcome_type is None else str(outcome_type)
    if normalized_outcome == "frame_event":
        normalized_outcome = "binary"
    if normalized_outcome == "frame_event":
        normalized_outcome = "binary"
    if normalized_outcome not in {None, "binary", "distribution", "trajectory"}:
        raise ValueError(f"unsupported outcome type {outcome_type!r}")
    analysis_frame = _with_skill(metrics)
    if normalized_outcome in {"binary", "trajectory"}:
        analysis_frame = analysis_frame.drop(
            columns=["coverage", "interval_width", "interval_width_sd"],
            errors="ignore",
        )
    ribbon_metrics: tuple[str, ...] = (
        "primary_loss",
        "game_equal_loss",
        "null_loss",
        "skill",
    )
    change_metrics: tuple[str, ...] = (
        "primary_loss",
        "game_equal_loss",
        "skill",
    )
    if normalized_outcome == "distribution":
        missing = {"coverage", "interval_width"} - set(analysis_frame.columns)
        if missing:
            raise ValueError(
                f"distribution task metrics lack inference columns {sorted(missing)}"
            )
        ribbon_metrics += ("coverage", "interval_width")
        change_metrics += ("coverage", "interval_width")
    elif normalized_outcome == "binary":
        binary_uncertainty = _available_metrics(
            analysis_frame,
            (
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
            ),
        )
        ribbon_metrics += binary_uncertainty
        change_metrics += binary_uncertainty
    elif normalized_outcome == "trajectory":
        trajectory_uncertainty = _available_metrics(
            analysis_frame,
            (
                "path_equal_rmse",
                "path_coverage",
                "game_equal_path_coverage",
                "mean_tube_diameter",
                "game_equal_tube_diameter",
            ),
        )
        ribbon_metrics += trajectory_uncertainty
        change_metrics += trajectory_uncertainty

    primary_contrasts = paired_max_t_intervals(
        analysis_frame,
        value="primary_loss",
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    stability_metrics = _available_metrics(
        analysis_frame, ("primary_loss", "game_equal_loss", "skill")
    )
    if normalized_outcome == "distribution":
        stability_metrics += ("coverage", "interval_width")
    elif normalized_outcome == "binary":
        stability_metrics += binary_uncertainty
    elif normalized_outcome == "trajectory":
        stability_metrics += trajectory_uncertainty
    stability_bootstrap = [
        bca_stability_intervals(
            analysis_frame,
            value=value,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        for value in stability_metrics
    ]
    sd_intervals = pd.concat(
        [result[0] for result in stability_bootstrap], ignore_index=True
    )
    log_sd_ratios = pd.concat(
        [result[1] for result in stability_bootstrap], ignore_index=True
    )
    tables: dict[str, pd.DataFrame] = {
        "summary": summarize_task(analysis_frame),
        "paired_contrasts": primary_contrasts,
        "paired_stability": pd.concat(
            [
                paired_difference_stability(analysis_frame, value=value)
                for value in stability_metrics
            ],
            ignore_index=True,
        ),
        "stability_ribbons": stability_ribbons(
            analysis_frame, values=ribbon_metrics
        ),
        "training_size_changes": training_size_changes(
            analysis_frame, values=change_metrics
        ),
        "training_size_difference_in_changes": training_size_difference_in_changes(
            analysis_frame, values=change_metrics
        ),
        "bca_sd_intervals": sd_intervals,
        "bca_log_sd_ratios": log_sd_ratios,
        "supported_best": supported_best_conclusions(primary_contrasts),
    }
    inferential_tables = [
        primary_contrasts,
        sd_intervals,
        log_sd_ratios,
    ]
    if normalized_outcome == "distribution":
        width_contrasts = paired_max_t_intervals(
            analysis_frame,
            value="interval_width",
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        coverage_bounds = simultaneous_coverage_lower_bounds(
            analysis_frame,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            target_coverage=target_coverage,
        )
        tables.update(
            {
                "interval_width_contrasts": width_contrasts,
                "coverage_lower_bounds": coverage_bounds,
                "controlled_sharpness": controlled_sharpness_conclusions(
                    width_contrasts,
                    coverage_bounds,
                    target_coverage=target_coverage,
                ),
            }
        )
        inferential_tables.extend([width_contrasts, coverage_bounds])

    registry_hashes = {
        str(table["bootstrap_index_sha256"].iloc[0])
        for table in inferential_tables
        if "bootstrap_index_sha256" in table and len(table)
    }
    if len(registry_hashes) != 1:
        raise ValueError("task inference products did not share one bootstrap registry")
    return tables


def task_analysis_receipt_fields(
    tables: Mapping[str, pd.DataFrame],
    *,
    bootstrap_seed: int,
    bootstrap_draws: int,
    outcome_type: str | None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> dict[str, Any]:
    """Return method metadata shared by filesystem and run aggregation receipts."""

    primary = tables.get("paired_contrasts")
    if primary is None or primary.empty:
        raise ValueError("analysis tables lack primary paired contrasts")
    primary_models = _model_order(
        pd.concat([primary["model_left"], primary["model_right"]]).astype(str)
    )
    primary_anchors = tuple(sorted(set(primary["n_train"].astype(int))))
    _validate_model_panel(primary_models, label="task analysis receipt")
    model_anchor_count, contrast_count = _family_size(
        primary_models, primary_anchors
    )
    observed_family_sizes = set(primary["family_contrasts"].astype(int))
    if observed_family_sizes != {contrast_count}:
        raise ValueError("analysis tables have an inconsistent contrast family size")
    pair_count = math.comb(len(primary_models), 2)
    pair_word = {6: "six", 10: "ten"}.get(pair_count, str(pair_count))
    model_word = {4: "four", 5: "five"}.get(
        len(primary_models), str(len(primary_models))
    )
    fields: dict[str, Any] = {
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_unit": f"complete_repeat_{len(primary_models)}x6_vector",
        "bootstrap_index_sha256": str(primary["bootstrap_index_sha256"].iloc[0]),
        "max_t_family": f"all_{pair_word}_model_pairs_by_all_six_task_anchors",
        "max_t_family_contrasts": contrast_count,
        "bca_stability": {
            "metrics": sorted(
                set(tables["bca_sd_intervals"]["metric"].astype(str))
            ),
            "statistics": ["between_repeat_sd", "paired_log_sd_ratio"],
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "jackknife_unit": (
                f"delete_one_complete_repeat_{len(primary_models)}x6_vector"
            ),
        },
        "supported_best_rule": (
            "simultaneous_mean_loss_ci_beats_every_comparator_at_anchor"
        ),
        "stability_ribbon": "mean_with_between_repeat_q10_q90",
        "training_size_change": "largest_anchor_minus_20_within_repeat",
        "training_size_difference_in_changes": (
            "paired_model_difference_of_largest_minus_20_changes_"
            "descriptive_outside_primary_max_t"
        ),
        "cross_task_inference": "none",
    }
    normalized_outcome = None if outcome_type is None else str(outcome_type)
    if normalized_outcome == "distribution":
        required = {
            "interval_width_contrasts",
            "coverage_lower_bounds",
            "controlled_sharpness",
        }
        missing = required - set(tables)
        if missing:
            raise ValueError(
                f"distribution analysis lacks tables {sorted(missing)}"
            )
        fields["distributional_inference"] = {
            "coverage_target": float(target_coverage),
            "coverage_family": f"{model_word}_models_by_six_anchors",
            "coverage_family_members": model_anchor_count,
            "coverage_bound": "simultaneous_one_sided_95pct_lower",
            "coverage_estimand": "game_equal_random_eligible_play",
            "calibration_unit": "one_semantically_selected_play_per_game",
            "width_family": (
                f"all_{pair_word}_model_pairs_by_all_six_task_anchors"
            ),
            "controlled_sharpness_rule": (
                "both_models_coverage_lcbs_meet_target_and_width_ci_excludes_zero"
            ),
        }
    elif normalized_outcome == "binary":
        fields["binary_uncertainty"] = {
            "primary_object": "calibrated_bernoulli_distribution",
            "reliability_metrics": [
                "calibrated_brier",
                "calibrated_log_loss",
                "calibration_bias",
                "rms_reliability",
            ],
            "concentration_metrics": ["mean_entropy", "mean_va_imprecision"],
            "label_conditional_sets": "secondary_diagnostic_only",
            "venn_abers_bounds_are_prediction_intervals": False,
        }
    elif normalized_outcome == "trajectory":
        fields["trajectory_uncertainty"] = {
            "coverage_target": float(target_coverage),
            "coverage_unit": "complete_requested_player_path",
            "calibration_unit": "one_semantically_selected_path_per_game",
            "region": "radial_horizon_scaled_conformal_tube",
            "simultaneous_over_all_players": False,
        }
    return fields


def write_task_analysis(
    metrics: pd.DataFrame,
    out_dir: str | Path,
    *,
    bootstrap_seed: int,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    outcome_type: str | None = None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
) -> dict[str, str]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = task_analysis_tables(
        metrics,
        bootstrap_seed=bootstrap_seed,
        bootstrap_draws=bootstrap_draws,
        outcome_type=outcome_type,
        target_coverage=target_coverage,
    )
    artifacts = {name: f"{name}.csv" for name in tables}
    for name, table in tables.items():
        table.to_csv(output / artifacts[name], index=False)
    receipt = {
        "task_id": _single_task_id(metrics),
        **task_analysis_receipt_fields(
            tables,
            bootstrap_seed=bootstrap_seed,
            bootstrap_draws=bootstrap_draws,
            outcome_type=outcome_type,
            target_coverage=target_coverage,
        ),
        "artifacts": artifacts,
    }
    (output / "analysis_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)
        + "\n"
    )
    return artifacts
