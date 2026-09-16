"""Statistical aggregation for the repeated rushing-yard uncertainty study.

The functions in this module operate on one row per complete study cell:
``repeat x model x n_train``.  Inference is deliberately performed at the
repeat level.  Test plays are not treated as independent replicates.

The public :func:`run_aggregation` entry point enforces either locked main design
(50 or 100 repeats, four models, and six anchor sizes), but the lower-level helpers
accept smaller explicit grids so they can also be used for prespecified
sensitivity analyses and unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .design import (
        CANONICAL_MODEL_IDS as _DESIGN_MODEL_IDS,
        DEFAULT_CONFIRMATORY_REPEATS as _DESIGN_DEFAULT_REPEATS,
        FULL_CONFIRMATORY_REPEATS as _DESIGN_FULL_REPEATS,
        MAIN_ANCHORS as _DESIGN_ANCHORS,
        SENSITIVITY_ANCHORS as _DESIGN_SENSITIVITY_ANCHORS,
        manifest_seed as _design_manifest_seed,
    )
    from .models import MODEL_DISPLAY_NAMES as _MODEL_DISPLAY_NAMES
except ImportError:  # pragma: no cover - keeps this module usable in isolation
    _DESIGN_MODEL_IDS = (
        "ridge_sgd_l2",
        "lightgbm_multiclass",
        "zoo_cnn",
        "set_transformer",
    )
    _DESIGN_ANCHORS = (20, 40, 80, 160, 240, 360)
    _DESIGN_SENSITIVITY_ANCHORS = (20, 160, 360)
    _DESIGN_DEFAULT_REPEATS = 50
    _DESIGN_FULL_REPEATS = 100
    _design_manifest_seed = None
    _MODEL_DISPLAY_NAMES = {
        "ridge_sgd_l2": "L2 one-vs-rest logistic",
        "lightgbm_multiclass": "LightGBM",
        "zoo_cnn": "Zoo CNN",
        "set_transformer": "Set Transformer",
    }

MAIN_MODELS: tuple[str, ...] = tuple(_DESIGN_MODEL_IDS)
MODEL_DISPLAY_NAMES: dict[str, str] = dict(_MODEL_DISPLAY_NAMES)
ANCHOR_SIZES: tuple[int, ...] = tuple(_DESIGN_ANCHORS)
DEFAULT_MAIN_REPEAT_IDS: tuple[int, ...] = tuple(
    range(1, _DESIGN_DEFAULT_REPEATS + 1)
)
FULL_MAIN_REPEAT_IDS: tuple[int, ...] = tuple(range(1, _DESIGN_FULL_REPEATS + 1))
# Compatibility alias and the fixed endpoint of the separate sensitivity branch.
MAIN_REPEAT_IDS: tuple[int, ...] = DEFAULT_MAIN_REPEAT_IDS
SENSITIVITY_SIZES: tuple[int, ...] = tuple(_DESIGN_SENSITIVITY_ANCHORS)
SENSITIVITY_REPEAT_IDS: tuple[int, ...] = tuple(range(1, 21))
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_817
DEFAULT_CONFIDENCE = 0.95
DEFAULT_COVERAGE_TARGET = 0.90

_ID_COLUMNS = {"repeat", "repeat_seed", "model", "n_train", "seed"}
_NON_METRIC_COLUMNS = {
    *_ID_COLUMNS,
    "n_plays",
    "n_test",
    "alpha",
    "timestamp",
    "width_definition",
    "selected_config",
    "configuration",
    "variant",
    "branch",
}
_KNOWN_BASE_METRICS = {
    "crps",
    "coverage",
    "mean_width",
    "mean_width_inclusive",
    "mean_q",
    "central_width",
    "conformal_padding",
}


@dataclass(frozen=True)
class StudyManifest:
    """Configuration consumed by :func:`run_aggregation`."""

    models: tuple[str, ...] = MAIN_MODELS
    sizes: tuple[int, ...] = ANCHOR_SIZES
    repeat_ids: tuple[int, ...] = MAIN_REPEAT_IDS
    metric_columns: tuple[str, ...] | None = None
    coverage_metrics: tuple[str, ...] | None = None
    confidence: float = DEFAULT_CONFIDENCE
    coverage_target: float = DEFAULT_COVERAGE_TARGET
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    plot_metrics: tuple[str, ...] | None = None


def _as_frame(records: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a defensive DataFrame copy and expand an optional metrics dict."""

    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("Study metrics are empty.")

    if "metrics" in frame.columns and frame["metrics"].map(lambda value: isinstance(value, Mapping)).all():
        expanded = pd.DataFrame(frame["metrics"].tolist(), index=frame.index)
        collisions = sorted(set(expanded.columns) & (set(frame.columns) - {"metrics"}))
        if collisions:
            raise ValueError(f"Nested metrics collide with top-level columns: {collisions}.")
        frame = pd.concat([frame.drop(columns="metrics"), expanded], axis=1)
    return frame


def _load_manifest(manifest: StudyManifest | Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest, (str, Path)):
        with Path(manifest).open() as handle:
            value = json.load(handle)
    elif is_dataclass(manifest):
        value = asdict(manifest)
    elif isinstance(manifest, Mapping):
        value = dict(manifest)
    else:
        raise TypeError("manifest must be a StudyManifest, mapping, or JSON path.")
    return value


def _metric_scope(metric: str) -> tuple[str, str]:
    """Return ``(base_metric, aggregation_scope)`` for a metric column."""

    name = metric.lower()
    scope = "unspecified"
    for prefix, suffix, label in (
        ("game_equal_", "", "game_equal"),
        ("", "_game_equal", "game_equal"),
        ("play_", "", "play"),
        ("", "_play", "play"),
    ):
        if prefix and name.startswith(prefix):
            name, scope = name[len(prefix) :], label
            break
        if suffix and name.endswith(suffix):
            name, scope = name[: -len(suffix)], label
            break
    if scope == "unspecified" and name in _KNOWN_BASE_METRICS:
        # Existing sweep columns are play-weighted unless explicitly suffixed.
        scope = "play"
    return name, scope


def infer_metric_columns(frame: pd.DataFrame) -> list[str]:
    """Infer supported play-level and game-equal metric columns.

    Identifier, count, configuration, and within-test standard-error columns
    are intentionally excluded.  Callers may always pass an explicit metric
    list when additional estimands are desired.
    """

    metrics: list[str] = []
    for column in frame.columns:
        if column in _NON_METRIC_COLUMNS or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        base, scope = _metric_scope(str(column))
        if base in _KNOWN_BASE_METRICS and scope in {"play", "game_equal"}:
            metrics.append(str(column))
    if not metrics:
        raise ValueError(
            "No metric columns were inferred. Pass metric_columns explicitly; "
            f"available columns are {list(frame.columns)}."
        )
    return metrics


def _validate_grid(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    models: Sequence[str],
    sizes: Sequence[int],
    repeat_ids: Sequence[Any],
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = _as_frame(records)
    required = {"repeat", "model", "n_train"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required grid columns: {missing_columns}.")

    models = tuple(str(model) for model in models)
    sizes = tuple(int(size) for size in sizes)
    repeat_ids = tuple(repeat_ids)
    if len(set(models)) != len(models) or len(set(sizes)) != len(sizes) or len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("models, sizes, and repeat_ids must each contain unique values.")

    if frame[["repeat", "model", "n_train"]].isna().any().any():
        raise ValueError("Grid identifiers cannot contain missing values.")
    frame["model"] = frame["model"].astype(str)
    try:
        frame["n_train"] = frame["n_train"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_train must contain integer-compatible values.") from exc

    duplicated = frame.duplicated(["repeat", "model", "n_train"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["repeat", "model", "n_train"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate repeat/model/size cells found, for example {examples}.")

    expected = {
        (repeat, model, size)
        for repeat in repeat_ids
        for model in models
        for size in sizes
    }
    observed = set(frame[["repeat", "model", "n_train"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed, key=str)
    extra = sorted(observed - expected, key=str)
    if missing or extra:
        raise ValueError(
            "Study grid is not the exact requested Cartesian product: "
            f"missing={len(missing)} {missing[:4]}, extra={len(extra)} {extra[:4]}."
        )
    if len(frame) != len(expected):
        raise ValueError(f"Expected {len(expected)} cells, found {len(frame)}.")

    if "repeat_seed" in frame.columns:
        seed_counts = frame.groupby("repeat", sort=False)["repeat_seed"].nunique(dropna=False)
        if not (seed_counts == 1).all():
            bad = seed_counts[seed_counts != 1].index.tolist()
            raise ValueError(f"repeat_seed must be constant within each repeat; bad repeats={bad}.")

    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    absent_metrics = sorted(set(metrics) - set(frame.columns))
    if absent_metrics:
        raise ValueError(f"Missing requested metric columns: {absent_metrics}.")
    for metric in metrics:
        try:
            values = frame[metric].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {metric!r} must be numeric.") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Metric {metric!r} contains missing or non-finite values.")
        base, _ = _metric_scope(metric)
        if base == "coverage" and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Coverage metric {metric!r} must lie in [0, 1].")
    return frame


def validate_main_grid(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
) -> pd.DataFrame:
    """Validate one and only one row for every required main-study cell."""

    return _validate_grid(
        records,
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
        metric_columns=metric_columns,
    )


def _describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty metric vector.")
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n_repeats": n,
        "mean": float(np.mean(values)),
        "sd": sd,
        "mcse": sd / math.sqrt(n),
        "median": float(median),
        "iqr": float(q75 - q25),
        "q10": float(q10),
        "q90": float(q90),
    }


def summarize_metrics(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    validate: bool = True,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
) -> pd.DataFrame:
    """Return long-form mean/SD/MCSE/median/IQR/q10/q90 summaries."""

    frame = (
        _validate_grid(
            records,
            models=models,
            sizes=sizes,
            repeat_ids=repeat_ids,
            metric_columns=metric_columns,
        )
        if validate
        else _as_frame(records)
    )
    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    rows: list[dict[str, Any]] = []
    for model in models:
        for size in sizes:
            group = frame[(frame["model"] == model) & (frame["n_train"] == int(size))]
            for metric in metrics:
                base, scope = _metric_scope(metric)
                rows.append(
                    {
                        "model": model,
                        "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
                        "n_train": int(size),
                        "metric": metric,
                        "base_metric": base,
                        "aggregation": scope,
                        **_describe(group[metric].to_numpy(dtype=np.float64)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_metrics_wide(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one row per model/size with metric-qualified statistics.

    The long summary remains the inference-friendly canonical form; this
    24-row representation is convenient for review tables and downstream
    spreadsheet use.
    """

    statistics = ["mean", "sd", "mcse", "median", "iqr", "q10", "q90"]
    required = {"model", "n_train", "metric", *statistics}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Long summary is missing columns required for widening: {missing}.")
    duplicated = summary.duplicated(["model", "n_train", "metric"], keep=False)
    if duplicated.any():
        raise ValueError("Long summary has duplicate model/size/metric rows.")
    identity = ["model", "n_train"]
    if "model_display_name" in summary.columns:
        identity.insert(1, "model_display_name")
    melted = summary.melt(
        id_vars=[*identity, "metric"],
        value_vars=statistics,
        var_name="statistic",
        value_name="value",
    )
    melted["output_column"] = melted["metric"].astype(str) + "_" + melted["statistic"]
    row_order = summary[identity].drop_duplicates(ignore_index=True)
    wide_values = melted.pivot(
        index=identity, columns="output_column", values="value"
    ).reset_index()
    wide = row_order.merge(wide_values, on=identity, validate="one_to_one")
    wide.columns.name = None
    return wide


def paired_model_contrasts(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    model_pairs: Sequence[tuple[str, str]] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    validate: bool = True,
) -> pd.DataFrame:
    """Create paired ``model_a - model_b`` values within repeat and size."""

    frame = (
        _validate_grid(records, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metric_columns)
        if validate
        else _as_frame(records)
    )
    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    pairs = list(model_pairs) if model_pairs is not None else list(combinations(models, 2))
    unknown = sorted({model for pair in pairs for model in pair} - set(models))
    if unknown:
        raise ValueError(f"Unknown models in model_pairs: {unknown}.")

    rows: list[dict[str, Any]] = []
    indexed = frame.set_index(["repeat", "model", "n_train"])
    for repeat in repeat_ids:
        for size in sizes:
            for metric in metrics:
                for model_a, model_b in pairs:
                    value_a = float(indexed.loc[(repeat, model_a, int(size)), metric])
                    value_b = float(indexed.loc[(repeat, model_b, int(size)), metric])
                    rows.append(
                        {
                            "repeat": repeat,
                            "metric": metric,
                            "n_train": int(size),
                            "model_a": model_a,
                            "model_b": model_b,
                            "contrast": f"{model_a} - {model_b}",
                            "value": value_a - value_b,
                        }
                    )
    return pd.DataFrame(rows)


def paired_size_contrasts(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    size_pairs: Sequence[tuple[int, int]] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    validate: bool = True,
) -> pd.DataFrame:
    """Create paired ``size_to - size_from`` values within repeat and model."""

    frame = (
        _validate_grid(records, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metric_columns)
        if validate
        else _as_frame(records)
    )
    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    pairs = list(size_pairs) if size_pairs is not None else list(combinations(sizes, 2))
    unknown = sorted({int(size) for pair in pairs for size in pair} - {int(size) for size in sizes})
    if unknown:
        raise ValueError(f"Unknown sizes in size_pairs: {unknown}.")

    rows: list[dict[str, Any]] = []
    indexed = frame.set_index(["repeat", "model", "n_train"])
    for repeat in repeat_ids:
        for model in models:
            for metric in metrics:
                for size_from, size_to in pairs:
                    before = float(indexed.loc[(repeat, model, int(size_from)), metric])
                    after = float(indexed.loc[(repeat, model, int(size_to)), metric])
                    rows.append(
                        {
                            "repeat": repeat,
                            "metric": metric,
                            "model": model,
                            "size_from": int(size_from),
                            "size_to": int(size_to),
                            "contrast": f"{int(size_to)} - {int(size_from)}",
                            "value": after - before,
                        }
                    )
    return pd.DataFrame(rows)


def endpoint_changes(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    validate: bool = True,
) -> pd.DataFrame:
    """Return each model's paired 20-to-360 change (360 minus 20)."""

    if 20 not in sizes or 360 not in sizes:
        raise ValueError("Endpoint changes require sizes 20 and 360.")
    return paired_size_contrasts(
        records,
        metric_columns=metric_columns,
        size_pairs=[(20, 360)],
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
        validate=validate,
    )


def endpoint_difference_in_differences(
    changes: pd.DataFrame,
    *,
    model_pairs: Sequence[tuple[str, str]] | None = None,
    models: Sequence[str] = MAIN_MODELS,
) -> pd.DataFrame:
    """Compare models' 20-to-360 changes within each repeat."""

    required = {"repeat", "metric", "model", "value"}
    missing = sorted(required - set(changes.columns))
    if missing:
        raise ValueError(f"Endpoint-change rows are missing columns {missing}.")
    pairs = list(model_pairs) if model_pairs is not None else list(combinations(models, 2))
    rows: list[dict[str, Any]] = []
    indexed = changes.set_index(["repeat", "metric", "model"])["value"]
    repeats = list(pd.unique(changes["repeat"]))
    metrics = list(pd.unique(changes["metric"]))
    for repeat in repeats:
        for metric in metrics:
            for model_a, model_b in pairs:
                value = float(indexed.loc[(repeat, metric, model_a)]) - float(indexed.loc[(repeat, metric, model_b)])
                rows.append(
                    {
                        "repeat": repeat,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "contrast": f"change({model_a}) - change({model_b})",
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def repeat_block_bootstrap_indices(
    repeats: int | Sequence[Any] = len(MAIN_REPEAT_IDS),
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Draw repeat-block indices; one row must index every outcome vector."""

    n_repeats = int(repeats) if isinstance(repeats, (int, np.integer)) else len(tuple(repeats))
    if n_repeats < 2:
        raise ValueError("Repeat-block bootstrap requires at least two repeats.")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n_repeats, size=(int(n_bootstrap), n_repeats), endpoint=False)


def _validate_bootstrap_indices(indices: np.ndarray, n_repeats: int) -> np.ndarray:
    out = np.asarray(indices)
    if out.ndim != 2 or out.shape[1] != n_repeats:
        raise ValueError(f"bootstrap_indices must have shape [B, {n_repeats}], got {out.shape}.")
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError("bootstrap_indices must be integer-valued.")
    if out.shape[0] < 1 or np.any((out < 0) | (out >= n_repeats)):
        raise ValueError("bootstrap_indices contain invalid repeat positions.")
    return out.astype(np.int64, copy=False)


def _bca_intervals(
    observed: np.ndarray,
    bootstrap_statistics: np.ndarray,
    jackknife_statistics: np.ndarray,
    *,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized bias-corrected and accelerated bootstrap intervals.

    Columns are estimands, bootstrap rows resample complete repeat vectors,
    and jackknife rows leave out one complete repeat.  A half-draw continuity
    clamp keeps the canonical strict-less-than bias correction finite at the
    empirical boundaries.
    """

    observed = np.asarray(observed, dtype=np.float64)
    boot = np.asarray(bootstrap_statistics, dtype=np.float64)
    jack = np.asarray(jackknife_statistics, dtype=np.float64)
    if observed.ndim != 1 or boot.ndim != 2 or jack.ndim != 2:
        raise ValueError("BCa inputs must be a vector and two matrices.")
    if boot.shape[1] != observed.size or jack.shape[1] != observed.size:
        raise ValueError("BCa statistic columns must match the observed vector.")
    if boot.shape[0] < 2 or jack.shape[0] < 3:
        raise ValueError("BCa intervals require at least two draws and three jackknife replicates.")
    if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(boot)) and np.all(np.isfinite(jack))):
        raise ValueError("BCa statistics must be finite.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1.")

    draws = boot.shape[0]
    less = np.sum(boot < observed[None, :], axis=0, dtype=np.float64)
    proportions = less / float(draws)
    proportions = np.clip(proportions, 0.5 / draws, 1.0 - 0.5 / draws)
    normal = NormalDist()
    bias_correction = np.array([normal.inv_cdf(float(value)) for value in proportions])

    jack_mean = jack.mean(axis=0)
    influence = jack_mean[None, :] - jack
    numerator = np.sum(influence**3, axis=0)
    denominator = 6.0 * np.sum(influence**2, axis=0) ** 1.5
    acceleration = np.zeros_like(numerator)
    np.divide(
        numerator,
        denominator,
        out=acceleration,
        where=denominator > np.finfo(np.float64).eps,
    )

    alpha = (1.0 - confidence) / 2.0
    nominal_z = (normal.inv_cdf(alpha), normal.inv_cdf(1.0 - alpha))
    adjusted = np.empty((2, observed.size), dtype=np.float64)
    for row, z_alpha in enumerate(nominal_z):
        shifted = bias_correction + z_alpha
        divisor = 1.0 - acceleration * shifted
        near_zero = np.abs(divisor) <= np.finfo(np.float64).eps
        divisor = np.where(near_zero, np.copysign(np.finfo(np.float64).eps, divisor), divisor)
        transformed = bias_correction + shifted / divisor
        adjusted[row] = np.array([normal.cdf(float(value)) for value in transformed])
    adjusted = np.clip(adjusted, 0.0, 1.0)

    lower = np.empty(observed.size, dtype=np.float64)
    upper = np.empty(observed.size, dtype=np.float64)
    for column in range(observed.size):
        probabilities = np.sort(adjusted[:, column])
        lower[column], upper[column] = np.quantile(boot[:, column], probabilities)
    return lower, upper, bias_correction, acceleration


def _t_critical(confidence: float, df: int, *, two_sided: bool) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between 0 and 1.")
    if df < 1:
        return float("nan")
    probability = (1.0 + confidence) / 2.0 if two_sided else confidence
    try:
        from scipy.stats import t as student_t  # type: ignore

        return float(student_t.ppf(probability, df=df))
    except (ImportError, ModuleNotFoundError):
        # Third-order Cornish-Fisher expansion; highly accurate at df=49 and
        # still a practical fallback for smaller sensitivity grids.
        z = NormalDist().inv_cdf(probability)
        nu = float(df)
        return float(
            z
            + (z**3 + z) / (4.0 * nu)
            + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * nu**2)
            + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * nu**3)
        )


def _bootstrap_max_t(
    matrix: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    confidence: float,
    two_sided: bool,
    chunk_size: int = 256,
) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    n_repeats, n_estimands = matrix.shape
    indices = _validate_bootstrap_indices(bootstrap_indices, n_repeats)
    observed = matrix.mean(axis=0)
    observed_se = matrix.std(axis=0, ddof=1) / math.sqrt(n_repeats)
    active = observed_se > np.finfo(np.float64).eps
    if not np.any(active):
        return _t_critical(confidence, n_repeats - 1, two_sided=two_sided)

    # Studentized bootstrap samples can be degenerate even when the original
    # repeat vector is not (for example, a resample may repeatedly select the
    # same observed value).  Mapping a nonzero mean displacement to t=0 in
    # that case is anti-conservative.  Use a scale-relative numerical floor so
    # the displacement remains visible, while leaving every ordinary positive
    # bootstrap SE unchanged.  ``sqrt(eps)`` is the usual threshold at which
    # division starts to lose roughly half of floating-point precision.
    se_floor = np.sqrt(np.finfo(np.float64).eps) * observed_se[active]

    maxima = np.empty(indices.shape[0], dtype=np.float64)
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        sample = matrix[indices[start:stop]][:, :, active]
        boot_mean = sample.mean(axis=1)
        boot_se = sample.std(axis=1, ddof=1) / math.sqrt(n_repeats)
        numerator = boot_mean - observed[active]
        stabilized_boot_se = np.maximum(boot_se, se_floor[None, :])
        standardized = numerator / stabilized_boot_se
        if two_sided:
            maxima[start:stop] = np.max(np.abs(standardized), axis=1)
        else:
            maxima[start:stop] = np.max(standardized, axis=1)
    critical = float(np.quantile(maxima, confidence))
    if not np.isfinite(critical):
        raise FloatingPointError("Bootstrap max-t critical value is non-finite.")
    return critical


def _contrast_matrix(
    contrasts: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    repeat_ids: Sequence[Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[Any]]:
    required = {"repeat", "value", *id_columns}
    missing = sorted(required - set(contrasts.columns))
    if missing:
        raise ValueError(f"Contrast rows are missing columns {missing}.")
    if contrasts.duplicated(["repeat", *id_columns]).any():
        raise ValueError("Contrast rows contain duplicate repeat/estimand values.")

    ids = contrasts[list(id_columns)].drop_duplicates(ignore_index=True)
    ids["_contrast_id"] = np.arange(len(ids), dtype=int)
    merged = contrasts.merge(ids, on=list(id_columns), how="left", validate="many_to_one")
    repeats = list(repeat_ids) if repeat_ids is not None else sorted(pd.unique(merged["repeat"]), key=str)
    wide = merged.pivot(index="repeat", columns="_contrast_id", values="value").reindex(index=repeats, columns=ids["_contrast_id"])
    if wide.isna().any().any():
        raise ValueError("Contrast vectors are incomplete; every repeat must contain every contrast.")
    matrix = wide.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Contrast values must be finite.")
    return matrix, ids.drop(columns="_contrast_id"), repeats


def summarize_paired_contrasts(
    contrasts: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    repeat_ids: Sequence[Any] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize paired differences with pointwise and max-t intervals."""

    matrix, ids, repeats = _contrast_matrix(contrasts, id_columns=id_columns, repeat_ids=repeat_ids)
    n = matrix.shape[0]
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    se = sd / math.sqrt(n)
    point_critical = _t_critical(confidence, n - 1, two_sided=True)
    indices = (
        repeat_block_bootstrap_indices(n, n_bootstrap=n_bootstrap, seed=seed)
        if bootstrap_indices is None
        else _validate_bootstrap_indices(bootstrap_indices, n)
    )
    # A finite empirical bootstrap can occasionally put its max-t quantile
    # below the marginal Student-t cutoff.  Flooring at the pointwise cutoff
    # preserves the defining property of a simultaneous interval: it cannot
    # be narrower than the corresponding pointwise interval.
    simultaneous_critical = max(
        _bootstrap_max_t(
            matrix,
            indices,
            confidence=confidence,
            two_sided=True,
        ),
        point_critical,
    )
    output = ids.copy()
    output["n_repeats"] = n
    output["mean"] = mean
    output["sd"] = sd
    output["mcse"] = se
    output["pointwise_lower"] = mean - point_critical * se
    output["pointwise_upper"] = mean + point_critical * se
    output["simultaneous_lower"] = mean - simultaneous_critical * se
    output["simultaneous_upper"] = mean + simultaneous_critical * se
    output["confidence"] = confidence
    output["max_t_critical"] = simultaneous_critical
    output["bootstrap_draws"] = indices.shape[0]
    return output


def _cell_matrix(
    frame: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    models: Sequence[str],
    sizes: Sequence[int],
    repeat_ids: Sequence[Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    indexed = frame.set_index(["repeat", "model", "n_train"])
    vectors: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []
    for metric in metric_columns:
        base, scope = _metric_scope(metric)
        for model in models:
            for size in sizes:
                vector = np.array(
                    [indexed.loc[(repeat, model, int(size)), metric] for repeat in repeat_ids],
                    dtype=np.float64,
                )
                vectors.append(vector)
                labels.append(
                    {
                        "metric": metric,
                        "base_metric": base,
                        "aggregation": scope,
                        "model": model,
                        "n_train": int(size),
                    }
                )
    return np.column_stack(vectors), pd.DataFrame(labels)


def coverage_qualification(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    coverage_metrics: Sequence[str] | None = None,
    target: float = DEFAULT_COVERAGE_TARGET,
    confidence: float = DEFAULT_CONFIDENCE,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_indices: np.ndarray | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Construct simultaneous one-sided lower bounds and coverage flags."""

    frame = _as_frame(records)
    if coverage_metrics is None:
        coverage_metrics = [metric for metric in infer_metric_columns(frame) if _metric_scope(metric)[0] == "coverage"]
    coverage_metrics = list(coverage_metrics)
    if not coverage_metrics:
        raise ValueError("No coverage metrics were supplied or inferred.")
    if not (0.0 <= target <= 1.0):
        raise ValueError("Coverage target must lie in [0, 1].")
    if validate:
        frame = _validate_grid(
            frame,
            models=models,
            sizes=sizes,
            repeat_ids=repeat_ids,
            metric_columns=coverage_metrics,
        )
    matrix, labels = _cell_matrix(
        frame,
        metric_columns=coverage_metrics,
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
    )
    n = matrix.shape[0]
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    se = sd / math.sqrt(n)
    indices = (
        repeat_block_bootstrap_indices(n, n_bootstrap=n_bootstrap, seed=seed)
        if bootstrap_indices is None
        else _validate_bootstrap_indices(bootstrap_indices, n)
    )
    point_t = _t_critical(confidence, n - 1, two_sided=False)
    max_t = max(
        _bootstrap_max_t(matrix, indices, confidence=confidence, two_sided=False),
        point_t,
    )
    output = labels.copy()
    output["n_repeats"] = n
    output["mean"] = mean
    output["sd"] = sd
    output["mcse"] = se
    output["pointwise_lower"] = mean - point_t * se
    output["simultaneous_lower"] = mean - max_t * se
    output["target"] = float(target)
    output["qualified"] = output["simultaneous_lower"] >= float(target)
    output["confidence"] = confidence
    output["max_t_critical"] = max_t
    output["bootstrap_draws"] = indices.shape[0]
    return output


def supported_best_declarations(
    model_contrasts: pd.DataFrame,
    *,
    models: Sequence[str] = MAIN_MODELS,
) -> pd.DataFrame:
    """Declare a CRPS model best only with support against every comparator.

    The input must contain the simultaneous intervals returned by
    :func:`summarize_paired_contrasts`.  Because each stored contrast is
    ``model_a - model_b`` and lower CRPS is better, model A is supported over B
    only when the simultaneous upper endpoint is below zero (and conversely
    for B when the lower endpoint is above zero).
    """

    required = {
        "metric",
        "n_train",
        "model_a",
        "model_b",
        "mean",
        "simultaneous_lower",
        "simultaneous_upper",
    }
    missing = sorted(required - set(model_contrasts.columns))
    if missing:
        raise ValueError(f"Model contrasts are missing supported-best columns: {missing}.")
    models = tuple(str(model) for model in models)
    crps = model_contrasts[
        model_contrasts["metric"].map(lambda metric: _metric_scope(str(metric))[0] == "crps")
    ].copy()
    if crps.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "aggregation",
                "n_train",
                "model",
                "model_display_name",
                "comparators_supported",
                "comparators_required",
                "supported_best",
                "comparison_evidence",
            ]
        )
    if crps.duplicated(["metric", "n_train", "model_a", "model_b"]).any():
        raise ValueError("CRPS model contrasts contain duplicate oriented comparisons.")
    unknown = (set(crps["model_a"]) | set(crps["model_b"])) - set(models)
    if unknown:
        raise ValueError(f"CRPS contrasts contain unknown models: {sorted(unknown)}.")
    endpoints = crps[["simultaneous_lower", "simultaneous_upper"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(endpoints)) or np.any(endpoints[:, 0] > endpoints[:, 1]):
        raise ValueError("CRPS simultaneous intervals must be finite and ordered.")

    rows: list[dict[str, Any]] = []
    for (metric, n_train), group in crps.groupby(["metric", "n_train"], sort=False):
        expected_pairs = {frozenset(pair) for pair in combinations(models, 2)}
        observed_pairs = {
            frozenset((str(row.model_a), str(row.model_b)))
            for row in group.itertuples(index=False)
        }
        if observed_pairs != expected_pairs or len(group) != len(expected_pairs):
            raise ValueError(
                f"CRPS contrasts for metric={metric}, n_train={n_train} are not the complete model-pair family."
            )
        _, scope = _metric_scope(str(metric))
        for model in models:
            evidence: list[dict[str, Any]] = []
            for comparator in models:
                if comparator == model:
                    continue
                match = group[
                    ((group["model_a"] == model) & (group["model_b"] == comparator))
                    | ((group["model_a"] == comparator) & (group["model_b"] == model))
                ]
                if len(match) != 1:
                    raise ValueError(f"Missing unique CRPS evidence for {model} versus {comparator}.")
                contrast = match.iloc[0]
                if contrast["model_a"] == model:
                    supported = float(contrast["simultaneous_upper"]) < 0.0
                else:
                    supported = float(contrast["simultaneous_lower"]) > 0.0
                evidence.append(
                    {
                        "comparator": comparator,
                        "supported_lower_crps": bool(supported),
                        "stored_orientation": f"{contrast['model_a']} - {contrast['model_b']}",
                        "simultaneous_lower": float(contrast["simultaneous_lower"]),
                        "simultaneous_upper": float(contrast["simultaneous_upper"]),
                    }
                )
            supported_count = sum(item["supported_lower_crps"] for item in evidence)
            rows.append(
                {
                    "metric": metric,
                    "aggregation": scope,
                    "n_train": int(n_train),
                    "model": model,
                    "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "comparators_supported": int(supported_count),
                    "comparators_required": len(models) - 1,
                    "supported_best": supported_count == len(models) - 1,
                    "comparison_evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":")),
                }
            )
    return pd.DataFrame(rows)


def controlled_sharpness_table(
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    model_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Join width comparisons to simultaneous coverage qualification.

    A pair is labelled ``sharper_at_controlled_coverage`` only if both models'
    corresponding play-weighted or game-equal coverage lower bounds reach the
    target and the simultaneous width contrast supports a lower-width model.
    """

    summary_required = {"metric", "model", "n_train", "mean"}
    coverage_required = {"metric", "model", "n_train", "simultaneous_lower", "target", "qualified"}
    contrast_required = {
        "metric",
        "n_train",
        "model_a",
        "model_b",
        "mean",
        "simultaneous_lower",
        "simultaneous_upper",
    }
    for name, frame, required in (
        ("summary", summary, summary_required),
        ("coverage", coverage, coverage_required),
        ("model_contrasts", model_contrasts, contrast_required),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing controlled-sharpness columns: {missing}.")

    widths = model_contrasts[
        model_contrasts["metric"].map(
            lambda metric: _metric_scope(str(metric))[0] in {"mean_width", "mean_width_inclusive"}
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    coverage_by_scope: dict[str, str] = {}
    for metric in pd.unique(coverage["metric"]):
        base, scope = _metric_scope(str(metric))
        if base == "coverage":
            if scope in coverage_by_scope and coverage_by_scope[scope] != str(metric):
                raise ValueError(f"Multiple coverage metrics were supplied for aggregation scope {scope!r}.")
            coverage_by_scope[scope] = str(metric)

    for contrast in widths.itertuples(index=False):
        base, scope = _metric_scope(str(contrast.metric))
        coverage_metric = coverage_by_scope.get(scope)
        if coverage_metric is None:
            raise ValueError(f"No corresponding {scope} coverage metric exists for {contrast.metric}.")

        def cell(model: str) -> tuple[float, pd.Series]:
            width_match = summary[
                (summary["metric"] == contrast.metric)
                & (summary["model"] == model)
                & (summary["n_train"] == int(contrast.n_train))
            ]
            coverage_match = coverage[
                (coverage["metric"] == coverage_metric)
                & (coverage["model"] == model)
                & (coverage["n_train"] == int(contrast.n_train))
            ]
            if len(width_match) != 1 or len(coverage_match) != 1:
                raise ValueError(
                    f"Expected one width and coverage row for {model}, {contrast.metric}, n={contrast.n_train}."
                )
            return float(width_match.iloc[0]["mean"]), coverage_match.iloc[0]

        width_a, coverage_a = cell(str(contrast.model_a))
        width_b, coverage_b = cell(str(contrast.model_b))
        target_a = float(coverage_a["target"])
        target_b = float(coverage_b["target"])
        if not math.isclose(target_a, target_b, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Compared models have different coverage targets.")
        qualified_a = bool(coverage_a["qualified"]) and float(coverage_a["simultaneous_lower"]) >= target_a
        qualified_b = bool(coverage_b["qualified"]) and float(coverage_b["simultaneous_lower"]) >= target_b
        both_qualified = qualified_a and qualified_b
        lower = float(contrast.simultaneous_lower)
        upper = float(contrast.simultaneous_upper)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError("Width simultaneous intervals must be finite and ordered.")
        sharper_model: str | None = None
        if both_qualified and upper < 0.0:
            sharper_model = str(contrast.model_a)
        elif both_qualified and lower > 0.0:
            sharper_model = str(contrast.model_b)
        if sharper_model is not None:
            status = "sharper_at_controlled_coverage"
        elif both_qualified:
            status = "inconclusive_at_controlled_coverage"
        else:
            status = "coverage_not_controlled"
        rows.append(
            {
                "metric": contrast.metric,
                "base_metric": base,
                "aggregation": scope,
                "n_train": int(contrast.n_train),
                "model_a": str(contrast.model_a),
                "model_a_display_name": MODEL_DISPLAY_NAMES.get(str(contrast.model_a), str(contrast.model_a)),
                "model_b": str(contrast.model_b),
                "model_b_display_name": MODEL_DISPLAY_NAMES.get(str(contrast.model_b), str(contrast.model_b)),
                "mean_width_a": width_a,
                "mean_width_b": width_b,
                "width_mean_difference_a_minus_b": float(contrast.mean),
                "width_simultaneous_lower": lower,
                "width_simultaneous_upper": upper,
                "coverage_metric": coverage_metric,
                "coverage_target": target_a,
                "coverage_lcb_a": float(coverage_a["simultaneous_lower"]),
                "coverage_lcb_b": float(coverage_b["simultaneous_lower"]),
                "coverage_qualified_a": qualified_a,
                "coverage_qualified_b": qualified_b,
                "both_coverage_qualified": both_qualified,
                "sharper_model": sharper_model,
                "sharper_model_display_name": (
                    MODEL_DISPLAY_NAMES.get(sharper_model, sharper_model) if sharper_model is not None else None
                ),
                "sharper_at_controlled_coverage": sharper_model is not None,
                "sharpness_status": status,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_cell_sd_intervals(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_indices: np.ndarray | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Repeat-block BCa intervals for across-rerun standard deviations."""

    frame = _as_frame(records)
    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    if validate:
        frame = _validate_grid(frame, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metrics)
    matrix, labels = _cell_matrix(
        frame,
        metric_columns=metrics,
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
    )
    n = matrix.shape[0]
    indices = (
        repeat_block_bootstrap_indices(n, n_bootstrap=n_bootstrap, seed=seed)
        if bootstrap_indices is None
        else _validate_bootstrap_indices(bootstrap_indices, n)
    )
    boot_sd = np.empty((indices.shape[0], matrix.shape[1]), dtype=np.float64)
    chunk_size = 256
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        boot_sd[start:stop] = matrix[indices[start:stop]].std(axis=1, ddof=1)
    observed_sd = matrix.std(axis=0, ddof=1)
    jackknife_sd = np.empty((n, matrix.shape[1]), dtype=np.float64)
    for omitted in range(n):
        jackknife_sd[omitted] = np.delete(matrix, omitted, axis=0).std(axis=0, ddof=1)
    lower, upper, bias_correction, acceleration = _bca_intervals(
        observed_sd,
        boot_sd,
        jackknife_sd,
        confidence=confidence,
    )
    output = labels.copy()
    output["n_repeats"] = n
    output["sd"] = observed_sd
    output["bootstrap_lower"] = lower
    output["bootstrap_upper"] = upper
    output["bca_bias_correction"] = bias_correction
    output["bca_acceleration"] = acceleration
    output["confidence"] = confidence
    output["bootstrap_draws"] = indices.shape[0]
    output["interval_method"] = "bca_repeat_block"
    return output


def bootstrap_log_sd_ratios(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    model_pairs: Sequence[tuple[str, str]] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = ANCHOR_SIZES,
    repeat_ids: Sequence[Any] = MAIN_REPEAT_IDS,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_indices: np.ndarray | None = None,
    sd_floor: float = 1e-12,
    validate: bool = True,
) -> pd.DataFrame:
    """Repeat-block BCa intervals for paired log ratios of rerun SDs."""

    if sd_floor <= 0:
        raise ValueError("sd_floor must be positive.")
    frame = _as_frame(records)
    metrics = list(metric_columns) if metric_columns is not None else infer_metric_columns(frame)
    if validate:
        frame = _validate_grid(frame, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metrics)
    matrix, cell_labels = _cell_matrix(
        frame,
        metric_columns=metrics,
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
    )
    lookup = {
        (row.metric, row.model, int(row.n_train)): idx
        for idx, row in enumerate(cell_labels.itertuples(index=False))
    }
    pairs = list(model_pairs) if model_pairs is not None else list(combinations(models, 2))
    ratio_labels: list[dict[str, Any]] = []
    numerator_idx: list[int] = []
    denominator_idx: list[int] = []
    for metric in metrics:
        base, scope = _metric_scope(metric)
        for size in sizes:
            for model_a, model_b in pairs:
                numerator_idx.append(lookup[(metric, model_a, int(size))])
                denominator_idx.append(lookup[(metric, model_b, int(size))])
                ratio_labels.append(
                    {
                        "metric": metric,
                        "base_metric": base,
                        "aggregation": scope,
                        "n_train": int(size),
                        "model_a": model_a,
                        "model_b": model_b,
                        "contrast": f"log SD({model_a}) / SD({model_b})",
                    }
                )
    numerator_idx_array = np.asarray(numerator_idx, dtype=int)
    denominator_idx_array = np.asarray(denominator_idx, dtype=int)
    observed_sd = matrix.std(axis=0, ddof=1)
    observed_log_ratio = np.log(np.maximum(observed_sd[numerator_idx_array], sd_floor)) - np.log(
        np.maximum(observed_sd[denominator_idx_array], sd_floor)
    )

    n = matrix.shape[0]
    indices = (
        repeat_block_bootstrap_indices(n, n_bootstrap=n_bootstrap, seed=seed)
        if bootstrap_indices is None
        else _validate_bootstrap_indices(bootstrap_indices, n)
    )
    boot_log_ratio = np.empty((indices.shape[0], len(ratio_labels)), dtype=np.float64)
    chunk_size = 256
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        boot_sd = matrix[indices[start:stop]].std(axis=1, ddof=1)
        boot_log_ratio[start:stop] = np.log(np.maximum(boot_sd[:, numerator_idx_array], sd_floor)) - np.log(
            np.maximum(boot_sd[:, denominator_idx_array], sd_floor)
        )
    jackknife_log_ratio = np.empty((n, len(ratio_labels)), dtype=np.float64)
    for omitted in range(n):
        jack_sd = np.delete(matrix, omitted, axis=0).std(axis=0, ddof=1)
        jackknife_log_ratio[omitted] = np.log(
            np.maximum(jack_sd[numerator_idx_array], sd_floor)
        ) - np.log(np.maximum(jack_sd[denominator_idx_array], sd_floor))
    lower, upper, bias_correction, acceleration = _bca_intervals(
        observed_log_ratio,
        boot_log_ratio,
        jackknife_log_ratio,
        confidence=confidence,
    )
    output = pd.DataFrame(ratio_labels)
    output["n_repeats"] = n
    output["log_sd_ratio"] = observed_log_ratio
    output["log_bootstrap_lower"] = lower
    output["log_bootstrap_upper"] = upper
    output["sd_ratio"] = np.exp(observed_log_ratio)
    output["ratio_bootstrap_lower"] = np.exp(lower)
    output["ratio_bootstrap_upper"] = np.exp(upper)
    output["bca_bias_correction"] = bias_correction
    output["bca_acceleration"] = acceleration
    output["regularized"] = (
        (observed_sd[numerator_idx_array] <= sd_floor) | (observed_sd[denominator_idx_array] <= sd_floor)
    )
    output["confidence"] = confidence
    output["bootstrap_draws"] = indices.shape[0]
    output["interval_method"] = "bca_repeat_block"
    return output


def plot_metric_ribbons(
    summary: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    model_order: Sequence[str] = MAIN_MODELS,
    output_path: str | Path | None = None,
):
    """Plot mean learning curves with q10--q90 whole-rerun ribbons."""

    required = {"model", "n_train", "metric", "mean", "q10", "q90"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing plot columns {missing}.")
    selected = list(metrics) if metrics is not None else list(pd.unique(summary["metric"]))
    if not selected:
        raise ValueError("At least one metric is required for plotting.")
    unknown = sorted(set(selected) - set(summary["metric"]))
    if unknown:
        raise ValueError(f"Metrics not found in summary: {unknown}.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes_array = plt.subplots(len(selected), 1, figsize=(9, max(3.2 * len(selected), 4.0)), squeeze=False)
    axes = axes_array[:, 0]
    for axis, metric in zip(axes, selected):
        metric_data = summary[summary["metric"] == metric]
        for model in model_order:
            group = metric_data[metric_data["model"] == model].sort_values("n_train")
            if group.empty:
                continue
            x = group["n_train"].to_numpy(dtype=float)
            mean = group["mean"].to_numpy(dtype=float)
            q10 = group["q10"].to_numpy(dtype=float)
            q90 = group["q90"].to_numpy(dtype=float)
            display_name = (
                str(group["model_display_name"].iloc[0])
                if "model_display_name" in group.columns
                else MODEL_DISPLAY_NAMES.get(model, model)
            )
            line = axis.plot(x, mean, marker="o", label=display_name)[0]
            axis.fill_between(x, q10, q90, color=line.get_color(), alpha=0.18)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
        if _metric_scope(metric)[0] == "coverage":
            axis.axhline(DEFAULT_COVERAGE_TARGET, color="black", linestyle="--", linewidth=1)
        axis.legend(fontsize="small")
    axes[-1].set_xlabel("Training games")
    fig.tight_layout()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
    return fig, axes


def flatten_sensitivity_candidate_scores(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    extended: bool = False,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = SENSITIVITY_SIZES,
    tie_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Validate and flatten every saved sensitivity candidate tune score.

    Input rows contain one complete sensitivity cell with
    ``branch/repeat/n_train/model/candidate_scores``.  The returned row per
    candidate includes a deterministic replay of the runner's exact
    frozen-first tie rule (strictly lower score replaces the incumbent; exact
    ties retain the earlier candidate) and is directly groupable for selection
    frequencies.
    """

    if not isinstance(extended, (bool, np.bool_)):
        raise TypeError("extended must be boolean.")
    if not math.isfinite(tie_tolerance) or tie_tolerance != 0.0:
        raise ValueError("tie_tolerance must be exactly 0.0 for the locked strict-lower tie rule.")
    source = _as_frame(rows)
    required = {"branch", "repeat", "n_train", "model", "candidate_scores"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Candidate-score cells are missing columns: {missing}.")
    if not (source["branch"].astype(str) == "sensitivity").all():
        bad = sorted(set(source.loc[source["branch"].astype(str) != "sensitivity", "branch"].astype(str)))
        raise ValueError(f"Candidate-score rows must all be sensitivity cells; found {bad}.")
    source = source.copy()
    source["model"] = source["model"].astype(str)
    source["repeat"] = source["repeat"].astype(int)
    source["n_train"] = source["n_train"].astype(int)
    models = tuple(str(model) for model in models)
    sizes = tuple(int(size) for size in sizes)
    if models != MAIN_MODELS:
        raise ValueError(f"Candidate-score model order must be the locked canonical order {MAIN_MODELS}.")
    if sizes != SENSITIVITY_SIZES:
        raise ValueError(f"Candidate-score sizes must be exactly {SENSITIVITY_SIZES}.")
    repeat_ids = MAIN_REPEAT_IDS if bool(extended) else SENSITIVITY_REPEAT_IDS
    if source.duplicated(["repeat", "n_train", "model"]).any():
        raise ValueError("Candidate-score input contains duplicate sensitivity cells.")
    expected_cells = {
        (repeat, size, model)
        for repeat in repeat_ids
        for size in sizes
        for model in models
    }
    observed_cells = set(source[["repeat", "n_train", "model"]].itertuples(index=False, name=None))
    missing_cells = expected_cells - observed_cells
    extra_cells = observed_cells - expected_cells
    if missing_cells or extra_cells:
        raise ValueError(
            "Candidate-score cells are not the exact sensitivity grid: "
            f"missing={len(missing_cells)}, extra={len(extra_cells)}."
        )

    candidate_counts = {model: (5 if model == "ridge_sgd_l2" else 4) for model in models}
    flattened: list[dict[str, Any]] = []
    for cell in source.itertuples(index=False):
        cell_values = cell._asdict()
        candidate_payload = cell_values["candidate_scores"]
        if isinstance(candidate_payload, str):
            try:
                candidate_payload = json.loads(candidate_payload)
            except json.JSONDecodeError as exc:
                raise ValueError("candidate_scores contains invalid JSON.") from exc
        if not isinstance(candidate_payload, Sequence) or isinstance(candidate_payload, (str, bytes)):
            raise ValueError("candidate_scores must be a list of candidate mappings.")
        candidates = list(candidate_payload)
        model = str(cell_values["model"])
        expected_count = candidate_counts[model]
        if len(candidates) != expected_count:
            raise ValueError(
                f"{model} repeat={cell_values['repeat']} n={cell_values['n_train']} must have "
                f"{expected_count} candidates, found {len(candidates)}."
            )
        parsed: list[dict[str, Any]] = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError("Every candidate score must be a mapping.")
            candidate_required = {
                "candidate_index",
                "is_frozen_main",
                "config",
                "config_hash",
                "tune_crps",
                "elapsed_seconds",
                "fit_seed",
                "refit_seed",
            }
            absent = sorted(candidate_required - set(candidate))
            if absent:
                raise ValueError(f"Candidate score is missing fields: {absent}.")
            index = candidate["candidate_index"]
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or int(index) != position:
                raise ValueError(
                    f"Candidate indices for {model} must be ordered exactly 0..{expected_count - 1}."
                )
            frozen = candidate["is_frozen_main"]
            if not isinstance(frozen, (bool, np.bool_)) or bool(frozen) != (position == 0):
                raise ValueError("Exactly candidate index 0 must be marked is_frozen_main=true.")
            config = candidate["config"]
            if not isinstance(config, Mapping):
                raise ValueError("Candidate config must be a mapping.")
            try:
                config_json = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("Candidate config is not canonical-JSON serializable.") from exc
            expected_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            if str(candidate["config_hash"]) != expected_hash:
                raise ValueError("Candidate config_hash does not match its canonical config JSON.")
            tune_crps = float(candidate["tune_crps"])
            elapsed_seconds = float(candidate["elapsed_seconds"])
            if not math.isfinite(tune_crps) or tune_crps < 0.0:
                raise ValueError("Candidate tune_crps must be finite and nonnegative.")
            if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
                raise ValueError("Candidate elapsed_seconds must be finite and nonnegative.")
            fit_seed = candidate["fit_seed"]
            if isinstance(fit_seed, bool) or not isinstance(fit_seed, (int, np.integer)) or int(fit_seed) <= 0:
                raise ValueError("Candidate fit_seed must be a positive integer.")
            refit_seed = candidate["refit_seed"]
            if model in {"ridge_sgd_l2", "lightgbm_multiclass"}:
                if refit_seed is not None:
                    raise ValueError("Classical sensitivity candidates must have refit_seed=null.")
            elif isinstance(refit_seed, bool) or not isinstance(refit_seed, (int, np.integer)) or int(refit_seed) <= 0:
                raise ValueError("Neural sensitivity candidates must have a positive refit_seed.")
            parsed.append(
                {
                    "candidate_index": position,
                    "is_frozen_main": bool(frozen),
                    "config_json": config_json,
                    "config_hash": expected_hash,
                    "tune_crps": tune_crps,
                    "elapsed_seconds": elapsed_seconds,
                    "fit_seed": int(fit_seed),
                    "refit_seed": None if refit_seed is None else int(refit_seed),
                }
            )

        # Replay runner.py's exact strict-improvement rule. Candidate zero is
        # the deterministic winner only for an exactly equal score.
        selected_index = 0
        selected_score = float("inf")
        for candidate in parsed:
            if candidate["tune_crps"] < selected_score:
                selected_score = candidate["tune_crps"]
                selected_index = int(candidate["candidate_index"])
        declared_index = cell_values.get("selected_candidate_index")
        if declared_index is not None and int(declared_index) != selected_index:
            raise ValueError("Saved selected_candidate_index disagrees with the frozen-first tie replay.")
        declared_config = cell_values.get("selected_config")
        if declared_config is not None:
            if isinstance(declared_config, str):
                try:
                    declared_config = json.loads(declared_config)
                except json.JSONDecodeError as exc:
                    raise ValueError("selected_config contains invalid JSON.") from exc
            declared_json = json.dumps(
                declared_config, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if declared_json != parsed[selected_index]["config_json"]:
                raise ValueError("Saved selected_config disagrees with the candidate-score tie replay.")

        for candidate in parsed:
            is_selected = int(candidate["candidate_index"]) == selected_index
            flattened.append(
                {
                    "branch": "sensitivity",
                    "repeat": int(cell_values["repeat"]),
                    "n_train": int(cell_values["n_train"]),
                    "model": model,
                    "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    **candidate,
                    "is_selected": is_selected,
                    "frozen_main_selected": selected_index == 0,
                    "selected_candidate_index": selected_index,
                    "within_tie_tolerance_of_selected": float(candidate["tune_crps"]) == selected_score,
                    "tie_tolerance": float(tie_tolerance),
                    "tie_break_rule": "lowest_ordered_candidate_on_exact_tie",
                }
            )

    output = pd.DataFrame(flattened)
    keys = ["repeat", "n_train", "model", "candidate_index"]
    if output.duplicated(keys).any():
        raise AssertionError("Flattened candidate-score keys are unexpectedly duplicated.")
    expected_rows = len(repeat_ids) * len(sizes) * sum(candidate_counts.values())
    if len(output) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} flattened candidates, found {len(output)}.")
    selected_counts = output.groupby(["repeat", "n_train", "model"])["is_selected"].sum()
    if not (selected_counts == 1).all():
        raise AssertionError("Every sensitivity cell must select exactly one candidate.")
    return output.sort_values(keys).reset_index(drop=True)


def _winner_table(frame: pd.DataFrame, metrics: Sequence[str], sizes: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    means = frame.groupby(["model", "n_train"], as_index=False)[list(metrics)].mean()
    for metric in metrics:
        if _metric_scope(metric)[0] not in {"crps", "mean_width", "mean_width_inclusive"}:
            continue
        for size in sizes:
            group = means[means["n_train"] == int(size)].sort_values([metric, "model"])
            ranking = [str(value) for value in group["model"]]
            rows.append(
                {
                    "metric": metric,
                    "n_train": int(size),
                    "winner": ranking[0],
                    "ranking": " > ".join(ranking),
                }
            )
    return pd.DataFrame(rows)


def _sensitivity_conclusions(
    fixed: pd.DataFrame,
    tuned: pd.DataFrame,
    *,
    metrics: Sequence[str],
    coverage_status: pd.DataFrame,
    models: Sequence[str],
    sizes: Sequence[int],
    repeat_ids: Sequence[Any],
    confidence: float,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    """Classify prespecified pairwise conclusions for the trigger rule.

    A negative model-a-minus-model-b interval supports model A having lower
    CRPS/width; a positive interval supports the opposite direction.  Width
    is only interpretable when both models meet the simultaneous coverage
    requirement in the corresponding play or game-equal scope.
    """

    conclusion_metrics = [
        metric
        for metric in metrics
        if _metric_scope(metric)[0] in {"crps", "mean_width", "mean_width_inclusive"}
    ]
    if not conclusion_metrics:
        return pd.DataFrame(
            columns=[
                "metric",
                "n_train",
                "model_a",
                "model_b",
                "fixed_conclusion",
                "tuned_conclusion",
                "conclusion_state_changed",
                "conclusion_changed",
            ]
        )

    def infer(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        values = paired_model_contrasts(
            frame,
            metric_columns=conclusion_metrics,
            models=models,
            sizes=sizes,
            repeat_ids=repeat_ids,
            validate=False,
        )
        result = summarize_paired_contrasts(
            values,
            id_columns=["metric", "n_train", "model_a", "model_b", "contrast"],
            repeat_ids=repeat_ids,
            confidence=confidence,
            bootstrap_indices=bootstrap_indices,
        )
        return result.rename(
            columns={
                "mean": f"{label}_mean_difference",
                "simultaneous_lower": f"{label}_lower",
                "simultaneous_upper": f"{label}_upper",
            }
        )[
            [
                "metric",
                "n_train",
                "model_a",
                "model_b",
                "contrast",
                f"{label}_mean_difference",
                f"{label}_lower",
                f"{label}_upper",
            ]
        ]

    conclusions = infer(fixed, "fixed").merge(
        infer(tuned, "tuned"),
        on=["metric", "n_train", "model_a", "model_b", "contrast"],
        validate="one_to_one",
    )
    coverage_lookup: dict[tuple[str, str, int], tuple[bool, bool]] = {}
    for row in coverage_status.itertuples(index=False):
        coverage_lookup[(str(row.metric), str(row.model), int(row.n_train))] = (
            bool(row.fixed_qualified),
            bool(row.tuned_qualified),
        )
    coverage_by_scope: dict[str, str] = {}
    for metric in pd.unique(coverage_status.get("metric", pd.Series(dtype=str))):
        base, scope = _metric_scope(str(metric))
        if base == "coverage":
            coverage_by_scope.setdefault(scope, str(metric))

    fixed_eligible: list[bool] = []
    tuned_eligible: list[bool] = []
    estimand_types: list[str] = []
    for row in conclusions.itertuples(index=False):
        base, scope = _metric_scope(str(row.metric))
        if base == "crps":
            fixed_eligible.append(True)
            tuned_eligible.append(True)
            estimand_types.append("crps")
            continue
        coverage_metric = coverage_by_scope.get(scope)
        fixed_flags: list[bool] = []
        tuned_flags: list[bool] = []
        if coverage_metric is not None:
            for model in (str(row.model_a), str(row.model_b)):
                fixed_flag, tuned_flag = coverage_lookup.get(
                    (coverage_metric, model, int(row.n_train)),
                    (False, False),
                )
                fixed_flags.append(fixed_flag)
                tuned_flags.append(tuned_flag)
        fixed_eligible.append(len(fixed_flags) == 2 and all(fixed_flags))
        tuned_eligible.append(len(tuned_flags) == 2 and all(tuned_flags))
        estimand_types.append("coverage_qualified_width")
    conclusions["estimand_type"] = estimand_types
    conclusions["fixed_eligible"] = fixed_eligible
    conclusions["tuned_eligible"] = tuned_eligible

    def classify(lower: float, upper: float, eligible: bool) -> str:
        if not eligible:
            return "inconclusive"
        if upper < 0.0:
            return "supported"
        if lower > 0.0:
            return "opposite"
        return "inconclusive"

    conclusions["fixed_conclusion"] = [
        classify(lower, upper, eligible)
        for lower, upper, eligible in zip(
            conclusions["fixed_lower"], conclusions["fixed_upper"], conclusions["fixed_eligible"]
        )
    ]
    conclusions["tuned_conclusion"] = [
        classify(lower, upper, eligible)
        for lower, upper, eligible in zip(
            conclusions["tuned_lower"], conclusions["tuned_upper"], conclusions["tuned_eligible"]
        )
    ]
    conclusions["conclusion_state_changed"] = (
        conclusions["fixed_conclusion"] != conclusions["tuned_conclusion"]
    )
    # ``supported`` means model A is supported and ``opposite`` means model B
    # is supported in the stored A-minus-B orientation.  Both are directional
    # conclusions.  The extension rule must therefore be orientation-neutral:
    # trigger when either fixed directional conclusion becomes inconclusive or
    # reverses under tuning.
    conclusions["conclusion_changed"] = (
        conclusions["fixed_conclusion"].isin(["supported", "opposite"])
        & (conclusions["tuned_conclusion"] != conclusions["fixed_conclusion"])
    )
    return conclusions


def summarize_tuning_sensitivity(
    fixed_records: pd.DataFrame | Iterable[Mapping[str, Any]],
    tuned_records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    metric_columns: Sequence[str] | None = None,
    models: Sequence[str] = MAIN_MODELS,
    sizes: Sequence[int] = SENSITIVITY_SIZES,
    repeat_ids: Sequence[Any] = SENSITIVITY_REPEAT_IDS,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    confidence: float = DEFAULT_CONFIDENCE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    selected_config_column: str = "selected_config",
) -> dict[str, Any]:
    """Summarize paired tuned-minus-fixed sensitivity and trigger extension.

    The locked stage-1 analysis uses 20 complete repeats at the same three
    anchors.  It extends the entire three-anchor sensitivity branch to 50
    repeats only if tuning changes a mean ranking or degrades a prespecified
    CRPS/coverage-qualified-width directional conclusion under the frozen
    configuration to ``inconclusive`` or the opposite direction under tuning.
    A tuned-minus-fixed shift or an inconclusive-to-supported improvement alone
    is descriptive and does not trigger extension.
    """

    fixed_all = _as_frame(fixed_records)
    tuned_all = _as_frame(tuned_records)
    if "branch" in fixed_all.columns:
        fixed_all = fixed_all[fixed_all["branch"].astype(str) == "main"].copy()
    if "branch" in tuned_all.columns:
        tuned_all = tuned_all[tuned_all["branch"].astype(str) == "sensitivity"].copy()
    selected_sizes = [int(size) for size in sizes]
    fixed = fixed_all[
        fixed_all["repeat"].isin(repeat_ids) & fixed_all["n_train"].isin(selected_sizes)
    ].copy()
    tuned = tuned_all[
        tuned_all["repeat"].isin(repeat_ids) & tuned_all["n_train"].isin(selected_sizes)
    ].copy()
    metrics = list(metric_columns) if metric_columns is not None else [
        metric for metric in infer_metric_columns(fixed) if metric in tuned.columns
    ]
    fixed = _validate_grid(fixed, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metrics)
    tuned = _validate_grid(tuned, models=models, sizes=sizes, repeat_ids=repeat_ids, metric_columns=metrics)

    keys = ["repeat", "model", "n_train"]
    merged = fixed[keys + metrics].merge(
        tuned[keys + metrics],
        on=keys,
        suffixes=("_fixed", "_tuned"),
        validate="one_to_one",
    )
    delta_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        values = row._asdict()
        for metric in metrics:
            delta_rows.append(
                {
                    "repeat": values["repeat"],
                    "model": values["model"],
                    "n_train": int(values["n_train"]),
                    "metric": metric,
                    "contrast": "tuned - fixed",
                    "value": float(values[f"{metric}_tuned"] - values[f"{metric}_fixed"]),
                }
            )
    deltas = pd.DataFrame(delta_rows)
    bootstrap_indices = repeat_block_bootstrap_indices(
        len(repeat_ids), n_bootstrap=n_bootstrap, seed=seed
    )
    inference = summarize_paired_contrasts(
        deltas,
        id_columns=["metric", "model", "n_train", "contrast"],
        repeat_ids=repeat_ids,
        confidence=confidence,
        bootstrap_indices=bootstrap_indices,
    )
    fixed_means = fixed.groupby(["model", "n_train"], as_index=False)[metrics].mean().melt(
        id_vars=["model", "n_train"], var_name="metric", value_name="fixed_mean"
    )
    tuned_means = tuned.groupby(["model", "n_train"], as_index=False)[metrics].mean().melt(
        id_vars=["model", "n_train"], var_name="metric", value_name="tuned_mean"
    )
    inference = inference.merge(fixed_means, on=["model", "n_train", "metric"], validate="one_to_one")
    inference = inference.merge(tuned_means, on=["model", "n_train", "metric"], validate="one_to_one")
    inference["relative_change"] = inference["mean"] / inference["fixed_mean"].abs().replace(0.0, np.nan)
    inference["significant_shift"] = (inference["simultaneous_lower"] > 0.0) | (
        inference["simultaneous_upper"] < 0.0
    )

    fixed_winners = _winner_table(fixed, metrics, sizes).rename(
        columns={"winner": "fixed_winner", "ranking": "fixed_ranking"}
    )
    tuned_winners = _winner_table(tuned, metrics, sizes).rename(
        columns={"winner": "tuned_winner", "ranking": "tuned_ranking"}
    )
    rankings = fixed_winners.merge(tuned_winners, on=["metric", "n_train"], validate="one_to_one")
    rankings["ranking_changed"] = rankings["fixed_ranking"] != rankings["tuned_ranking"]

    coverage_metrics = [metric for metric in metrics if _metric_scope(metric)[0] == "coverage"]
    if coverage_metrics:
        fixed_coverage = coverage_qualification(
            fixed,
            coverage_metrics=coverage_metrics,
            target=coverage_target,
            confidence=confidence,
            models=models,
            sizes=sizes,
            repeat_ids=repeat_ids,
            bootstrap_indices=bootstrap_indices,
        )
        tuned_coverage = coverage_qualification(
            tuned,
            coverage_metrics=coverage_metrics,
            target=coverage_target,
            confidence=confidence,
            models=models,
            sizes=sizes,
            repeat_ids=repeat_ids,
            bootstrap_indices=bootstrap_indices,
        )
        coverage_status = fixed_coverage[["metric", "model", "n_train", "qualified"]].rename(
            columns={"qualified": "fixed_qualified"}
        ).merge(
            tuned_coverage[["metric", "model", "n_train", "qualified"]].rename(
                columns={"qualified": "tuned_qualified"}
            ),
            on=["metric", "model", "n_train"],
            validate="one_to_one",
        )
        coverage_status["qualification_changed"] = (
            coverage_status["fixed_qualified"] != coverage_status["tuned_qualified"]
        )
    else:
        coverage_status = pd.DataFrame(
            columns=["metric", "model", "n_train", "fixed_qualified", "tuned_qualified", "qualification_changed"]
        )

    conclusions = _sensitivity_conclusions(
        fixed,
        tuned,
        metrics=metrics,
        coverage_status=coverage_status,
        models=models,
        sizes=sizes,
        repeat_ids=repeat_ids,
        confidence=confidence,
        bootstrap_indices=bootstrap_indices,
    )

    if selected_config_column in tuned.columns:
        configuration_frequencies = (
            tuned.groupby(["model", "n_train", selected_config_column], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        configuration_frequencies["fraction"] = configuration_frequencies["count"] / len(repeat_ids)
    else:
        configuration_frequencies = pd.DataFrame(
            columns=["model", "n_train", selected_config_column, "count", "fraction"]
        )

    reasons: list[str] = []
    if not rankings.empty and bool(rankings["ranking_changed"].any()):
        reasons.append("at least one mean model ranking changes")
    if not conclusions.empty and bool(conclusions["conclusion_changed"].any()):
        reasons.append(
            "at least one prespecified CRPS or coverage-qualified-width directional conclusion degrades or reverses"
        )

    return {
        "delta_values": deltas,
        "summary": inference,
        "rankings": rankings,
        "coverage_status": coverage_status,
        "conclusions": conclusions,
        "configuration_frequencies": configuration_frequencies,
        "extension_trigger": bool(reasons),
        "trigger_reasons": reasons,
    }


def _locked_manifest(manifest: StudyManifest | Mapping[str, Any] | str | Path) -> StudyManifest:
    raw = _load_manifest(manifest)
    # ``build_study_manifest`` wraps the immutable study config under
    # ``config`` and adds a seed registry/provenance.  Retain support for the
    # lightweight flat mappings used by downstream notebooks and tests.
    full_manifest = raw if isinstance(raw.get("config"), Mapping) else None
    config = dict(raw["config"]) if full_manifest is not None else raw
    analysis = config.get("analysis", {})
    if not isinstance(analysis, Mapping):
        raise ValueError("Manifest analysis configuration must be a mapping.")

    configured_models = config.get("models", MAIN_MODELS)
    if isinstance(configured_models, Mapping):
        models = tuple(configured_models)
    else:
        models = tuple(configured_models)
    split = config.get("splits", {})
    if not isinstance(split, Mapping):
        raise ValueError("Manifest splits configuration must be a mapping.")
    configured_sizes = split.get(
        "nested_train_anchors",
        config.get("sizes", config.get("training_sizes", ANCHOR_SIZES)),
    )
    sizes = tuple(int(value) for value in configured_sizes)

    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("Manifest execution configuration must be a mapping.")
    n_repeats = int(execution.get("confirmatory_repeats", config.get("n_repeats", 50)))
    configured_repeat_ids = config.get("repeat_ids")
    repeat_ids = (
        tuple(configured_repeat_ids)
        if configured_repeat_ids is not None
        else tuple(range(1, n_repeats + 1))
    )
    if set(models) != set(MAIN_MODELS) or len(models) != len(MAIN_MODELS):
        raise ValueError(f"Main manifest must contain exactly the locked models {MAIN_MODELS}, got {models}.")
    if sizes != ANCHOR_SIZES:
        raise ValueError(f"Main manifest sizes must be exactly {ANCHOR_SIZES}, got {sizes}.")
    allowed_repeat_ids = {DEFAULT_MAIN_REPEAT_IDS, FULL_MAIN_REPEAT_IDS}
    if repeat_ids not in allowed_repeat_ids or n_repeats != len(repeat_ids):
        raise ValueError(
            "Main manifest repeat IDs must be exactly 1--50 for the default "
            "profile or 1--100 for the full profile."
        )
    metric_columns = analysis.get("metric_columns", config.get("metric_columns"))
    if metric_columns is None and full_manifest is None:
        candidate_metrics = config.get("metrics")
        if candidate_metrics is not None and not isinstance(candidate_metrics, Mapping):
            metric_columns = candidate_metrics
    coverage_metrics = analysis.get("coverage_metrics", config.get("coverage_metrics"))
    plot_metrics = analysis.get("plot_metrics", config.get("plot_metrics"))

    if full_manifest is not None and isinstance(full_manifest.get("seed_registry"), Mapping):
        if _design_manifest_seed is None:  # pragma: no cover - design normally imports above
            raise RuntimeError("Cannot resolve the design manifest bootstrap seed.")
        seed_parts = analysis.get("bootstrap_seed_key", ("analysis", "bootstrap"))
        if isinstance(seed_parts, (str, bytes)) or not isinstance(seed_parts, Sequence):
            raise ValueError("analysis.bootstrap_seed_key must be a sequence of semantic key parts.")
        bootstrap_seed = int(_design_manifest_seed(full_manifest, *seed_parts))
    else:
        bootstrap_seed = int(
            analysis.get(
                "bootstrap_seed",
                config.get("bootstrap_seed", config.get("base_seed", DEFAULT_BOOTSTRAP_SEED)),
            )
        )
    return StudyManifest(
        models=MAIN_MODELS,
        sizes=ANCHOR_SIZES,
        repeat_ids=repeat_ids,
        metric_columns=tuple(metric_columns) if metric_columns is not None else None,
        coverage_metrics=tuple(coverage_metrics) if coverage_metrics is not None else None,
        confidence=float(analysis.get("confidence", config.get("confidence", DEFAULT_CONFIDENCE))),
        coverage_target=float(
            analysis.get("coverage_target", config.get("coverage_target", DEFAULT_COVERAGE_TARGET))
        ),
        bootstrap_seed=bootstrap_seed,
        plot_metrics=tuple(plot_metrics) if plot_metrics is not None else None,
    )


def _sensitivity_contract(
    manifest: StudyManifest | Mapping[str, Any] | str | Path,
) -> tuple[StudyManifest, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Resolve the locked stage-1 and extension sensitivity design."""

    main = _locked_manifest(manifest)
    raw = _load_manifest(manifest)
    config = dict(raw["config"]) if isinstance(raw.get("config"), Mapping) else raw
    sensitivity = config.get("sensitivity", {})
    if not isinstance(sensitivity, Mapping):
        raise ValueError("Manifest sensitivity configuration must be a mapping.")
    stage1 = sensitivity.get("stage1", {})
    extension = sensitivity.get("extension", {})
    if not isinstance(stage1, Mapping) or not isinstance(extension, Mapping):
        raise ValueError("Sensitivity stage1 and extension configurations must be mappings.")
    repeat_ids = tuple(int(value) for value in stage1.get("repeat_ids", SENSITIVITY_REPEAT_IDS))
    sizes = tuple(int(value) for value in stage1.get("anchors", SENSITIVITY_SIZES))
    additional_repeat_ids = tuple(
        int(value) for value in extension.get("additional_repeat_ids", tuple(range(21, 51)))
    )
    if repeat_ids != SENSITIVITY_REPEAT_IDS:
        raise ValueError(
            f"Sensitivity stage 1 must use repeat IDs {SENSITIVITY_REPEAT_IDS}, got {repeat_ids}."
        )
    if sizes != SENSITIVITY_SIZES:
        raise ValueError(f"Sensitivity anchors must be exactly {SENSITIVITY_SIZES}, got {sizes}.")
    if additional_repeat_ids != tuple(range(21, 51)):
        raise ValueError("Sensitivity extension must add repeat IDs 21 through 50.")
    target_total = int(extension.get("target_total_repeats", 50))
    if target_total != 50:
        raise ValueError("Sensitivity extension target must be 50 complete repeats.")
    return main, repeat_ids, sizes, additional_repeat_ids


def run_aggregation(
    metrics: pd.DataFrame | Iterable[Mapping[str, Any]],
    manifest: StudyManifest | Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Validate, aggregate, infer, plot, and persist the locked main study.

    Parameters
    ----------
    metrics:
        A DataFrame or iterable of per-cell dictionaries.  An optional nested
        ``metrics`` dictionary is expanded automatically.
    manifest:
        StudyManifest, mapping, or JSON path.  The locked main grid cannot be
        overridden, but metric columns, repeat IDs, coverage target, plotting
        choices, confidence, and bootstrap seed may be supplied.
    out_dir:
        Destination directory for CSV/JSON/PNG artifacts.
    bootstrap_draws:
        Number of deterministic repeat-block draws; defaults to 10,000.
    """

    config = _locked_manifest(manifest)
    frame = _as_frame(metrics)
    if "branch" in frame.columns:
        frame = frame[frame["branch"].astype(str) == "main"].copy()
        if frame.empty:
            raise ValueError("No main-branch metric rows were supplied.")
    metric_columns = list(config.metric_columns) if config.metric_columns is not None else infer_metric_columns(frame)
    frame = validate_main_grid(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
    )
    bootstrap_indices = repeat_block_bootstrap_indices(
        len(config.repeat_ids),
        n_bootstrap=bootstrap_draws,
        seed=config.bootstrap_seed,
    )

    summary = summarize_metrics(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
    )
    summary_wide = summarize_metrics_wide(summary)
    model_values = paired_model_contrasts(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
    )
    model_inference = summarize_paired_contrasts(
        model_values,
        id_columns=["metric", "n_train", "model_a", "model_b", "contrast"],
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )
    adjacent_pairs = list(zip(config.sizes[:-1], config.sizes[1:]))
    size_values = paired_size_contrasts(
        frame,
        metric_columns=metric_columns,
        size_pairs=adjacent_pairs,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
    )
    size_inference = summarize_paired_contrasts(
        size_values,
        id_columns=["metric", "model", "size_from", "size_to", "contrast"],
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )
    endpoint_values = endpoint_changes(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
    )
    endpoint_inference = summarize_paired_contrasts(
        endpoint_values,
        id_columns=["metric", "model", "size_from", "size_to", "contrast"],
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )
    did_values = endpoint_difference_in_differences(endpoint_values, models=config.models)
    did_inference = summarize_paired_contrasts(
        did_values,
        id_columns=["metric", "model_a", "model_b", "contrast"],
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )

    coverage_metrics = (
        list(config.coverage_metrics)
        if config.coverage_metrics is not None
        else [metric for metric in metric_columns if _metric_scope(metric)[0] == "coverage"]
    )
    coverage = (
        coverage_qualification(
            frame,
            coverage_metrics=coverage_metrics,
            target=config.coverage_target,
            confidence=config.confidence,
            models=config.models,
            sizes=config.sizes,
            repeat_ids=config.repeat_ids,
            bootstrap_indices=bootstrap_indices,
        )
        if coverage_metrics
        else pd.DataFrame()
    )
    supported_best = supported_best_declarations(model_inference, models=config.models)
    has_width_metric = any(
        _metric_scope(metric)[0] in {"mean_width", "mean_width_inclusive"}
        for metric in metric_columns
    )
    controlled_sharpness = (
        controlled_sharpness_table(summary, coverage, model_inference)
        if has_width_metric and not coverage.empty
        else pd.DataFrame()
    )
    sd_intervals = bootstrap_cell_sd_intervals(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )
    log_sd_ratios = bootstrap_log_sd_ratios(
        frame,
        metric_columns=metric_columns,
        models=config.models,
        sizes=config.sizes,
        repeat_ids=config.repeat_ids,
        confidence=config.confidence,
        bootstrap_indices=bootstrap_indices,
    )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, pd.DataFrame] = {
        "summary": summary,
        "summary_wide": summary_wide,
        "model_contrasts": model_inference,
        "adjacent_size_contrasts": size_inference,
        "endpoint_changes": endpoint_inference,
        "endpoint_difference_in_differences": did_inference,
        "coverage_qualification": coverage,
        "supported_best_crps": supported_best,
        "controlled_sharpness": controlled_sharpness,
        "sd_intervals": sd_intervals,
        "log_sd_ratios": log_sd_ratios,
    }
    output_paths: dict[str, Path] = {}
    for name, table in artifacts.items():
        path = destination / f"{name}.csv"
        table.to_csv(path, index=False)
        output_paths[name] = path

    preferred_plot_metrics = list(config.plot_metrics) if config.plot_metrics is not None else [
        metric for metric in metric_columns if _metric_scope(metric)[0] in {"crps", "mean_width", "coverage"}
    ]
    plot_path = destination / "metric_ribbons.png"
    figure, _ = plot_metric_ribbons(
        summary,
        metrics=preferred_plot_metrics or metric_columns[:1],
        model_order=config.models,
        output_path=plot_path,
    )
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except ImportError:
        pass
    output_paths["plot"] = plot_path

    analysis_manifest = {
        **asdict(config),
        "models": list(config.models),
        "sizes": list(config.sizes),
        "repeat_ids": list(config.repeat_ids),
        "metric_columns": metric_columns,
        "coverage_metrics": coverage_metrics,
        "bootstrap_draws": int(bootstrap_indices.shape[0]),
        "bootstrap_unit": "whole repeat vector",
        "stability_interval_method": "BCa repeat-block bootstrap with leave-one-repeat jackknife acceleration",
        "contrast_orientation": "model_a - model_b and size_to - size_from",
    }
    manifest_path = destination / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(analysis_manifest, indent=2))
    output_paths["manifest"] = manifest_path

    return {
        **artifacts,
        "output_paths": output_paths,
        "manifest": analysis_manifest,
    }


def run_sensitivity_aggregation(
    fixed_metrics: pd.DataFrame | Iterable[Mapping[str, Any]],
    tuned_metrics: pd.DataFrame | Iterable[Mapping[str, Any]],
    manifest: StudyManifest | Mapping[str, Any] | str | Path,
    out_dir: str | Path,
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    extended: bool = False,
) -> dict[str, Any]:
    """Aggregate the locked tuning sensitivity and write its trigger.

    ``fixed_metrics`` may be the complete 50- or 100-repeat main table; it is paired to
    the selected sensitivity stage by repeat ID and anchor.  ``tuned_metrics`` must contain
    exactly the 20 x 4 x 3 stage-1 cells by default.  With ``extended=True``,
    both inputs are reduced to and validated against exactly repeats 1--50 at
    those same three anchors, and only that all-50 analysis is reported.  Rows
    for another explicitly labelled branch are ignored.
    """

    config, stage1_repeat_ids, sizes, additional_repeat_ids = _sensitivity_contract(manifest)
    if not isinstance(extended, (bool, np.bool_)):
        raise TypeError("extended must be boolean.")
    repeat_ids = MAIN_REPEAT_IDS if bool(extended) else stage1_repeat_ids
    fixed = _as_frame(fixed_metrics)
    tuned = _as_frame(tuned_metrics)
    if "branch" in fixed.columns:
        fixed = fixed[fixed["branch"].astype(str) == "main"].copy()
    if "branch" in tuned.columns:
        tuned = tuned[tuned["branch"].astype(str) == "sensitivity"].copy()
    fixed = fixed[
        fixed["repeat"].isin(repeat_ids) & fixed["n_train"].isin(sizes)
    ].copy()
    tuned = tuned[
        tuned["repeat"].isin(repeat_ids) & tuned["n_train"].isin(sizes)
    ].copy()
    if fixed.empty or tuned.empty:
        raise ValueError("Both fixed and tuned stage-1 metric grids are required.")

    if config.metric_columns is not None:
        metric_columns = [metric for metric in config.metric_columns if metric in tuned.columns]
    else:
        metric_columns = [metric for metric in infer_metric_columns(fixed) if metric in tuned.columns]
    if not metric_columns:
        raise ValueError("Fixed and tuned sensitivity inputs have no common configured metrics.")
    result = summarize_tuning_sensitivity(
        fixed,
        tuned,
        metric_columns=metric_columns,
        models=config.models,
        sizes=sizes,
        repeat_ids=repeat_ids,
        coverage_target=config.coverage_target,
        confidence=config.confidence,
        n_bootstrap=bootstrap_draws,
        seed=config.bootstrap_seed,
    )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_names = {
        "delta_values": "sensitivity_delta_values.csv",
        "summary": "sensitivity_summary.csv",
        "rankings": "sensitivity_rankings.csv",
        "coverage_status": "sensitivity_coverage_status.csv",
        "conclusions": "sensitivity_conclusions.csv",
        "configuration_frequencies": "sensitivity_config_frequencies.csv",
    }
    output_paths: dict[str, Path] = {}
    for key, filename in table_names.items():
        path = destination / filename
        result[key].to_csv(path, index=False)
        output_paths[key] = path

    trigger = {
        "analysis_stage": "all_50_extension" if extended else "stage1",
        "repeats_analyzed": list(repeat_ids),
        "stage1_repeat_ids": list(stage1_repeat_ids),
        "sensitivity_anchors": list(sizes),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(config.bootstrap_seed),
        "extension_trigger": bool(result["extension_trigger"]),
        "trigger_reasons": list(result["trigger_reasons"]),
        "extension_target_total_repeats": 50,
        "additional_repeat_ids": (
            list(additional_repeat_ids) if result["extension_trigger"] and not extended else []
        ),
        "extension_complete": bool(extended),
        "extension_scope": "whole sensitivity branch at the same three anchors",
    }
    trigger_path = destination / "sensitivity_extension_trigger.json"
    trigger_path.write_text(json.dumps(trigger, indent=2))
    output_paths["extension_trigger"] = trigger_path
    return {
        **result,
        "output_paths": output_paths,
        "trigger_manifest": trigger,
    }


__all__ = [
    "ANCHOR_SIZES",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "MAIN_MODELS",
    "MAIN_REPEAT_IDS",
    "MODEL_DISPLAY_NAMES",
    "SENSITIVITY_REPEAT_IDS",
    "SENSITIVITY_SIZES",
    "StudyManifest",
    "bootstrap_cell_sd_intervals",
    "bootstrap_log_sd_ratios",
    "coverage_qualification",
    "controlled_sharpness_table",
    "endpoint_changes",
    "endpoint_difference_in_differences",
    "infer_metric_columns",
    "flatten_sensitivity_candidate_scores",
    "paired_model_contrasts",
    "paired_size_contrasts",
    "plot_metric_ribbons",
    "repeat_block_bootstrap_indices",
    "run_aggregation",
    "run_sensitivity_aggregation",
    "summarize_metrics",
    "summarize_metrics_wide",
    "summarize_paired_contrasts",
    "summarize_tuning_sensitivity",
    "supported_best_declarations",
    "validate_main_grid",
]
