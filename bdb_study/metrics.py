"""Task-neutral metrics for the multi-release BDB stability study.

The primary scores are deliberately kept separate from post-hoc calibration:
raw CRPS for ordered distributions, raw Brier score for binary tasks, and
pooled-coordinate RMSE for trajectories.  Venn--Abers outputs are retained as
secondary calibration artifacts without relabeling their multiprobability
bounds as confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import warnings
from typing import Iterable

import numpy as np


EPS = 1e-12


def _semantic_index(seed: int, group: str, size: int, *, namespace: str) -> int:
    """Choose one row deterministically without depending on input row order."""

    if size <= 0:
        raise ValueError("semantic subsampling requires a non-empty group")
    payload = f"{namespace}|{int(seed)}|{group}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % int(size)


def hierarchical_subsample_indices(
    game_ids: np.ndarray | Iterable[object],
    *,
    seed: int,
    labels: np.ndarray | Iterable[int] | None = None,
    target_label: int | None = None,
) -> np.ndarray:
    """Select one eligible example per game using a frozen semantic seed.

    When ``target_label`` is supplied, only games containing that label enter
    the registry.  Selection is made against the checksum-bound canonical
    prepared-row order and is therefore stable across machines and resumes.
    """

    games = np.asarray(game_ids).astype(str).reshape(-1)
    if len(games) == 0:
        raise ValueError("hierarchical subsampling requires calibration games")
    if labels is None:
        eligible = np.ones(len(games), dtype=bool)
        label_namespace = "all"
    else:
        observed = np.asarray(labels).reshape(-1)
        if len(observed) != len(games) or target_label not in (0, 1):
            raise ValueError("label-conditional subsampling inputs are incompatible")
        eligible = observed == int(target_label)
        label_namespace = f"label-{int(target_label)}"
    selected: list[int] = []
    for game in sorted(np.unique(games[eligible])):
        candidates = np.flatnonzero(eligible & (games == game))
        # ``candidates`` is sorted by the checksum-bound prepared-row index.
        # The semantic hash chooses a position, not a process-random hash.
        offset = _semantic_index(
            seed,
            str(game),
            len(candidates),
            namespace=f"bdb-hierarchical-subsample-v1|{label_namespace}",
        )
        selected.append(int(candidates[offset]))
    if not selected:
        raise ValueError("no calibration game contains the requested examples")
    return np.asarray(selected, dtype=np.int64)


def _split_conformal_quantile(
    scores: np.ndarray | Iterable[float],
    *,
    alpha: float,
    fallback: float,
) -> tuple[float, int]:
    values = _one_dimensional(scores, "conformal scores")
    if len(values) == 0 or not 0.0 < alpha < 1.0:
        raise ValueError("split conformal requires scores and alpha in (0,1)")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    if rank > len(values):
        return float(fallback), rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def _one_dimensional(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_binary_probability(values: np.ndarray | Iterable[float], name: str = "probability") -> np.ndarray:
    array = _one_dimensional(values, name)
    if np.any((array < -1e-10) | (array > 1.0 + 1e-10)):
        raise ValueError(f"{name} is outside [0, 1]")
    return np.clip(array, 0.0, 1.0)


def validate_probability_matrix(values: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] == 0:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix")
    if n_classes is not None and probability.shape[1] != int(n_classes):
        raise ValueError(f"expected {n_classes} classes, got {probability.shape[1]}")
    if not np.all(np.isfinite(probability)) or np.any(probability < -1e-10):
        raise ValueError("probabilities must be finite and non-negative")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("probability rows must sum to one")
    return np.clip(probability, 0.0, 1.0) / row_sum[:, None]


def brier_contributions(y_true: np.ndarray, p_positive: np.ndarray) -> np.ndarray:
    y = _one_dimensional(y_true, "binary labels")
    p = validate_binary_probability(p_positive)
    if len(y) != len(p) or np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("binary labels and probabilities are incompatible")
    return np.square(p - y)


def binary_log_loss_contributions(y_true: np.ndarray, p_positive: np.ndarray) -> np.ndarray:
    y = _one_dimensional(y_true, "binary labels")
    p = np.clip(validate_binary_probability(p_positive), EPS, 1.0 - EPS)
    if len(y) != len(p) or np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("binary labels and probabilities are incompatible")
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def crps_contributions(y_index: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probability = validate_probability_matrix(probabilities)
    y = np.asarray(y_index, dtype=np.int64).reshape(-1)
    if len(y) != len(probability) or np.any((y < 0) | (y >= probability.shape[1])):
        raise ValueError("ordered class labels are incompatible with probabilities")
    cdf = np.cumsum(probability, axis=1)
    observed_cdf = np.arange(probability.shape[1])[None, :] >= y[:, None]
    return np.mean(np.square(cdf - observed_cdf), axis=1)


def central_interval_from_probability(
    probabilities: np.ndarray, alpha: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    probability = validate_probability_matrix(probabilities)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    cdf = np.cumsum(probability, axis=1)
    lower = np.argmax(cdf >= alpha / 2.0, axis=1)
    upper = np.argmax(cdf >= 1.0 - alpha / 2.0, axis=1)
    return lower.astype(int), upper.astype(int)


def _probability_locality(probabilities: np.ndarray) -> np.ndarray:
    probability = validate_probability_matrix(probabilities)
    classes = np.arange(probability.shape[1], dtype=float)
    mean = probability @ classes
    second = probability @ np.square(classes)
    standard_deviation = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    entropy = -np.sum(probability * np.log(np.clip(probability, EPS, None)), axis=1)
    return np.column_stack([mean, standard_deviation, entropy])


def conformal_distribution_intervals(
    calibration_probability: np.ndarray,
    calibration_y: np.ndarray,
    test_probability: np.ndarray,
    *,
    alpha: float = 0.10,
    local_k: int = 200,
) -> dict[str, np.ndarray]:
    """Current locally weighted central-interval conformal construction."""

    cal = validate_probability_matrix(calibration_probability)
    test = validate_probability_matrix(test_probability, cal.shape[1])
    cal_y = np.asarray(calibration_y, dtype=int).reshape(-1)
    if len(cal_y) != len(cal) or np.any((cal_y < 0) | (cal_y >= cal.shape[1])):
        raise ValueError("calibration labels and distributions are incompatible")
    cal_lower, cal_upper = central_interval_from_probability(cal, alpha)
    test_lower, test_upper = central_interval_from_probability(test, alpha)
    scores = np.maximum.reduce(
        [cal_lower - cal_y, cal_y - cal_upper, np.zeros_like(cal_y)]
    ).astype(float)
    cal_features = _probability_locality(cal)
    test_features = _probability_locality(test)
    scale = np.std(cal_features, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    cal_scaled = cal_features / scale
    test_scaled = test_features / scale
    neighbors = min(max(int(local_k), 5), len(cal_y))
    target_quantile = min(1.0, (1.0 - alpha) * (len(cal_y) + 1) / len(cal_y))
    padding = np.zeros(len(test), dtype=int)
    for index, feature in enumerate(test_scaled):
        distance = np.sum(np.square(cal_scaled - feature), axis=1)
        kth = np.partition(distance, neighbors - 1)[neighbors - 1]
        bandwidth = math.sqrt(max(float(kth), 1e-12))
        weights = np.exp(-0.5 * distance / bandwidth**2)
        order = np.argsort(scores)
        ordered_score, ordered_weight = scores[order], weights[order]
        if ordered_weight.sum() <= 0:
            ordered_weight = np.ones_like(ordered_weight)
        cumulative = np.cumsum(ordered_weight) / ordered_weight.sum()
        selected = min(
            int(np.searchsorted(cumulative, target_quantile, side="left")), len(scores) - 1
        )
        padding[index] = int(math.ceil(ordered_score[selected]))
    lower = np.maximum(0, test_lower - padding)
    upper = np.minimum(cal.shape[1] - 1, test_upper + padding)
    return {
        "lower": lower,
        "upper": upper,
        "padding": padding,
        "width": upper - lower,
    }


def hierarchical_distribution_intervals(
    calibration_probability: np.ndarray,
    calibration_y: np.ndarray,
    calibration_game_ids: np.ndarray,
    test_probability: np.ndarray,
    *,
    alpha: float = 0.10,
    seed: int = 20260820,
) -> dict[str, np.ndarray | float | int | str]:
    """Game-clustered central intervals for one random play in a new game.

    Exactly one calibration example is selected from each game.  This makes
    the game, rather than a correlated play row, the conformal exchangeability
    unit.  A rank beyond the available calibration games returns full-support
    intervals instead of silently weakening the target coverage.
    """

    cal = validate_probability_matrix(calibration_probability)
    test = validate_probability_matrix(test_probability, cal.shape[1])
    cal_y = np.asarray(calibration_y, dtype=int).reshape(-1)
    games = np.asarray(calibration_game_ids).reshape(-1)
    if len(cal_y) != len(cal) or len(games) != len(cal):
        raise ValueError("calibration probabilities, labels, and games are incompatible")
    if np.any((cal_y < 0) | (cal_y >= cal.shape[1])):
        raise ValueError("calibration labels are outside the ordered support")
    selected = hierarchical_subsample_indices(games, seed=seed)
    cal_lower, cal_upper = central_interval_from_probability(cal, alpha)
    scores = np.maximum.reduce(
        [cal_lower - cal_y, cal_y - cal_upper, np.zeros_like(cal_y)]
    ).astype(float)
    quantile, rank = _split_conformal_quantile(
        scores[selected], alpha=alpha, fallback=float(cal.shape[1] - 1)
    )
    padding_value = int(math.ceil(quantile))
    test_lower, test_upper = central_interval_from_probability(test, alpha)
    lower = np.maximum(0, test_lower - padding_value)
    upper = np.minimum(cal.shape[1] - 1, test_upper + padding_value)
    return {
        "lower": lower,
        "upper": upper,
        "padding": np.full(len(test), padding_value, dtype=np.int64),
        "width": upper - lower,
        "selected_calibration_indices": selected,
        "quantile": float(quantile),
        "rank": int(rank),
        "n_calibration_games": int(len(selected)),
        "seed": int(seed),
    }


def bernoulli_entropy(probability: np.ndarray | Iterable[float]) -> np.ndarray:
    p = np.clip(validate_binary_probability(probability), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def calibration_bias(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = _one_dimensional(y_true, "binary labels")
    p = validate_binary_probability(probability)
    if len(y) != len(p) or np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("binary labels and probabilities are incompatible")
    return float(np.mean(y - p))


def rms_reliability_error(
    calibration_probability: np.ndarray,
    test_probability: np.ndarray,
    test_y: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, np.ndarray]:
    """RMS reliability using bin boundaries frozen from calibration scores."""

    cal = validate_binary_probability(calibration_probability, "calibration probability")
    test = validate_binary_probability(test_probability, "test probability")
    y = _one_dimensional(test_y, "test labels")
    if len(test) != len(y) or np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("test probabilities and labels are incompatible")
    if int(n_bins) < 2:
        raise ValueError("reliability requires at least two bins")
    boundaries = np.quantile(cal, np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1])
    boundaries = np.unique(np.asarray(boundaries, dtype=np.float64))
    membership = np.searchsorted(boundaries, test, side="right")
    weighted = 0.0
    total = 0
    for value in np.unique(membership):
        chosen = membership == value
        count = int(chosen.sum())
        if count == 0:
            continue
        difference = float(y[chosen].mean() - test[chosen].mean())
        weighted += count * difference**2
        total += count
    if total == 0:
        raise ValueError("reliability bins contain no test examples")
    return float(math.sqrt(weighted / total)), boundaries


def label_conditional_prediction_sets(
    calibration_probability: np.ndarray,
    calibration_y: np.ndarray,
    calibration_game_ids: np.ndarray,
    test_probability: np.ndarray,
    *,
    alpha: float = 0.10,
    seed: int = 20260820,
) -> dict[str, np.ndarray | float | int]:
    """Secondary label-conditional binary sets with game-level subsampling."""

    cal_p = validate_binary_probability(calibration_probability, "calibration probability")
    test_p = validate_binary_probability(test_probability, "test probability")
    cal_y = np.asarray(calibration_y, dtype=int).reshape(-1)
    games = np.asarray(calibration_game_ids).reshape(-1)
    if len(cal_p) != len(cal_y) or len(cal_p) != len(games):
        raise ValueError("binary calibration probabilities, labels, and games are incompatible")
    if set(np.unique(cal_y)) != {0, 1}:
        raise ValueError("label-conditional sets require both calibration classes")
    included = np.zeros((len(test_p), 2), dtype=bool)
    quantiles = np.zeros(2, dtype=np.float64)
    ranks = np.zeros(2, dtype=np.int64)
    registries: list[np.ndarray] = []
    candidate_probability = np.column_stack([1.0 - test_p, test_p])
    calibration_by_label = np.column_stack([1.0 - cal_p, cal_p])
    for label in (0, 1):
        selected = hierarchical_subsample_indices(
            games,
            seed=seed,
            labels=cal_y,
            target_label=label,
        )
        score = 1.0 - calibration_by_label[selected, label]
        quantile, rank = _split_conformal_quantile(score, alpha=alpha, fallback=1.0)
        included[:, label] = (1.0 - candidate_probability[:, label]) <= quantile + EPS
        quantiles[label] = quantile
        ranks[label] = rank
        registries.append(selected)
    return {
        "included": included,
        "set_size": included.sum(axis=1).astype(np.int64),
        "quantiles": quantiles,
        "ranks": ranks,
        "selected_label0_indices": registries[0],
        "selected_label1_indices": registries[1],
        "seed": int(seed),
    }


def empirical_distribution_null(y_index: np.ndarray, n_classes: int, smoothing: float = 0.0) -> np.ndarray:
    y = np.asarray(y_index, dtype=np.int64).reshape(-1)
    if len(y) == 0 or np.any((y < 0) | (y >= n_classes)):
        raise ValueError("null distribution labels are empty or outside support")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64) + float(smoothing)
    return counts / counts.sum()


def prevalence_null(y_true: np.ndarray, smoothing: float = 0.5) -> float:
    y = _one_dimensional(y_true, "binary labels")
    if len(y) == 0 or np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("prevalence null requires non-empty binary labels")
    return float((y.sum() + smoothing) / (len(y) + 2.0 * smoothing))


def game_equal_mean(contributions: np.ndarray, game_ids: np.ndarray) -> float:
    values = _one_dimensional(contributions, "metric contributions")
    groups = np.asarray(game_ids).reshape(-1)
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("metric contributions and game IDs are incompatible")
    unique = np.unique(groups)
    return float(np.mean([values[groups == group].mean() for group in unique]))


def skill_score(model_loss: float, null_loss: float) -> float:
    if not math.isfinite(model_loss) or not math.isfinite(null_loss) or null_loss <= 0.0:
        raise ValueError("skill score losses must be finite and the null loss positive")
    return float(1.0 - model_loss / null_loss)


@dataclass(frozen=True)
class VennAbersOutput:
    raw_probability: np.ndarray
    calibrated_probability: np.ndarray
    p0: np.ndarray
    p1: np.ndarray

    @property
    def lower(self) -> np.ndarray:
        return np.minimum(self.p0, self.p1)

    @property
    def upper(self) -> np.ndarray:
        return np.maximum(self.p0, self.p1)

    @property
    def imprecision(self) -> np.ndarray:
        return self.upper - self.lower


def fit_venn_abers(
    calibration_probability: np.ndarray,
    calibration_y: np.ndarray,
    test_probability: np.ndarray,
    *,
    precision: int | None = None,
) -> VennAbersOutput:
    """Fit inductive Venn--Abers and retain its calibrated point prediction.

    ``venn_abers`` returns ``(p_prime, p0_p1)``.  The legacy sack pipeline
    discarded ``p_prime``; this implementation intentionally preserves it.
    """

    from venn_abers import VennAbers

    raw_calibration = validate_binary_probability(calibration_probability)
    raw_test = validate_binary_probability(test_probability)
    # Clipping is an internal numerical accommodation for the calibrator.  It
    # must not redefine the raw model probability used by the primary Brier.
    cal_p = np.clip(raw_calibration, 1e-6, 1.0 - 1e-6)
    test_p = np.clip(raw_test, 1e-6, 1.0 - 1e-6)
    cal_y = _one_dimensional(calibration_y, "calibration labels").astype(np.int64)
    if len(cal_p) != len(cal_y) or set(np.unique(cal_y)) != {0, 1}:
        raise ValueError("Venn--Abers requires aligned calibration data containing both classes")
    calibrator = VennAbers()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")
        calibrator.fit(
            p_cal=np.column_stack([1.0 - cal_p, cal_p]),
            y_cal=cal_y,
            precision=precision,
        )
        p_prime, p0_p1 = calibrator.predict_proba(
            p_test=np.column_stack([1.0 - test_p, test_p])
        )
    p_prime = np.asarray(p_prime, dtype=np.float64)
    bounds = np.asarray(p0_p1, dtype=np.float64)
    if p_prime.shape != (len(test_p), 2) or bounds.shape != (len(test_p), 2):
        raise RuntimeError("Venn--Abers returned unexpected shapes")
    if not np.all(np.isfinite(p_prime)) or not np.all(np.isfinite(bounds)):
        raise FloatingPointError("Venn--Abers returned non-finite values")
    return VennAbersOutput(
        raw_probability=raw_test,
        calibrated_probability=np.clip(p_prime[:, 1], 0.0, 1.0),
        p0=np.clip(bounds[:, 0], 0.0, 1.0),
        p1=np.clip(bounds[:, 1], 0.0, 1.0),
    )


def trajectory_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return one mean squared-coordinate error per trajectory."""

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[-1] != 2:
        raise ValueError("trajectory arrays must share shape [examples, horizon, 2]")
    if valid.shape != truth.shape[:2] or not np.all(np.isfinite(truth[valid])):
        raise ValueError("trajectory mask is incompatible with targets")
    if not np.all(np.isfinite(prediction[valid])) or np.any(valid.sum(axis=1) == 0):
        raise ValueError("every trajectory needs finite predictions and at least one target frame")
    # NaN is an intentional padding value in the official variable-horizon
    # target tensor. Multiplication by a false mask is not sufficient because
    # IEEE NaN * 0 remains NaN; remove invalid deltas before squaring.
    delta = np.where(valid[..., None], prediction - truth, 0.0)
    squared = np.square(delta).sum(axis=2)
    return squared.sum(axis=1) / (2.0 * valid.sum(axis=1))


def trajectory_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    per_trajectory = trajectory_squared_error(y_true, y_pred, mask)
    valid = np.asarray(mask, dtype=bool)
    weights = 2.0 * valid.sum(axis=1)
    return float(np.sqrt(np.average(per_trajectory, weights=weights)))


def trajectory_path_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """One coordinate-RMSE value per requested-player path."""

    return np.sqrt(trajectory_squared_error(y_true, y_pred, mask))


def pathwise_conformal_tube(
    calibration_y: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_mask: np.ndarray,
    calibration_game_ids: np.ndarray,
    test_y: np.ndarray,
    test_prediction: np.ndarray,
    test_mask: np.ndarray,
    horizon_scale: np.ndarray,
    *,
    alpha: float = 0.10,
    seed: int = 20260820,
    field_bounds: tuple[float, float, float, float] = (
        0.0,
        120.0,
        0.0,
        160.0 / 3.0,
    ),
) -> dict[str, np.ndarray | float | int]:
    """Calibrate a whole-path radial tube on one requested path per game.

    The predictive region at each horizon is the calibrated disk intersected
    with the legal field rectangle. Official targets are required to lie on
    that rectangle, so the intersection preserves the disk's path-coverage
    event. Reported diameter is the conservative pre-intersection disk
    diameter; boundary clipping is never allowed to create artificial
    sharpness differences between models.
    """

    cal_truth = np.asarray(calibration_y, dtype=np.float64)
    cal_prediction = np.asarray(calibration_prediction, dtype=np.float64)
    cal_valid = np.asarray(calibration_mask, dtype=bool)
    test_truth = np.asarray(test_y, dtype=np.float64)
    test_prediction = np.asarray(test_prediction, dtype=np.float64)
    test_valid = np.asarray(test_mask, dtype=bool)
    if (
        cal_truth.shape != cal_prediction.shape
        or test_truth.shape != test_prediction.shape
        or cal_truth.ndim != 3
        or test_truth.ndim != 3
        or cal_truth.shape[-1] != 2
        or test_truth.shape[-1] != 2
        or cal_truth.shape[1:] != test_truth.shape[1:]
        or cal_valid.shape != cal_truth.shape[:2]
        or test_valid.shape != test_truth.shape[:2]
    ):
        raise ValueError("pathwise conformal arrays have incompatible shapes")
    scale = np.asarray(horizon_scale, dtype=np.float64).reshape(-1)
    if (
        len(scale) != cal_truth.shape[1]
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or np.any(np.diff(scale) < -1e-12)
    ):
        raise ValueError("horizon scale must be positive, finite, and nondecreasing")
    if np.any(cal_valid.sum(axis=1) == 0) or np.any(test_valid.sum(axis=1) == 0):
        raise ValueError("every trajectory needs at least one valid future frame")
    bounds = np.asarray(field_bounds, dtype=np.float64).reshape(-1)
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError("field bounds must be finite (x_min,x_max,y_min,y_max)")
    tolerance = 1e-6
    for truth, valid, label in (
        (cal_truth, cal_valid, "calibration"),
        (test_truth, test_valid, "test"),
    ):
        on_field = (
            (truth[..., 0] >= bounds[0] - tolerance)
            & (truth[..., 0] <= bounds[1] + tolerance)
            & (truth[..., 1] >= bounds[2] - tolerance)
            & (truth[..., 1] <= bounds[3] + tolerance)
        )
        if np.any(valid & ~on_field):
            raise ValueError(f"{label} trajectory target lies outside legal field bounds")
    cal_radial = np.linalg.norm(
        np.where(cal_valid[..., None], cal_prediction - cal_truth, 0.0), axis=2
    )
    selected = hierarchical_subsample_indices(calibration_game_ids, seed=seed)
    cal_scaled = np.where(cal_valid, cal_radial / scale[None, :], -np.inf)
    scores = np.max(cal_scaled, axis=1)
    quantile, rank = _split_conformal_quantile(
        scores[selected], alpha=alpha, fallback=1.0e6
    )
    radius = quantile * scale
    test_radial = np.linalg.norm(
        np.where(test_valid[..., None], test_prediction - test_truth, 0.0), axis=2
    )
    within = (~test_valid) | (test_radial <= radius[None, :] + EPS)
    path_covered = np.all(within, axis=1)
    per_path_mean_diameter = (
        (test_valid * (2.0 * radius[None, :])).sum(axis=1) / test_valid.sum(axis=1)
    )
    return {
        "path_covered": path_covered,
        "per_path_mean_diameter": per_path_mean_diameter,
        "radius": radius,
        "horizon_scale": scale,
        "field_bounds": bounds,
        "region_contract": "disk_intersect_legal_field_rectangle",
        "diameter_contract": "conservative_pre_intersection_disk_diameter",
        "quantile": float(quantile),
        "rank": int(rank),
        "selected_calibration_indices": selected,
        "n_calibration_games": int(len(selected)),
        "seed": int(seed),
    }
