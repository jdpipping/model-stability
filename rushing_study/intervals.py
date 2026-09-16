"""Shared probability normalization and conformal intervals for rushing studies."""

from __future__ import annotations

import math

import numpy as np


MIN_IDX_Y = 71
MAX_IDX_Y = 150
NUM_CLASSES = MAX_IDX_Y - MIN_IDX_Y + 1


def central_interval_from_proba(proba: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(alpha/2, 1-alpha/2) quantiles of predictive CDF per sample → lower/upper bin indices."""
    proba = normalize_proba_rows(proba)
    cdf = np.cumsum(proba, axis=1)
    lower = np.argmax(cdf >= (alpha / 2.0), axis=1)
    upper = np.argmax(cdf >= (1.0 - alpha / 2.0), axis=1)
    return lower.astype(int), upper.astype(int)


def conformal_padding(cal_y: np.ndarray, cal_l: np.ndarray, cal_u: np.ndarray, alpha: float) -> int:
    """Nonconformity scores on cal set; return empirical (1-alpha) quantile as padding q."""
    scores = np.maximum.reduce([cal_l - cal_y, cal_y - cal_u, np.zeros_like(cal_y)])
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    q = int(np.partition(scores, k - 1)[k - 1])
    return q


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile with nonnegative weights and q in [0, 1]."""
    if len(values) == 0:
        return 0.0
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if np.sum(w) <= 0:
        w = np.ones_like(w)

    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    idx = int(np.searchsorted(cdf, np.clip(q, 0.0, 1.0), side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


def _locality_features_from_proba(proba: np.ndarray) -> np.ndarray:
    """Construct low-dim locality features from predictive distributions."""
    p = normalize_proba_rows(proba)
    cls = np.arange(NUM_CLASSES, dtype=np.float64)
    mean = p @ cls
    second = p @ (cls**2)
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return np.stack([mean, std, entropy], axis=1)


def local_conformal_padding(
    cal_y: np.ndarray,
    cal_l: np.ndarray,
    cal_u: np.ndarray,
    cal_feat: np.ndarray,
    test_feat: np.ndarray,
    alpha: float,
    local_k: int,
) -> np.ndarray:
    """Tibshirani-style locally weighted conformal padding q(x) for each test point."""
    scores = np.maximum.reduce([cal_l - cal_y, cal_y - cal_u, np.zeros_like(cal_y)]).astype(np.float64)

    # Standardize locality features so one coordinate cannot dominate distances.
    feat_scale = np.std(cal_feat, axis=0, ddof=1)
    feat_scale = np.where(feat_scale > 1e-8, feat_scale, 1.0)
    cal_scaled = cal_feat / feat_scale
    test_scaled = test_feat / feat_scale

    n_cal = len(cal_scaled)
    k = min(max(int(local_k), 5), n_cal)
    qhat = np.zeros(len(test_scaled), dtype=np.float64)
    target_q = min(1.0, (1.0 - alpha) * (n_cal + 1) / n_cal)

    for j in range(len(test_scaled)):
        d2 = np.sum((cal_scaled - test_scaled[j]) ** 2, axis=1)
        # Adaptive bandwidth from kth-nearest calibration point.
        kth = np.partition(d2, k - 1)[k - 1]
        bw = float(np.sqrt(max(kth, 1e-12)))
        w = np.exp(-0.5 * d2 / (bw**2))
        qhat[j] = _weighted_quantile(scores, w, target_q)

    return np.ceil(qhat).astype(int)


def normalize_proba_rows(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-normalized probabilities for stable metrics and interval construction."""
    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, eps, None)
    row_sums = p.sum(axis=1, keepdims=True)
    # Guard against pathological all-zero/NaN rows from upstream models.
    bad = ~np.isfinite(row_sums) | (row_sums <= 0)
    if np.any(bad):
        p[bad[:, 0]] = 1.0 / p.shape[1]
        row_sums = p.sum(axis=1, keepdims=True)
    return p / row_sums
