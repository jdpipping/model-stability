"""Prediction-level artifacts and metrics for the rushing study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .intervals import (
    NUM_CLASSES,
    _locality_features_from_proba,
    central_interval_from_proba,
    local_conformal_padding,
    normalize_proba_rows,
)


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics: dict[str, float | int]
    calibration: pd.DataFrame
    test: pd.DataFrame
    arrays: dict[str, np.ndarray]


def crps_contributions(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    p = normalize_proba_rows(proba)
    y = np.asarray(y, dtype=int)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return np.sum((np.cumsum(p, axis=1) - np.cumsum(one_hot, axis=1)) ** 2, axis=1) / 199.0


def _distribution_columns(proba: np.ndarray) -> dict[str, np.ndarray]:
    p = normalize_proba_rows(proba)
    classes = np.arange(NUM_CLASSES, dtype=np.float64)
    mean_class = p @ classes
    second = p @ (classes**2)
    std_class = np.sqrt(np.maximum(second - mean_class**2, 0.0))
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    return {
        "predictive_mean_class": mean_class,
        "predictive_mean_yards": mean_class - 28.0,
        "predictive_std_class": std_class,
        "predictive_entropy": entropy,
    }


def evaluate_cell(
    cal_proba: np.ndarray,
    test_proba: np.ndarray,
    cal_y: np.ndarray,
    test_y: np.ndarray,
    cal_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    alpha: float,
    local_k: int,
) -> EvaluationArtifacts:
    """Evaluate a fitted distribution and retain independently auditable rows."""
    cal_p = normalize_proba_rows(cal_proba)
    test_p = normalize_proba_rows(test_proba)
    if len(cal_p) != len(cal_y) or len(test_p) != len(test_y):
        raise ValueError("Prediction and outcome row counts differ.")

    cal_l, cal_u = central_interval_from_proba(cal_p, alpha)
    test_l, test_u = central_interval_from_proba(test_p, alpha)
    cal_features = _locality_features_from_proba(cal_p)
    test_features = _locality_features_from_proba(test_p)
    q = local_conformal_padding(
        cal_y,
        cal_l,
        cal_u,
        cal_features,
        test_features,
        alpha,
        local_k=local_k,
    )
    lo = np.maximum(0, test_l - q)
    hi = np.minimum(NUM_CLASSES - 1, test_u + q)
    width = hi - lo
    covered = (test_y >= lo) & (test_y <= hi)
    crps_row = crps_contributions(test_y, test_p)

    calibration = cal_metadata[["game_id", "play_id", "season"]].reset_index(drop=True).copy()
    calibration["y_class"] = np.asarray(cal_y, dtype=int)
    calibration["central_lo"] = cal_l
    calibration["central_hi"] = cal_u
    for name, values in _distribution_columns(cal_p).items():
        calibration[name] = values

    test = test_metadata[["game_id", "play_id", "season"]].reset_index(drop=True).copy()
    test["y_class"] = np.asarray(test_y, dtype=int)
    test["central_lo"] = test_l
    test["central_hi"] = test_u
    test["conformal_q"] = q
    test["conformal_lo"] = lo
    test["conformal_hi"] = hi
    test["width"] = width
    test["width_inclusive"] = width + 1
    test["covered"] = covered.astype(np.int8)
    test["crps_contribution"] = crps_row
    for name, values in _distribution_columns(test_p).items():
        test[name] = values

    by_game = test.groupby("game_id", sort=False).agg(
        crps=("crps_contribution", "mean"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        mean_width_inclusive=("width_inclusive", "mean"),
    )
    metrics: dict[str, float | int] = {
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test)),
        "n_test_games": int(test["game_id"].nunique()),
        "crps": float(np.mean(crps_row)),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "mean_width_inclusive": float(np.mean(width + 1)),
        "std_width": float(np.std(width, ddof=1)) if len(width) > 1 else 0.0,
        "mean_q": float(np.mean(q)),
        "std_q": float(np.std(q, ddof=1)) if len(q) > 1 else 0.0,
        "crps_game_equal": float(by_game["crps"].mean()),
        "coverage_game_equal": float(by_game["coverage"].mean()),
        "mean_width_game_equal": float(by_game["mean_width"].mean()),
        "mean_width_inclusive_game_equal": float(by_game["mean_width_inclusive"].mean()),
    }
    return EvaluationArtifacts(
        metrics=metrics,
        calibration=calibration,
        test=test,
        arrays={
            # Preserve the evaluated precision so interval and aggregate
            # metrics can be independently rebuilt without boundary flips.
            "calibration_proba": cal_p.astype(np.float64, copy=False),
            "test_proba": test_p.astype(np.float64, copy=False),
        },
    )
