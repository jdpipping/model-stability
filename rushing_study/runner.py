"""Cell execution for the immutable four-model rushing stability study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from .data import StudyData
from .metrics import EvaluationArtifacts, crps_contributions, evaluate_cell
from .models import (
    MODEL_DISPLAY_NAMES,
    build_neural_with_weights,
    canonical_model_id,
    fit_lightgbm,
    fit_neural_select_and_refit,
    fit_ovr_logistic,
    frozen_model_config,
    predict_classical,
    predict_neural,
    release_neural,
    sensitivity_candidates,
)


@dataclass(frozen=True)
class CellPayload:
    metrics: dict[str, Any]
    calibration_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    arrays: dict[str, np.ndarray]
    history: dict[str, Any]
    candidate_scores: list[dict[str, Any]]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("config", manifest)


def _split_for_repeat(manifest: dict[str, Any], repeat_id: int) -> dict[str, Any]:
    records = manifest.get("split_manifests", manifest.get("repeat_splits", []))
    for record in records:
        if int(record.get("repeat_id", record.get("repeat", -1))) == int(repeat_id):
            return record
    raise KeyError(f"Manifest has no split for repeat {repeat_id}.")


def _manifest_seed(
    manifest: dict[str, Any],
    branch: str,
    repeat_id: int,
    n_train: int,
    model_id: str,
    stage: str,
) -> int:
    from .design import manifest_seed

    if branch not in {"main", "sensitivity"}:
        raise ValueError(f"Unknown study branch: {branch}")
    return int(
        manifest_seed(
            manifest,
            "cell",
            int(repeat_id),
            stage,
            canonical_model_id(model_id),
            int(n_train),
        )
    )


def _anchor_record(split: dict[str, Any], n_train: int) -> dict[str, Any]:
    anchors = split.get("anchors", split.get("training_subsets", {}))
    record = anchors.get(str(n_train), anchors.get(n_train))
    if record is None:
        raise KeyError(f"Repeat split has no {n_train}-game anchor.")
    if isinstance(record, list):
        return {"train_games": record}
    return record


def _games(record: dict[str, Any], *names: str) -> list[Any]:
    for name in names:
        if name in record:
            value = record[name]
            if isinstance(value, dict):
                return [game for season_games in value.values() for game in season_games]
            return list(value)
    raise KeyError(f"None of the game-list fields exists: {names}")


def _features(data: StudyData, model_id: str) -> np.ndarray:
    model_id = canonical_model_id(model_id)
    if model_id in {"ridge_sgd_l2", "lightgbm_multiclass"}:
        return data.tabular
    if model_id == "zoo_cnn":
        return data.spatial
    if model_id == "set_transformer":
        return data.player_set
    raise AssertionError(model_id)


def _neural_inner_indices(
    data: StudyData,
    train_indices: np.ndarray,
    anchor: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    fit_games = {str(v) for v in _games(anchor, "neural_fit_games", "inner_train_games")}
    val_games = {str(v) for v in _games(anchor, "neural_validation_games", "inner_validation_games")}
    if fit_games & val_games:
        raise ValueError("Neural inner fit and validation games overlap.")
    selected_games = data.metadata.iloc[train_indices]["game_id"].astype(str).to_numpy()
    fit_local = np.flatnonzero(np.isin(selected_games, list(fit_games)))
    val_local = np.flatnonzero(np.isin(selected_games, list(val_games)))
    if len(fit_local) + len(val_local) != len(train_indices):
        raise ValueError("Neural inner split does not partition the complete n-game training subset.")
    return fit_local, val_local


def _predict_and_evaluate(
    data: StudyData,
    model_id: str,
    model: Any,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
    local_k: int,
) -> EvaluationArtifacts:
    features = _features(data, model_id)
    canonical = canonical_model_id(model_id)
    if canonical in {"ridge_sgd_l2", "lightgbm_multiclass"}:
        cal_proba = predict_classical(model, features[calibration_indices])
        test_proba = predict_classical(model, features[test_indices])
    else:
        cal_proba = predict_neural(model, features[calibration_indices])
        test_proba = predict_neural(model, features[test_indices])
    return evaluate_cell(
        cal_proba,
        test_proba,
        data.y[calibration_indices],
        data.y[test_indices],
        data.metadata.iloc[calibration_indices],
        data.metadata.iloc[test_indices],
        alpha=alpha,
        local_k=local_k,
    )


def _fit_primary(
    data: StudyData,
    manifest: dict[str, Any],
    split: dict[str, Any],
    repeat_id: int,
    n_train: int,
    model_id: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], int]:
    config = _manifest_config(manifest)
    canonical = canonical_model_id(model_id)
    model_config = frozen_model_config(config, canonical)
    anchor = _anchor_record(split, n_train)
    train_indices = data.indices_for_games(_games(anchor, "train_games", "games"))
    if data.metadata.iloc[train_indices]["game_id"].nunique() != n_train:
        raise ValueError(f"Anchor {n_train} did not select exactly {n_train} games.")
    features = _features(data, canonical)
    history: dict[str, Any] = {}
    if canonical == "ridge_sgd_l2":
        seed = _manifest_seed(manifest, "main", repeat_id, n_train, canonical, "fit")
        model = fit_ovr_logistic(features[train_indices], data.y[train_indices], model_config, seed)
        parameter_count = int(model.coef_.size + model.intercept_.size)
        history = {"fit_seed": seed}
    elif canonical == "lightgbm_multiclass":
        seed = _manifest_seed(manifest, "main", repeat_id, n_train, canonical, "fit")
        model = fit_lightgbm(features[train_indices], data.y[train_indices], model_config, seed)
        parameter_count = int(model.booster_.num_trees())
        history = {"fit_seed": seed}
    else:
        selection_seed = _manifest_seed(manifest, "main", repeat_id, n_train, canonical, "epoch_selection")
        refit_seed = _manifest_seed(manifest, "main", repeat_id, n_train, canonical, "refit")
        fit_local, val_local = _neural_inner_indices(data, train_indices, anchor)
        fitted = fit_neural_select_and_refit(
            canonical,
            features[train_indices],
            data.y[train_indices],
            fit_local,
            val_local,
            model_config,
            selection_seed,
            refit_seed,
        )
        model = fitted.model
        parameter_count = fitted.parameter_count
        expected = model_config.get("expected_parameters")
        if expected is not None and int(expected) != parameter_count:
            raise RuntimeError(f"{canonical} expected {expected} parameters; got {parameter_count}.")
        history = {
            "best_epoch": fitted.best_epoch,
            "selection": fitted.selection_history,
            "refit": fitted.refit_history,
            "selection_seed": selection_seed,
            "refit_seed": refit_seed,
        }
    return model, model_config, history, parameter_count


def _fit_sensitivity(
    data: StudyData,
    manifest: dict[str, Any],
    split: dict[str, Any],
    repeat_id: int,
    n_train: int,
    model_id: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], int, list[dict[str, Any]]]:
    config = _manifest_config(manifest)
    canonical = canonical_model_id(model_id)
    anchor = _anchor_record(split, n_train)
    train_indices = data.indices_for_games(_games(anchor, "train_games", "games"))
    tune_indices = data.indices_for_games(_games(split, "tune_games"))
    if data.metadata.iloc[train_indices]["game_id"].nunique() != n_train:
        raise ValueError("Sensitivity training subset has the wrong number of games.")
    if set(data.metadata.iloc[train_indices]["game_id"]) & set(data.metadata.iloc[tune_indices]["game_id"]):
        raise ValueError("Sensitivity training and tuning games overlap.")
    features = _features(data, canonical)
    candidates = sensitivity_candidates(config, canonical)
    candidate_scores: list[dict[str, Any]] = []
    best_score = float("inf")
    best_index = 0
    best_model: Any | None = None
    best_weights: list[np.ndarray] | None = None
    best_history: dict[str, Any] = {}
    best_parameter_count = 0
    fit_seed = _manifest_seed(manifest, "sensitivity", repeat_id, n_train, canonical, "fit")
    selection_seed = _manifest_seed(manifest, "sensitivity", repeat_id, n_train, canonical, "epoch_selection")
    refit_seed = _manifest_seed(manifest, "sensitivity", repeat_id, n_train, canonical, "refit")
    rebuild_seed = _manifest_seed(manifest, "sensitivity", repeat_id, n_train, canonical, "prediction_rebuild")
    for index, candidate in enumerate(candidates):
        started = time.perf_counter()
        if canonical == "ridge_sgd_l2":
            fitted_model = fit_ovr_logistic(features[train_indices], data.y[train_indices], candidate, fit_seed)
            tune_proba = predict_classical(fitted_model, features[tune_indices])
            parameter_count = int(fitted_model.coef_.size + fitted_model.intercept_.size)
            candidate_history: dict[str, Any] = {}
        elif canonical == "lightgbm_multiclass":
            fitted_model = fit_lightgbm(features[train_indices], data.y[train_indices], candidate, fit_seed)
            tune_proba = predict_classical(fitted_model, features[tune_indices])
            parameter_count = int(fitted_model.booster_.num_trees())
            candidate_history = {}
        else:
            fit_local, val_local = _neural_inner_indices(data, train_indices, anchor)
            neural = fit_neural_select_and_refit(
                canonical,
                features[train_indices],
                data.y[train_indices],
                fit_local,
                val_local,
                candidate,
                selection_seed,
                refit_seed,
            )
            fitted_model = neural.model
            tune_proba = predict_neural(fitted_model, features[tune_indices])
            parameter_count = neural.parameter_count
            candidate_history = {
                "best_epoch": neural.best_epoch,
                "selection": neural.selection_history,
                "refit": neural.refit_history,
            }
        score = float(np.mean(crps_contributions(data.y[tune_indices], tune_proba)))
        candidate_scores.append(
            {
                "candidate_index": index,
                "is_frozen_main": index == 0,
                "config": candidate,
                "config_hash": _canonical_hash(candidate),
                "tune_crps": score,
                "elapsed_seconds": time.perf_counter() - started,
                "fit_seed": fit_seed if canonical in {"ridge_sgd_l2", "lightgbm_multiclass"} else selection_seed,
                "refit_seed": None if canonical in {"ridge_sgd_l2", "lightgbm_multiclass"} else refit_seed,
            }
        )
        # Candidates are in manifest order with the frozen primary setting
        # first, so strict comparison implements the prespecified exact-tie
        # rule without turning near-ties into ties.
        if score < best_score:
            best_score = score
            best_index = index
            best_parameter_count = parameter_count
            best_history = candidate_history
            if canonical in {"ridge_sgd_l2", "lightgbm_multiclass"}:
                best_model = fitted_model
            else:
                best_weights = [np.array(value, copy=True) for value in fitted_model.get_weights()]
        if canonical in {"zoo_cnn", "set_transformer"}:
            release_neural(fitted_model)
        elif best_model is not fitted_model:
            del fitted_model
        gc.collect()

    chosen_config = candidates[best_index]
    if canonical in {"zoo_cnn", "set_transformer"}:
        if best_weights is None:
            raise RuntimeError("No neural sensitivity candidate was retained.")
        best_model = build_neural_with_weights(canonical, chosen_config, best_weights, rebuild_seed)
    if best_model is None:
        raise RuntimeError("No sensitivity candidate was selected.")
    best_history.update(
        {
            "selected_candidate_index": best_index,
            "selected_tune_crps": best_score,
            "candidate_count": len(candidates),
            "fit_seed": fit_seed if canonical in {"ridge_sgd_l2", "lightgbm_multiclass"} else None,
            "selection_seed": selection_seed if canonical in {"zoo_cnn", "set_transformer"} else None,
            "refit_seed": refit_seed if canonical in {"zoo_cnn", "set_transformer"} else None,
            "prediction_rebuild_seed": rebuild_seed
            if canonical in {"zoo_cnn", "set_transformer"}
            else None,
        }
    )
    return best_model, chosen_config, best_history, best_parameter_count, candidate_scores


def run_cell(
    data: StudyData,
    manifest: dict[str, Any],
    branch: str,
    repeat_id: int,
    n_train: int,
    model_id: str,
) -> CellPayload:
    """Fit, calibrate, evaluate, and package one immutable study cell."""
    if branch not in {"main", "sensitivity"}:
        raise ValueError("branch must be 'main' or 'sensitivity'.")
    canonical = canonical_model_id(model_id)
    config = _manifest_config(manifest)
    split = _split_for_repeat(manifest, repeat_id)
    anchor = _anchor_record(split, n_train)
    train_indices = data.indices_for_games(_games(anchor, "train_games", "games"))
    calibration_indices = data.indices_for_games(_games(split, "calibration_games"))
    test_indices = data.indices_for_games(_games(split, "test_games"))
    partition_sets = [
        set(data.metadata.iloc[idx]["game_id"].astype(str))
        for idx in (train_indices, calibration_indices, test_indices)
    ]
    if any(partition_sets[i] & partition_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train, calibration, and test games are not disjoint.")

    started = time.perf_counter()
    if branch == "main":
        model, model_config, history, parameter_count = _fit_primary(
            data, manifest, split, repeat_id, n_train, canonical
        )
        candidate_scores: list[dict[str, Any]] = []
    else:
        model, model_config, history, parameter_count, candidate_scores = _fit_sensitivity(
            data, manifest, split, repeat_id, n_train, canonical
        )
    evaluation = _predict_and_evaluate(
        data,
        canonical,
        model,
        calibration_indices,
        test_indices,
        alpha=float(config["uncertainty"]["alpha"]),
        local_k=int(config["uncertainty"]["local_k"]),
    )
    if canonical in {"zoo_cnn", "set_transformer"}:
        release_neural(model)
    else:
        del model
        gc.collect()

    manifest_hash = manifest.get("manifest_hash", manifest.get("run_hash"))
    metrics = {
        **evaluation.metrics,
        "branch": branch,
        "repeat": int(repeat_id),
        "n_train": int(n_train),
        "n_train_plays": int(len(train_indices)),
        "model": canonical,
        "model_display_name": MODEL_DISPLAY_NAMES[canonical],
        "model_config_hash": _canonical_hash(model_config),
        "manifest_hash": manifest_hash,
        "parameter_count": int(parameter_count),
        "elapsed_seconds": float(time.perf_counter() - started),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if branch == "sensitivity":
        metrics["selected_config"] = json.dumps(
            model_config, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    history = {
        **history,
        "model_config": model_config,
        "model_config_hash": metrics["model_config_hash"],
    }
    return CellPayload(
        metrics=metrics,
        calibration_predictions=evaluation.calibration,
        test_predictions=evaluation.test,
        arrays=evaluation.arrays,
        history=history,
        candidate_scores=candidate_scores,
    )
