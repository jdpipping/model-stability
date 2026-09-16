"""Pinned BDB 2024 winner-fidelity utilities.

This module is a small, dependency-light transcription of the behavior in
``mpchang/uncovering-missed-tackle-opportunities`` at commit
``8b3de97f1e42351d14e5b69d8cb03f51f244806a``.  It intentionally records
the difference between the model that was actually executed in the notebook
and the hyperparameters printed in the submission appendix.

The resulting score is trained on a case-control sample (charted solo tackles
versus charted misses).  It must not be described as an unconditional tackle
probability without an additional calibration argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


WINNER_COMMIT = "8b3de97f1e42351d14e5b69d8cb03f51f244806a"
WINNER_REPOSITORY = "https://github.com/mpchang/uncovering-missed-tackle-opportunities"

# This order is copied from build_tackle_sequence_input and the executed
# feature_names cell in train_model.ipynb at WINNER_COMMIT.
FEATURE_NAMES: tuple[str, ...] = (
    "ballcarrier_speed",
    "relative_x_speed",
    "absolute_relative_y_speed",
    "euclidean_distance",
    "angle_of_attack_cosine",
    "ballcarrier_voronoi_area",
    "team_influence_at_ballcarrier",
    "blocker_influence_at_tackler",
    "is_run",
)

PUBLISHED_DISPLAY_NAMES: tuple[str, ...] = (
    "Ballcarrier Speed",
    "Relative X Speed",
    "|Relative Y Speed|",
    "Euclidian Distance",  # Preserve the notebook spelling in the receipt.
    "Angle of Attack",
    "Voronoi Area",
    "Team Influence",
    "Blocker Influence",
    "Is_run",
)

EXPECTED_FIDELITY_COUNTS: Mapping[str, Mapping[str, int]] = {
    "weeks_1_8": {"made": 8000, "missed": 1583, "total": 9583},
    "week_9": {"made": 836, "missed": 180, "total": 1016},
}

EXECUTED_XGBOOST_RECEIPT: Mapping[str, Any] = {
    "implementation": "xgboost.XGBClassifier",
    "objective": "binary:logistic",
    "max_depth": 7,
    "eta": 0.1,
    "n_estimators": 150,
    "reg_lambda": 150,
    "subsample": 0.75,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": "n_negative / n_positive on the fitted split",
    "eval_metric": "sklearn.metrics.log_loss",
    "early_stopping_rounds": 20,
    "validation_fraction": 0.1,
    "validation_split_seed": 1234,
}

APPENDIX_XGBOOST_RECEIPT: Mapping[str, Any] = {
    "n_estimators": 250,
    "max_depth": 7,
    "eta": 0.1,
    "subsample": 0.75,
    "scale_pos_weight": 0.20,
    "reg_lambda": 150,
}

REFERENCE_SCHEMA_VERSION = "bdb2024-xgboost-fidelity-v2"
FRAMEWISE_SCHEMA_VERSION = "bdb2024-framewise-fidelity-v1"
FRAMEWISE_SCORE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "play_id",
    "frame_id",
    "nfl_id",
    "week",
    *FEATURE_NAMES,
    "case_control_score",
)
OPPORTUNITY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "play_id",
    "nfl_id",
    "week",
    "first_frame_id",
    "last_frame_id",
    "scored_frames",
    "opportunities",
    "missed_opportunities",
    "opportunity_frame_ids_json",
    "missed_frame_ids_json",
)


def executed_xgboost_parameters(labels: Iterable[int]) -> dict[str, Any]:
    """Materialize the data-dependent parameters used by the notebook fit.

    XGBoost is intentionally imported only by :func:`fit_executed_xgboost`,
    so cohort preparation, receipts, and tests do not require that optional
    reference dependency.
    """

    raw_target = np.asarray(list(labels)).reshape(-1)
    if len(raw_target) == 0 or not np.all(np.isin(raw_target, [0, 1])):
        raise ValueError("fidelity labels must be a nonempty binary sequence")
    target = raw_target.astype(np.int8)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("fidelity XGBoost requires both made and missed tackles")
    return {
        "objective": "binary:logistic",
        "max_depth": 7,
        "eta": 0.1,
        "n_estimators": 150,
        "reg_lambda": 150,
        "subsample": 0.75,
        "colsample_bytree": 1,
        "colsample_bylevel": 1,
        "scale_pos_weight": negatives / positives,
        "early_stopping_rounds": 20,
        "verbosity": 2,
    }


@dataclass(frozen=True)
class ExecutedXGBoostFit:
    """Optional fitted winner reference plus its exact notebook split."""

    model: Any
    train_indices: np.ndarray
    validation_indices: np.ndarray
    parameters: Mapping[str, Any]


def _fidelity_feature_array(features: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(features, pd.DataFrame) and tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("fidelity DataFrame columns do not match the exact feature order")
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"fidelity features must have shape [N,{len(FEATURE_NAMES)}]"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("fidelity features must be finite")
    return values


def fit_executed_xgboost(
    features: np.ndarray | pd.DataFrame,
    labels: Iterable[int],
    *,
    verbose: bool = True,
) -> ExecutedXGBoostFit:
    """Fit the executed 150-tree notebook reference on its 90/10 split.

    This is the fidelity branch, not a member of the common five-role grid.
    It deliberately reproduces the notebook's unstratified split with seed
    1234 and its validation/train evaluation-set order. A clear runtime error
    is raised when the optional ``xgboost`` package is unavailable.
    """

    try:
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "the BDB2024 fidelity fit requires the optional xgboost dependency"
        ) from exc

    values = _fidelity_feature_array(features)
    raw_target = np.asarray(list(labels)).reshape(-1)
    if not np.all(np.isin(raw_target, [0, 1])):
        raise ValueError("fidelity labels must be binary")
    target = raw_target.astype(np.int8)
    if len(values) != len(target):
        raise ValueError("fidelity features and labels are not row-aligned")
    indices = np.arange(len(target), dtype=np.int64)
    train_index, validation_index = train_test_split(
        indices, test_size=0.1, random_state=1234
    )
    parameters = executed_xgboost_parameters(target[train_index])
    model = XGBClassifier(**parameters, eval_metric=log_loss)
    model.fit(
        values[train_index],
        target[train_index],
        eval_set=[
            (values[validation_index], target[validation_index]),
            (values[train_index], target[train_index]),
        ],
        verbose=verbose,
    )
    return ExecutedXGBoostFit(
        model=model,
        train_indices=np.asarray(train_index, dtype=np.int64),
        validation_indices=np.asarray(validation_index, dtype=np.int64),
        parameters=dict(parameters),
    )


def predict_fidelity_score(
    fitted: ExecutedXGBoostFit | Any,
    features: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Return the winner's made-versus-missed score with shape checks."""

    values = _fidelity_feature_array(features)
    model = fitted.model if isinstance(fitted, ExecutedXGBoostFit) else fitted
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    if probabilities.shape != (len(values), 2) or not np.all(np.isfinite(probabilities)):
        raise ValueError("fidelity model returned invalid binary probabilities")
    score = probabilities[:, 1]
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("fidelity score lies outside [0, 1]")
    return score


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    """Convert common NumPy/XGBoost receipt values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(
    path: Path,
    root: Path,
    *,
    rows: int | None = None,
    columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    if columns is not None:
        record["columns"] = [str(column) for column in columns]
    return record


def _commit_immutable_file(temporary: Path, target: Path) -> Path:
    """Commit a completed file without replacing a different artifact."""

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, target)
    except FileExistsError:
        if (
            not target.is_file()
            or target.stat().st_size != temporary.stat().st_size
            or _sha256_file(target) != _sha256_file(temporary)
        ):
            raise FileExistsError(
                f"refusing to replace a different immutable fidelity artifact: {target}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    temporary = _temporary_path(path)
    try:
        payload = _canonical_json_bytes(_json_value(value))
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return _commit_immutable_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_array_immutable(path: Path, value: np.ndarray) -> Path:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        return _commit_immutable_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv_immutable(path: Path, frame: pd.DataFrame) -> Path:
    temporary = _temporary_path(path)
    try:
        frame.to_csv(
            temporary,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        return _commit_immutable_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_model_immutable(path: Path, model: Any) -> Path:
    if not callable(getattr(model, "save_model", None)):
        raise TypeError("fidelity model must expose save_model(path)")
    temporary = _temporary_path(path)
    try:
        # XGBoost determines the stable UBJSON format from the suffix.
        model.save_model(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError("fidelity model serializer produced no artifact")
        return _commit_immutable_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_training_receipt(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "configured_tree_budget": 150,
    }
    try:
        result["xgboost_version"] = importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        result["xgboost_version"] = "unavailable_or_test_double"
    for field in ("best_iteration", "best_score"):
        try:
            result[field] = _json_value(getattr(model, field))
        except (AttributeError, TypeError, ValueError):
            result[field] = None
    try:
        result["boosted_rounds"] = int(model.get_booster().num_boosted_rounds())
    except (AttributeError, TypeError, ValueError):
        result["boosted_rounds"] = None
    try:
        result["evaluation_history"] = _json_value(model.evals_result())
    except (AttributeError, TypeError, ValueError):
        result["evaluation_history"] = None
    return result


def persist_reference_artifacts(
    fitted: ExecutedXGBoostFit,
    week9_examples: pd.DataFrame,
    week9_labels: Iterable[int],
    week9_scores: Iterable[float],
    output_dir: str | Path,
    *,
    prepared_hash: str,
) -> dict[str, Any]:
    """Persist the fitted notebook reference and row-level week-nine scores.

    The output is deliberately independent of the common five-role study.
    Every artifact is immutable and the returned receipt binds its byte hash,
    feature order, exact 90/10 split, and case-control interpretation.
    """

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    train = np.asarray(fitted.train_indices, dtype=np.int64)
    validation = np.asarray(fitted.validation_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1:
        raise ValueError("fidelity split indices must be one-dimensional")
    if len(np.intersect1d(train, validation)):
        raise ValueError("fidelity training and validation indices overlap")
    expected = np.arange(len(train) + len(validation), dtype=np.int64)
    if not np.array_equal(np.sort(np.concatenate([train, validation])), expected):
        raise ValueError("fidelity split indices do not partition development rows")

    labels = np.asarray(list(week9_labels)).reshape(-1)
    scores = np.asarray(list(week9_scores), dtype=np.float64).reshape(-1)
    if len(week9_examples) != len(labels) or len(labels) != len(scores):
        raise ValueError("week-nine examples, labels, and scores are not row-aligned")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("week-nine fidelity labels must be binary")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("week-nine fidelity scores must be finite values in [0, 1]")
    required = {"game_id", "play_id", "nfl_id", "week"}
    missing = required.difference(week9_examples.columns)
    if missing:
        raise ValueError(f"week-nine examples are missing identity columns: {sorted(missing)}")
    if not week9_examples["week"].astype(int).eq(9).all():
        raise ValueError("candidate-score artifact may contain only week-nine examples")

    score_columns = [
        column
        for column in ("example_id", "game_id", "play_id", "nfl_id", "week")
        if column in week9_examples
    ]
    score_frame = week9_examples.loc[:, score_columns].copy().reset_index(drop=True)
    score_frame["target"] = labels.astype(np.int8)
    score_frame["case_control_score"] = scores
    score_frame = score_frame.sort_values(
        ["game_id", "play_id", "nfl_id"], kind="stable"
    ).reset_index(drop=True)

    model_path = _save_model_immutable(root / "executed_xgboost_150.ubj", fitted.model)
    train_path = _write_array_immutable(root / "development_train_indices.npy", train)
    validation_path = _write_array_immutable(
        root / "development_validation_indices.npy", validation
    )
    scores_path = _write_csv_immutable(root / "week9_candidate_scores.csv", score_frame)
    artifacts = {
        "model": _artifact_record(model_path, root),
        "development_train_indices": _artifact_record(train_path, root),
        "development_validation_indices": _artifact_record(validation_path, root),
        "week9_candidate_scores": _artifact_record(
            scores_path,
            root,
            rows=len(score_frame),
            columns=score_frame.columns,
        ),
    }
    receipt: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "branch": "bdb2024_executed_notebook_xgboost_reference",
        "prepared_hash": str(prepared_hash),
        "winner_commit": WINNER_COMMIT,
        "feature_names": list(FEATURE_NAMES),
        "parameters": _json_value(dict(fitted.parameters)),
        "training": _model_training_receipt(fitted.model),
        "development_examples": int(len(expected)),
        "development_train_examples": int(len(train)),
        "development_validation_examples": int(len(validation)),
        "week9_examples": int(len(score_frame)),
        "score_interpretation": "case_control_made_versus_missed_score_not_unconditional_probability",
        "artifacts": artifacts,
    }
    receipt["receipt_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt).rstrip(b"\n")
    ).hexdigest()
    _write_json_immutable(root / "reference_receipt.json", receipt)
    return receipt


def fidelity_receipt() -> dict[str, Any]:
    """Return the immutable provenance and known notebook/report discrepancy."""

    return {
        "repository": WINNER_REPOSITORY,
        "commit": WINNER_COMMIT,
        "feature_names": list(FEATURE_NAMES),
        "published_display_names": list(PUBLISHED_DISPLAY_NAMES),
        "training_offset_frames": 10,
        "sequence_length": 1,
        "frame_rate_hz": 10,
        "expected_counts": {
            key: dict(value) for key, value in EXPECTED_FIDELITY_COUNTS.items()
        },
        "executed_notebook_model": dict(EXECUTED_XGBOOST_RECEIPT),
        "executable_reference": {
            "fit_entrypoint": "bdb_study.fidelity.bdb2024.fit_executed_xgboost",
            "score_entrypoint": "bdb_study.fidelity.bdb2024.predict_fidelity_score",
            "persist_entrypoint": "bdb_study.fidelity.bdb2024.persist_reference_artifacts",
            "framewise_entrypoint": "bdb_study.fidelity.bdb2024.run_framewise_fidelity_from_raw",
            "optional_dependencies": ["xgboost", "pyarrow"],
            "reference_schema_version": REFERENCE_SCHEMA_VERSION,
            "framewise_schema_version": FRAMEWISE_SCHEMA_VERSION,
        },
        "submission_appendix_model": dict(APPENDIX_XGBOOST_RECEIPT),
        "known_discrepancy": {
            "field": "n_estimators",
            "executed_notebook": 150,
            "submission_appendix": 250,
            "fidelity_choice": 150,
        },
        "score_interpretation": "case-control made-versus-missed tackle score",
    }


def validate_fidelity_counts(
    made_train: int,
    missed_train: int,
    made_test: int,
    missed_test: int,
) -> None:
    """Fail loudly unless preprocessing reproduces the published cohort sizes."""

    observed = {
        "weeks_1_8": {"made": int(made_train), "missed": int(missed_train)},
        "week_9": {"made": int(made_test), "missed": int(missed_test)},
    }
    mismatches: list[str] = []
    for split, counts in observed.items():
        for label, value in counts.items():
            expected = EXPECTED_FIDELITY_COUNTS[split][label]
            if value != expected:
                mismatches.append(f"{split}.{label}: expected {expected}, found {value}")
    if mismatches:
        raise ValueError("BDB2024 fidelity counts do not match: " + "; ".join(mismatches))


def rotate_direction_and_orientation(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the winner's unit-circle convention and left-to-right reflection."""

    required = {"x", "y", "s", "a", "dir", "o", "playDirection"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"tracking frame is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["x_clean"] = np.where(
        result["playDirection"].eq("left"), 120.0 - result["x"], result["x"]
    )
    # The pinned winner reflected only x when standardizing play direction.
    result["y_clean"] = result["y"].astype(float)
    result["s_clean"] = result["s"].astype(float)
    result["a_clean"] = result["a"].astype(float)
    result["dir_clean"] = (-(result["dir"] - 90.0)) % 360.0
    result["o_clean"] = (-(result["o"] - 90.0)) % 360.0
    left = result["playDirection"].eq("left")
    result.loc[left, "dir_clean"] = (180.0 - result.loc[left, "dir_clean"]) % 360.0
    result.loc[left, "o_clean"] = (180.0 - result.loc[left, "o_clean"]) % 360.0
    return result


def quantize_winner_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply and restore the winner's memory-optimization quantization.

    The pinned load notebook rounds position, speed, and acceleration to
    hundredths and angles to tenths before event annotation and feature
    construction.  Returning restored floats keeps the rest of this module
    simple while preserving those executed values.
    """

    result = frame.copy()
    for column in ("x_clean", "y_clean", "s_clean", "a_clean"):
        if column not in result:
            raise ValueError(f"cannot quantize missing geometry column {column}")
        result[column] = np.round(result[column].astype(float) * 100.0) / 100.0
    for column in ("dir_clean", "o_clean"):
        if column not in result:
            raise ValueError(f"cannot quantize missing angle column {column}")
        result[column] = np.round(result[column].astype(float) * 10.0) / 10.0
    return result


def strict_penalty_play_mask(plays: pd.DataFrame) -> pd.Series:
    """Return the exact strict penalty filter used by the pinned winner."""

    required = {"foulName1", "playNullifiedByPenalty"}
    missing = required.difference(plays.columns)
    if missing:
        raise ValueError(f"plays are missing strict penalty columns: {sorted(missing)}")
    # ``read_csv`` interprets the literal NFL sentinel ``"NA"`` as missing,
    # which is the representation consumed by the pinned notebook.  The
    # canonical Parquet lake intentionally preserves that literal string, so
    # normalize both representations before applying the same scientific
    # rule.  Empty strings are treated equivalently for hand-built fixtures.
    foul = plays["foulName1"]
    no_foul = foul.isna() | foul.astype(str).str.strip().isin({"", "NA"})
    nullified = plays["playNullifiedByPenalty"].astype(str).str.strip().str.upper()
    return no_foul & nullified.ne("Y")


def _unique_event_frame(play_tracking: pd.DataFrame) -> int | None:
    frames = np.sort(
        play_tracking.loc[
            play_tracking["event"].isin(["tackle", "out_of_bounds"]), "frameId"
        ].dropna().unique()
    )
    if len(frames) == 0:
        return None
    if len(frames) != 1:
        raise ValueError(f"made tackle play has {len(frames)} tackle/OOB event frames")
    return int(frames[0])


def infer_candidate_event_frame(
    play_tracking: pd.DataFrame,
    *,
    tackler_id: int,
    ballcarrier_id: int,
    made: bool,
) -> int | None:
    """Infer a made or missed tackle frame using the winner's rules."""

    if made:
        return _unique_event_frame(play_tracking)

    tackler = play_tracking.loc[
        play_tracking["nflId"].eq(tackler_id), ["frameId", "x_clean", "y_clean"]
    ].drop_duplicates("frameId")
    carrier = play_tracking.loc[
        play_tracking["nflId"].eq(ballcarrier_id), ["frameId", "x_clean", "y_clean"]
    ].drop_duplicates("frameId")
    aligned = carrier.merge(tackler, on="frameId", suffixes=("_carrier", "_tackler"))
    if aligned.empty:
        return None
    squared_distance = np.square(aligned["x_clean_carrier"] - aligned["x_clean_tackler"]) + np.square(
        aligned["y_clean_carrier"] - aligned["y_clean_tackler"]
    )
    # pandas idxmin/iloc returns the first frame in an exact tie, matching the
    # original ordered tracking data after deterministic frame sorting.
    return int(aligned.iloc[int(np.argmin(squared_distance.to_numpy()))]["frameId"])


def _cell_area(vertices: np.ndarray) -> float:
    shifted = np.roll(vertices, -1, axis=0)
    return float(abs(np.sum(vertices[:, 0] * shifted[:, 1] - shifted[:, 0] * vertices[:, 1])) / 2.0)


def _voronoi_areas(points: np.ndarray) -> dict[tuple[float, float], float]:
    from scipy.spatial import Voronoi

    bounds = np.asarray([(0.0, 0.0), (120.0, 0.0), (120.0, 53.0), (120.0, 0.0)])
    x_min, x_max = float(bounds[:, 0].min()), float(bounds[:, 0].max())
    y_min, y_max = float(bounds[:, 1].min()), float(bounds[:, 1].max())
    reflected: list[tuple[float, float]] = []
    for x_value, y_value in points:
        reflected.extend(
            [
                (x_value, y_min - (y_value - y_min)),
                (x_value, y_max + (y_max - y_value)),
                (x_min - (x_value - x_min), y_value),
                (x_max + (x_max - x_value), y_value),
            ]
        )
    total = np.concatenate([points, np.asarray(reflected, dtype=float)])
    voronoi = Voronoi(total)
    areas: dict[tuple[float, float], float] = {}
    for index, point in enumerate(total):
        if not (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
            continue
        region = voronoi.regions[voronoi.point_region[index]]
        if not region or -1 in region:
            raise ValueError("unbounded Voronoi cell after winner-style reflection")
        areas[(float(point[0]), float(point[1]))] = _cell_area(voronoi.vertices[region])
    return areas


def ballcarrier_voronoi_area(
    positions: Mapping[int, tuple[float, float]],
    ballcarrier_id: int,
    *,
    restriction: float = 5.0,
) -> float:
    """Winner-style bounded Voronoi area with the virtual trailing player."""

    if ballcarrier_id not in positions:
        raise ValueError("ballcarrier is absent from the tracking frame")
    clipped = {
        player: (min(max(float(x), 0.0), 120.0), min(max(float(y), 0.0), 53.0))
        for player, (x, y) in positions.items()
    }
    carrier = clipped[ballcarrier_id]
    points = np.asarray(list(clipped.values()), dtype=float)
    virtual = np.asarray([[carrier[0] - 2.0 * restriction, carrier[1]]], dtype=float)
    areas = _voronoi_areas(np.concatenate([points, virtual]))
    return float(areas[carrier])


def _covariance(direction: float, speed: float) -> np.ndarray:
    theta = np.radians(direction)
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )
    influence_radius = 3.0
    max_speed = 18.0
    sx = influence_radius + influence_radius * speed / max_speed
    sy = influence_radius - influence_radius * speed / max_speed
    scaling = np.asarray([[sx / 2.0, 0.0], [0.0, sy / 2.0]], dtype=float)
    return rotation @ scaling @ scaling @ rotation.T


def _single_player_influence(
    position: tuple[float, float], direction: float, speed: float, point: tuple[float, float]
) -> float:
    covariance = _covariance(float(direction), float(speed))
    determinant = float(np.linalg.det(covariance))
    if determinant == 0.0:
        raise ValueError("winner influence covariance is singular")
    mean = np.asarray(
        [
            position[0] + 0.5 * speed * np.cos(np.radians(direction)),
            position[1] + 0.5 * speed * np.sin(np.radians(direction)),
        ]
    )
    delta = np.asarray(point, dtype=float) - mean
    exponent = -0.5 * float(delta @ np.linalg.inv(covariance) @ delta.T)
    return float(np.exp(exponent) / np.sqrt((2.0 * np.pi) ** 2 * determinant))


def _sum_influence(rows: pd.DataFrame, point: tuple[float, float]) -> float:
    return float(
        sum(
            _single_player_influence(
                (float(row.x_clean), float(row.y_clean)),
                float(row.dir_clean),
                float(row.s_clean),
                point,
            )
            for row in rows.itertuples(index=False)
        )
    )


def _compile_influences(
    rows: pd.DataFrame,
) -> tuple[tuple[np.ndarray, np.ndarray, float], ...]:
    """Precompute the frame-constant part of each Gaussian influence."""

    compiled: list[tuple[np.ndarray, np.ndarray, float]] = []
    for row in rows.itertuples(index=False):
        covariance = _covariance(float(row.dir_clean), float(row.s_clean))
        determinant = float(np.linalg.det(covariance))
        if determinant == 0.0:
            raise ValueError("winner influence covariance is singular")
        direction = float(row.dir_clean)
        speed = float(row.s_clean)
        mean = np.asarray(
            [
                float(row.x_clean) + 0.5 * speed * np.cos(np.radians(direction)),
                float(row.y_clean) + 0.5 * speed * np.sin(np.radians(direction)),
            ]
        )
        compiled.append(
            (
                mean,
                np.linalg.inv(covariance),
                float(1.0 / np.sqrt((2.0 * np.pi) ** 2 * determinant)),
            )
        )
    return tuple(compiled)


def _sum_compiled_influence(
    compiled: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    point: tuple[float, float],
) -> float:
    target = np.asarray(point, dtype=float)
    return float(
        sum(
            normalization
            * np.exp(-0.5 * float((target - mean) @ inverse @ (target - mean).T))
            for mean, inverse, normalization in compiled
        )
    )


def winner_feature_vector(
    state_frame: pd.DataFrame,
    *,
    tackler_id: int,
    ballcarrier_id: int,
    is_run: bool,
    offset_frames: int = 10,
) -> np.ndarray:
    """Construct the exact nine published inputs in their executed order."""

    required = {
        "nflId",
        "club",
        "possessionTeam",
        "defensiveTeam",
        "x_clean",
        "y_clean",
        "s_clean",
        "dir_clean",
    }
    missing = required.difference(state_frame.columns)
    if missing:
        raise ValueError(f"state frame is missing winner feature columns: {sorted(missing)}")
    frame = state_frame.sort_values("nflId", kind="stable")
    carrier_rows = frame.loc[frame["nflId"].eq(ballcarrier_id)]
    tackler_rows = frame.loc[frame["nflId"].eq(tackler_id)]
    if len(carrier_rows) != 1 or len(tackler_rows) != 1:
        raise ValueError("candidate state needs exactly one ballcarrier and one tackler row")
    carrier = carrier_rows.iloc[0]
    tackler = tackler_rows.iloc[0]

    carrier_direction = float(carrier["dir_clean"])
    tackler_direction = float(tackler["dir_clean"])
    carrier_speed = float(carrier["s_clean"])
    tackler_speed = float(tackler["s_clean"])
    carrier_vx = carrier_speed * np.cos(np.radians(carrier_direction))
    carrier_vy = carrier_speed * np.sin(np.radians(carrier_direction))
    tackler_vx = tackler_speed * np.cos(np.radians(tackler_direction))
    tackler_vy = tackler_speed * np.sin(np.radians(tackler_direction))
    carrier_position = np.asarray([carrier["x_clean"], carrier["y_clean"]], dtype=float)
    tackler_position = np.asarray([tackler["x_clean"], tackler["y_clean"]], dtype=float)
    separation = carrier_position - tackler_position
    distance = float(np.linalg.norm(separation))
    projected_carrier = carrier_position + np.asarray([carrier_vx, carrier_vy]) * (
        float(offset_frames) / 10.0
    )
    separation_angle = float(
        np.degrees(
            np.arctan2(
                projected_carrier[1] - tackler_position[1],
                projected_carrier[0] - tackler_position[0],
            )
        )
    )
    attack_cosine = float(np.cos(np.radians(separation_angle - tackler_direction)))

    # The pinned ``get_positions_from_dataframe`` clips points to its
    # 120-by-53 influence/Voronoi field. Distance and attack angle above use
    # the unmodified tracking coordinates, exactly as the notebook did.
    positions = {
        int(row.nflId): (
            min(max(float(row.x_clean), 0.0), 120.0),
            min(max(float(row.y_clean), 0.0), 53.0),
        )
        for row in frame.itertuples(index=False)
    }
    voronoi_area = ballcarrier_voronoi_area(positions, int(ballcarrier_id), restriction=5.0)
    carrier_point = positions[int(ballcarrier_id)]
    tackler_point = positions[int(tackler_id)]
    offense = frame.loc[
        frame["club"].eq(frame["possessionTeam"]) & frame["nflId"].ne(ballcarrier_id)
    ]
    defense = frame.loc[frame["club"].eq(frame["defensiveTeam"])]
    team_influence = _sum_influence(offense, carrier_point) - _sum_influence(
        defense, carrier_point
    )
    blocker_influence = _sum_influence(offense, tackler_point)

    return np.asarray(
        [
            carrier_speed,
            tackler_vx - carrier_vx,
            abs(tackler_vy - carrier_vy),
            distance,
            attack_cosine,
            voronoi_area,
            team_influence,
            blocker_influence,
            float(bool(is_run)),
        ],
        dtype=np.float32,
    )


def winner_feature_matrix_for_defenders(
    state_frame: pd.DataFrame,
    *,
    is_run: bool,
    offset_frames: int = 10,
) -> pd.DataFrame:
    """Build exact winner inputs for every defender in one retained frame.

    Frame-shared Voronoi and team-influence quantities are evaluated once;
    the result is numerically equivalent to calling :func:`winner_feature_vector`
    separately for every defender but is practical for whole-release scoring.
    """

    required = {
        "nflId",
        "club",
        "possessionTeam",
        "defensiveTeam",
        "ballCarrierId",
        "x_clean",
        "y_clean",
        "s_clean",
        "dir_clean",
    }
    missing = required.difference(state_frame.columns)
    if missing:
        raise ValueError(f"state frame is missing winner feature columns: {sorted(missing)}")
    frame = state_frame.sort_values("nflId", kind="stable")
    if frame["ballCarrierId"].nunique(dropna=True) != 1:
        raise ValueError("state frame must have exactly one ballcarrier ID")
    ballcarrier_id = int(frame["ballCarrierId"].dropna().iloc[0])
    carrier_rows = frame.loc[frame["nflId"].eq(ballcarrier_id)]
    if len(carrier_rows) != 1:
        raise ValueError("state frame needs exactly one tracked ballcarrier")
    carrier = carrier_rows.iloc[0]
    defense = frame.loc[frame["club"].eq(frame["defensiveTeam"])]
    if defense.empty:
        raise ValueError("state frame contains no tracked defenders")

    positions = {
        int(row.nflId): (
            min(max(float(row.x_clean), 0.0), 120.0),
            min(max(float(row.y_clean), 0.0), 53.0),
        )
        for row in frame.itertuples(index=False)
    }
    if len(positions) != len(frame):
        raise ValueError("state frame contains duplicate player rows")
    carrier_position = np.asarray(
        [float(carrier["x_clean"]), float(carrier["y_clean"])], dtype=float
    )
    carrier_direction = float(carrier["dir_clean"])
    carrier_speed = float(carrier["s_clean"])
    carrier_vx = carrier_speed * np.cos(np.radians(carrier_direction))
    carrier_vy = carrier_speed * np.sin(np.radians(carrier_direction))
    projected_carrier = carrier_position + np.asarray([carrier_vx, carrier_vy]) * (
        float(offset_frames) / 10.0
    )
    voronoi_area = ballcarrier_voronoi_area(
        positions, ballcarrier_id, restriction=5.0
    )
    carrier_point = positions[ballcarrier_id]
    offense = frame.loc[
        frame["club"].eq(frame["possessionTeam"])
        & frame["nflId"].ne(ballcarrier_id)
    ]
    offense_influences = _compile_influences(offense)
    defense_influences = _compile_influences(defense)
    team_influence = _sum_compiled_influence(
        offense_influences, carrier_point
    ) - _sum_compiled_influence(defense_influences, carrier_point)

    rows: list[list[float]] = []
    defender_ids: list[int] = []
    for tackler in defense.itertuples(index=False):
        tackler_id = int(tackler.nflId)
        tackler_position = np.asarray(
            [float(tackler.x_clean), float(tackler.y_clean)], dtype=float
        )
        tackler_direction = float(tackler.dir_clean)
        tackler_speed = float(tackler.s_clean)
        tackler_vx = tackler_speed * np.cos(np.radians(tackler_direction))
        tackler_vy = tackler_speed * np.sin(np.radians(tackler_direction))
        separation_angle = float(
            np.degrees(
                np.arctan2(
                    projected_carrier[1] - tackler_position[1],
                    projected_carrier[0] - tackler_position[0],
                )
            )
        )
        rows.append(
            [
                carrier_speed,
                tackler_vx - carrier_vx,
                abs(tackler_vy - carrier_vy),
                float(np.linalg.norm(carrier_position - tackler_position)),
                float(np.cos(np.radians(separation_angle - tackler_direction))),
                voronoi_area,
                team_influence,
                _sum_compiled_influence(offense_influences, positions[tackler_id]),
                float(bool(is_run)),
            ]
        )
        defender_ids.append(tackler_id)
    values = np.asarray(rows, dtype=np.float32)
    if values.shape != (len(defender_ids), len(FEATURE_NAMES)) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("winner framewise features are non-finite or malformed")
    result = pd.DataFrame(values, columns=FEATURE_NAMES)
    result.insert(0, "nfl_id", np.asarray(defender_ids, dtype=np.int64))
    return result


@dataclass(frozen=True)
class OpportunitySummary:
    opportunities: int
    missed_opportunities: int
    opportunity_indices: tuple[int, ...]
    missed_indices: tuple[int, ...]


def opportunity_state_machine(
    probabilities: Iterable[float],
    *,
    threshold: float = 0.75,
    required_frames: int = 5,
    frame_ids: Iterable[int] | None = None,
) -> OpportunitySummary:
    """Apply the exact strict-high/five-up then five-down notebook automaton.

    A value equal to ``threshold`` follows the low branch, because the pinned
    notebook tests ``probability > 0.75``.  A renewed high value during the
    down counter returns to the same opportunity steady state rather than
    creating another opportunity.
    """

    values = np.asarray(list(probabilities), dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite one-dimensional sequence")
    if required_frames < 1:
        raise ValueError("required_frames must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    frames = None if frame_ids is None else np.asarray(list(frame_ids), dtype=np.int64)
    if frames is not None:
        if frames.ndim != 1 or len(frames) != len(values):
            raise ValueError("frame_ids must be one-dimensional and score-aligned")
        if len(frames) and np.any(np.diff(frames) <= 0):
            raise ValueError("frame_ids must be strictly increasing")

    state = "none"
    high_count = 0
    low_count = 0
    opportunities: list[int] = []
    misses: list[int] = []
    for index, probability in enumerate(values):
        if frames is not None and index and frames[index] != frames[index - 1] + 1:
            # Missing tracking cannot count toward consecutive 10-Hz frames.
            state = "none"
            high_count = 0
            low_count = 0
        if probability > threshold:
            if state == "none":
                state = "building"
                high_count = 1
            elif state == "building":
                high_count += 1
            elif state == "declining":
                state = "steady"
                low_count = 0
            if state == "building" and high_count == required_frames:
                state = "steady"
                opportunities.append(index)
        else:
            if state in {"none", "building"}:
                state = "none"
                high_count = 0
            elif state == "steady":
                state = "declining"
                low_count = 1
            elif state == "declining":
                low_count += 1
            if state == "declining" and low_count == required_frames:
                state = "none"
                high_count = 0
                low_count = 0
                misses.append(index)
    return OpportunitySummary(
        opportunities=len(opportunities),
        missed_opportunities=len(misses),
        opportunity_indices=tuple(opportunities),
        missed_indices=tuple(misses),
    )


class _StreamingTableWriter:
    """Atomic row-group writer for large framewise fidelity tables."""

    def __init__(self, path: Path, artifact_format: str):
        if artifact_format not in {"csv", "parquet"}:
            raise ValueError("fidelity artifact format must be csv or parquet")
        self.path = path
        self.artifact_format = artifact_format
        self.temporary = _temporary_path(path)
        self._rows = 0
        self._columns: tuple[str, ...] | None = None
        self._csv_handle: Any | None = None
        self._parquet_writer: Any | None = None
        self._parquet_schema: Any | None = None
        if artifact_format == "csv":
            self._csv_handle = self.temporary.open("w", encoding="utf-8", newline="")

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns or ()

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        columns = tuple(str(column) for column in frame.columns)
        if self._columns is None:
            self._columns = columns
        elif columns != self._columns:
            raise ValueError("streamed fidelity table changed column order")
        if self.artifact_format == "csv":
            frame.to_csv(
                self._csv_handle,
                index=False,
                header=self._rows == 0,
                float_format="%.17g",
                lineterminator="\n",
            )
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError(
                    "Parquet fidelity artifacts require the pinned pyarrow dependency"
                ) from exc
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_schema = table.schema
                self._parquet_writer = pq.ParquetWriter(
                    self.temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=True,
                )
            elif table.schema != self._parquet_schema:
                table = table.cast(self._parquet_schema)
            self._parquet_writer.write_table(table)
        self._rows += int(len(frame))

    def finish(self, empty_columns: Iterable[str]) -> Path:
        if self._rows == 0:
            empty = pd.DataFrame(columns=list(empty_columns))
            self.write(empty)
            # ``write`` deliberately ignores empty frames; create an explicit
            # schema-only artifact here instead.
            if self.artifact_format == "csv":
                empty.to_csv(self._csv_handle, index=False, lineterminator="\n")
                self._columns = tuple(empty.columns)
            else:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "Parquet fidelity artifacts require the pinned pyarrow dependency"
                    ) from exc
                table = pa.Table.from_pandas(empty, preserve_index=False)
                self._parquet_writer = pq.ParquetWriter(
                    self.temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=True,
                )
                self._parquet_writer.write_table(table)
                self._columns = tuple(empty.columns)
        if self._csv_handle is not None:
            self._csv_handle.flush()
            os.fsync(self._csv_handle.fileno())
            self._csv_handle.close()
            self._csv_handle = None
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        return _commit_immutable_file(self.temporary, self.path)

    def abort(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        self.temporary.unlink(missing_ok=True)


def _frame_score_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, FRAMEWISE_SCORE_COLUMNS].copy()
    for column in ("game_id", "play_id", "frame_id", "nfl_id"):
        result[column] = result[column].astype(np.int64)
    result["week"] = result["week"].astype(np.int16)
    for column in FEATURE_NAMES:
        result[column] = result[column].astype(np.float32)
    result["case_control_score"] = result["case_control_score"].astype(np.float64)
    return result


def _opportunity_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=OPPORTUNITY_SUMMARY_COLUMNS)
    if result.empty:
        return result
    for column in (
        "game_id",
        "play_id",
        "nfl_id",
        "first_frame_id",
        "last_frame_id",
        "scored_frames",
        "opportunities",
        "missed_opportunities",
    ):
        result[column] = result[column].astype(np.int64)
    result["week"] = result["week"].astype(np.int16)
    return result


def _normalize_raw_tracking(tracking: pd.DataFrame) -> pd.DataFrame:
    result = tracking.copy()
    for column in ("nflId", "jerseyNumber", "o", "dir"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _prepare_week_tracking(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
    *,
    week: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_tracking = {
        "gameId",
        "playId",
        "frameId",
        "nflId",
        "club",
        "x",
        "y",
        "s",
        "a",
        "dir",
        "o",
        "event",
        "playDirection",
    }
    missing = required_tracking.difference(tracking.columns)
    if missing:
        raise ValueError(f"raw tracking is missing fidelity columns: {sorted(missing)}")
    required_play = {
        "gameId",
        "playId",
        "ballCarrierId",
        "possessionTeam",
        "defensiveTeam",
        "passResult",
        "foulName1",
        "playNullifiedByPenalty",
    }
    missing = required_play.difference(plays.columns)
    if missing:
        raise ValueError(f"plays are missing fidelity columns: {sorted(missing)}")
    if "nflId" not in players:
        raise ValueError("players are missing nflId")

    raw = _normalize_raw_tracking(tracking)
    present_keys = raw[["gameId", "playId"]].drop_duplicates()
    play_rows = plays.drop_duplicates(["gameId", "playId"])
    if plays.duplicated(["gameId", "playId"], keep=False).any():
        raise ValueError("plays contain duplicate game/play rows")
    present_plays = play_rows.merge(present_keys, on=["gameId", "playId"], how="inner")
    eligible_plays = present_plays.loc[strict_penalty_play_mask(present_plays)].copy()
    play_columns = [
        "gameId",
        "playId",
        "ballCarrierId",
        "possessionTeam",
        "defensiveTeam",
        "passResult",
    ]
    if "playDirection" in eligible_plays:
        play_columns.append("playDirection")
    result = raw.merge(
        eligible_plays[play_columns],
        on=["gameId", "playId"],
        how="inner",
        suffixes=("", "_play"),
    )
    if "playDirection_play" in result:
        result["playDirection"] = result["playDirection"].fillna(
            result["playDirection_play"]
        )
        result = result.drop(columns="playDirection_play")
    result = result.loc[result["nflId"].notna()].merge(
        players[["nflId"]].drop_duplicates("nflId"), on="nflId", how="inner"
    )
    result["week"] = int(week)
    result = quantize_winner_geometry(rotate_direction_and_orientation(result))
    return result, {
        "input_tracking_rows": int(len(tracking)),
        "input_plays": int(len(present_plays)),
        "strict_penalty_excluded_plays": int(len(present_plays) - len(eligible_plays)),
        "strict_eligible_plays": int(len(eligible_plays)),
        "player_tracking_rows": int(len(result)),
    }


def _retained_play_frames(play: pd.DataFrame) -> pd.DataFrame:
    snaps = np.sort(play.loc[play["event"].eq("ball_snap"), "frameId"].dropna().unique())
    tackles = np.sort(play.loc[play["event"].eq("tackle"), "frameId"].dropna().unique())
    out_of_bounds = np.sort(
        play.loc[play["event"].eq("out_of_bounds"), "frameId"].dropna().unique()
    )
    if len(snaps) > 1 or len(tackles) > 1 or len(out_of_bounds) > 1:
        key = (int(play["gameId"].iloc[0]), int(play["playId"].iloc[0]))
        raise ValueError(f"multiple active-window marker frames for play {key}")
    minimum = int(snaps[0]) + 5 if len(snaps) else int(play["frameId"].min())
    tackle_frame = int(tackles[0]) if len(tackles) else int(play["frameId"].max())
    oob_frame = (
        int(out_of_bounds[0]) if len(out_of_bounds) else int(play["frameId"].max())
    )
    return play.loc[play["frameId"].between(minimum, min(tackle_frame, oob_frame))]


def _validate_week_receipt(
    receipt_path: Path,
    *,
    output_root: Path,
    week: int,
    source_sha256: str,
    model_sha256: str,
    artifact_format: str,
) -> dict[str, Any] | None:
    if not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"framewise week receipt is unreadable: {receipt_path}") from exc
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    claimed = receipt.get("receipt_hash")
    actual = hashlib.sha256(_canonical_json_bytes(unsigned).rstrip(b"\n")).hexdigest()
    if claimed != actual:
        raise ValueError(f"framewise week receipt hash is invalid: {receipt_path}")
    if (
        receipt.get("schema_version") != FRAMEWISE_SCHEMA_VERSION
        or receipt.get("winner_commit") != WINNER_COMMIT
        or receipt.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError(f"framewise week receipt has incompatible semantics: {receipt_path}")
    expected = {
        "week": int(week),
        "source_sha256": str(source_sha256),
        "model_sha256": str(model_sha256),
        "artifact_format": str(artifact_format),
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise FileExistsError(
            f"existing framewise week receipt has a different source or model: {receipt_path}"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "frame_scores",
        "opportunity_summaries",
    }:
        raise ValueError(
            f"framewise week receipt has an invalid artifact registry: {receipt_path}"
        )
    for artifact in artifacts.values():
        relative = Path(str(artifact.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"framewise receipt contains an unsafe artifact path: {relative}"
            )
        path = output_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact.get("size_bytes", -1))
            or _sha256_file(path) != artifact.get("sha256")
        ):
            raise ValueError(f"framewise artifact failed checksum validation: {path}")
    return receipt


def process_framewise_tracking_week(
    model: Any,
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
    output_dir: str | Path,
    *,
    week: int,
    source_sha256: str,
    model_sha256: str,
    artifact_format: str = "parquet",
    chunk_rows: int = 50_000,
) -> dict[str, Any]:
    """Score every defender/frame in one raw weekly shard without full-release load."""

    if week not in range(1, 10):
        raise ValueError("BDB2024 framewise week must lie in 1..9")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "weeks" / f"week_{week:02d}.json"
    existing = _validate_week_receipt(
        receipt_path,
        output_root=root,
        week=week,
        source_sha256=source_sha256,
        model_sha256=model_sha256,
        artifact_format=artifact_format,
    )
    if existing is not None:
        return existing

    suffix = "parquet" if artifact_format == "parquet" else "csv"
    scores_path = root / "frame_scores" / f"week_{week:02d}.{suffix}"
    events_path = root / "opportunity_summaries" / f"week_{week:02d}.{suffix}"
    score_writer = _StreamingTableWriter(scores_path, artifact_format)
    event_writer = _StreamingTableWriter(events_path, artifact_format)
    score_buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    event_buffer: list[dict[str, Any]] = []
    try:
        cleaned, audit = _prepare_week_tracking(tracking, plays, players, week=week)
        retained_plays = 0
        retained_frames = 0
        bad_ballcarrier_plays = 0
        for (game_id, play_id), raw_play in cleaned.groupby(
            ["gameId", "playId"], sort=True
        ):
            play = _retained_play_frames(raw_play).sort_values(
                ["frameId", "nflId"], kind="stable"
            )
            if play.empty:
                continue
            carriers = play["ballCarrierId"].dropna().astype(np.int64).unique()
            if len(carriers) != 1 or int(carriers[0]) not in set(
                play["nflId"].dropna().astype(np.int64)
            ):
                bad_ballcarrier_plays += 1
                continue
            retained_plays += 1
            play_parts: list[pd.DataFrame] = []
            pass_result = play["passResult"].iloc[0]
            # The pinned notebook used read_csv, which parsed the NFL literal
            # ``NA`` as null. The canonical Parquet lake preserves that token.
            is_run = bool(
                pd.isna(pass_result)
                or str(pass_result).strip().upper() in {"", "NA", "R"}
            )
            for frame_id, state in play.groupby("frameId", sort=True):
                try:
                    features = winner_feature_matrix_for_defenders(
                        state,
                        is_run=is_run,
                        offset_frames=10,
                    )
                except (KeyError, IndexError, TypeError, ValueError, FloatingPointError) as exc:
                    raise ValueError(
                        f"invalid winner feature state at {int(game_id)}:{int(play_id)}:{int(frame_id)}"
                    ) from exc
                features.insert(0, "week", int(week))
                features.insert(0, "frame_id", int(frame_id))
                features.insert(0, "play_id", int(play_id))
                features.insert(0, "game_id", int(game_id))
                play_parts.append(features)
                retained_frames += 1
            if not play_parts:
                continue
            scored = pd.concat(play_parts, ignore_index=True)
            scored["case_control_score"] = predict_fidelity_score(
                model, scored.loc[:, FEATURE_NAMES]
            )
            scored = _frame_score_table(scored)
            if scored.duplicated(["game_id", "play_id", "frame_id", "nfl_id"]).any():
                raise ValueError("framewise fidelity output has duplicate identities")
            for nfl_id, player_scores in scored.groupby("nfl_id", sort=True):
                player_scores = player_scores.sort_values("frame_id", kind="stable")
                frames = player_scores["frame_id"].to_numpy(dtype=np.int64)
                summary = opportunity_state_machine(
                    player_scores["case_control_score"].to_numpy(dtype=float),
                    frame_ids=frames,
                )
                event_buffer.append(
                    {
                        "game_id": int(game_id),
                        "play_id": int(play_id),
                        "nfl_id": int(nfl_id),
                        "week": int(week),
                        "first_frame_id": int(frames[0]),
                        "last_frame_id": int(frames[-1]),
                        "scored_frames": int(len(frames)),
                        "opportunities": int(summary.opportunities),
                        "missed_opportunities": int(summary.missed_opportunities),
                        "opportunity_frame_ids_json": json.dumps(
                            [int(frames[index]) for index in summary.opportunity_indices],
                            separators=(",", ":"),
                        ),
                        "missed_frame_ids_json": json.dumps(
                            [int(frames[index]) for index in summary.missed_indices],
                            separators=(",", ":"),
                        ),
                    }
                )
            score_buffer.append(scored)
            buffered_rows += len(scored)
            if buffered_rows >= chunk_rows:
                score_writer.write(pd.concat(score_buffer, ignore_index=True))
                score_buffer.clear()
                buffered_rows = 0
            if len(event_buffer) >= chunk_rows:
                event_writer.write(_opportunity_summary_table(event_buffer))
                event_buffer.clear()
        if score_buffer:
            score_writer.write(pd.concat(score_buffer, ignore_index=True))
        if event_buffer:
            event_writer.write(_opportunity_summary_table(event_buffer))
        score_writer.finish(FRAMEWISE_SCORE_COLUMNS)
        event_writer.finish(OPPORTUNITY_SUMMARY_COLUMNS)
    except BaseException:
        score_writer.abort()
        event_writer.abort()
        raise

    score_record = _artifact_record(
        scores_path,
        root,
        rows=score_writer.rows,
        columns=score_writer.columns,
    )
    event_record = _artifact_record(
        events_path,
        root,
        rows=event_writer.rows,
        columns=event_writer.columns,
    )
    receipt: dict[str, Any] = {
        "schema_version": FRAMEWISE_SCHEMA_VERSION,
        "week": int(week),
        "source_sha256": str(source_sha256),
        "model_sha256": str(model_sha256),
        "artifact_format": artifact_format,
        "winner_commit": WINNER_COMMIT,
        "feature_names": list(FEATURE_NAMES),
        "active_window": "snap_plus_5_through_min_tackle_out_of_bounds_inclusive",
        "opportunity_rule": {
            "threshold": 0.75,
            "comparison": "strictly_greater",
            "consecutive_high_frames": 5,
            "consecutive_low_frames_for_miss": 5,
        },
        "score_interpretation": "case_control_made_versus_missed_score_not_unconditional_probability",
        "audit": {
            **audit,
            "retained_plays": int(retained_plays),
            "retained_frames": int(retained_frames),
            "bad_ballcarrier_plays": int(bad_ballcarrier_plays),
            "scored_defender_frames": int(score_writer.rows),
            "play_defender_summaries": int(event_writer.rows),
        },
        "artifacts": {
            "frame_scores": score_record,
            "opportunity_summaries": event_record,
        },
    }
    receipt["receipt_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt).rstrip(b"\n")
    ).hexdigest()
    _write_json_immutable(receipt_path, receipt)
    return receipt


def _read_release_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "raw BDB2024 Parquet input requires the pinned pyarrow dependency"
            ) from exc
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported BDB2024 table format: {path}")


def _find_release_table(raw_dir: Path, basename: str) -> Path:
    matches = sorted(
        path
        for path in raw_dir.rglob(f"{basename}.*")
        if path.suffix.lower() in {".parquet", ".csv", ".gz"}
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {basename}.csv/parquet below {raw_dir}, found {len(matches)}"
        )
    return matches[0]


def _release_tracking_paths(raw_dir: Path) -> list[tuple[int, Path]]:
    matches: list[tuple[int, Path]] = []
    for path in raw_dir.rglob("tracking_week_*.*"):
        if path.suffix.lower() not in {".parquet", ".csv", ".gz"}:
            continue
        match = re.search(r"tracking_week_(\d+)", path.name)
        if match:
            matches.append((int(match.group(1)), path))
    matches.sort(key=lambda item: item[0])
    if [week for week, _ in matches] != list(range(1, 10)):
        raise FileNotFoundError("expected exactly one BDB2024 tracking shard for weeks 1..9")
    return matches


def run_framewise_fidelity_from_raw(
    model: Any,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    model_sha256: str,
    artifact_format: str = "parquet",
    chunk_rows: int = 50_000,
) -> dict[str, Any]:
    """Stream all nine raw tracking shards through the winner reference.

    Only one weekly tracking table is resident at a time. Per-week receipts
    make interruption recovery safe; a completed shard is checksum-validated
    and skipped on resume.
    """

    raw_root = Path(raw_dir).resolve()
    output_root = Path(output_dir).resolve()
    plays_path = _find_release_table(raw_root, "plays")
    players_path = _find_release_table(raw_root, "players")
    tracking_paths = _release_tracking_paths(raw_root)
    tracking_by_week = {week: path for week, path in tracking_paths}
    plays = _read_release_table(plays_path)
    players = _read_release_table(players_path)
    weeks: list[dict[str, Any]] = []
    for week, path in tracking_paths:
        source_sha = _sha256_file(path)
        receipt_path = output_root / "weeks" / f"week_{week:02d}.json"
        existing = _validate_week_receipt(
            receipt_path,
            output_root=output_root,
            week=week,
            source_sha256=source_sha,
            model_sha256=model_sha256,
            artifact_format=artifact_format,
        )
        if existing is not None:
            weeks.append(existing)
            continue
        tracking = _read_release_table(path)
        weeks.append(
            process_framewise_tracking_week(
                model,
                tracking,
                plays,
                players,
                output_root,
                week=week,
                source_sha256=source_sha,
                model_sha256=model_sha256,
                artifact_format=artifact_format,
                chunk_rows=chunk_rows,
            )
        )
        del tracking

    receipt: dict[str, Any] = {
        "schema_version": FRAMEWISE_SCHEMA_VERSION,
        "branch": "bdb2024_all_defender_all_retained_frame_scores",
        "winner_commit": WINNER_COMMIT,
        "model_sha256": str(model_sha256),
        "artifact_format": artifact_format,
        "complete_weeks": list(range(1, 10)),
        "static_sources": {
            "plays": {
                "path": plays_path.relative_to(raw_root).as_posix(),
                "sha256": _sha256_file(plays_path),
                "size_bytes": int(plays_path.stat().st_size),
            },
            "players": {
                "path": players_path.relative_to(raw_root).as_posix(),
                "sha256": _sha256_file(players_path),
                "size_bytes": int(players_path.stat().st_size),
            },
        },
        "week_receipts": [
            {
                "week": int(item["week"]),
                "receipt_path": f"weeks/week_{int(item['week']):02d}.json",
                "receipt_file_sha256": _sha256_file(
                    output_root / "weeks" / f"week_{int(item['week']):02d}.json"
                ),
                "receipt_hash": item["receipt_hash"],
                "source_path": tracking_by_week[int(item["week"])]
                .relative_to(raw_root)
                .as_posix(),
                "source_sha256": item["source_sha256"],
                "source_size_bytes": int(
                    tracking_by_week[int(item["week"])].stat().st_size
                ),
                "scored_defender_frames": int(item["audit"]["scored_defender_frames"]),
                "play_defender_summaries": int(item["audit"]["play_defender_summaries"]),
            }
            for item in weeks
        ],
        "score_interpretation": "case_control_made_versus_missed_score_not_unconditional_probability",
    }
    receipt["receipt_hash"] = hashlib.sha256(
        _canonical_json_bytes(receipt).rstrip(b"\n")
    ).hexdigest()
    _write_json_immutable(output_root / "framewise_receipt.json", receipt)
    return receipt
