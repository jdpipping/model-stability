"""Task-neutral model families used by the additive BDB suite.

TensorFlow is imported only inside neural functions.  This is intentional: a
CPU tabular worker must never initialize or reserve the GPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np
from sklearn.linear_model import Ridge, SGDClassifier

from .determinism import configure_tensorflow_determinism


MODEL_ROLES = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
LEGACY_FAMILY_ALIASES = {
    "linear_structure": "glm",
    "boosted_structure": "lightgbm",
    "glm": "glm",
    "lightgbm": "lightgbm",
    "cnn": "cnn",
    "transformer": "transformer",
    "set_transformer": "set_transformer",
    "relnet": "relnet",
    "attn_relnet": "attn_relnet",
}
MODEL_FAMILIES = MODEL_ROLES
NEURAL_FAMILIES = frozenset(
    {"cnn", "transformer", "set_transformer", "relnet", "attn_relnet"}
)
GLOBAL_SET_FAMILIES = frozenset({"transformer", "set_transformer"})
RELATIONAL_FAMILIES = frozenset({"relnet", "attn_relnet"})
RELATIONAL_EDGE_VOCAB_SIZE = 23
PUNT_CDF_RESIDUAL_CONTRACT = "punt_cdf_residual_v1"


def implementation_family(value: str) -> str:
    try:
        return LEGACY_FAMILY_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError(f"unsupported model role/family {value!r}") from exc


def empirical_cdf_baseline(labels: np.ndarray, n_outputs: int) -> np.ndarray:
    """Fit an ordered CDF baseline using only the supplied training labels."""

    y = np.asarray(labels, dtype=int).reshape(-1)
    if len(y) == 0 or n_outputs < 2 or np.any((y < 0) | (y >= n_outputs)):
        raise ValueError("ordered labels or output support are invalid")
    counts = np.bincount(y, minlength=int(n_outputs)).astype(np.float64)
    cdf = np.cumsum(counts) / float(len(y))
    cdf[-1] = 1.0
    return cdf


def _bounded_pava_row(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    level: list[float] = []
    weight: list[int] = []
    for value in clipped:
        level.append(float(value))
        weight.append(1)
        while len(level) >= 2 and level[-2] > level[-1]:
            total = weight[-2] + weight[-1]
            pooled = (level[-2] * weight[-2] + level[-1] * weight[-1]) / total
            level[-2:] = [pooled]
            weight[-2:] = [total]
    return np.repeat(np.asarray(level, dtype=np.float64), weight)


def bounded_pava(values: np.ndarray) -> np.ndarray:
    """Project every final-axis vector onto bounded nondecreasing CDFs."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError("PAVA input must be finite with a nonempty final axis")
    flattened = array.reshape(-1, array.shape[-1])
    projected = np.vstack([_bounded_pava_row(row) for row in flattened])
    return projected.reshape(array.shape)


def cdf_to_pmf(cdf: np.ndarray) -> np.ndarray:
    values = np.asarray(cdf, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("CDF must have shape [N,K] with K >= 2")
    complete = bounded_pava(values)
    complete[:, -1] = 1.0
    probability = np.concatenate(
        [complete[:, :1], np.diff(complete, axis=1)], axis=1
    )
    return _normalize_rows(np.clip(probability, 0.0, None))


def punt_cdf_residual_to_pmf(
    residual: np.ndarray, baseline_cdf: np.ndarray
) -> np.ndarray:
    baseline = np.asarray(baseline_cdf, dtype=np.float64).reshape(-1)
    values = np.asarray(residual, dtype=np.float64)
    if baseline.ndim != 1 or len(baseline) != values.shape[-1] + 1:
        raise ValueError("CDF residual and baseline dimensions differ")
    raw = baseline[None, :-1] + values.reshape(-1, values.shape[-1])
    projected = bounded_pava(raw)
    complete = np.concatenate(
        [projected, np.ones((len(projected), 1), dtype=np.float64)], axis=1
    )
    return cdf_to_pmf(complete)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    probability = np.clip(probability, 0.0, None)
    denominator = probability.sum(axis=1, keepdims=True)
    bad = denominator[:, 0] <= 0.0
    if np.any(bad):
        probability[bad] = 1.0 / probability.shape[1]
        denominator = probability.sum(axis=1, keepdims=True)
    return probability / denominator


@dataclass
class ClassicalFit:
    family: str
    outcome_type: str
    models: tuple[Any, ...]
    n_outputs: int
    config: dict[str, Any]
    cdf_baseline: np.ndarray | None = None
    trajectory_horizon: int | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.outcome_type == "binary":
            model = self.models[0]
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(x)
                classes = np.asarray(model.classes_, dtype=int)
                if 1 in classes:
                    return probability[:, int(np.where(classes == 1)[0][0])]
            return np.full(len(x), float(getattr(model, "constant_probability")))
        if self.outcome_type == "distribution":
            if self.config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT:
                if self.cdf_baseline is None:
                    raise RuntimeError("ordered CDF residual fit has no train-only baseline")
                if bool(self.config.get("threshold_query", False)):
                    expanded = _ordered_threshold_features(x, self.cdf_baseline)
                    residual = np.asarray(self.models[0].predict(expanded)).reshape(
                        len(x), self.n_outputs - 1
                    )
                else:
                    residual = np.asarray(self.models[0].predict(x)).reshape(
                        len(x), self.n_outputs - 1
                    )
                return punt_cdf_residual_to_pmf(residual, self.cdf_baseline)
            model = self.models[0]
            probability = model.predict_proba(x)
            result = np.zeros((len(x), self.n_outputs), dtype=np.float64)
            for column, label in enumerate(np.asarray(model.classes_, dtype=int)):
                if 0 <= label < self.n_outputs:
                    result[:, label] = probability[:, column]
            return _normalize_rows(result)
        if self.outcome_type == "trajectory":
            if self.trajectory_horizon is None:
                return np.column_stack([model.predict(x) for model in self.models])
            expanded = _horizon_conditioned_features(
                np.asarray(x, dtype=np.float32), self.trajectory_horizon
            )
            coordinates = np.column_stack(
                [model.predict(expanded) for model in self.models]
            )
            return coordinates.reshape(len(x), self.trajectory_horizon, 2)
        raise ValueError(f"unsupported outcome type {self.outcome_type!r}")


class _ConstantBinary:
    def __init__(self, probability: float) -> None:
        self.constant_probability = float(probability)


class _ConstantRegression:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)


CLASSICAL_PARAMETER_COUNT_DEFINITION = (
    "prediction-bearing fitted scalar state: linear coefficients and intercepts; "
    "constant predictions; or LightGBM split thresholds and leaf values"
)
NEURAL_PARAMETER_COUNT_DEFINITION = "Keras model.count_params() scalar state"


def _lightgbm_tree_parameter_count(node: dict[str, Any]) -> int:
    """Count prediction-bearing fitted scalars in one dumped LightGBM tree."""

    if "leaf_value" in node:
        return 1
    if not all(name in node for name in ("threshold", "left_child", "right_child")):
        raise ValueError("LightGBM tree dump lacks a threshold or child node")
    return (
        1
        + _lightgbm_tree_parameter_count(dict(node["left_child"]))
        + _lightgbm_tree_parameter_count(dict(node["right_child"]))
    )


def classical_parameter_count(fit: ClassicalFit) -> int:
    """Return an explicitly defined fitted-state scalar count for a classical fit.

    A tree count is not a parameter count.  For LightGBM this traverses the
    actual fitted model and counts every learned split threshold and leaf
    prediction; for linear models it counts every stored coefficient and
    intercept.  Constant fallbacks contain one fitted prediction scalar.
    """

    total = 0
    for model in fit.models:
        coefficients = getattr(model, "coef_", None)
        if coefficients is not None:
            intercept = getattr(model, "intercept_", None)
            if intercept is None:
                raise ValueError("linear model has coefficients but no intercept")
            total += int(np.asarray(coefficients).size + np.asarray(intercept).size)
            continue
        booster = getattr(model, "booster_", None)
        if booster is not None:
            dump = booster.dump_model()
            tree_info = dump.get("tree_info") if isinstance(dump, dict) else None
            if not isinstance(tree_info, list) or not tree_info:
                raise ValueError("LightGBM model dump contains no trees")
            total += sum(
                _lightgbm_tree_parameter_count(dict(tree["tree_structure"]))
                for tree in tree_info
            )
            continue
        if isinstance(model, (_ConstantBinary, _ConstantRegression)):
            total += 1
            continue
        raise ValueError(f"unsupported classical fitted model {type(model).__name__}")
    if total <= 0:
        raise ValueError("classical fit contains no fitted scalar parameters")
    return total


def _trajectory_output_mask(target: np.ndarray, target_mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("trajectory targets must have shape [N,H,2]")
    if target_mask is None:
        return np.ones((len(values), values.shape[1] * 2), dtype=bool)
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != values.shape[:2]:
        raise ValueError("trajectory target mask must have shape [N,H]")
    return np.repeat(mask, 2, axis=1)


def _horizon_conditioned_features(
    features: np.ndarray, max_horizon: int
) -> np.ndarray:
    """Expand examples over horizon with a shared deterministic horizon basis."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or max_horizon < 1:
        raise ValueError("horizon-conditioned features require [N,F] and H >= 1")
    horizon = np.arange(1, int(max_horizon) + 1, dtype=np.float32)
    fraction = horizon / float(max_horizon)
    basis = np.column_stack(
        [fraction, np.sqrt(fraction), np.square(fraction)]
    ).astype(np.float32)
    return np.concatenate(
        [np.repeat(x, max_horizon, axis=0), np.tile(basis, (len(x), 1))],
        axis=1,
    )


def _ordered_threshold_features(
    features: np.ndarray, baseline_cdf: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    baseline = np.asarray(baseline_cdf, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or len(baseline) < 2:
        raise ValueError("ordered threshold features are invalid")
    thresholds = np.arange(len(baseline) - 1, dtype=np.float32) / float(
        len(baseline) - 1
    )
    basis = np.column_stack(
        [thresholds, np.sqrt(thresholds), np.square(thresholds), baseline[:-1]]
    ).astype(np.float32)
    return np.concatenate(
        [np.repeat(x, len(thresholds), axis=0), np.tile(basis, (len(x), 1))],
        axis=1,
    )


def _horizon_conditioned_training(
    features: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(target, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("trajectory targets must have shape [N,H,2]")
    if target_mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    else:
        mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != values.shape[:2] or np.any(mask.sum(axis=1) == 0):
        raise ValueError("trajectory target mask is invalid")
    expanded = _horizon_conditioned_features(features, values.shape[1])
    valid = mask.reshape(-1)
    return expanded[valid], values.reshape(-1, 2)[valid]


def fit_glm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    outcome_type: str,
    config: dict[str, Any],
    seed: int,
    n_outputs: int = 1,
    target_mask: np.ndarray | None = None,
) -> ClassicalFit:
    features = np.asarray(x, dtype=np.float32)
    target = np.asarray(y)
    alpha = float(config.get("alpha", 1.0 / 3.0))
    epochs = int(config.get("epochs", 50))
    batch_size = int(config.get("batch_size", 64))
    if outcome_type in {"binary", "distribution"}:
        classes = np.arange(2 if outcome_type == "binary" else n_outputs, dtype=int)
        labels = target.astype(int).reshape(-1)
        if outcome_type == "binary" and len(np.unique(labels)) == 1:
            probability = (labels.sum() + 0.5) / (len(labels) + 1.0)
            return ClassicalFit("glm", outcome_type, (_ConstantBinary(probability),), n_outputs, dict(config))
        if (
            outcome_type == "distribution"
            and config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT
        ):
            baseline = empirical_cdf_baseline(labels, n_outputs)
            observed_cdf = labels[:, None] <= np.arange(n_outputs - 1)[None, :]
            residual = observed_cdf.astype(np.float32) - baseline[None, :-1]
            model = Ridge(alpha=alpha, solver="auto")
            model.fit(features, residual)
            return ClassicalFit(
                "glm",
                outcome_type,
                (model,),
                n_outputs,
                dict(config),
                cdf_baseline=baseline,
            )
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate=str(config.get("learning_rate", "optimal")),
            eta0=float(config.get("eta0", 0.01)),
            random_state=seed,
            max_iter=1,
            tol=None,
        )
        rng = np.random.default_rng(seed)
        first = True
        for _ in range(epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                if first:
                    model.partial_fit(features[batch], labels[batch], classes=classes)
                    first = False
                else:
                    model.partial_fit(features[batch], labels[batch])
        return ClassicalFit("glm", outcome_type, (model,), n_outputs, dict(config))
    if outcome_type == "trajectory":
        expanded, conditioned_target = _horizon_conditioned_training(
            features, target, target_mask
        )
        models: list[Any] = []
        for coordinate in range(2):
            model = Ridge(alpha=alpha, solver="auto")
            model.fit(expanded, conditioned_target[:, coordinate])
            models.append(model)
        return ClassicalFit(
            "glm",
            outcome_type,
            tuple(models),
            int(target.shape[1] * 2),
            dict(config),
            trajectory_horizon=int(target.shape[1]),
        )
    raise ValueError(f"unsupported outcome type {outcome_type!r}")


def fit_lightgbm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    outcome_type: str,
    config: dict[str, Any],
    seed: int,
    n_outputs: int = 1,
    target_mask: np.ndarray | None = None,
) -> ClassicalFit:
    import lightgbm as lgb

    common = {
        "n_estimators": int(config.get("n_estimators", 200)),
        "learning_rate": float(config.get("learning_rate", 0.05)),
        "max_depth": int(config.get("max_depth", 5)),
        "min_child_samples": int(config.get("min_child_samples", 50)),
        "reg_alpha": float(config.get("reg_alpha", 0.5)),
        "reg_lambda": float(config.get("reg_lambda", 0.5)),
        "random_state": int(seed),
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    features = np.asarray(x, dtype=np.float32)
    target = np.asarray(y)
    if outcome_type == "binary":
        labels = target.astype(int).reshape(-1)
        if len(np.unique(labels)) == 1:
            probability = (labels.sum() + 0.5) / (len(labels) + 1.0)
            return ClassicalFit("lightgbm", outcome_type, (_ConstantBinary(probability),), 1, dict(config))
        model = lgb.LGBMClassifier(objective="binary", **common)
        model.fit(features, labels)
        return ClassicalFit("lightgbm", outcome_type, (model,), 1, dict(config))
    if outcome_type == "distribution":
        if config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT:
            labels = target.astype(int).reshape(-1)
            baseline = empirical_cdf_baseline(labels, n_outputs)
            observed_cdf = labels[:, None] <= np.arange(n_outputs - 1)[None, :]
            residual = observed_cdf.astype(np.float32) - baseline[None, :-1]
            expanded = _ordered_threshold_features(features, baseline)
            model = lgb.LGBMRegressor(objective="regression_l2", **common)
            model.fit(expanded, residual.reshape(-1))
            fitted_config = {**dict(config), "threshold_query": True}
            return ClassicalFit(
                "lightgbm",
                outcome_type,
                (model,),
                n_outputs,
                fitted_config,
                cdf_baseline=baseline,
            )
        model = lgb.LGBMClassifier(objective="multiclass", num_class=int(n_outputs), **common)
        model.fit(features, target.astype(int).reshape(-1))
        return ClassicalFit("lightgbm", outcome_type, (model,), n_outputs, dict(config))
    if outcome_type == "trajectory":
        expanded, conditioned_target = _horizon_conditioned_training(
            features, target, target_mask
        )
        models = []
        for coordinate in range(2):
            params = dict(common)
            params["random_state"] = seed + coordinate
            model = lgb.LGBMRegressor(objective="regression_l2", **params)
            model.fit(expanded, conditioned_target[:, coordinate])
            models.append(model)
        return ClassicalFit(
            "lightgbm",
            outcome_type,
            tuple(models),
            int(target.shape[1] * 2),
            dict(config),
            trajectory_horizon=int(target.shape[1]),
        )
    raise ValueError(f"unsupported outcome type {outcome_type!r}")


def _neural_loss(
    outcome_type: str, config: dict[str, Any] | None = None
) -> Callable[[Any, Any], Any]:
    import tensorflow as tf

    output_config = {} if config is None else dict(config)

    if outcome_type == "binary":
        if output_config.get("binary_loss") == "log_loss":
            def log_loss(y_true: Any, y_pred: Any) -> Any:
                return tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
                    )
                )
            return log_loss
        def brier(y_true: Any, y_pred: Any) -> Any:
            return tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))
        return brier
    if outcome_type == "distribution":
        if output_config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT:
            baseline = np.asarray(output_config.get("cdf_baseline"), dtype=np.float32)
            if baseline.ndim != 1 or len(baseline) < 2:
                raise ValueError("punt CDF residual loss requires a fitted baseline")
            base = tf.constant(baseline[:-1], dtype=tf.float32)

            def ordered_residual_crps(y_true: Any, y_pred: Any) -> Any:
                truth_cdf = tf.cumsum(tf.cast(y_true, tf.float32), axis=-1)[..., :-1]
                raw_cdf = tf.clip_by_value(
                    base[None, :] + tf.cast(y_pred, tf.float32), 0.0, 1.0
                )
                return tf.reduce_mean(tf.square(raw_cdf - truth_cdf))

            return ordered_residual_crps
        def crps(y_true: Any, y_pred: Any) -> Any:
            return tf.reduce_mean(
                tf.square(tf.cumsum(tf.cast(y_pred, tf.float32), axis=-1) - tf.cumsum(tf.cast(y_true, tf.float32), axis=-1))
            )
        return crps
    if outcome_type == "trajectory":
        def masked_mse(y_true: Any, y_pred: Any) -> Any:
            target = tf.cast(y_true[..., :2], tf.float32)
            mask = tf.cast(y_true[..., 2], tf.float32)
            squared = tf.reduce_sum(tf.square(target - tf.cast(y_pred, tf.float32)), axis=-1)
            return tf.reduce_sum(squared * mask) / tf.maximum(2.0 * tf.reduce_sum(mask), 1.0)
        return masked_mse
    raise ValueError(f"unsupported outcome type {outcome_type!r}")


def neural_head_loss_signature(
    outcome_type: str,
    *,
    n_outputs: int,
    max_horizon: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Describe the output head, target semantics, and optimized loss exactly."""

    output_contract = config.get("output_contract")
    if outcome_type == "binary":
        head = {"type": "dense_sigmoid", "shape": [1]}
        loss = (
            "mean_binary_crossentropy"
            if config.get("binary_loss") == "log_loss"
            else "mean_brier"
        )
        target = "zero_one_label"
    elif outcome_type == "distribution":
        if output_contract == PUNT_CDF_RESIDUAL_CONTRACT:
            baseline = np.asarray(config.get("cdf_baseline"), dtype=np.float64)
            if baseline.shape != (int(n_outputs),) or not np.all(np.isfinite(baseline)):
                raise ValueError("ordered head/loss signature requires its train-only CDF")
            head = {
                "type": "dense_linear_cdf_residual",
                "shape": [int(n_outputs) - 1],
                "prediction_projection": "bounded_pava_then_full_pmf",
                "train_cdf_sha256": hashlib.sha256(
                    np.ascontiguousarray(baseline).view(np.uint8)
                ).hexdigest(),
            }
            loss = "mean_squared_cdf_error_crps_equivalent"
            target = "one_hot_support_label_cumulative_through_k_minus_2"
        else:
            head = {"type": "dense_softmax", "shape": [int(n_outputs)]}
            loss = "mean_crps"
            target = "one_hot_support_label"
    elif outcome_type == "trajectory":
        if int(max_horizon) < 1:
            raise ValueError("trajectory head signature requires a positive horizon")
        trajectory_target = str(config.get("trajectory_target", "residual"))
        if trajectory_target not in {"residual", "absolute"}:
            raise ValueError("trajectory target signature is invalid")
        head = {
            "type": "shared_horizon_conditioned_xy_decoder",
            "shape": [int(max_horizon), 2],
        }
        loss = "masked_mean_squared_coordinate_error"
        target = f"masked_{trajectory_target}_xy"
    else:
        raise ValueError(f"unsupported outcome type {outcome_type!r}")
    payload = {
        "schema_version": "bdb-neural-head-loss-v1",
        "outcome_type": str(outcome_type),
        "output_contract": output_contract,
        "head": head,
        "loss": loss,
        "training_target": target,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "signature_sha256": digest}


def _output_head(
    encoded: Any,
    outcome_type: str,
    n_outputs: int,
    max_horizon: int,
    layers: Any,
    config: dict[str, Any] | None = None,
) -> Any:
    output_config = {} if config is None else config
    if outcome_type == "binary":
        return layers.Dense(1, activation="sigmoid", name="probability")(encoded)
    if outcome_type == "distribution":
        if output_config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT:
            return layers.Dense(
                n_outputs - 1, activation=None, name="cdf_residual"
            )(encoded)
        return layers.Dense(n_outputs, activation="softmax", name="distribution")(encoded)
    if outcome_type == "trajectory":
        import tensorflow as tf

        horizon = layers.Lambda(
            lambda value: tf.tile(
                tf.reshape(
                    tf.linspace(1.0 / float(max_horizon), 1.0, max_horizon),
                    (1, max_horizon, 1),
                ),
                (tf.shape(value)[0], 1, 1),
            ),
            name="normalized_horizon",
        )(encoded)
        horizon_basis = layers.Lambda(
            lambda value: tf.concat(
                [value, tf.sqrt(value), tf.square(value)], axis=-1
            ),
            name="horizon_basis",
        )(horizon)
        repeated = layers.RepeatVector(max_horizon, name="repeat_encoded_over_horizon")(
            encoded
        )
        decoded = layers.Concatenate(name="horizon_conditioned_state")(
            [repeated, horizon_basis]
        )
        decoded = layers.TimeDistributed(
            layers.Dense(64, activation="gelu"), name="shared_horizon_decoder"
        )(decoded)
        return layers.TimeDistributed(
            layers.Dense(2), name="trajectory_residual"
        )(decoded)
    raise ValueError(f"unsupported outcome type {outcome_type!r}")


def build_cnn(
    input_shape: tuple[int | None, int, int, int],
    *,
    outcome_type: str,
    n_outputs: int,
    max_horizon: int,
    dropout: float,
    context_dim: int = 0,
    output_config: dict[str, Any] | None = None,
) -> Any:
    import tensorflow as tf
    from tensorflow.keras import layers

    raster = layers.Input(shape=input_shape, name="raster")
    frame_mask = layers.Input(shape=(input_shape[0],), dtype="bool", name="frame_mask")
    value = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(raster)
    value = layers.TimeDistributed(layers.MaxPool2D(2))(value)
    value = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"))(value)
    value = layers.TimeDistributed(layers.GlobalAveragePooling2D())(value)
    mask_float = layers.Lambda(lambda item: tf.cast(item, tf.float32)[..., None])(frame_mask)
    # Spatial convolution biases make an all-zero padded frame non-zero. Mask
    # before temporal convolution so left padding cannot alter the first real
    # frame, then mask again before pooling. This also makes batch-local removal
    # of all-padding frame columns scientifically equivalent.
    value = layers.Multiply()([value, mask_float])
    value = layers.Conv1D(96, 3, padding="same", activation="relu")(value)
    value = layers.Dropout(dropout)(value)
    summed = layers.Lambda(lambda item: tf.reduce_sum(item[0] * item[1], axis=1))([value, mask_float])
    count = layers.Lambda(lambda item: tf.maximum(tf.reduce_sum(item, axis=1), 1.0))(mask_float)
    encoded = layers.Lambda(lambda item: item[0] / item[1])([summed, count])
    encoded = layers.Dense(128, activation="relu")(encoded)
    model_inputs: list[Any] = [raster, frame_mask]
    if context_dim:
        context = layers.Input(shape=(int(context_dim),), name="global_context")
        context_encoded = layers.Dense(64, activation="relu", name="context_encoding")(context)
        encoded = layers.Concatenate(name="tracking_and_context")([encoded, context_encoded])
        model_inputs.append(context)
    output = _output_head(
        encoded, outcome_type, n_outputs, max_horizon, layers, output_config
    )
    return tf.keras.Model(model_inputs, output, name="bdb_spatiotemporal_cnn")


def build_transformer(
    token_shape: tuple[int | None, int, int],
    *,
    outcome_type: str,
    n_outputs: int,
    max_horizon: int,
    dropout: float,
    d_model: int = 64,
    heads: int = 2,
    ff_dim: int = 256,
    layers_count: int = 3,
    context_dim: int = 0,
    output_config: dict[str, Any] | None = None,
) -> Any:
    import tensorflow as tf
    from tensorflow.keras import layers

    class FrameSetBlock(layers.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)
            self.norm1 = layers.LayerNormalization()
            self.ff1 = layers.Dense(ff_dim, activation="gelu")
            self.ff2 = layers.Dense(d_model)
            self.norm2 = layers.LayerNormalization()

        def call(self, inputs: Any) -> Any:
            value, mask = inputs
            shape = tf.shape(value)
            flattened = tf.reshape(value, (-1, shape[2], d_model))
            flat_mask = tf.reshape(mask, (-1, shape[2]))
            attention_mask = tf.logical_and(flat_mask[:, :, None], flat_mask[:, None, :])
            attended = self.attention(flattened, flattened, attention_mask=attention_mask)
            flattened = self.norm1(flattened + attended)
            flattened = self.norm2(flattened + self.ff2(self.ff1(flattened)))
            return tf.reshape(flattened, shape)

    tokens = layers.Input(shape=token_shape, name="player_tokens")
    player_mask = layers.Input(shape=token_shape[:2], dtype="bool", name="player_mask")
    frame_mask = layers.Input(shape=(token_shape[0],), dtype="bool", name="frame_mask")
    time_to_event = layers.Input(
        shape=(token_shape[0], 1), name="time_to_event"
    )
    if layers_count < 1:
        raise ValueError("transformer layers_count must be positive")
    value = layers.Dense(d_model, name="global_set_token_projection")(tokens)
    # The same causal, masked time-to-event tensor is supplied to all three
    # protocol-v2 neural families.  Adding it before within-frame set attention
    # preserves player permutation equivariance while preventing the temporal
    # block from becoming frame-order invariant.
    time_encoding = layers.Dense(
        d_model, use_bias=False, name="global_set_time_projection"
    )(time_to_event)
    time_encoding = layers.Lambda(
        lambda item: item[:, :, None, :], name="broadcast_global_set_time"
    )(time_encoding)
    value = layers.Add(name="token_plus_time")([value, time_encoding])
    # One temporal block follows the player-set blocks.  Thus layers_count=3
    # reproduces the locked two-set-plus-one-temporal architecture.
    for _ in range(max(0, layers_count - 1)):
        value = FrameSetBlock()([value, player_mask])
    player_weight = layers.Lambda(lambda item: tf.cast(item, tf.float32)[..., None])(player_mask)
    pooled = layers.Lambda(lambda item: tf.reduce_sum(item[0] * item[1], axis=2))([value, player_weight])
    denominator = layers.Lambda(lambda item: tf.maximum(tf.reduce_sum(item, axis=2), 1.0))(player_weight)
    pooled = layers.Lambda(lambda item: item[0] / item[1])([pooled, denominator])
    temporal_mask = layers.Lambda(
        lambda item: tf.logical_and(item[:, :, None], item[:, None, :]),
        output_shape=(token_shape[0], token_shape[0]),
        name="temporal_attention_mask",
    )(frame_mask)
    attended = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)(
        pooled, pooled, attention_mask=temporal_mask
    )
    pooled = layers.LayerNormalization()(pooled + attended)
    feed = layers.Dense(ff_dim, activation="gelu")(pooled)
    pooled = layers.LayerNormalization()(pooled + layers.Dense(d_model)(feed))
    frame_weight = layers.Lambda(lambda item: tf.cast(item, tf.float32)[..., None])(frame_mask)
    encoded = layers.Lambda(lambda item: tf.reduce_sum(item[0] * item[1], axis=1))([pooled, frame_weight])
    frame_count = layers.Lambda(lambda item: tf.maximum(tf.reduce_sum(item, axis=1), 1.0))(frame_weight)
    encoded = layers.Lambda(lambda item: item[0] / item[1])([encoded, frame_count])
    model_inputs: list[Any] = [tokens, player_mask, frame_mask, time_to_event]
    if context_dim:
        context = layers.Input(shape=(int(context_dim),), name="global_context")
        context_encoded = layers.Dense(
            64, activation="gelu", name="context_encoding"
        )(context)
        encoded = layers.Concatenate(name="tracking_and_context")([encoded, context_encoded])
        model_inputs.append(context)
    output = _output_head(
        encoded, outcome_type, n_outputs, max_horizon, layers, output_config
    )
    model = tf.keras.Model(
        model_inputs, output, name="bdb_global_set_transformer_v1"
    )
    # This is deliberately distinct from the 2020 single-snapshot Set
    # Transformer: it factorizes global player-set attention within frame and
    # global temporal attention across the complete causal history.
    model._bdb_architecture_id = "bdb_global_set_transformer_v1"
    model._bdb_attention_scope = (
        "all_observed_players_within_frame_then_all_observed_frames"
    )
    return model


def build_relational_network(
    token_shape: tuple[int | None, int, int],
    *,
    aggregation: str,
    outcome_type: str,
    n_outputs: int,
    max_horizon: int,
    dropout: float,
    d_model: int = 48,
    edge_dim: int = 8,
    edge_hidden: int = 64,
    context_dim: int = 0,
    output_config: dict[str, Any] | None = None,
) -> Any:
    """Build the matched RelNet/AttnRelNet architecture.

    The aggregation switch has no learned parameters. Everything before and
    after it—including edge messages and the temporal encoder—is shared.
    """

    import tensorflow as tf
    from tensorflow.keras import layers

    if aggregation not in {"fixed", "attention"}:
        raise ValueError("relational aggregation must be fixed or attention")

    class EdgeMessageAggregation(layers.Layer):
        def __init__(self) -> None:
            super().__init__(name=f"{aggregation}_edge_aggregation")
            self.edge_embedding = layers.Embedding(
                RELATIONAL_EDGE_VOCAB_SIZE, edge_dim, mask_zero=False
            )
            self.message_hidden = layers.Dense(edge_hidden, activation="gelu")
            self.message_output = layers.Dense(d_model)

        def call(self, inputs: Any) -> Any:
            node, edge_type = inputs
            shape = tf.shape(node)
            target = tf.broadcast_to(
                node[:, :, :, None, :],
                (shape[0], shape[1], shape[2], shape[2], d_model),
            )
            source = tf.broadcast_to(
                node[:, :, None, :, :],
                (shape[0], shape[1], shape[2], shape[2], d_model),
            )
            edge = self.edge_embedding(edge_type)
            message = self.message_output(
                self.message_hidden(tf.concat([target, source, edge], axis=-1))
            )
            mask = tf.cast(edge_type > 0, tf.float32)
            if aggregation == "fixed":
                numerator = tf.reduce_sum(message * mask[..., None], axis=3)
                denominator = tf.maximum(
                    tf.reduce_sum(mask, axis=3, keepdims=True), 1.0
                )
                return numerator / denominator
            score = tf.reduce_sum(target * message, axis=-1) / tf.sqrt(
                tf.cast(d_model, tf.float32)
            )
            score = tf.where(mask > 0, score, tf.constant(-1e9, tf.float32))
            weight = tf.nn.softmax(score, axis=3) * mask
            weight /= tf.maximum(tf.reduce_sum(weight, axis=3, keepdims=True), 1e-8)
            return tf.reduce_sum(message * weight[..., None], axis=3)

    tokens = layers.Input(shape=token_shape, name="player_tokens")
    player_mask = layers.Input(shape=token_shape[:2], dtype="bool", name="player_mask")
    frame_mask = layers.Input(shape=(token_shape[0],), dtype="bool", name="frame_mask")
    edge_type = layers.Input(
        shape=(token_shape[0], token_shape[1], token_shape[1]),
        dtype="uint8",
        name="edge_type",
    )
    time_to_event = layers.Input(
        shape=(token_shape[0], 1), name="time_to_event"
    )
    node = layers.Dense(d_model, name="shared_node_encoder")(tokens)
    time_encoding = layers.Dense(d_model, use_bias=False, name="shared_time_encoder")(
        time_to_event
    )
    node = layers.Add()([node, time_encoding[:, :, None, :]])
    node_weight = layers.Lambda(lambda value: tf.cast(value, tf.float32)[..., None])(
        player_mask
    )
    node = layers.Multiply()([node, node_weight])
    # Kernel width one over player slots prevents accidental adjacency meaning;
    # only the time axis is convolved and stable slots preserve identity.
    temporal_node = layers.Conv2D(
        d_model, (3, 1), padding="same", activation="gelu", name="shared_slot_temporal"
    )(node)
    temporal_node = layers.Multiply()([temporal_node, node_weight])
    message = EdgeMessageAggregation()([temporal_node, edge_type])
    node = layers.LayerNormalization(name="message_residual_norm")(
        temporal_node + layers.Dropout(dropout)(message)
    )
    feed = layers.Dense(edge_hidden, activation="gelu", name="shared_node_ff1")(node)
    feed = layers.Dense(d_model, name="shared_node_ff2")(feed)
    node = layers.LayerNormalization(name="node_ff_norm")(node + feed)
    node = layers.Multiply()([node, node_weight])
    frame_state = layers.Lambda(
        lambda value: tf.reduce_sum(value[0] * value[1], axis=2)
        / tf.maximum(tf.reduce_sum(value[1], axis=2), 1.0),
        name="masked_player_pool",
    )([node, node_weight])
    frame_state = layers.Conv1D(
        d_model, 3, padding="same", activation="gelu", name="shared_frame_temporal"
    )(frame_state)
    frame_weight = layers.Lambda(
        lambda value: tf.cast(value, tf.float32)[..., None]
    )(frame_mask)
    encoded = layers.Lambda(
        lambda value: tf.reduce_sum(value[0] * value[1], axis=1)
        / tf.maximum(tf.reduce_sum(value[1], axis=1), 1.0),
        name="masked_time_pool",
    )([frame_state, frame_weight])
    model_inputs: list[Any] = [
        tokens,
        player_mask,
        frame_mask,
        edge_type,
        time_to_event,
    ]
    if context_dim:
        context = layers.Input(shape=(int(context_dim),), name="global_context")
        context_encoded = layers.Dense(
            64, activation="gelu", name="context_encoding"
        )(context)
        encoded = layers.Concatenate(name="tracking_and_context")(
            [encoded, context_encoded]
        )
        model_inputs.append(context)
    output = _output_head(
        encoded, outcome_type, n_outputs, max_horizon, layers, output_config
    )
    name = "bdb_relnet" if aggregation == "fixed" else "bdb_attn_relnet"
    model = tf.keras.Model(model_inputs, output, name=name)
    model._bdb_output_contract = (output_config or {}).get("output_contract")
    model._bdb_cdf_baseline = (output_config or {}).get("cdf_baseline")
    model._bdb_aggregation = aggregation
    return model


@dataclass
class NeuralFit:
    family: str
    model: Any
    best_epoch: int
    selector_history: dict[str, list[float]]
    refit_history: dict[str, list[float]]
    parameter_count: int
    validation_game_ids: tuple[str, ...]
    validation_split_seed: int
    fairness_metadata: dict[str, Any] = field(default_factory=dict)
    complexity_metadata: dict[str, Any] = field(default_factory=dict)


NEURAL_SELECTION_SCHEMA_VERSION = "bdb-neural-epoch-selection-v1"


@dataclass(frozen=True)
class NeuralSelection:
    """Outcome-free epoch-selection state sufficient for an exact neural refit."""

    family: str
    outcome_type: str
    n_outputs: int
    max_horizon: int
    best_epoch: int
    selector_history: dict[str, list[float]]
    parameter_count: int
    input_shapes: dict[str, tuple[int | None, ...]]
    validation_game_ids: tuple[str, ...]
    validation_split_seed: int
    selection_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEURAL_SELECTION_SCHEMA_VERSION,
            "family": self.family,
            "outcome_type": self.outcome_type,
            "n_outputs": int(self.n_outputs),
            "max_horizon": int(self.max_horizon),
            "best_epoch": int(self.best_epoch),
            "selector_history": {
                str(key): [float(value) for value in values]
                for key, values in self.selector_history.items()
            },
            "parameter_count": int(self.parameter_count),
            "input_shapes": {
                str(key): [None if value is None else int(value) for value in shape]
                for key, shape in self.input_shapes.items()
            },
            "validation_game_ids": list(self.validation_game_ids),
            "validation_split_seed": int(self.validation_split_seed),
            "selection_seed": int(self.selection_seed),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NeuralSelection":
        expected = {
            "schema_version",
            "family",
            "outcome_type",
            "n_outputs",
            "max_horizon",
            "best_epoch",
            "selector_history",
            "parameter_count",
            "input_shapes",
            "validation_game_ids",
            "validation_split_seed",
            "selection_seed",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("neural selection receipt fields are invalid")
        family = implementation_family(str(value["family"]))
        outcome_type = str(value["outcome_type"])
        history = value["selector_history"]
        shapes = value["input_shapes"]
        game_ids = value["validation_game_ids"]
        if (
            value.get("schema_version") != NEURAL_SELECTION_SCHEMA_VERSION
            or family not in NEURAL_FAMILIES
            or outcome_type not in {"binary", "distribution", "trajectory"}
            or not isinstance(history, Mapping)
            or not isinstance(shapes, Mapping)
            or not isinstance(game_ids, list)
        ):
            raise ValueError("neural selection receipt contract is invalid")
        normalized_history: dict[str, list[float]] = {}
        for key, raw in history.items():
            if not isinstance(raw, list):
                raise ValueError("neural selection history is invalid")
            observed = [float(item) for item in raw]
            if not observed or not np.all(np.isfinite(observed)):
                raise ValueError("neural selection history is non-finite or empty")
            normalized_history[str(key)] = observed
        if len({len(items) for items in normalized_history.values()}) != 1:
            raise ValueError("neural selection history lengths differ")
        validation_losses = normalized_history.get("val_loss")
        best_epoch = int(value["best_epoch"])
        if (
            validation_losses is None
            or best_epoch != int(np.argmin(validation_losses)) + 1
            or best_epoch < 1
        ):
            raise ValueError("neural selection best epoch is invalid")
        normalized_shapes: dict[str, tuple[int | None, ...]] = {}
        for key, raw in shapes.items():
            if not isinstance(raw, list) or not raw:
                raise ValueError("neural selection input shapes are invalid")
            shape: list[int | None] = []
            for item in raw:
                if item is None:
                    shape.append(None)
                elif isinstance(item, bool) or int(item) <= 0:
                    raise ValueError("neural selection input shape is invalid")
                else:
                    shape.append(int(item))
            normalized_shapes[str(key)] = tuple(shape)
        n_outputs = int(value["n_outputs"])
        max_horizon = int(value["max_horizon"])
        parameter_count = int(value["parameter_count"])
        if n_outputs < 1 or max_horizon < 1 or parameter_count < 1:
            raise ValueError("neural selection dimensions or capacity are invalid")
        normalized_games = tuple(sorted(str(item) for item in game_ids))
        if list(normalized_games) != game_ids or len(set(normalized_games)) != len(
            normalized_games
        ):
            raise ValueError("neural selection validation games are not canonical")
        return cls(
            family=family,
            outcome_type=outcome_type,
            n_outputs=n_outputs,
            max_horizon=max_horizon,
            best_epoch=best_epoch,
            selector_history=normalized_history,
            parameter_count=parameter_count,
            input_shapes=normalized_shapes,
            validation_game_ids=normalized_games,
            validation_split_seed=int(value["validation_split_seed"]),
            selection_seed=int(value["selection_seed"]),
        )


def representative_transformer_forward_flops(
    input_shapes: dict[str, tuple[int | None, ...]],
    config: dict[str, Any],
) -> int:
    """Return a deterministic one-example Global Set Transformer FLOP estimate."""

    token_shape = tuple(input_shapes["player_tokens"])
    if len(token_shape) != 3:
        raise ValueError("Set Transformer player-token shape must be [T,P,F]")
    time = int(config.get("representative_time_steps", token_shape[0] or 20))
    players = int(token_shape[1])
    channels = int(token_shape[2])
    d_model = int(config.get("d_model", 64))
    heads = int(config.get("heads", 2))
    ff_dim = int(config.get("ff_dim", 256))
    layers_count = int(config.get("layers", 3))
    context = int(input_shapes.get("global_context", (0,))[0])
    if (
        time <= 0
        or players <= 0
        or channels <= 0
        or d_model <= 0
        or heads <= 0
        or d_model % heads
        or ff_dim <= 0
        or layers_count < 1
    ):
        raise ValueError("Set Transformer representative workload is invalid")

    # Dense projections use the conventional two FLOPs per multiply-add. Each
    # attention block includes Q/K/V/output projections and both score/value
    # matrix products. Layer norms, masks, activations, and softmax are omitted
    # consistently because they do not affect the locked comparative receipt.
    flops = 2 * time * players * channels * d_model
    flops += 2 * time * d_model  # scalar time-to-event projection
    player_blocks = max(0, layers_count - 1)
    flops += player_blocks * (
        8 * time * players * d_model * d_model
        + 4 * time * players * players * d_model
        + 4 * time * players * d_model * ff_dim
    )
    flops += 8 * time * d_model * d_model
    flops += 4 * time * time * d_model
    flops += 4 * time * d_model * ff_dim
    flops += 2 * context * 64
    return int(flops)


def transformer_complexity_metadata(
    family: str,
    input_shapes: dict[str, tuple[int | None, ...]],
    config: dict[str, Any],
    parameter_count: int,
) -> dict[str, Any]:
    """Bind the protocol-v2 Global Set Transformer capacity and workload."""

    implementation = implementation_family(family)
    if implementation not in GLOBAL_SET_FAMILIES:
        raise ValueError("complexity metadata is only defined for Set Transformer")
    count = int(parameter_count)
    if count <= 0 or count > 350_000:
        raise ValueError("Set Transformer violates the 350k parameter cap")
    return {
        "architecture_id": "bdb_global_set_transformer_v1",
        "inputs": "task_tokens_player_frame_masks_time_context",
        "attention_contract": "global_set_time_attention_v1",
        "relation_scope": "global_masked_all_player_self_attention",
        "typed_graph_edges_consumed": False,
        "temporal_encoder": "factorized_masked_temporal_attention",
        "protocol_note": "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot",
        "parameter_count": count,
        "parameter_cap": 350_000,
        "representative_forward_flops": representative_transformer_forward_flops(
            input_shapes, config
        ),
    }


def representative_relational_forward_flops(
    input_shapes: dict[str, tuple[int | None, ...]],
    config: dict[str, Any],
    *,
    aggregation: str,
) -> int:
    """Return a deterministic dense-graph one-example forward FLOP estimate."""

    if aggregation not in {"fixed", "attention"}:
        raise ValueError("unknown relational aggregation")
    token_shape = tuple(input_shapes["player_tokens"])
    time = int(config.get("representative_time_steps", token_shape[0] or 20))
    players = int(token_shape[1])
    channels = int(token_shape[2])
    d_model = int(config.get("d_model", 48))
    edge_dim = int(config.get("edge_dim", 8))
    edge_hidden = int(config.get("edge_hidden", 64))
    context = int(input_shapes.get("global_context", (0,))[0])
    directed_edges = players * max(players - 1, 0)
    flops = 2 * time * players * channels * d_model
    flops += 2 * time * players * 3 * d_model * d_model
    flops += 2 * time * directed_edges * (2 * d_model + edge_dim) * edge_hidden
    flops += 2 * time * directed_edges * edge_hidden * d_model
    flops += 4 * time * players * d_model * edge_hidden
    flops += 2 * time * 3 * d_model * d_model
    flops += 2 * context * 64
    if aggregation == "attention":
        flops += time * directed_edges * (2 * d_model + 8)
    else:
        flops += time * directed_edges * (d_model + 2)
    return int(flops)


def relational_fairness_metadata(
    family: str,
    input_shapes: dict[str, tuple[int | None, ...]],
    config: dict[str, Any],
    parameter_count: int,
) -> dict[str, Any]:
    implementation = implementation_family(family)
    if implementation not in RELATIONAL_FAMILIES:
        raise ValueError("fairness metadata is only defined for relational models")
    aggregation = "fixed" if implementation == "relnet" else "attention"
    count = int(parameter_count)
    if count <= 0 or count > 350_000:
        raise ValueError("relational model violates the 350k parameter cap")
    return {
        "contract": "matched_relational_aggregation_v1",
        "aggregation": aggregation,
        "shared_components": [
            "tokenization",
            "causal_history",
            "edge_types",
            "slot_temporal_encoder",
            "context_encoder",
            "output_head",
            "loss",
            "tuning_grid",
            "training_protocol",
        ],
        "only_architecture_difference": "fixed_masked_mean_vs_attention_weighted_mean",
        "parameter_count": count,
        "parameter_cap": 350_000,
        "representative_forward_flops": representative_relational_forward_flops(
            input_shapes, config, aggregation=aggregation
        ),
        "parameter_tolerance_fraction": 0.05,
        "flop_tolerance_fraction": 0.15,
    }


def validate_relational_pair_fairness(
    relnet: dict[str, Any], attn_relnet: dict[str, Any]
) -> dict[str, float]:
    """Fail closed unless a paired architecture receipt meets both bounds."""

    parameters = [int(relnet["parameter_count"]), int(attn_relnet["parameter_count"])]
    flops = [
        int(relnet["representative_forward_flops"]),
        int(attn_relnet["representative_forward_flops"]),
    ]
    parameter_gap = abs(parameters[0] - parameters[1]) / max(parameters)
    flop_gap = abs(flops[0] - flops[1]) / max(flops)
    if max(parameters) > 350_000 or parameter_gap > 0.05 or flop_gap > 0.15:
        raise ValueError("RelNet/AttnRelNet fairness guard failed")
    return {
        "parameter_gap_fraction": float(parameter_gap),
        "flop_gap_fraction": float(flop_gap),
    }


def grouped_validation_indices(
    game_ids: np.ndarray,
    seed: int,
    fraction: float = 0.2,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("validation fraction must be in (0,1)")
    groups = np.asarray(game_ids).reshape(-1)
    unique = np.asarray(sorted(np.unique(groups), key=str), dtype=object)
    if len(unique) < 2:
        raise ValueError("neural epoch selection requires at least two games")
    rng = np.random.default_rng(seed)
    n_validation = min(max(1, int(np.ceil(len(unique) * fraction))), len(unique) - 1)
    if strata is None:
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        validation_groups = shuffled[:n_validation]
    else:
        labels = np.asarray(strata).astype(str).reshape(-1)
        if len(labels) != len(groups):
            raise ValueError("validation strata do not align with game IDs")
        game_stratum: dict[object, str] = {}
        for game in unique:
            values = np.unique(labels[groups == game])
            if len(values) != 1:
                raise ValueError(f"game {game} occurs in multiple validation strata")
            game_stratum[game] = str(values[0])
        by_stratum = {
            stratum: [game for game in unique if game_stratum[game] == stratum]
            for stratum in sorted(set(game_stratum.values()))
        }
        exact = {key: n_validation * len(value) / len(unique) for key, value in by_stratum.items()}
        allocation = {key: int(np.floor(value)) for key, value in exact.items()}
        remaining = n_validation - sum(allocation.values())
        tie = list(by_stratum)
        rng.shuffle(tie)
        tie_rank = {key: index for index, key in enumerate(tie)}
        for key in sorted(by_stratum, key=lambda item: (-(exact[item] - allocation[item]), tie_rank[item]))[:remaining]:
            allocation[key] += 1
        selected: list[object] = []
        for key in sorted(by_stratum):
            values = np.asarray(by_stratum[key], dtype=object)
            rng.shuffle(values)
            selected.extend(values[: allocation[key]])
        validation_groups = np.asarray(selected, dtype=object)
    validation = np.flatnonzero(np.isin(groups, validation_groups))
    training = np.flatnonzero(~np.isin(groups, validation_groups))
    return training, validation


def _keras_targets(y: np.ndarray, outcome_type: str, n_outputs: int, target_mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(y)
    if outcome_type == "binary":
        return values.astype(np.float32).reshape(-1, 1)
    if outcome_type == "distribution":
        return np.eye(n_outputs, dtype=np.float32)[values.astype(int).reshape(-1)]
    if outcome_type == "trajectory":
        if target_mask is None or np.asarray(target_mask).shape != values.shape[:2]:
            raise ValueError("trajectory neural targets require a compatible mask")
        return np.concatenate([values.astype(np.float32), np.asarray(target_mask, dtype=np.float32)[..., None]], axis=-1)
    raise ValueError(f"unsupported outcome type {outcome_type!r}")


def _make_neural_model(
    family: str,
    inputs: Any,
    *,
    outcome_type: str,
    n_outputs: int,
    max_horizon: int,
    config: dict[str, Any],
) -> Any:
    dropout = float(config.get("dropout", 0.3))
    if hasattr(inputs, "input_shapes"):
        shapes = dict(inputs.input_shapes)
    else:
        shapes = {name: tuple(value.shape[1:]) for name, value in inputs.items()}
    context_dim = int(shapes.get("global_context", (0,))[0])
    family = implementation_family(family)
    if family == "cnn":
        model = build_cnn(
            tuple(shapes["raster"]),
            outcome_type=outcome_type,
            n_outputs=n_outputs,
            max_horizon=max_horizon,
            dropout=dropout,
            context_dim=context_dim,
            output_config=config,
        )
    elif family in GLOBAL_SET_FAMILIES:
        model = build_transformer(
            tuple(shapes["player_tokens"]),
            outcome_type=outcome_type,
            n_outputs=n_outputs,
            max_horizon=max_horizon,
            dropout=dropout,
            d_model=int(config.get("d_model", 64)),
            heads=int(config.get("heads", 2)),
            ff_dim=int(config.get("ff_dim", 256)),
            layers_count=int(config.get("layers", 3)),
            context_dim=context_dim,
            output_config=config,
        )
    elif family in RELATIONAL_FAMILIES:
        model = build_relational_network(
            tuple(shapes["player_tokens"]),
            aggregation="fixed" if family == "relnet" else "attention",
            outcome_type=outcome_type,
            n_outputs=n_outputs,
            max_horizon=max_horizon,
            dropout=dropout,
            d_model=int(config.get("d_model", 48)),
            edge_dim=int(config.get("edge_dim", 8)),
            edge_hidden=int(config.get("edge_hidden", 64)),
            context_dim=context_dim,
            output_config=config,
        )
    else:
        raise ValueError("neural family is unsupported")
    model._bdb_output_contract = config.get("output_contract")
    model._bdb_cdf_baseline = config.get("cdf_baseline")
    return model


def _neural_input_signature(
    inputs: Any,
) -> tuple[int, dict[str, tuple[int | None, ...]]]:
    """Return example count and fixed model-input shapes for eager or lazy data."""

    if hasattr(inputs, "input_shapes"):
        count = len(inputs)
        shapes = {str(name): tuple(shape) for name, shape in inputs.input_shapes.items()}
    elif isinstance(inputs, dict) and inputs:
        counts = {len(np.asarray(value)) for value in inputs.values()}
        if len(counts) != 1:
            raise ValueError("neural input arrays are not example-aligned")
        count = counts.pop()
        shapes = {
            str(name): tuple(np.asarray(value).shape[1:])
            for name, value in inputs.items()
        }
    else:
        raise ValueError("neural inputs must be a non-empty mapping or lazy input source")
    if count <= 0 or not shapes:
        raise ValueError("neural inputs cannot be empty")
    return int(count), shapes


def _keras_batch_dataset(
    source: Any,
    *,
    targets: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    import tensorflow as tf

    if int(batch_size) <= 0:
        raise ValueError("neural batch size must be positive")
    count, _ = _neural_input_signature(source)
    if targets is not None and len(np.asarray(targets)) != count:
        raise ValueError("neural targets do not align with lazy inputs")

    class BatchDataset(tf.keras.utils.PyDataset):
        def __init__(self) -> None:
            super().__init__()
            self.order = np.arange(len(source), dtype=int)
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self) -> int:
            return int(np.ceil(len(self.order) / batch_size))

        def __getitem__(self, batch: int) -> Any:
            selected = self.order[batch * batch_size : (batch + 1) * batch_size]
            values = source.batch(selected)
            if targets is None:
                return values
            return values, targets[selected]

        def on_epoch_end(self) -> None:
            if shuffle:
                self.rng.shuffle(self.order)

    return BatchDataset()


def predict_neural(model: Any, inputs: Any, *, batch_size: int = 64) -> np.ndarray:
    if hasattr(inputs, "batch"):
        dataset = _keras_batch_dataset(
            inputs, targets=None, batch_size=batch_size, shuffle=False, seed=0
        )
        prediction = np.asarray(model.predict(dataset, verbose=0))
    else:
        prediction = np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0))
    if getattr(model, "_bdb_output_contract", None) == PUNT_CDF_RESIDUAL_CONTRACT:
        return punt_cdf_residual_to_pmf(
            prediction, np.asarray(model._bdb_cdf_baseline, dtype=np.float64)
        )
    return prediction


def select_neural_epoch_explicit_preprocessing(
    family: str,
    selector_train_inputs: Any,
    selector_train_y: np.ndarray,
    selector_validation_inputs: Any,
    selector_validation_y: np.ndarray,
    *,
    outcome_type: str,
    n_outputs: int,
    selector_train_mask: np.ndarray | None,
    selector_validation_mask: np.ndarray | None,
    config: dict[str, Any],
    selection_seed: int,
    validation_game_ids: tuple[str, ...],
    validation_split_seed: int,
) -> NeuralSelection:
    """Select the deterministic refit epoch without retaining learned weights."""

    import tensorflow as tf

    family = implementation_family(family)
    if family not in NEURAL_FAMILIES:
        raise ValueError("neural family is unsupported")
    learning_rate = float(config.get("learning_rate", 1e-3))
    max_epochs = int(config.get("max_epochs", 50))
    patience = int(config.get("patience", 10))
    batch_size = int(config.get("batch_size", 64))
    if max_epochs <= 0 or patience < 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("neural training configuration is invalid")
    max_horizon = (
        int(np.asarray(selector_train_y).shape[1])
        if outcome_type == "trajectory"
        else 1
    )
    selector_count, selector_shapes = _neural_input_signature(selector_train_inputs)
    validation_count, validation_shapes = _neural_input_signature(selector_validation_inputs)
    if selector_shapes != validation_shapes:
        raise ValueError("selector training and validation neural input shapes differ")
    if selector_count != len(np.asarray(selector_train_y)):
        raise ValueError("selector targets do not align with selector inputs")
    if validation_count != len(np.asarray(selector_validation_y)):
        raise ValueError("validation targets do not align with validation inputs")
    selector_target = _keras_targets(
        selector_train_y, outcome_type, n_outputs, selector_train_mask
    )
    selector_validation_target = _keras_targets(
        selector_validation_y, outcome_type, n_outputs, selector_validation_mask
    )
    configure_tensorflow_determinism(
        tf,
        selection_seed,
        deterministic=bool(config.get("deterministic", True)),
    )
    selector_config = dict(config)
    if (
        outcome_type == "distribution"
        and config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT
    ):
        selector_config["cdf_baseline"] = empirical_cdf_baseline(
            selector_train_y, n_outputs
        ).tolist()
    selector = _make_neural_model(
        family,
        selector_train_inputs,
        outcome_type=outcome_type,
        n_outputs=n_outputs,
        max_horizon=max_horizon,
        config=selector_config,
    )
    selector.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=_neural_loss(outcome_type, selector_config),
        jit_compile=False,
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
    )
    if hasattr(selector_train_inputs, "batch"):
        selector_training_data = _keras_batch_dataset(
            selector_train_inputs,
            targets=selector_target,
            batch_size=batch_size,
            shuffle=True,
            seed=selection_seed,
        )
        selector_validation_data = _keras_batch_dataset(
            selector_validation_inputs,
            targets=selector_validation_target,
            batch_size=batch_size,
            shuffle=False,
            seed=selection_seed,
        )
        selector_history = selector.fit(
            selector_training_data,
            validation_data=selector_validation_data,
            epochs=max_epochs,
            verbose=0,
            callbacks=[callback],
        ).history
    else:
        selector_history = selector.fit(
            selector_train_inputs,
            selector_target,
            validation_data=(selector_validation_inputs, selector_validation_target),
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[callback],
        ).history
    validation_losses = np.asarray(selector_history["val_loss"], dtype=float)
    if len(validation_losses) == 0 or not np.all(np.isfinite(validation_losses)):
        raise RuntimeError("neural epoch selection produced invalid validation losses")
    best_epoch = int(np.argmin(validation_losses)) + 1
    selector_parameter_count = int(selector.count_params())
    del selector
    tf.keras.backend.clear_session()
    return NeuralSelection(
        family=family,
        outcome_type=outcome_type,
        n_outputs=int(n_outputs),
        max_horizon=max_horizon,
        best_epoch=best_epoch,
        selector_history={
            key: [float(value) for value in values]
            for key, values in selector_history.items()
        },
        parameter_count=selector_parameter_count,
        input_shapes=selector_shapes,
        validation_game_ids=tuple(
            sorted(str(value) for value in validation_game_ids)
        ),
        validation_split_seed=int(validation_split_seed),
        selection_seed=int(selection_seed),
    )


def refit_neural_from_selection_explicit_preprocessing(
    family: str,
    final_inputs: Any,
    final_y: np.ndarray,
    *,
    outcome_type: str,
    n_outputs: int,
    final_mask: np.ndarray | None,
    config: dict[str, Any],
    refit_seed: int,
    selection: NeuralSelection | Mapping[str, Any],
) -> NeuralFit:
    """Validate a frozen selector result and perform only the final refit."""

    import tensorflow as tf

    selected = (
        selection
        if isinstance(selection, NeuralSelection)
        else NeuralSelection.from_mapping(selection)
    )
    family = implementation_family(family)
    max_horizon = (
        int(np.asarray(final_y).shape[1]) if outcome_type == "trajectory" else 1
    )
    final_count, final_shapes = _neural_input_signature(final_inputs)
    if final_count != len(np.asarray(final_y)):
        raise ValueError("refit targets do not align with refit inputs")
    if (
        selected.family != family
        or selected.outcome_type != outcome_type
        or selected.n_outputs != int(n_outputs)
        or selected.max_horizon != max_horizon
        or selected.input_shapes != final_shapes
    ):
        raise ValueError("neural selection is incompatible with the requested refit")
    learning_rate = float(config.get("learning_rate", 1e-3))
    batch_size = int(config.get("batch_size", 64))
    max_epochs = int(config.get("max_epochs", 50))
    observed_epochs = len(selected.selector_history["val_loss"])
    if (
        learning_rate <= 0.0
        or batch_size <= 0
        or max_epochs <= 0
        or observed_epochs > max_epochs
        or selected.best_epoch > max_epochs
    ):
        raise ValueError("neural refit configuration is invalid")
    final_target = _keras_targets(final_y, outcome_type, n_outputs, final_mask)
    final_config = dict(config)
    if (
        outcome_type == "distribution"
        and config.get("output_contract") == PUNT_CDF_RESIDUAL_CONTRACT
    ):
        final_config["cdf_baseline"] = empirical_cdf_baseline(
            final_y, n_outputs
        ).tolist()
    configure_tensorflow_determinism(
        tf,
        refit_seed,
        deterministic=bool(config.get("deterministic", True)),
    )
    final_model = _make_neural_model(
        family,
        final_inputs,
        outcome_type=outcome_type,
        n_outputs=n_outputs,
        max_horizon=max_horizon,
        config=final_config,
    )
    if int(final_model.count_params()) != selected.parameter_count:
        raise RuntimeError("selector and refit neural capacities differ")
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=_neural_loss(outcome_type, final_config),
        jit_compile=False,
    )
    if hasattr(final_inputs, "batch"):
        final_data = _keras_batch_dataset(
            final_inputs,
            targets=final_target,
            batch_size=batch_size,
            shuffle=True,
            seed=refit_seed,
        )
        refit_history = final_model.fit(
            final_data, epochs=selected.best_epoch, verbose=0
        ).history
    else:
        refit_history = final_model.fit(
            final_inputs,
            final_target,
            epochs=selected.best_epoch,
            batch_size=batch_size,
            verbose=0,
        ).history
    return NeuralFit(
        family=family,
        model=final_model,
        best_epoch=selected.best_epoch,
        selector_history=selected.selector_history,
        refit_history={key: [float(value) for value in values] for key, values in refit_history.items()},
        parameter_count=int(final_model.count_params()),
        validation_game_ids=selected.validation_game_ids,
        validation_split_seed=selected.validation_split_seed,
        fairness_metadata=(
            relational_fairness_metadata(
                family,
                final_shapes,
                final_config,
                int(final_model.count_params()),
            )
            if family in RELATIONAL_FAMILIES
            else {}
        ),
        complexity_metadata=(
            transformer_complexity_metadata(
                family,
                final_shapes,
                final_config,
                int(final_model.count_params()),
            )
            if family in GLOBAL_SET_FAMILIES
            else {}
        ),
    )


def fit_neural_explicit_preprocessing(
    family: str,
    selector_train_inputs: Any,
    selector_train_y: np.ndarray,
    selector_validation_inputs: Any,
    selector_validation_y: np.ndarray,
    final_inputs: Any,
    final_y: np.ndarray,
    *,
    outcome_type: str,
    n_outputs: int,
    selector_train_mask: np.ndarray | None,
    selector_validation_mask: np.ndarray | None,
    final_mask: np.ndarray | None,
    config: dict[str, Any],
    selection_seed: int,
    refit_seed: int,
    validation_game_ids: tuple[str, ...],
    validation_split_seed: int,
) -> NeuralFit:
    """Select an epoch with fold-fitted preprocessing, then refit on all n.

    This remains the default monolithic path.  The two explicit phase helpers
    are also used independently by resumable cluster execution.
    """

    selection = select_neural_epoch_explicit_preprocessing(
        family,
        selector_train_inputs,
        selector_train_y,
        selector_validation_inputs,
        selector_validation_y,
        outcome_type=outcome_type,
        n_outputs=n_outputs,
        selector_train_mask=selector_train_mask,
        selector_validation_mask=selector_validation_mask,
        config=config,
        selection_seed=selection_seed,
        validation_game_ids=validation_game_ids,
        validation_split_seed=validation_split_seed,
    )
    return refit_neural_from_selection_explicit_preprocessing(
        family,
        final_inputs,
        final_y,
        outcome_type=outcome_type,
        n_outputs=n_outputs,
        final_mask=final_mask,
        config=config,
        refit_seed=refit_seed,
        selection=selection,
    )


def fit_neural_with_epoch_refit(
    family: str,
    inputs: dict[str, np.ndarray],
    y: np.ndarray,
    game_ids: np.ndarray,
    *,
    outcome_type: str,
    n_outputs: int,
    target_mask: np.ndarray | None,
    config: dict[str, Any],
    validation_split_seed: int,
    selection_seed: int,
    refit_seed: int,
    strata: np.ndarray | None = None,
) -> NeuralFit:
    fit_index, validation_index = grouped_validation_indices(
        game_ids, validation_split_seed, strata=strata
    )
    return fit_neural_explicit_preprocessing(
        family,
        {name: values[fit_index] for name, values in inputs.items()},
        np.asarray(y)[fit_index],
        {name: values[validation_index] for name, values in inputs.items()},
        np.asarray(y)[validation_index],
        inputs,
        y,
        outcome_type=outcome_type,
        n_outputs=n_outputs,
        selector_train_mask=None if target_mask is None else np.asarray(target_mask)[fit_index],
        selector_validation_mask=None if target_mask is None else np.asarray(target_mask)[validation_index],
        final_mask=target_mask,
        config=config,
        selection_seed=selection_seed,
        refit_seed=refit_seed,
        validation_game_ids=tuple(
            sorted(str(value) for value in np.unique(np.asarray(game_ids)[validation_index]))
        ),
        validation_split_seed=validation_split_seed,
    )
