"""Frozen model specifications and fitting helpers for the rushing study."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import random
from typing import Any, Callable

import numpy as np


NUM_CLASSES = 80
MODEL_IDS = (
    "ridge_sgd_l2",
    "lightgbm_multiclass",
    "zoo_cnn",
    "set_transformer",
)
LEGACY_MODEL_ALIASES = {
    "ridge_multinomial": "ridge_sgd_l2",
    "tree_multiclass_hgbt": "lightgbm_multiclass",
    "zoo_cnn_distributional": "zoo_cnn",
    "set_transformer_distributional": "set_transformer",
}
MODEL_DISPLAY_NAMES = {
    "ridge_sgd_l2": "L2 one-vs-rest logistic",
    "lightgbm_multiclass": "LightGBM",
    "zoo_cnn": "Zoo CNN",
    "set_transformer": "Set Transformer",
}


def frozen_model_config(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Return one self-contained frozen model/training configuration."""
    canonical = canonical_model_id(model_id)
    reverse_alias = {value: key for key, value in LEGACY_MODEL_ALIASES.items()}
    raw = dict(config["models"].get(canonical, config["models"].get(reverse_alias.get(canonical, ""), {})))
    if not raw:
        raise KeyError(f"Configuration has no entry for {canonical}.")
    if canonical in {"zoo_cnn", "set_transformer"}:
        training = config.get("neural_training", {})
        raw.setdefault("learning_rate", training.get("learning_rate", 1e-3))
        raw.setdefault("batch_size", training.get("batch_size", 64))
        raw.setdefault("max_epochs", training.get("max_epochs", 50))
        raw.setdefault("patience", training.get("early_stopping_patience", 10))
    if canonical == "lightgbm_multiclass":
        raw.setdefault("n_jobs", config.get("execution", {}).get("lightgbm_n_jobs", 1))
    return raw


def sensitivity_candidates(config: dict[str, Any], model_id: str) -> list[dict[str, Any]]:
    """Build the locked, ordered sensitivity grid, with the main setting first."""
    canonical = canonical_model_id(model_id)
    main = frozen_model_config(config, canonical)
    configured = config.get("sensitivity", {}).get("grids", {}).get(canonical)
    if canonical == "ridge_sgd_l2" and isinstance(configured, dict) and "values" in configured:
        parameter = str(configured.get("parameter", "alpha"))
        candidates = [{**main, parameter: value} for value in configured["values"]]
    elif configured:
        candidates = [{**main, **overrides} for overrides in configured]
    elif canonical == "ridge_sgd_l2":
        candidates = [{**main, "alpha": value} for value in (1 / 3, 10.0, 10 / 3, 1.0, 0.1)]
    elif canonical == "lightgbm_multiclass":
        grid = (
            {"learning_rate": 0.05, "max_depth": 5, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 0.5},
            {"learning_rate": 0.05, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"learning_rate": 0.05, "max_depth": 7, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
            {"learning_rate": 0.1, "max_depth": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1},
        )
        candidates = [{**main, **overrides} for overrides in grid]
    else:
        candidates = [
            {**main, "learning_rate": learning_rate, "dropout": dropout}
            for learning_rate, dropout in ((1e-3, 0.3), (3e-4, 0.1), (3e-4, 0.3), (1e-3, 0.1))
        ]
    # Canonical JSON equality is not needed here: the explicitly first candidate
    # is the frozen main setting and is therefore the deterministic tie winner.
    return candidates


def canonical_model_id(model_id: str) -> str:
    """Return the canonical model identifier while accepting legacy result IDs."""
    canonical = LEGACY_MODEL_ALIASES.get(model_id, model_id)
    if canonical not in MODEL_IDS:
        raise ValueError(f"Unknown model id: {model_id}")
    return canonical


def make_tabular_features(spatial_tensor: np.ndarray) -> np.ndarray:
    """Create the frozen 40-feature min/mean/max/std representation."""
    flat = np.asarray(spatial_tensor).reshape(len(spatial_tensor), -1, spatial_tensor.shape[-1])
    return np.concatenate(
        [flat.min(axis=1), flat.mean(axis=1), flat.max(axis=1), flat.std(axis=1)],
        axis=1,
    )


def one_hot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if np.any((y < 0) | (y >= NUM_CLASSES)):
        raise ValueError("Class labels must be in [0, 79].")
    out = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def normalize_probabilities(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected probability shape (n, {NUM_CLASSES}); got {p.shape}.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def _expand_class_probabilities(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), NUM_CLASSES), dtype=np.float64)
    for col, cls in enumerate(np.asarray(classes, dtype=int)):
        if 0 <= cls < NUM_CLASSES:
            out[:, cls] = proba[:, col]
    return normalize_probabilities(out)


def fit_ovr_logistic(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int):
    """Fit the frozen manually-epochized SGD one-vs-rest logistic model."""
    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config["alpha"]),
        fit_intercept=True,
        max_iter=1,
        tol=None,
        shuffle=True,
        random_state=int(seed),
        learning_rate="optimal",
        eta0=float(config.get("eta0", 0.0)),
        average=False,
        warm_start=False,
    )
    rng = np.random.default_rng(seed)
    batch_size = int(config.get("batch_size", 64))
    classes = np.arange(NUM_CLASSES)
    first = True
    for _ in range(int(config.get("epochs", 50))):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if first:
                model.partial_fit(x[idx], y[idx], classes=classes)
                first = False
            else:
                model.partial_fit(x[idx], y[idx])
    return model


def fit_lightgbm(x: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int):
    """Fit the fully specified frozen LightGBM multiclass model."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        num_class=NUM_CLASSES,
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        num_leaves=int(config.get("num_leaves", 31)),
        min_child_samples=int(config["min_child_samples"]),
        min_child_weight=float(config.get("min_child_weight", 1e-3)),
        min_split_gain=float(config.get("min_split_gain", 0.0)),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        subsample=float(config.get("subsample", 1.0)),
        subsample_freq=int(config.get("subsample_freq", 0)),
        colsample_bytree=float(config.get("colsample_bytree", 1.0)),
        random_state=int(seed),
        n_jobs=int(config.get("n_jobs", 1)),
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )
    model.fit(x, y)
    return model


def predict_classical(model: Any, x: np.ndarray) -> np.ndarray:
    return _expand_class_probabilities(model.predict_proba(x), model.classes_)


def _set_neural_determinism(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # Determinism is part of the locked protocol.  The pinned TensorFlow build
    # must support this call; an incompatible runtime is a hard failure that
    # must be resolved before the definitive manifest is frozen.
    tf.config.experimental.enable_op_determinism()


def _neural_builder(model_id: str, config: dict[str, Any]) -> Callable[[], Any]:
    model_id = canonical_model_id(model_id)

    def build():
        from .neural_models import get_conv_net, get_set_transformer

        if model_id == "zoo_cnn":
            return get_conv_net(NUM_CLASSES, dropout=float(config["dropout"]))
        if model_id == "set_transformer":
            return get_set_transformer(
                NUM_CLASSES,
                d_model=int(config["d_model"]),
                num_layers=int(config["num_layers"]),
                num_heads=int(config["num_heads"]),
                ff_dim=int(config["ff_dim"]),
                dropout=float(config["dropout"]),
            )
        raise ValueError(f"{model_id} is not a neural model.")

    return build


@dataclass
class NeuralFit:
    model: Any
    best_epoch: int
    selection_history: dict[str, list[float]]
    refit_history: dict[str, list[float]]
    parameter_count: int


def fit_neural_select_and_refit(
    model_id: str,
    x: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    selection_seed: int,
    refit_seed: int,
) -> NeuralFit:
    """Select an epoch on game-disjoint data, then refit on every selected row."""
    import tensorflow as tf
    from .neural_models import crps

    if len(fit_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Neural epoch selection needs nonempty fit and validation rows.")
    build = _neural_builder(model_id, config)
    batch_size = int(config.get("batch_size", 64))
    max_epochs = int(config.get("max_epochs", 50))
    patience = int(config.get("patience", 10))
    learning_rate = float(config["learning_rate"])

    tf.keras.backend.clear_session()
    _set_neural_determinism(selection_seed)
    selector = build()
    parameter_count = int(selector.count_params())
    selector.compile(
        loss=crps,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        jit_compile=False,
    )
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )
    selected = selector.fit(
        x[fit_indices],
        one_hot(y[fit_indices]),
        validation_data=(x[validation_indices], one_hot(y[validation_indices])),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[stop],
        verbose=0,
    )
    val_loss = np.asarray(selected.history.get("val_loss", []), dtype=float)
    if len(val_loss) == 0 or not np.all(np.isfinite(val_loss)):
        raise RuntimeError("Neural epoch selection produced no finite validation CRPS.")
    best_epoch = int(np.argmin(val_loss)) + 1
    selection_history = {
        key: [float(value) for value in values]
        for key, values in selected.history.items()
    }
    del selector
    tf.keras.backend.clear_session()
    gc.collect()

    _set_neural_determinism(refit_seed)
    final_model = build()
    if int(final_model.count_params()) != parameter_count:
        raise RuntimeError("Neural model parameter count changed between selection and refit.")
    final_model.compile(
        loss=crps,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        jit_compile=False,
    )
    refit = final_model.fit(
        x,
        one_hot(y),
        epochs=best_epoch,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    refit_history = {
        key: [float(value) for value in values]
        for key, values in refit.history.items()
    }
    return NeuralFit(
        model=final_model,
        best_epoch=best_epoch,
        selection_history=selection_history,
        refit_history=refit_history,
        parameter_count=parameter_count,
    )


def predict_neural(model: Any, x: np.ndarray) -> np.ndarray:
    return normalize_probabilities(model.predict(x, batch_size=256, verbose=0))


def build_neural_with_weights(
    model_id: str,
    config: dict[str, Any],
    weights: list[np.ndarray],
    seed: int,
):
    """Rebuild a prediction-only neural model from retained candidate weights."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    _set_neural_determinism(seed)
    model = _neural_builder(model_id, config)()
    model.set_weights(weights)
    return model


def release_neural(model: Any) -> None:
    del model
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    finally:
        gc.collect()
