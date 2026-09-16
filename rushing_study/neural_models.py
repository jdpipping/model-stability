"""Shared Keras models and utilities for rushing-yards and sack-probability training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import (
    Add,
    AvgPool1D,
    AvgPool2D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
    MaxPooling2D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


SET_TRANSFORMER_D_MODEL = 64
SET_TRANSFORMER_NUM_LAYERS = 3
SET_TRANSFORMER_NUM_HEADS = 2
SET_TRANSFORMER_FF_DIM = 256
SET_TRANSFORMER_DROPOUT = 0.3
SET_TRANSFORMER_EXPECTED_PARAMS = 157_416


def crps(y_true, y_pred):
    """CRPS for ordinal outcome: L2 on cumulative distributions (scaled by 199)."""
    return K.mean(K.sum((K.cumsum(y_pred, axis=1) - K.cumsum(y_true, axis=1)) ** 2, axis=1)) / 199


def get_conv_net(num_classes_y: int, dropout: float = 0.3) -> Model:
    """Zoo CNN: 11x10x10 input -> conv/pool -> softmax over yard bins."""
    inp = Input(shape=(11, 10, 10), name="playersfeatures_input")
    x = Conv2D(128, (1, 1), activation="relu")(inp)
    x = Conv2D(160, (1, 1), activation="relu")(x)
    x = Conv2D(128, (1, 1), activation="relu")(x)
    xmax = Lambda(lambda y: y * 0.3)(MaxPooling2D((1, 10))(x))
    xavg = Lambda(lambda y: y * 0.7)(AvgPool2D((1, 10))(x))
    x = Add()([xmax, xavg])
    x = Lambda(lambda y: K.squeeze(y, 2))(x)
    x = BatchNormalization()(x)
    x = Conv1D(160, 1, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, 1, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv1D(96, 1, activation="relu")(x)
    x = BatchNormalization()(x)
    xmax = Lambda(lambda y: y * 0.3)(MaxPooling1D(11)(x))
    xavg = Lambda(lambda y: y * 0.7)(AvgPool1D(11)(x))
    x = Add()([xmax, xavg])
    x = Lambda(lambda y: K.squeeze(y, 1))(x)
    x = Dense(96, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    out = Dense(num_classes_y, activation="softmax", name="output")(x)
    return Model(inputs=inp, outputs=out)


def get_set_transformer(
    num_classes_y: int,
    feature_len: int = 10,
    d_model: int = SET_TRANSFORMER_D_MODEL,
    num_layers: int = SET_TRANSFORMER_NUM_LAYERS,
    num_heads: int = SET_TRANSFORMER_NUM_HEADS,
    ff_dim: int = SET_TRANSFORMER_FF_DIM,
    dropout: float = SET_TRANSFORMER_DROPOUT,
) -> Model:
    """Single-frame player-set transformer for yardage distribution prediction."""
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

    inp = Input(shape=(22, feature_len), name="player_tokens_input")
    x = BatchNormalization(axis=-1)(inp)
    x = Dense(d_model, activation="relu")(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)

    for layer_idx in range(num_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"self_attention_{layer_idx + 1}",
        )(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)

        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dropout(dropout)(ff)
        ff = Dense(d_model)(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(d_model, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    x = LayerNormalization()(x)
    out = Dense(num_classes_y, activation="softmax", name="output")(x)
    return Model(inputs=inp, outputs=out)


def _encode_set_tokens(
    inp,
    d_model: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    prefix: str,
):
    """Encode a small set of tokens with self-attention and pooled readout."""
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

    x = BatchNormalization(axis=-1, name=f"{prefix}_bn")(inp)
    x = Dense(d_model, activation="relu", name=f"{prefix}_proj")(x)
    x = LayerNormalization(name=f"{prefix}_proj_ln")(x)
    x = Dropout(dropout, name=f"{prefix}_proj_dropout")(x)

    for layer_idx in range(num_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"{prefix}_self_attention_{layer_idx + 1}",
        )(x, x)
        x = Add(name=f"{prefix}_attn_resid_{layer_idx + 1}")([x, attn])
        x = LayerNormalization(name=f"{prefix}_attn_ln_{layer_idx + 1}")(x)

        ff = Dense(ff_dim, activation="relu", name=f"{prefix}_ff_1_{layer_idx + 1}")(x)
        ff = Dropout(dropout, name=f"{prefix}_ff_dropout_{layer_idx + 1}")(ff)
        ff = Dense(d_model, name=f"{prefix}_ff_2_{layer_idx + 1}")(ff)
        x = Add(name=f"{prefix}_ff_resid_{layer_idx + 1}")([x, ff])
        x = LayerNormalization(name=f"{prefix}_ff_ln_{layer_idx + 1}")(x)

    x = GlobalAveragePooling1D(name=f"{prefix}_pool")(x)
    x = Dense(d_model, activation="relu", name=f"{prefix}_dense")(x)
    x = Dropout(dropout, name=f"{prefix}_dense_dropout")(x)
    return x


def _encode_pair_grid(inp, prefix: str, dropout: float) -> tf.Tensor:
    """Encode a padded pairwise interaction grid with simple 1x1 convs and global pooling."""
    x = BatchNormalization(axis=-1, name=f"{prefix}_bn")(inp)
    x = Conv2D(64, (1, 1), activation="relu", name=f"{prefix}_conv1")(x)
    x = Conv2D(96, (1, 1), activation="relu", name=f"{prefix}_conv2")(x)
    xmax = GlobalMaxPooling2D(name=f"{prefix}_gmax")(x)
    xavg = GlobalAveragePooling2D(name=f"{prefix}_gavg")(x)
    x = Concatenate(name=f"{prefix}_pooled")([xmax, xavg])
    x = Dense(96, activation="relu", name=f"{prefix}_dense")(x)
    x = LayerNormalization(name=f"{prefix}_ln")(x)
    x = Dropout(dropout, name=f"{prefix}_dropout")(x)
    return x


def get_sack_multiview_model(
    player_feature_len: int,
    rush_block_feature_len: int,
    coverage_feature_len: int,
    context_feature_len: int,
    d_model: int = SET_TRANSFORMER_D_MODEL,
    num_layers: int = 2,
    num_heads: int = SET_TRANSFORMER_NUM_HEADS,
    ff_dim: int = 128,
    dropout: float = 0.2,
    output_bias: float | None = None,
) -> Model:
    """Binary sack-probability model from pairwise pass-protection, coverage, player-set, and context views."""
    player_inp = Input(shape=(22, player_feature_len), name="player_tokens_input")
    rush_block_inp = Input(shape=(8, 9, rush_block_feature_len), name="rush_block_pairs_input")
    coverage_inp = Input(shape=(11, 5, coverage_feature_len), name="coverage_route_pairs_input")
    context_inp = Input(shape=(context_feature_len,), name="play_context_input")

    pair_branch = _encode_pair_grid(rush_block_inp, prefix="rush_block", dropout=dropout)
    coverage_branch = _encode_pair_grid(coverage_inp, prefix="coverage_route", dropout=dropout)
    set_branch = _encode_set_tokens(
        player_inp,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        prefix="player_tokens",
    )

    ctx = BatchNormalization(name="context_bn")(context_inp)
    ctx = Dense(64, activation="relu", name="context_dense1")(ctx)
    ctx = LayerNormalization(name="context_ln1")(ctx)
    ctx = Dropout(dropout, name="context_dropout1")(ctx)
    ctx = Dense(32, activation="relu", name="context_dense2")(ctx)

    x = Concatenate(name="fusion")([pair_branch, coverage_branch, set_branch, ctx])
    x = Dense(160, activation="relu", name="fusion_dense1")(x)
    x = LayerNormalization(name="fusion_ln1")(x)
    x = Dropout(dropout, name="fusion_dropout1")(x)
    x = Dense(64, activation="relu", name="fusion_dense2")(x)
    x = LayerNormalization(name="fusion_ln2")(x)
    x = Dropout(dropout, name="fusion_dropout2")(x)

    bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
    out = Dense(1, activation="sigmoid", bias_initializer=bias_initializer, name="output")(x)
    return Model(
        inputs={
            "player_tokens_input": player_inp,
            "rush_block_pairs_input": rush_block_inp,
            "coverage_route_pairs_input": coverage_inp,
            "play_context_input": context_inp,
        },
        outputs=out,
    )


def get_binary_set_transformer(
    player_feature_len: int,
    d_model: int = SET_TRANSFORMER_D_MODEL,
    num_layers: int = 2,
    num_heads: int = SET_TRANSFORMER_NUM_HEADS,
    ff_dim: int = 128,
    dropout: float = 0.2,
    output_bias: float | None = None,
) -> Model:
    """Binary sack-probability model using only the 22-player QB-centered token set."""
    player_inp = Input(shape=(22, player_feature_len), name="player_tokens_input")
    x = _encode_set_tokens(
        player_inp,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        prefix="binary_player_tokens",
    )
    x = Dense(64, activation="relu", name="binary_set_dense1")(x)
    x = LayerNormalization(name="binary_set_ln1")(x)
    x = Dropout(dropout, name="binary_set_dropout1")(x)
    x = Dense(16, activation="relu", name="binary_set_dense2")(x)

    bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
    out = Dense(1, activation="sigmoid", bias_initializer=bias_initializer, name="output")(x)
    return Model(inputs=player_inp, outputs=out)


def get_set_transformer_context_binary(
    player_feature_len: int,
    context_feature_len: int,
    d_model: int = SET_TRANSFORMER_D_MODEL,
    num_layers: int = 2,
    num_heads: int = SET_TRANSFORMER_NUM_HEADS,
    ff_dim: int = 128,
    dropout: float = 0.2,
    output_bias: float | None = None,
) -> Model:
    """Binary sack model from player-token self-attention plus play context."""
    player_inp = Input(shape=(22, player_feature_len), name="player_tokens_input")
    context_inp = Input(shape=(context_feature_len,), name="play_context_input")

    set_branch = _encode_set_tokens(
        player_inp,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        prefix="set_ctx_player_tokens",
    )

    ctx = BatchNormalization(name="set_ctx_context_bn")(context_inp)
    ctx = Dense(64, activation="relu", name="set_ctx_context_dense1")(ctx)
    ctx = LayerNormalization(name="set_ctx_context_ln1")(ctx)
    ctx = Dropout(dropout, name="set_ctx_context_dropout1")(ctx)
    ctx = Dense(32, activation="relu", name="set_ctx_context_dense2")(ctx)

    x = Concatenate(name="set_ctx_fusion")([set_branch, ctx])
    x = Dense(96, activation="relu", name="set_ctx_fusion_dense1")(x)
    x = LayerNormalization(name="set_ctx_fusion_ln1")(x)
    x = Dropout(dropout, name="set_ctx_fusion_dropout1")(x)
    x = Dense(24, activation="relu", name="set_ctx_fusion_dense2")(x)

    bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
    out = Dense(1, activation="sigmoid", bias_initializer=bias_initializer, name="output")(x)
    return Model(
        inputs={
            "player_tokens_input": player_inp,
            "play_context_input": context_inp,
        },
        outputs=out,
    )


def get_pair_grid_cnn_context_binary(
    rush_block_feature_len: int,
    coverage_feature_len: int,
    context_feature_len: int,
    dropout: float = 0.2,
    output_bias: float | None = None,
) -> Model:
    """Binary sack CNN over pair grids with an additive play-context branch."""
    rush_block_inp = Input(shape=(8, 9, rush_block_feature_len), name="rush_block_pairs_input")
    coverage_inp = Input(shape=(11, 5, coverage_feature_len), name="coverage_route_pairs_input")
    context_inp = Input(shape=(context_feature_len,), name="play_context_input")

    rush_branch = _encode_pair_grid(rush_block_inp, prefix="pair_ctx_rush_block", dropout=dropout)
    coverage_branch = _encode_pair_grid(coverage_inp, prefix="pair_ctx_coverage_route", dropout=dropout)

    ctx = BatchNormalization(name="pair_ctx_context_bn")(context_inp)
    ctx = Dense(64, activation="relu", name="pair_ctx_context_dense1")(ctx)
    ctx = LayerNormalization(name="pair_ctx_context_ln1")(ctx)
    ctx = Dropout(dropout, name="pair_ctx_context_dropout1")(ctx)
    ctx = Dense(32, activation="relu", name="pair_ctx_context_dense2")(ctx)

    x = Concatenate(name="pair_ctx_fusion")([rush_branch, coverage_branch, ctx])
    x = Dense(128, activation="relu", name="pair_ctx_fusion_dense1")(x)
    x = LayerNormalization(name="pair_ctx_fusion_ln1")(x)
    x = Dropout(dropout, name="pair_ctx_fusion_dropout1")(x)
    x = Dense(32, activation="relu", name="pair_ctx_fusion_dense2")(x)
    x = LayerNormalization(name="pair_ctx_fusion_ln2")(x)
    x = Dropout(dropout, name="pair_ctx_fusion_dropout2")(x)

    bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
    out = Dense(1, activation="sigmoid", bias_initializer=bias_initializer, name="output")(x)
    return Model(
        inputs={
            "rush_block_pairs_input": rush_block_inp,
            "coverage_route_pairs_input": coverage_inp,
            "play_context_input": context_inp,
        },
        outputs=out,
    )


def get_sack_pair_cnn_model(
    rush_block_feature_len: int,
    coverage_feature_len: int,
    dropout: float = 0.2,
    output_bias: float | None = None,
) -> Model:
    """Binary sack-probability CNN over protection and coverage pair grids."""
    rush_block_inp = Input(shape=(8, 9, rush_block_feature_len), name="rush_block_pairs_input")
    coverage_inp = Input(shape=(11, 5, coverage_feature_len), name="coverage_route_pairs_input")

    rush_branch = _encode_pair_grid(rush_block_inp, prefix="pair_cnn_rush_block", dropout=dropout)
    coverage_branch = _encode_pair_grid(coverage_inp, prefix="pair_cnn_coverage_route", dropout=dropout)

    x = Concatenate(name="pair_cnn_fusion")([rush_branch, coverage_branch])
    x = Dense(128, activation="relu", name="pair_cnn_dense1")(x)
    x = LayerNormalization(name="pair_cnn_ln1")(x)
    x = Dropout(dropout, name="pair_cnn_dropout1")(x)
    x = Dense(32, activation="relu", name="pair_cnn_dense2")(x)
    x = LayerNormalization(name="pair_cnn_ln2")(x)
    x = Dropout(dropout, name="pair_cnn_dropout2")(x)

    bias_initializer = tf.keras.initializers.Constant(output_bias) if output_bias is not None else "zeros"
    out = Dense(1, activation="sigmoid", bias_initializer=bias_initializer, name="output")(x)
    return Model(
        inputs={
            "rush_block_pairs_input": rush_block_inp,
            "coverage_route_pairs_input": coverage_inp,
        },
        outputs=out,
    )


class Metric(Callback):
    """Compute val_CRPS each epoch so EarlyStopping can monitor it."""

    def __init__(self, model: Model, callbacks: list[Callback], data: list[np.ndarray]):
        super().__init__()
        self._predict_model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        x_valid, y_valid = self.data[0], self.data[1]
        y_pred = self._predict_model.predict(x_valid, verbose=0)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        logs["val_CRPS"] = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * x_valid.shape[0])
        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


def fit_distribution_model(
    model: Model,
    x_train: np.ndarray,
    y_train_oh: np.ndarray,
    x_val: np.ndarray,
    y_val_oh: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Model:
    """Train a Keras distribution model with CRPS early stopping."""
    es = EarlyStopping(monitor="val_CRPS", mode="min", restore_best_weights=True, verbose=0, patience=10)
    es.set_model(model)
    metric = Metric(model, [es], [x_val, y_val_oh])
    model.compile(
        loss=crps,
        optimizer=Adam(learning_rate=learning_rate),
        jit_compile=False,
    )
    model.fit(
        x_train,
        y_train_oh,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[metric],
        validation_data=(x_val, y_val_oh),
    )
    return model


def fit_binary_model(
    model: Model,
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Model:
    """Train a binary model with BCE/log loss early stopping."""
    es = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, verbose=0, patience=10)
    y_train = np.asarray(y_train, dtype="float32").reshape(-1, 1)
    y_val = np.asarray(y_val, dtype="float32").reshape(-1, 1)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        jit_compile=False,
        metrics=[
            tf.keras.metrics.BinaryCrossentropy(name="log_loss"),
            tf.keras.metrics.MeanSquaredError(name="brier"),
        ],
    )
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es],
        validation_data=(x_val, y_val),
    )
    return model


def build_grouped_play_folds(train_x: np.ndarray, train_y: pd.DataFrame, n_splits: int = 8):
    """Group folds by base PlayId so original and mirrored _aug rows stay in the same fold."""
    if "PlayId" not in train_y.columns:
        raise ValueError("train_y must include PlayId for grouped CV.")

    groups = train_y["PlayId"].astype(str).str.replace("_aug", "", regex=False).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    fold_splits = list(gkf.split(train_x, train_y, groups=groups))

    for i, (tdx, vdx) in enumerate(fold_splits):
        overlap = np.intersect1d(groups[tdx], groups[vdx], assume_unique=False)
        if overlap.size > 0:
            raise RuntimeError(f"Group leakage detected in fold {i}: {overlap.size} overlapping base PlayIds.")

    return fold_splits, groups
