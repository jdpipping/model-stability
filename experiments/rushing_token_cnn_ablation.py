#!/usr/bin/env python3
"""Exploratory player-token CNN ablation against one frozen rushing split.

This script deliberately writes outside the immutable confirmatory run. It
matches the Set Transformer's 22x10 input, parameter scale, split, loss,
dropout, epoch-selection rule, refit rule, calibration, and test evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rushing_study.neural_models import crps
from rushing_study.data import load_study_data
from rushing_study.metrics import evaluate_cell
from rushing_study.models import normalize_probabilities
from rushing_study.runner import _games, _manifest_seed, _neural_inner_indices


DEFAULT_RUN = ROOT / "results" / "rushing" / "full100-confirmatory-20260820"
DEFAULT_OUTPUT = ROOT / "results" / "rushing" / "exploratory-token-cnn"


def build_token_cnn(dropout: float = 0.3) -> Model:
    """Near-parameter-matched 1D CNN over the exact 22x10 player tokens."""

    inputs = Input(shape=(22, 10), name="player_tokens_input")
    x = BatchNormalization(axis=-1, name="input_bn")(inputs)
    x = Conv1D(128, 1, padding="same", activation="relu", name="token_conv")(x)
    x = BatchNormalization(name="token_bn")(x)
    x = Conv1D(160, 3, padding="same", activation="relu", name="local_conv_1")(x)
    x = BatchNormalization(name="local_bn_1")(x)
    x = Conv1D(128, 3, padding="same", activation="relu", name="local_conv_2")(x)
    x = BatchNormalization(name="local_bn_2")(x)
    pooled = Concatenate(name="global_pool")(
        [GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)]
    )
    x = Dense(96, activation="relu", name="dense")(pooled)
    x = LayerNormalization(name="dense_ln")(x)
    x = Dropout(dropout, name="dropout")(x)
    outputs = Dense(80, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="matched_player_token_cnn")


def _repeat_split(manifest: dict, repeat_id: int) -> dict:
    matches = [
        split
        for split in manifest["split_manifests"]
        if int(split["repeat_id"]) == repeat_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one split for repeat {repeat_id}; found {len(matches)}")
    return matches[0]


def _anchor(split: dict, n_train: int) -> dict:
    anchors = split["anchors"]
    record = anchors.get(str(n_train), anchors.get(n_train))
    if not isinstance(record, dict):
        raise ValueError(f"Split has no training anchor {n_train}")
    return record


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    fit_local: np.ndarray,
    validation_local: np.ndarray,
    *,
    selection_seed: int,
    refit_seed: int,
    learning_rate: float,
    dropout: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[Model, int, dict[str, list[float]], dict[str, list[float]]]:
    tf.keras.backend.clear_session()
    _set_seed(selection_seed)
    selector = build_token_cnn(dropout)
    selector.compile(loss=crps, optimizer=Adam(learning_rate=learning_rate))
    stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )
    selected = selector.fit(
        x[fit_local],
        tf.one_hot(y[fit_local], 80),
        validation_data=(x[validation_local], tf.one_hot(y[validation_local], 80)),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[stop],
        verbose=2,
    )
    val_loss = np.asarray(selected.history["val_loss"], dtype=float)
    if not np.isfinite(val_loss).all():
        raise RuntimeError("Epoch selection produced non-finite validation CRPS")
    best_epoch = int(np.argmin(val_loss)) + 1
    selection_history = {
        key: [float(value) for value in values]
        for key, values in selected.history.items()
    }

    del selector
    tf.keras.backend.clear_session()
    _set_seed(refit_seed)
    final_model = build_token_cnn(dropout)
    final_model.compile(loss=crps, optimizer=Adam(learning_rate=learning_rate))
    refit = final_model.fit(
        x,
        tf.one_hot(y, 80),
        epochs=best_epoch,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
    )
    refit_history = {
        key: [float(value) for value in values]
        for key, values in refit.history.items()
    }
    return final_model, best_epoch, selection_history, refit_history


def _saved_comparators(run_dir: Path, repeat_id: int, n_train: int) -> list[dict]:
    rows = []
    cell_root = (
        run_dir
        / "cells"
        / "main"
        / f"repeat_{repeat_id:03d}"
        / f"n_{n_train:04d}"
    )
    for path in sorted(cell_root.glob("*/metrics.json")):
        row = json.loads(path.read_text())
        rows.append(
            {
                "model": row["model"],
                "model_display_name": row["model_display_name"],
                "parameter_count": int(row["parameter_count"]),
                "crps": float(row["crps"]),
                "coverage": float(row["coverage"]),
                "mean_width": float(row["mean_width"]),
                "mean_q": float(row["mean_q"]),
            }
        )
    if len(rows) != 4:
        raise ValueError(f"Expected four saved comparators; found {len(rows)}")
    return rows


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data = load_study_data(manifest["config"], mmap_mode="r")
    split = _repeat_split(manifest, args.repeat_id)
    anchor = _anchor(split, args.n_train)
    train_indices = data.indices_for_games(_games(anchor, "train_games", "games"))
    calibration_indices = data.indices_for_games(_games(split, "calibration_games"))
    test_indices = data.indices_for_games(_games(split, "test_games"))
    if data.metadata.iloc[train_indices]["game_id"].nunique() != args.n_train:
        raise ValueError("Training split does not contain the requested number of games")
    fit_local, validation_local = _neural_inner_indices(data, train_indices, anchor)

    config = manifest["config"]["models"]["set_transformer"]
    training = manifest["config"]["neural_training"]
    selection_seed = _manifest_seed(
        manifest,
        "main",
        args.repeat_id,
        args.n_train,
        "set_transformer",
        "epoch_selection",
    )
    refit_seed = _manifest_seed(
        manifest,
        "main",
        args.repeat_id,
        args.n_train,
        "set_transformer",
        "refit",
    )
    started = time.perf_counter()
    model, best_epoch, selection_history, refit_history = _fit(
        data.player_set[train_indices],
        data.y[train_indices],
        fit_local,
        validation_local,
        selection_seed=selection_seed,
        refit_seed=refit_seed,
        learning_rate=float(training["learning_rate"]),
        dropout=float(config["dropout"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["early_stopping_patience"]),
    )
    cal_proba = normalize_probabilities(
        model.predict(data.player_set[calibration_indices], batch_size=256, verbose=0)
    )
    test_proba = normalize_probabilities(
        model.predict(data.player_set[test_indices], batch_size=256, verbose=0)
    )
    evaluation = evaluate_cell(
        cal_proba,
        test_proba,
        data.y[calibration_indices],
        data.y[test_indices],
        data.metadata.iloc[calibration_indices],
        data.metadata.iloc[test_indices],
        alpha=float(manifest["config"]["uncertainty"]["alpha"]),
        local_k=int(manifest["config"]["uncertainty"]["local_k"]),
    )
    elapsed_seconds = time.perf_counter() - started
    parameter_count = int(model.count_params())
    transformer_parameter_count = int(config["expected_parameters"])
    metrics = {
        **evaluation.metrics,
        "model": "matched_player_token_cnn",
        "model_display_name": "Matched player-token CNN",
        "repeat": args.repeat_id,
        "n_train": args.n_train,
        "n_train_plays": int(len(train_indices)),
        "parameter_count": parameter_count,
        "transformer_parameter_count": transformer_parameter_count,
        "parameter_ratio_to_transformer": parameter_count / transformer_parameter_count,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed_seconds,
    }
    payload = {
        "status": "exploratory_nonconfirmatory_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_hash": manifest["manifest_hash"],
        "source_split_hash": split["split_hash"],
        "hardware": {
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "devices": [device.name for device in tf.config.list_physical_devices()],
        },
        "protocol": {
            "input": "same 22x10 player tokens as Set Transformer",
            "architecture": "Conv1D(128,k1)-Conv1D(160,k3)-Conv1D(128,k3), global mean+max pooling, Dense(96)",
            "learning_rate": float(training["learning_rate"]),
            "dropout": float(config["dropout"]),
            "batch_size": int(training["batch_size"]),
            "max_epochs": int(training["max_epochs"]),
            "patience": int(training["early_stopping_patience"]),
            "selection_seed_reused_from": "set_transformer",
            "selection_seed": selection_seed,
            "refit_seed": refit_seed,
        },
        "metrics": metrics,
        "comparators_same_split": _saved_comparators(run_dir, args.repeat_id, args.n_train),
        "history": {
            "selection": selection_history,
            "refit": refit_history,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"repeat_{args.repeat_id:03d}_n_{args.n_train:04d}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=20)
    args = parser.parse_args()
    output = run(args)
    payload = json.loads(output.read_text())
    print(json.dumps({"output": str(output), **payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
