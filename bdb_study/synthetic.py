"""Fast two-size end-to-end exercise of suite storage and analysis contracts."""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import summarize_task
from .metrics import skill_score
from .storage import (
    CellKey,
    atomic_write_csv,
    atomic_write_json,
    bind_storage_backend,
    initialize_run_dir,
)


SYNTHETIC_TASK_ID = "synthetic_bdb_contract"
SYNTHETIC_MODELS = ("glm", "lightgbm", "cnn", "transformer")
SYNTHETIC_ANCHORS = (2, 4)
SYNTHETIC_REPEATS = 2


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest(repo_root: Path) -> dict[str, Any]:
    cells = [
        {
            "branch": "fixed_main",
            "repeat": repeat,
            "n_train": anchor,
            "model": model,
            "queue": "cpu_tabular" if model in {"glm", "lightgbm"} else "gpu_neural",
        }
        for repeat in range(1, SYNTHETIC_REPEATS + 1)
        for anchor in SYNTHETIC_ANCHORS
        for model in SYNTHETIC_MODELS
    ]
    payload = {
        "schema_version": "bdb-synthetic-study-v1",
        "task_id": SYNTHETIC_TASK_ID,
        "repeats": SYNTHETIC_REPEATS,
        "anchors": list(SYNTHETIC_ANCHORS),
        "models": list(SYNTHETIC_MODELS),
        "required_cells": cells,
        "purpose": "fast_two_size_storage_resume_analysis_contract",
    }
    payload = bind_storage_backend(payload, repo_root=repo_root)
    manifest_hash = _hash(payload)
    payload["manifest_hash"] = manifest_hash
    payload["run_hash"] = manifest_hash[:24]
    return payload


def _cell_artifacts(repeat: int, anchor: int, model: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    model_offset = {"glm": 0.030, "lightgbm": 0.020, "cnn": 0.010, "transformer": 0.0}[model]
    primary = 0.18 + model_offset - 0.004 * np.log2(anchor) + 0.001 * repeat
    null = 0.25 + 0.001 * repeat
    metrics = {
        "task_id": SYNTHETIC_TASK_ID,
        "branch": "fixed_main",
        "repeat": repeat,
        "n_train": anchor,
        "model": model,
        "family": model,
        "primary_loss": float(primary),
        "game_equal_loss": float(primary + 0.002),
        "null_loss": float(null),
        "skill": skill_score(float(primary), float(null)),
        "n_train_examples": anchor * 3,
        "n_test_examples": 4,
    }
    predictions = pd.DataFrame(
        {
            "example_id": [f"r{repeat}-test-{index}" for index in range(4)],
            "game_id": [f"g{index // 2}" for index in range(4)],
            "target": [0, 1, 0, 1],
            "prediction": np.asarray([0.2, 0.8, 0.3, 0.7]) + model_offset / 10.0,
        }
    )
    history = {
        "synthetic": True,
        "fit_games": anchor,
        "seed": int(20260817 + 100 * repeat + 10 * anchor + SYNTHETIC_MODELS.index(model)),
    }
    return metrics, predictions, history


def run_synthetic_study(
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Write, resume, validate, aggregate, and finalize a 16-cell toy study."""

    root = Path(repo_root).resolve()
    storage = initialize_run_dir(
        output_dir, _manifest(root), repo_root=root
    )
    existing_receipt = storage.run_dir / "final" / "receipt.json"
    if storage.validate_final() and existing_receipt.is_file():
        return json.loads(existing_receipt.read_text(encoding="utf-8"))
    keys: list[CellKey] = []
    for repeat in range(1, SYNTHETIC_REPEATS + 1):
        for anchor in SYNTHETIC_ANCHORS:
            for model in SYNTHETIC_MODELS:
                key = CellKey("fixed_main", repeat, anchor, model)
                keys.append(key)
                if storage.validate_cell(key).is_complete:
                    continue
                metrics, predictions, history = _cell_artifacts(repeat, anchor, model)
                storage.write_cell_artifacts(
                    key,
                    metrics=metrics,
                    predictions=predictions,
                    history=history,
                    preferred_predictions="csv",
                )
                if not storage.validate_cell(key).is_complete:
                    raise RuntimeError(f"synthetic cell failed its checksum gate: {key}")
    rows = pd.DataFrame([storage.load_metrics(key) for key in keys])
    expected_grid = {
        (repeat, anchor, model)
        for repeat in range(1, SYNTHETIC_REPEATS + 1)
        for anchor in SYNTHETIC_ANCHORS
        for model in SYNTHETIC_MODELS
    }
    observed_grid = set(
        zip(rows["repeat"].astype(int), rows["n_train"].astype(int), rows["model"])
    )
    if len(rows) != len(expected_grid) or observed_grid != expected_grid:
        raise RuntimeError("synthetic two-size result grid is incomplete")
    rows = rows.sort_values(["repeat", "n_train", "model"]).reset_index(drop=True)
    summary = summarize_task(rows)
    contrast_rows = []
    for anchor in SYNTHETIC_ANCHORS:
        pivot = rows[rows["n_train"] == anchor].pivot(
            index="repeat", columns="model", values="primary_loss"
        )
        for left, right in combinations(SYNTHETIC_MODELS, 2):
            difference = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            contrast_rows.append(
                {
                    "n_train": anchor,
                    "model_left": left,
                    "model_right": right,
                    "mean_difference": float(difference.mean()),
                    "difference_sd": float(difference.std(ddof=1)),
                    "inference": "synthetic_contract_only",
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    stability = contrasts.copy()
    final = storage.run_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    records = {
        "metrics": atomic_write_csv(final / "metrics.csv", rows),
        "summary": atomic_write_csv(final / "summary.csv", summary),
        "paired_contrasts": atomic_write_csv(final / "paired_contrasts.csv", contrasts),
        "paired_stability": atomic_write_csv(final / "paired_stability.csv", stability),
    }
    receipt = {
        "schema_version": "bdb-synthetic-study-receipt-v1",
        "manifest_hash": storage.manifest_hash,
        "cells": len(rows),
        "summary_groups": len(summary),
        "artifacts": {name: record.as_dict() for name, record in records.items()},
    }
    receipt_record = atomic_write_json(final / "receipt.json", receipt)
    if not storage.validate_final():
        storage.finalize_run(
            keys,
            final_artifacts=[final / record.file for record in records.values()]
            + [final / receipt_record.file],
        )
    return receipt
