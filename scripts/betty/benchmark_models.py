"""Benchmark the project's actual frozen model implementations on Betty."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np


# Support direct execution as well as the wrappers' configured PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware() -> dict[str, Any]:
    value: dict[str, Any] = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "packages": {
            name: _version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "tensorflow",
                "keras",
                "venn-abers",
            )
        },
    }
    try:
        import tensorflow as tf

        value["tensorflow_build"] = tf.sysconfig.get_build_info()
        value["gpus"] = []
        for item in tf.config.list_physical_devices("GPU"):
            details = tf.config.experimental.get_device_details(item)
            value["gpus"].append(
                {
                    "name": details.get("device_name", item.name),
                    "logical_name": item.name,
                    "device_type": item.device_type,
                    "compute_capability": details.get("compute_capability"),
                }
            )
    except Exception as exc:  # pragma: no cover - diagnostic path
        value["tensorflow_error"] = type(exc).__name__
    return value


def _cpu_benchmark(examples: int, seed: int) -> dict[str, Any]:
    from rushing_study.models import fit_lightgbm, fit_ovr_logistic

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(examples, 40)).astype(np.float32)
    target = rng.integers(0, 80, size=examples, dtype=np.int64)
    configurations = {
        "ridge_sgd_l2": {
            "alpha": 1.0 / 3.0,
            "eta0": 0.01,
            "epochs": 50,
            "batch_size": 64,
        },
        "lightgbm_multiclass": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 50,
            "min_child_weight": 0.001,
            "min_split_gain": 0.0,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "n_jobs": 1,
        },
    }
    timings: dict[str, float] = {}
    started = time.perf_counter()
    fit_ovr_logistic(features, target, configurations["ridge_sgd_l2"], seed)
    timings["ridge_sgd_l2"] = time.perf_counter() - started
    started = time.perf_counter()
    fit_lightgbm(features, target, configurations["lightgbm_multiclass"], seed)
    timings["lightgbm_multiclass"] = time.perf_counter() - started
    return {"examples": examples, "model_seconds": timings}


def _gpu_benchmark(examples: int, epochs: int, seed: int) -> dict[str, Any]:
    from rushing_study.models import fit_neural_select_and_refit

    rng = np.random.default_rng(seed)
    target = rng.integers(0, 80, size=examples, dtype=np.int64)
    fit_count = max(1, int(examples * 0.8))
    fit_index = np.arange(fit_count)
    validation_index = np.arange(fit_count, examples)
    common = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": epochs,
        "patience": 0,
        "dropout": 0.3,
    }
    cases = {
        "zoo_cnn": (
            rng.normal(size=(examples, 11, 10, 10)).astype(np.float32),
            dict(common),
        ),
        "set_transformer": (
            rng.normal(size=(examples, 22, 10)).astype(np.float32),
            {
                **common,
                "d_model": 64,
                "num_layers": 3,
                "num_heads": 2,
                "ff_dim": 256,
            },
        ),
    }
    timings: dict[str, float] = {}
    parameters: dict[str, int] = {}
    for offset, (model_id, (features, config)) in enumerate(cases.items()):
        started = time.perf_counter()
        fitted = fit_neural_select_and_refit(
            model_id,
            features,
            target,
            fit_index,
            validation_index,
            config,
            seed + 10 * offset,
            seed + 10 * offset + 1,
        )
        timings[model_id] = time.perf_counter() - started
        parameters[model_id] = int(fitted.parameter_count)
    return {
        "examples": examples,
        "selection_epochs": epochs,
        "refit_epochs": epochs,
        "model_seconds": timings,
        "parameters": parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260817)
    arguments = parser.parse_args()
    if arguments.examples < 100:
        parser.error("--examples must be at least 100")
    if arguments.epochs < 1:
        parser.error("--epochs must be positive")

    output = {
        "schema_version": "zoo-betty-benchmark-v1",
        "device": arguments.device,
        "hardware": _hardware(),
        "benchmark": (
            _cpu_benchmark(arguments.examples, arguments.seed)
            if arguments.device == "cpu"
            else _gpu_benchmark(
                arguments.examples, arguments.epochs, arguments.seed
            )
        ),
    }
    print("ZOO_BENCHMARK=" + json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
