from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from bdb_study.storage import (
    BackendPinError,
    CellKey,
    CellStatus,
    DEFAULT_QUEUE_SPECS,
    bind_storage_backend,
    cell_keys_from_design,
    estimate_queue_eta_seconds,
    initialize_run_dir,
    immutable_write_json,
    pending_cells_by_queue,
    queue_manifest,
    recommended_cpu_workers,
    status_by_queue,
    storage_backend_receipt,
)


def mini_design():
    return {
        "required_cells": [
            {
                "branch": "fixed_main",
                "repeat": 1,
                "n_train": 10,
                "model": "l2_glm",
                "queue": "cpu_tabular",
            },
            {
                "branch": "fixed_main",
                "repeat": 1,
                "n_train": 10,
                "model": "cnn",
                "queue": "gpu_neural",
            },
        ]
    }


class GenericStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = bind_storage_backend({"study": "synthetic-bdb", "version": 1})

    def test_backend_pin_is_required_and_exact(self):
        with self.assertRaises(BackendPinError):
            initialize_run_dir(self.root / "missing", {"study": "unbound"})
        changed = dict(self.manifest)
        changed["storage_backend"] = dict(changed["storage_backend"])
        changed["storage_backend"]["sha256"] = "0" * 64
        with self.assertRaises(BackendPinError):
            initialize_run_dir(self.root / "wrong", changed)
        self.assertEqual(self.manifest["storage_backend"], storage_backend_receipt())

    def test_atomic_cell_resume_and_corruption_status(self):
        storage = initialize_run_dir(self.root / "run", self.manifest)
        key = CellKey("fixed_main", 1, 10, "l2_glm")
        written = storage.write_cell_artifacts(
            key,
            metrics={"brier": 0.1},
            predictions={"raw_probability": np.array([0.2, 0.8])},
        )
        self.assertTrue(written)
        marker_before = (storage.cell_dir(key) / "_SUCCESS").read_bytes()
        self.assertFalse(
            storage.write_cell_artifacts(
                key,
                metrics={"brier": 999},
                predictions={"raw_probability": np.array([1.0])},
            )
        )
        self.assertEqual((storage.cell_dir(key) / "_SUCCESS").read_bytes(), marker_before)
        self.assertEqual(storage.validate_cell(key).status, CellStatus.COMPLETE)

        (storage.cell_dir(key) / "metrics.json").write_text("{}", encoding="utf-8")
        self.assertEqual(storage.validate_cell(key).status, CellStatus.CORRUPT)
        self.assertEqual(storage.pending_cells([key]), [key])

    def test_queue_resume_status_and_eta_use_slower_concurrent_queue(self):
        design = mini_design()
        storage = initialize_run_dir(self.root / "run", self.manifest)
        cpu, gpu = cell_keys_from_design(design)
        storage.write_cell_artifacts(
            cpu,
            metrics={"brier": 0.1},
            predictions={"p": np.array([0.1])},
        )
        statuses = status_by_queue(storage, design)
        self.assertEqual(statuses["cpu_tabular"].complete, 1)
        self.assertEqual(statuses["gpu_neural"].missing, 1)
        pending = pending_cells_by_queue(storage, design)
        self.assertEqual(pending["cpu_tabular"], [])
        self.assertEqual(pending["gpu_neural"], [gpu])
        eta = estimate_queue_eta_seconds(
            storage,
            design,
            {"l2_glm": 10.0, "cnn": 100.0},
            cpu_workers=12,
            gpu_workers=1,
        )
        self.assertEqual(eta["queue_seconds"], {"cpu_tabular": 0.0, "gpu_neural": 100.0})
        self.assertEqual(eta["eta_seconds"], 100.0)

    def test_ram_aware_cpu_worker_rule(self):
        gib = 1024**3
        self.assertEqual(recommended_cpu_workers(64 * gib, 2 * gib), 12)
        self.assertEqual(recommended_cpu_workers(8 * gib, 2 * gib), 3)
        self.assertEqual(recommended_cpu_workers(1 * gib, 2 * gib), 1)

    def test_gpu_queue_registry_contains_all_three_neural_families(self):
        expected = ("relnet", "attn_relnet", "set_transformer")
        default_gpu = next(
            queue for queue in DEFAULT_QUEUE_SPECS if queue.name == "gpu_neural"
        )
        self.assertEqual(default_gpu.model_families, expected)
        self.assertEqual(
            queue_manifest()["queues"]["gpu_neural"]["model_families"],
            list(expected),
        )
        self.assertEqual(
            queue_manifest(include_set_transformer=False)["queues"]["gpu_neural"]
            ["model_families"],
            ["relnet", "attn_relnet"],
        )

    def test_immutable_json_create_or_match_refuses_different_payload(self):
        path = self.root / "receipt.json"
        immutable_write_json(path, {"study": "BDB", "label": "stability"})
        before = path.read_bytes()
        immutable_write_json(path, {"study": "BDB", "label": "stability"})
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
            immutable_write_json(path, {"study": "BDB", "label": "different"})
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
