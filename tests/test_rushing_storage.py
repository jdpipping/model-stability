"""Tests for immutable and resumable rushing-study artifact storage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rushing_study import storage as storage_module
from rushing_study.storage import (
    CellKey,
    CellStatus,
    CorruptArtifactError,
    IncompleteRunError,
    ManifestMismatchError,
    RunDirectoryCollisionError,
    RunFinalizedError,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    cell_dir,
    checksum_path,
    classify_cells,
    finalize_run,
    initialize_run_dir,
    load_cell_metrics,
    manifest_sha256,
    validate_cell,
    validate_checksum,
    write_cell_artifacts,
)


class RushingStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            "study": "rushing-four-model",
            "seed": 20260817,
            "anchors": [50, 100, 250, 500, 1000, 2000],
            "repeats": 50,
        }
        self.key = CellKey("fixed-main", 0, 50, "ridge")

    def init_storage(self):
        return initialize_run_dir(self.root / "run", self.manifest)

    def test_initialize_is_idempotent_and_manifest_bound(self):
        run_dir = self.root / "run"
        expected_hash = manifest_sha256(self.manifest)

        first = initialize_run_dir(run_dir, self.manifest, manifest_hash=expected_hash)
        second = initialize_run_dir(run_dir, dict(reversed(list(self.manifest.items()))))

        self.assertEqual(first.manifest_hash, expected_hash)
        self.assertEqual(second.manifest_hash, expected_hash)
        self.assertTrue(validate_checksum(run_dir / "manifest.json"))
        self.assertTrue(checksum_path(run_dir / "manifest.json").is_file())

        with self.assertRaises(ManifestMismatchError):
            initialize_run_dir(run_dir, {**self.manifest, "repeats": 51})
        with self.assertRaises(ManifestMismatchError):
            initialize_run_dir(run_dir, self.manifest, manifest_hash="0" * 64)

    def test_initialize_refuses_nonempty_collision_and_file_path(self):
        collision = self.root / "collision"
        collision.mkdir()
        (collision / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")

        with self.assertRaises(RunDirectoryCollisionError):
            initialize_run_dir(collision, self.manifest)
        self.assertEqual(
            (collision / "unrelated.txt").read_text(encoding="utf-8"), "do not overwrite"
        )

        file_path = self.root / "not-a-directory"
        file_path.write_text("occupied", encoding="utf-8")
        with self.assertRaises(RunDirectoryCollisionError):
            initialize_run_dir(file_path, self.manifest)

    def test_initialize_recovers_only_a_missing_manifest_sidecar(self):
        run_dir = self.root / "run"
        initialize_run_dir(run_dir, self.manifest)
        sidecar = checksum_path(run_dir / "manifest.json")
        sidecar.unlink()

        recovered = initialize_run_dir(run_dir, self.manifest)
        self.assertEqual(recovered.manifest_hash, manifest_sha256(self.manifest))
        self.assertTrue(validate_checksum(run_dir / "manifest.json"))

        sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
        with self.assertRaises(CorruptArtifactError):
            initialize_run_dir(run_dir, self.manifest)

    def test_cell_paths_are_deterministic_and_traversal_safe(self):
        expected = (
            self.root
            / "run"
            / "cells"
            / "fixed-main"
            / "repeat_000"
            / "n_0050"
            / "ridge"
        )
        self.assertEqual(cell_dir(self.root / "run", self.key), expected)
        self.assertEqual(cell_dir(self.root / "run", "fixed-main", 0, 50, "ridge"), expected)

        with self.assertRaises(ValueError):
            CellKey("../escape", 0, 50, "ridge")
        with self.assertRaises(ValueError):
            CellKey("fixed-main", 0, 50, "../model")
        with self.assertRaises(ValueError):
            CellKey("fixed-main", -1, 50, "ridge")

    def test_generic_atomic_writers_create_valid_checksum_sidecars(self):
        output = self.root / "artifacts"
        json_record = atomic_write_json(output / "metrics.json", {"rmse": np.float64(1.25)})
        csv_record = atomic_write_csv(
            output / "rows.csv", pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        )
        npz_record = atomic_write_npz(output / "values.npz", x=np.array([1, 2, 3]))

        for record in (json_record, csv_record, npz_record):
            path = output / record.file
            self.assertTrue(path.is_file())
            self.assertTrue(checksum_path(path).is_file())
            self.assertTrue(validate_checksum(path, record.sha256))
        with np.load(output / npz_record.file, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["x"], np.array([1, 2, 3]))

    def test_dataframe_writer_has_explicit_csv_fallback(self):
        frame = pd.DataFrame({"play_id": [10, 11], "prediction": [1.5, -0.25]})
        with mock.patch.object(storage_module, "_parquet_engine", return_value=None):
            record = storage_module.atomic_write_dataframe(self.root, frame)

        self.assertEqual(record.format, "csv")
        self.assertEqual(record.file, "predictions.csv")
        self.assertFalse((self.root / "predictions.parquet").exists())
        assert_frame_equal(pd.read_csv(self.root / record.file), frame)

    def test_write_load_and_resume_skip_valid_dataframe_cell(self):
        storage = self.init_storage()
        predictions = pd.DataFrame(
            {"play_id": [1, 2], "actual": [3.0, -1.0], "prediction": [2.5, -0.5]}
        )
        metrics = {"rmse": 0.5, "n_test": 2}

        with mock.patch.object(storage_module, "_parquet_engine", return_value=None):
            written = write_cell_artifacts(
                storage,
                self.key,
                metrics=metrics,
                predictions=predictions,
                history={"loss": [1.0, 0.5]},
                arrays={"test_index": np.array([4, 8])},
            )

        self.assertTrue(written)
        self.assertEqual(validate_cell(storage, self.key).status, CellStatus.COMPLETE)
        self.assertEqual(load_cell_metrics(storage, self.key), metrics)
        assert_frame_equal(storage.load_predictions(self.key), predictions)
        self.assertEqual(storage.load_history(self.key), {"loss": [1.0, 0.5]})
        np.testing.assert_array_equal(storage.load_arrays(self.key)["test_index"], [4, 8])

        skipped = write_cell_artifacts(
            storage,
            self.key,
            metrics={"rmse": 999.0},
            predictions=pd.DataFrame({"prediction": [999.0]}),
        )
        self.assertFalse(skipped)
        self.assertEqual(load_cell_metrics(storage, self.key), metrics)
        self.assertEqual(storage.pending_cells([self.key]), [])

    def test_dict_predictions_are_stored_and_loaded_as_npz(self):
        storage = self.init_storage()
        expected = {
            "play_id": np.array([100, 101], dtype=np.int64),
            "prediction": np.array([2.0, -1.0], dtype=np.float32),
        }
        storage.write_cell_artifacts(
            self.key,
            metrics={"mae": 0.25},
            predictions=expected,
        )

        loaded = storage.load_predictions(self.key)
        self.assertIsInstance(loaded, dict)
        for name, values in expected.items():
            np.testing.assert_array_equal(loaded[name], values)

    def test_tampering_or_missing_sidecar_marks_cell_corrupt_and_rewrite_repairs(self):
        storage = self.init_storage()
        storage.write_cell_artifacts(
            self.key,
            metrics={"rmse": 1.0},
            predictions={"prediction": np.array([0.0])},
        )
        metrics_path = storage.cell_dir(self.key) / "metrics.json"
        metrics_path.write_text('{"rmse": 999}\n', encoding="utf-8")

        result = storage.validate_cell(self.key)
        self.assertEqual(result.status, CellStatus.CORRUPT)
        self.assertIn("checksum", result.reason)
        self.assertEqual(storage.pending_cells([self.key]), [self.key])
        with self.assertRaises(CorruptArtifactError):
            storage.load_metrics(self.key)

        storage.write_cell_artifacts(
            self.key,
            metrics={"rmse": 0.75},
            predictions={"prediction": np.array([1.0])},
        )
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.COMPLETE)

        checksum_path(storage.cell_dir(self.key) / "metrics.json").unlink()
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.CORRUPT)

    def test_missing_partial_and_complete_cells_are_classified_and_counted(self):
        storage = self.init_storage()
        complete = self.key
        partial = CellKey("fixed-main", 1, 50, "ridge")
        missing = CellKey("fixed-main", 2, 50, "ridge")
        storage.write_cell_artifacts(
            complete,
            metrics={"rmse": 1.0},
            predictions={"prediction": np.array([0.0])},
        )
        storage.cell_dir(partial).mkdir(parents=True)
        atomic_write_json(storage.cell_dir(partial) / "metrics.json", {"rmse": 2.0})

        classified = classify_cells(storage, [complete, partial, missing])
        self.assertEqual(classified[complete].status, CellStatus.COMPLETE)
        self.assertEqual(classified[partial].status, CellStatus.CORRUPT)
        self.assertEqual(classified[missing].status, CellStatus.MISSING)
        self.assertEqual(
            storage.status_counts([complete, partial, missing]).as_dict(),
            {"complete": 1, "running": 0, "corrupt": 1, "missing": 1, "total": 3},
        )
        self.assertEqual(storage.pending_cells([complete, partial, missing]), [partial, missing])

    def test_running_marker_is_distinct_and_cleared_explicitly(self):
        storage = self.init_storage()
        storage.mark_cell_running(self.key)

        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.RUNNING)
        self.assertEqual(
            storage.status_counts([self.key]).as_dict(),
            {"complete": 0, "running": 1, "corrupt": 0, "missing": 0, "total": 1},
        )

        storage.clear_cell_running(self.key)
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.CORRUPT)

    def test_dead_local_running_pid_is_stale_and_can_be_reclaimed(self):
        storage = self.init_storage()
        storage.mark_cell_running(self.key)
        marker_path = storage.cell_dir(self.key) / "_RUNNING"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["pid"] = 99_999_999
        marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")

        stale = storage.validate_cell(self.key)
        self.assertEqual(stale.status, CellStatus.CORRUPT)
        self.assertIn("dead local pid", stale.reason)
        storage.mark_cell_running(self.key, overwrite=True)
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.RUNNING)

    def test_finalize_rejects_committed_cells_outside_required_grid(self):
        storage = self.init_storage()
        unexpected = CellKey("fixed-main", 9, 50, "ridge")
        for key in (self.key, unexpected):
            storage.write_cell_artifacts(
                key,
                metrics={"rmse": 1.0},
                predictions={"prediction": np.array([0.0])},
            )

        with self.assertRaisesRegex(IncompleteRunError, "outside the required grid"):
            storage.finalize_run([self.key])
        self.assertFalse((storage.run_dir / "_SUCCESS").exists())

    def test_failed_payload_write_leaves_no_commit_and_is_resumable(self):
        storage = self.init_storage()
        predictions = pd.DataFrame({"prediction": [1.0]})
        with mock.patch.object(
            storage_module, "atomic_write_dataframe", side_effect=OSError("injected fault")
        ):
            with self.assertRaisesRegex(OSError, "injected fault"):
                storage.write_cell_artifacts(
                    self.key,
                    metrics={"rmse": 1.0},
                    predictions=predictions,
                )

        directory = storage.cell_dir(self.key)
        self.assertFalse((directory / "_SUCCESS").exists())
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.CORRUPT)

        target = self.root / "atomic-failure.csv"
        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk fault")):
            with self.assertRaisesRegex(OSError, "disk fault"):
                atomic_write_csv(target, predictions)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob(".atomic-failure.csv.*")), [])

        storage.write_cell_artifacts(
            self.key,
            metrics={"rmse": 0.5},
            predictions={"prediction": np.array([1.0])},
        )
        self.assertEqual(storage.validate_cell(self.key).status, CellStatus.COMPLETE)

    def test_finalize_requires_all_cells_and_freezes_run(self):
        storage = self.init_storage()
        second = CellKey("fixed-main", 0, 100, "ridge")
        storage.write_cell_artifacts(
            self.key,
            metrics={"rmse": 1.0},
            predictions={"prediction": np.array([0.0])},
        )

        with self.assertRaises(IncompleteRunError):
            finalize_run(storage, [self.key, second])
        self.assertFalse((storage.run_dir / "_SUCCESS").exists())

        storage.write_cell_artifacts(
            second,
            metrics={"rmse": 0.8},
            predictions={"prediction": np.array([0.1])},
        )
        summary = atomic_write_json(storage.run_dir / "summary.json", {"cells": 2})
        self.assertTrue(
            finalize_run(storage, [self.key, second], final_artifacts=[summary.file])
        )
        self.assertTrue(storage.validate_final([self.key, second]))
        self.assertFalse(finalize_run(storage, [self.key, second]))

        with self.assertRaises(RunFinalizedError):
            storage.write_cell_artifacts(
                CellKey("fixed-main", 1, 50, "ridge"),
                metrics={"rmse": 0.5},
                predictions={"prediction": np.array([0.2])},
            )

        (storage.cell_dir(self.key) / "metrics.json").write_text("tampered", encoding="utf-8")
        self.assertFalse(storage.validate_final([self.key, second]))

    def test_path_api_requires_matching_manifest_hash(self):
        storage = self.init_storage()
        storage.write_cell_artifacts(
            self.key,
            metrics={"rmse": 1.0},
            predictions={"prediction": np.array([0.0])},
        )

        self.assertEqual(
            load_cell_metrics(
                storage.run_dir, self.key, manifest_hash=storage.manifest_hash
            )["rmse"],
            1.0,
        )
        with self.assertRaises(ManifestMismatchError):
            load_cell_metrics(storage.run_dir, self.key, manifest_hash="0" * 64)


if __name__ == "__main__":
    unittest.main()
