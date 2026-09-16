from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bdb_study.execution import (
    ExecutionError,
    _build_selector_receipt,
    _immutable_write_bytes,
    _pretty_json_bytes,
    _read_selector_receipt,
    _runtime_plan_external_gpu_lanes,
    _run_worker_subprocess,
    _selector_cell_identity,
    _selector_run_binding,
    _write_selector_receipt,
    neural_selector_receipt_path,
    run_missing_cells,
    run_repeat_queue,
)
from bdb_study.models import (
    NEURAL_SELECTION_SCHEMA_VERSION,
    NeuralSelection,
    fit_neural_explicit_preprocessing,
    refit_neural_from_selection_explicit_preprocessing,
    select_neural_epoch_explicit_preprocessing,
)
from bdb_study.storage import CellKey, CellStatus
from bdb_study.storage import (
    PhaseLeaseError,
    claim_phase_lease,
    release_phase_lease,
)
from bdb_study.runtime_phases import betty_runtime_v3_resource_contract


ROOT = Path(__file__).resolve().parents[2]


def _selector_core() -> dict:
    return {
        "schema_version": "bdb-neural-cell-selector-v1",
        "binding": {"test": "binding"},
        "selector_preprocessing": {"fit_scope": "selector"},
        "selector_train_game_ids": ["game-a", "game-b"],
        "selection": {
            "schema_version": NEURAL_SELECTION_SCHEMA_VERSION,
            "family": "cnn",
            "outcome_type": "binary",
            "n_outputs": 1,
            "max_horizon": 1,
            "best_epoch": 1,
            "selector_history": {"loss": [0.4, 0.3], "val_loss": [0.2, 0.25]},
            "parameter_count": 17,
            "input_shapes": {"raster": [2, 8, 8, 1], "frame_mask": [2]},
            "validation_game_ids": ["game-c"],
            "validation_split_seed": 31,
            "selection_seed": 41,
        },
    }


def _storage() -> SimpleNamespace:
    return SimpleNamespace(
        manifest_hash="a" * 64,
        manifest={
            "task_id": "synthetic",
            "task_spec_hash": "b" * 64,
            "prepared": {"prepared_hash": "c" * 64},
            "execution": {"profile": "pilot10"},
        },
    )


class NeuralModelPhaseEquivalenceTests(unittest.TestCase):
    def test_monolithic_and_selector_refit_are_deterministically_equivalent(self) -> None:
        import tensorflow as tf

        # Compare phase decomposition on a CPU reference, independently of a
        # workstation GPU's determinism. Hardware replay is tested by preflight.
        self.enterContext(tf.device("/CPU:0"))
        rng = np.random.default_rng(20260822)
        inputs = {
            "raster": rng.normal(size=(12, 2, 8, 8, 1)).astype(np.float32),
            "frame_mask": np.ones((12, 2), dtype=bool),
        }
        labels = (np.arange(12) % 2).astype(np.float32)
        selector_train = np.arange(8)
        selector_validation = np.arange(8, 12)
        config = {
            "learning_rate": 1e-3,
            "dropout": 0.0,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 4,
            "deterministic": True,
        }
        common = {
            "outcome_type": "binary",
            "n_outputs": 1,
            "config": config,
            "validation_game_ids": tuple(f"game-{value}" for value in range(8, 12)),
            "validation_split_seed": 31,
        }
        monolithic = fit_neural_explicit_preprocessing(
            "cnn",
            {name: value[selector_train] for name, value in inputs.items()},
            labels[selector_train],
            {name: value[selector_validation] for name, value in inputs.items()},
            labels[selector_validation],
            inputs,
            labels,
            selector_train_mask=None,
            selector_validation_mask=None,
            final_mask=None,
            selection_seed=41,
            refit_seed=51,
            **common,
        )
        monolithic_prediction = np.asarray(monolithic.model.predict(inputs, verbose=0))
        monolithic_weights = [np.asarray(value).copy() for value in monolithic.model.get_weights()]
        tf.keras.backend.clear_session()

        selection = select_neural_epoch_explicit_preprocessing(
            "cnn",
            {name: value[selector_train] for name, value in inputs.items()},
            labels[selector_train],
            {name: value[selector_validation] for name, value in inputs.items()},
            labels[selector_validation],
            selector_train_mask=None,
            selector_validation_mask=None,
            selection_seed=41,
            **common,
        )
        split = refit_neural_from_selection_explicit_preprocessing(
            "cnn",
            inputs,
            labels,
            outcome_type="binary",
            n_outputs=1,
            final_mask=None,
            config=config,
            refit_seed=51,
            selection=selection.as_dict(),
        )
        split_prediction = np.asarray(split.model.predict(inputs, verbose=0))
        self.assertEqual(monolithic.best_epoch, selection.best_epoch)
        self.assertEqual(monolithic.selector_history, selection.selector_history)
        self.assertEqual(monolithic.refit_history, split.refit_history)
        for expected, observed in zip(monolithic_weights, split.model.get_weights()):
            np.testing.assert_array_equal(expected, observed)
        np.testing.assert_array_equal(monolithic_prediction, split_prediction)
        tf.keras.backend.clear_session()

    def test_refit_rejects_semantically_tampered_selection(self) -> None:
        value = _selector_core()["selection"]
        tampered = copy.deepcopy(value)
        tampered["best_epoch"] = 2
        with self.assertRaisesRegex(ValueError, "best epoch"):
            NeuralSelection.from_mapping(tampered)


class NeuralPhaseLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = CellKey("fixed_main", 1, 20, "relnet")
        self.manifest_hash = "a" * 64

    @staticmethod
    def _slurm_environment(
        job_id: str, *, dependency: str | None = None, restart_count: int = 0
    ) -> dict[str, str]:
        value = {
            "SLURM_JOB_ID": job_id,
            "SLURM_RESTART_COUNT": str(restart_count),
            "SLURM_CLUSTER_NAME": "betty",
        }
        if dependency is not None:
            value["SLURM_JOB_DEPENDENCY"] = dependency
        return value

    def test_fresh_claim_never_queries_slurm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observed = claim_phase_lease(
                Path(directory) / "_RUNNING",
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="selector",
                environment=self._slurm_environment("101"),
                scheduler_state_lookup=lambda job_id: self.fail(
                    "fresh execution must not query Slurm"
                ),
            )
            self.assertEqual(observed.status, "claimed")
            release_phase_lease(observed)

    def test_completion_requires_explicit_valid_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "_RUNNING"
            completion = Path(directory) / "_SUCCESS"
            completion.write_text('{"storage_valid": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                PhaseLeaseError, "explicit validity callback"
            ):
                claim_phase_lease(
                    marker,
                    manifest_hash=self.manifest_hash,
                    key=self.key,
                    phase="refit",
                    completion_path=completion,
                    environment=self._slurm_environment("101"),
                )
            observed = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                completion_path=completion,
                completion_validator=lambda: True,
                environment=self._slurm_environment("101"),
            )
            self.assertEqual(observed.status, "complete")
            self.assertTrue(completion.is_file())
            self.assertFalse(marker.exists())

    def test_live_slurm_marker_is_skipped_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "_RUNNING"
            prior = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("101"),
            )
            before = marker.read_bytes()
            observed = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("202"),
                scheduler_state_lookup=lambda job_id: "RUNNING",
            )
            self.assertEqual(observed.status, "running")
            self.assertEqual(marker.read_bytes(), before)
            self.assertFalse((Path(directory) / "._RUNNING.takeovers").exists())
            release_phase_lease(prior)

    def test_terminal_slurm_marker_is_archived_then_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "_RUNNING"
            prior = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="selector",
                environment=self._slurm_environment("101"),
            )
            prior_payload = marker.read_bytes()
            observed = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="selector",
                environment=self._slurm_environment("202"),
                scheduler_state_lookup=lambda job_id: "TIMEOUT",
            )
            self.assertEqual(observed.status, "claimed")
            self.assertEqual(
                observed.reclaimed_marker_sha256,
                prior.marker_sha256,
            )
            archive = (
                Path(directory)
                / "._RUNNING.takeovers"
                / f"{prior.marker_sha256}.json"
            )
            self.assertEqual(archive.read_bytes(), prior_payload)
            takeover = json.loads(marker.read_text(encoding="utf-8"))[
                "phase_lease"
            ]["takeover"]
            self.assertEqual(
                takeover["evidence"],
                {
                    "kind": "slurm_terminal_job",
                    "prior_job_id": "101",
                    "terminal_state": "TIMEOUT",
                },
            )
            with self.assertRaisesRegex(PhaseLeaseError, "different phase marker"):
                release_phase_lease(prior)
            self.assertTrue(marker.is_file())
            release_phase_lease(observed)
            self.assertFalse(marker.exists())
            self.assertEqual(archive.read_bytes(), prior_payload)

    def test_satisfied_afterok_dependency_is_scheduler_takeover_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "_RUNNING"
            prior = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("101"),
            )
            observed = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment(
                    "202", dependency="afterok:99:101,singleton"
                ),
                scheduler_state_lookup=lambda job_id: self.fail(
                    "a satisfied direct dependency must not need a client query"
                ),
            )
            self.assertEqual(observed.status, "claimed")
            takeover = json.loads(marker.read_text(encoding="utf-8"))[
                "phase_lease"
            ]["takeover"]
            self.assertEqual(
                takeover["evidence"],
                {
                    "kind": "slurm_satisfied_dependency",
                    "prior_job_id": "101",
                    "dependency_type": "afterok",
                    "current_job_id": "202",
                },
            )
            self.assertEqual(observed.reclaimed_marker_sha256, prior.marker_sha256)
            release_phase_lease(observed)

    def test_same_job_restart_count_reclaims_without_slurm_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "_RUNNING"
            prior = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("101", restart_count=0),
            )
            observed = claim_phase_lease(
                marker,
                manifest_hash=self.manifest_hash,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("101", restart_count=1),
                scheduler_state_lookup=lambda job_id: self.fail(
                    "a later incarnation of the same job must not query Slurm"
                ),
            )
            self.assertEqual(observed.status, "claimed")
            takeover = json.loads(marker.read_text(encoding="utf-8"))[
                "phase_lease"
            ]["takeover"]
            self.assertEqual(
                takeover["evidence"]["kind"],
                "slurm_restart_count_advanced",
            )
            self.assertEqual(observed.reclaimed_marker_sha256, prior.marker_sha256)
            release_phase_lease(observed)

    def test_corrupt_and_foreign_markers_are_rejected_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt" / "_RUNNING"
            corrupt.parent.mkdir(parents=True)
            corrupt.write_bytes(b"not-json\n")
            with self.assertRaisesRegex(PhaseLeaseError, "unreadable"):
                claim_phase_lease(
                    corrupt,
                    manifest_hash=self.manifest_hash,
                    key=self.key,
                    phase="refit",
                    environment=self._slurm_environment("202"),
                )
            self.assertEqual(corrupt.read_bytes(), b"not-json\n")

            foreign = Path(directory) / "foreign" / "_RUNNING"
            prior = claim_phase_lease(
                foreign,
                manifest_hash="b" * 64,
                key=self.key,
                phase="refit",
                environment=self._slurm_environment("101"),
            )
            before = foreign.read_bytes()
            with self.assertRaisesRegex(PhaseLeaseError, "different run"):
                claim_phase_lease(
                    foreign,
                    manifest_hash=self.manifest_hash,
                    key=self.key,
                    phase="refit",
                    environment=self._slurm_environment("202"),
                    scheduler_state_lookup=lambda job_id: "TIMEOUT",
                )
            self.assertEqual(foreign.read_bytes(), before)
            release_phase_lease(prior)

    def test_refit_takeover_leaves_final_artifact_payloads_unchanged(self) -> None:
        record = {
            "branch": self.key.branch,
            "repeat": self.key.repeat,
            "n_train": self.key.n_train,
            "model": self.key.model,
            "queue": "gpu_neural",
            "ablation_id": None,
            "sensitivity_id": None,
        }

        class CapturingStorage:
            def __init__(inner_self, run_dir: Path) -> None:
                inner_self.run_dir = run_dir
                inner_self.manifest_hash = self.manifest_hash
                inner_self.manifest = {
                    "task_id": "synthetic",
                    "task_spec_hash": "b" * 64,
                    "prepared": {
                        "prepared_hash": "c" * 64,
                        "path": "prepared",
                    },
                    "execution": {
                        "profile": "pilot10",
                        "smoke": {"enabled": False},
                        "benchmark": {"enabled": False},
                    },
                    "task_design": {"required_cells": [record]},
                }
                inner_self.completed = False
                inner_self.artifacts = None

            def validate_cell(inner_self, key):
                if inner_self.completed:
                    return SimpleNamespace(status=CellStatus.COMPLETE)
                marker = inner_self.run_dir / key.relative_dir / "_RUNNING"
                success = inner_self.run_dir / key.relative_dir / "_SUCCESS"
                if success.exists():
                    return SimpleNamespace(status=CellStatus.CORRUPT)
                return SimpleNamespace(
                    status=(CellStatus.RUNNING if marker.exists() else CellStatus.MISSING)
                )

            def write_cell_artifacts(
                inner_self,
                key,
                *,
                metrics,
                predictions,
                history,
                arrays,
                preferred_predictions,
            ):
                inner_self.artifacts = {
                    "metrics": copy.deepcopy(metrics),
                    "predictions": copy.deepcopy(predictions),
                    "history": copy.deepcopy(history),
                    "arrays": copy.deepcopy(arrays),
                }
                inner_self.completed = True
                return True

        def execute(
            directory: str, *, stale: bool, invalid_completion: bool = False
        ) -> dict:
            run_dir = Path(directory)
            storage = CapturingStorage(run_dir)
            selector_path = neural_selector_receipt_path(run_dir, self.key)
            receipt = _build_selector_receipt(
                storage,
                self.key,
                record,
                _selector_core(),
                elapsed_seconds=1.0,
            )
            _write_selector_receipt(selector_path, receipt)
            if stale:
                prior = claim_phase_lease(
                    run_dir / self.key.relative_dir / "_RUNNING",
                    manifest_hash=self.manifest_hash,
                    key=self.key,
                    phase="refit",
                    environment=self._slurm_environment("101"),
                )
                self.assertEqual(prior.status, "claimed")
            if invalid_completion:
                success = run_dir / self.key.relative_dir / "_SUCCESS"
                success.parent.mkdir(parents=True, exist_ok=True)
                success.write_text(
                    '{"storage_valid_but_scientifically_invalid": true}\n',
                    encoding="utf-8",
                )

            def fitted_result(*args, **kwargs):
                return SimpleNamespace(
                    metrics={"scientific_metric": [1, 2, 3]},
                    predictions={"prediction": np.asarray([0.25, 0.75])},
                    history={"loss": [0.5, 0.25]},
                    arrays={"weights": np.asarray([3.0, 4.0])},
                )

            with (
                patch("bdb_study.execution.open_run", return_value=storage),
                patch("bdb_study.execution._effective_task_spec", return_value=object()),
                patch("bdb_study.execution.load_prepared_task", return_value=object()),
                patch("bdb_study.execution.TaskRuntime", return_value=object()),
                patch("bdb_study.execution.run_cell", side_effect=fitted_result),
                patch("bdb_study.execution.validate_neural_selector_for_cell"),
                patch("bdb_study.execution.time.monotonic", side_effect=[10.0, 12.0]),
                patch("bdb_study.preflight.process_peak_rss_bytes", return_value=123),
                patch(
                    "bdb_study.storage._query_slurm_job_state",
                    return_value="TIMEOUT",
                ),
                patch.dict(
                    os.environ,
                    self._slurm_environment("202"),
                    clear=True,
                ),
            ):
                result = run_repeat_queue(
                    run_dir,
                    repeat=1,
                    queue="gpu_neural",
                    repo_root=ROOT,
                    neural_phase="refit",
                )
            self.assertEqual(result["completed"], 1)
            if invalid_completion:
                archives = list(
                    (run_dir / self.key.relative_dir / "._SUCCESS.takeovers").glob(
                        "*.json"
                    )
                )
                self.assertEqual(len(archives), 1)
                self.assertIn(
                    b'"storage_valid_but_scientifically_invalid": true',
                    archives[0].read_bytes(),
                )
            return storage.artifacts

        with (
            tempfile.TemporaryDirectory() as clean_dir,
            tempfile.TemporaryDirectory() as stale_dir,
            tempfile.TemporaryDirectory() as corrupt_dir,
        ):
            clean = execute(clean_dir, stale=False)
            reclaimed = execute(stale_dir, stale=True)
            repaired = execute(
                corrupt_dir, stale=False, invalid_completion=True
            )
        self.assertEqual(clean["metrics"], reclaimed["metrics"])
        self.assertEqual(clean["history"], reclaimed["history"])
        np.testing.assert_array_equal(
            clean["predictions"]["prediction"],
            reclaimed["predictions"]["prediction"],
        )
        np.testing.assert_array_equal(
            clean["arrays"]["weights"], reclaimed["arrays"]["weights"]
        )
        self.assertEqual(clean["metrics"], repaired["metrics"])
        self.assertEqual(clean["history"], repaired["history"])
        np.testing.assert_array_equal(
            clean["predictions"]["prediction"],
            repaired["predictions"]["prediction"],
        )
        np.testing.assert_array_equal(
            clean["arrays"]["weights"], repaired["arrays"]["weights"]
        )


class NeuralSelectorReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = _storage()
        self.key = CellKey("fixed_main", 1, 20, "relnet")
        self.record = {"ablation_id": None, "sensitivity_id": None}
        self.run_binding = _selector_run_binding(self.storage)
        self.cell = _selector_cell_identity(self.key, self.record)

    def test_receipt_round_trip_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = neural_selector_receipt_path(directory, self.key)
            receipt = _build_selector_receipt(
                self.storage,
                self.key,
                self.record,
                _selector_core(),
                elapsed_seconds=2.5,
            )
            _write_selector_receipt(path, receipt)
            loaded = _read_selector_receipt(
                path,
                expected_run_binding=self.run_binding,
                expected_cell=self.cell,
            )
            self.assertEqual(loaded, receipt)
            self.assertEqual(loaded["selector_observed_epochs"], 2)
            self.assertEqual(loaded["best_epoch"], 1)
            self.assertEqual(loaded["refit_epochs"], 1)
            payload = path.read_bytes()
            path.write_bytes(payload.replace(b'"best_epoch": 1', b'"best_epoch": 2', 1))
            with self.assertRaisesRegex(ExecutionError, "checksum"):
                _read_selector_receipt(
                    path,
                    expected_run_binding=self.run_binding,
                    expected_cell=self.cell,
                )

    def test_payload_only_crash_is_repaired_but_conflict_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = neural_selector_receipt_path(directory, self.key)
            receipt = _build_selector_receipt(
                self.storage,
                self.key,
                self.record,
                _selector_core(),
                elapsed_seconds=2.5,
            )
            _immutable_write_bytes(path, _pretty_json_bytes(receipt))
            self.assertFalse(path.with_name("selector.json.sha256").exists())
            loaded = _read_selector_receipt(
                path,
                expected_run_binding=self.run_binding,
                expected_cell=self.cell,
                repair_missing_sidecar=True,
            )
            self.assertEqual(loaded, receipt)
            self.assertTrue(path.with_name("selector.json.sha256").is_file())
            different = copy.deepcopy(receipt)
            different["selector_elapsed_seconds"] = 3.0
            with self.assertRaisesRegex(ExecutionError, "refusing to replace"):
                _write_selector_receipt(path, different)

    def test_missing_receipt_and_wrong_run_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = neural_selector_receipt_path(directory, self.key)
            with self.assertRaises(FileNotFoundError):
                _read_selector_receipt(
                    path,
                    expected_run_binding=self.run_binding,
                    expected_cell=self.cell,
                )
            receipt = _build_selector_receipt(
                self.storage,
                self.key,
                self.record,
                _selector_core(),
                elapsed_seconds=1.0,
            )
            _write_selector_receipt(path, receipt)
            wrong = {**self.run_binding, "manifest_hash": "d" * 64}
            with self.assertRaisesRegex(ExecutionError, "different run"):
                _read_selector_receipt(
                    path,
                    expected_run_binding=wrong,
                    expected_cell=self.cell,
                )

    def test_phase_cli_propagation_and_exact_gpu_queue_guard(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"completed": 1}\n',
            stderr="",
        )
        with patch("bdb_study.execution.subprocess.run", return_value=completed) as run:
            _run_worker_subprocess(
                ROOT,
                3,
                "gpu_neural",
                ROOT,
                neural_phase="selector",
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--neural-phase") + 1], "selector")
        with self.assertRaisesRegex(ExecutionError, "exactly --queue gpu_neural"):
            run_missing_cells(
                ROOT,
                repo_root=ROOT,
                queues=["cpu_tabular"],
                neural_phase="refit",
            )

    def test_expanded_lanes_require_exact_v3_resource_binding(self) -> None:
        self.assertEqual(_runtime_plan_external_gpu_lanes({}), 4)
        self.assertEqual(
            _runtime_plan_external_gpu_lanes(
                {"runtime_plan": {"schema_version": "bdb-runtime-plan-v2"}}
            ),
            4,
        )
        v3 = {
            "schema_version": "bdb-runtime-plan-v3",
            "gpu_lanes": 4,
            "resource_contract": betty_runtime_v3_resource_contract(),
        }
        self.assertEqual(_runtime_plan_external_gpu_lanes({"runtime_plan": v3}), 4)
        drifted = copy.deepcopy(v3)
        drifted["resource_contract"]["gpu"]["lanes"] = 3
        with self.assertRaisesRegex(ExecutionError, "resource contract"):
            _runtime_plan_external_gpu_lanes({"runtime_plan": drifted})

    def test_selector_phase_resume_skips_one_valid_immutable_receipt(self) -> None:
        record = {
            "branch": self.key.branch,
            "repeat": self.key.repeat,
            "n_train": self.key.n_train,
            "model": self.key.model,
            "queue": "gpu_neural",
            "ablation_id": None,
            "sensitivity_id": None,
        }
        storage = SimpleNamespace(
            manifest={
                **self.storage.manifest,
                "task_design": {"required_cells": [record]},
                "execution": {
                    "profile": "pilot10",
                    "smoke": {"enabled": False},
                    "benchmark": {"enabled": False},
                },
                "prepared": {
                    **self.storage.manifest["prepared"],
                    "path": "prepared",
                },
            },
            manifest_hash=self.storage.manifest_hash,
            validate_cell=lambda key: SimpleNamespace(status=CellStatus.MISSING),
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("bdb_study.execution.open_run", return_value=storage),
                patch("bdb_study.execution._effective_task_spec", return_value=object()),
                patch("bdb_study.execution.load_prepared_task", return_value=object()),
                patch("bdb_study.execution.TaskRuntime", return_value=object()),
                patch(
                    "bdb_study.execution.run_neural_selector",
                    return_value=_selector_core(),
                ) as selector,
                patch("bdb_study.execution.validate_neural_selector_for_cell"),
            ):
                first = run_repeat_queue(
                    directory,
                    repeat=1,
                    queue="gpu_neural",
                    repo_root=ROOT,
                    neural_phase="selector",
                )
                second = run_repeat_queue(
                    directory,
                    repeat=1,
                    queue="gpu_neural",
                    repo_root=ROOT,
                    neural_phase="selector",
                )
            self.assertEqual(first["completed"], 1)
            self.assertEqual(second["completed"], 0)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(selector.call_count, 1)


if __name__ == "__main__":
    unittest.main()
