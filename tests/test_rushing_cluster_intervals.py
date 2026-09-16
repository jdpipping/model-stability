"""Focused tests for the immutable BDB2020 no-refit interval companion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from rushing_study.cli import build_parser
from rushing_study.cluster_intervals import (
    ImmutableArtifactError,
    NUM_CLASSES,
    SourceValidationError,
    _finite_sample_padding,
    _generate_for_keys,
    validate_cluster_interval_artifact,
)
from rushing_study.design import SeedRegistry, manifest_seed
from rushing_study.metrics import evaluate_cell
from rushing_study.models import MODEL_IDS, frozen_model_config
from rushing_study.storage import (
    CellKey,
    atomic_write_json,
    initialize_run_dir,
    validate_checksum,
)


def _manifest() -> dict:
    config = {
        "study_id": "rushing_confirmatory_v1",
        "base_seed": 20260817,
        "seed_algorithm": "hmac-sha256-31bit-v1",
        "models": {model: {"stub": True} for model in MODEL_IDS},
        "splits": {"nested_train_anchors": [20, 40, 80, 160, 240, 360]},
        "execution": {"confirmatory_repeats": 1},
        "sensitivity": {
            "stage1": {"anchors": [20, 160, 360], "repeats": 1},
            "extension": {"target_total_repeats": 1},
        },
        "uncertainty": {
            "alpha": 0.1,
            "local_k": 5,
            "central_interval": "alpha_over_2_equal_tail",
        },
    }
    registry = SeedRegistry()
    for model in MODEL_IDS:
        for stage in ("fit", "epoch_selection", "refit", "prediction_rebuild"):
            registry.get("cell", 1, stage, model, 20)
    return {
        "study_id": config["study_id"],
        "config": config,
        "seed_derivation": {
            "algorithm": config["seed_algorithm"],
            "base_seed": config["base_seed"],
            "namespace": config["study_id"],
        },
        "seed_registry": registry.snapshot(),
    }


def _write_cell(storage, key: CellKey, *, probability_seed: int) -> None:
    rng = np.random.default_rng(probability_seed)
    calibration_proba = rng.dirichlet(np.ones(NUM_CLASSES), size=12)
    test_proba = rng.dirichlet(np.ones(NUM_CLASSES), size=8)
    calibration_metadata = pd.DataFrame(
        {
            "game_id": [f"cal-{index // 3}" for index in range(12)],
            "play_id": [f"cal-play-{index}" for index in range(12)],
            "season": [2017] * 12,
        }
    )
    test_metadata = pd.DataFrame(
        {
            "game_id": ["test-a"] * 3 + ["test-b"] * 5,
            "play_id": [f"test-play-{index}" for index in range(8)],
            "season": [2018] * 8,
        }
    )
    # Outcomes are held fixed across models, as they are in one frozen repeat.
    calibration_y = np.asarray([0, 10, 20, 30, 40, 50, 60, 70, 79, 5, 25, 55])
    test_y = np.asarray([0, 12, 24, 36, 48, 60, 72, 79])
    evaluated = evaluate_cell(
        calibration_proba,
        test_proba,
        calibration_y,
        test_y,
        calibration_metadata,
        test_metadata,
        alpha=0.1,
        local_k=5,
    )
    model_config = frozen_model_config(storage.manifest["config"], key.model)
    config_hash = hashlib.sha256(
        json.dumps(
            model_config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    history = {
        "fit_seed": manifest_seed(
            storage.manifest, "cell", key.repeat, "fit", key.model, key.n_train
        ),
        "model_config": model_config,
        "model_config_hash": config_hash,
    }
    predictions = pd.concat(
        [
            evaluated.calibration.assign(partition="calibration"),
            evaluated.test.assign(partition="test"),
        ],
        ignore_index=True,
    )
    storage.write_cell_artifacts(
        key,
        metrics={
            **evaluated.metrics,
            "branch": key.branch,
            "repeat": key.repeat,
            "n_train": key.n_train,
            "model": key.model,
            "manifest_hash": storage.manifest_hash,
            "model_config_hash": config_hash,
            "n_train_plays": 100,
            "parameter_count": 10,
            "elapsed_seconds": 0.1,
        },
        predictions=predictions,
        history=history,
        arrays=evaluated.arrays,
        preferred_predictions="csv",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ClusterIntervalSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.storage = initialize_run_dir(self.root / "source", _manifest())
        self.keys = [
            CellKey("main", 1, 20, "ridge_sgd_l2"),
            CellKey("main", 1, 20, "lightgbm_multiclass"),
        ]
        for index, key in enumerate(self.keys):
            _write_cell(self.storage, key, probability_seed=100 + index)

    def test_no_fit_deterministic_receipted_artifact(self):
        first = self.root / "cluster-first"
        second = self.root / "cluster-second"
        source_before = _tree_bytes(self.storage.run_dir)
        with mock.patch(
            "rushing_study.runner.run_cell", side_effect=AssertionError("fit called")
        ), mock.patch(
            "rushing_study.models.fit_ovr_logistic",
            side_effect=AssertionError("fit called"),
        ), mock.patch(
            "rushing_study.models.fit_neural_select_and_refit",
            side_effect=AssertionError("fit called"),
        ):
            first_result = _generate_for_keys(
                self.storage,
                self.keys,
                first,
                source_final_marker_sha256=None,
            )
            second_result = _generate_for_keys(
                self.storage,
                self.keys,
                second,
                source_final_marker_sha256=None,
            )

        self.assertEqual(first_result["receipt_sha256"], second_result["receipt_sha256"])
        self.assertEqual(_tree_bytes(first), _tree_bytes(second))
        self.assertEqual(_tree_bytes(self.storage.run_dir), source_before)
        receipt = validate_cluster_interval_artifact(first)
        validate_cluster_interval_artifact(
            first,
            source_run_dir=self.storage.run_dir,
        )
        self.assertTrue(receipt["no_fit_attestation"])
        self.assertEqual(receipt["cell_count"], 2)
        metrics = pd.read_csv(first / "cell_metrics.csv")
        self.assertEqual(metrics["uncertainty_subsample_seed"].nunique(), 1)
        self.assertEqual(metrics["selected_registry_semantic_sha256"].nunique(), 1)
        self.assertEqual(metrics["n_selected_calibration_plays"].tolist(), [4, 4])
        self.assertTrue(
            {
                "example_equal_coverage",
                "example_equal_mean_width",
                "game_equal_coverage",
                "game_equal_mean_width",
                "q",
            }.issubset(metrics.columns)
        )
        self.assertTrue(validate_checksum(first / "summary.csv"))
        self.assertTrue(validate_checksum(first / "source_binding.json"))

    def test_source_tamper_fails_before_output_is_created(self):
        history = self.storage.cell_dir(self.keys[0]) / "history.json"
        history.write_text("{}\n", encoding="utf-8")
        output = self.root / "should-not-exist"
        with self.assertRaisesRegex(SourceValidationError, "validation"):
            _generate_for_keys(
                self.storage,
                self.keys,
                output,
                source_final_marker_sha256=None,
            )
        self.assertFalse(output.exists())

    def test_existing_output_and_payload_tamper_fail_closed(self):
        output = self.root / "cluster"
        _generate_for_keys(
            self.storage,
            self.keys,
            output,
            source_final_marker_sha256=None,
        )
        with self.assertRaises(ImmutableArtifactError):
            _generate_for_keys(
                self.storage,
                self.keys,
                output,
                source_final_marker_sha256=None,
            )
        (output / "summary.csv").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ImmutableArtifactError, "failed checksum"):
            validate_cluster_interval_artifact(output)

    def test_source_replay_rejects_self_consistent_but_wrong_metric(self):
        output = self.root / "cluster-receipted-tamper"
        _generate_for_keys(
            self.storage,
            self.keys,
            output,
            source_final_marker_sha256=None,
        )
        key = self.keys[0]
        cell_dir = output / key.relative_dir
        metrics_path = cell_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["q"] = int(metrics["q"]) + 1
        metrics_record = atomic_write_json(metrics_path, metrics)

        cell_marker_path = cell_dir / "_SUCCESS"
        cell_marker = json.loads(cell_marker_path.read_text(encoding="utf-8"))
        cell_marker["artifacts"]["metrics"] = metrics_record.as_dict()
        cell_marker_record = atomic_write_json(cell_marker_path, cell_marker)

        receipt_path = output / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for entry in receipt["cell_markers"]:
            if entry["key"] == {
                "branch": key.branch,
                "repeat": key.repeat,
                "n_train": key.n_train,
                "model": key.model,
            }:
                entry["marker_sha256"] = cell_marker_record.sha256
                break
        receipt_record = atomic_write_json(receipt_path, receipt)
        root_marker_path = output / "_SUCCESS"
        root_marker = json.loads(root_marker_path.read_text(encoding="utf-8"))
        root_marker["receipt_sha256"] = receipt_record.sha256
        atomic_write_json(root_marker_path, root_marker)

        # The complete checksum chain is internally consistent.
        validate_cluster_interval_artifact(output)
        # Source-aware verification independently rebuilds q and fails closed.
        with self.assertRaisesRegex(SourceValidationError, "independent no-refit replay"):
            validate_cluster_interval_artifact(
                output,
                source_run_dir=self.storage.run_dir,
            )

    def test_finite_sample_rule_and_full_support_fallback(self):
        q, rank, fallback = _finite_sample_padding(np.arange(9))
        self.assertEqual(rank, 9)
        self.assertEqual(q, 8)
        self.assertFalse(fallback)
        q, rank, fallback = _finite_sample_padding(np.arange(8))
        self.assertEqual(rank, 9)
        self.assertEqual(q, NUM_CLASSES - 1)
        self.assertTrue(fallback)

    def test_cli_exposes_generate_and_verify_commands(self):
        parser = build_parser()
        generated = parser.parse_args(
            [
                "cluster-interval-sensitivity",
                "--run-dir",
                "source",
                "--output-dir",
                "companion",
            ]
        )
        verified = parser.parse_args(
            [
                "verify-cluster-interval-sensitivity",
                "--run-dir",
                "source",
                "--output-dir",
                "companion",
            ]
        )
        self.assertEqual(generated.run_dir, Path("source"))
        self.assertEqual(verified.output_dir, Path("companion"))


if __name__ == "__main__":
    unittest.main()
