from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from bdb_study.analysis import descriptive_suite_summary
from bdb_study.cli import build_parser, command_pilot_report
from bdb_study.pilot_report import (
    COMMON_ANCHORS,
    MODEL_FAMILIES,
    MODELS,
    PILOT_REPEATS,
    TASK_ANCHORS,
    TASKS,
    generate_pilot_report,
    load_validated_pilot_aggregate,
)
from bdb_study.storage import atomic_write_csv, atomic_write_json


class PilotReportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        frames: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, object]] = []
        for task_index, task_id in enumerate(TASKS):
            task_rows = []
            for repeat in PILOT_REPEATS:
                for anchor in TASK_ANCHORS[task_id]:
                    for model_index, model in enumerate(MODELS):
                        primary = (
                            0.210
                            + 0.006 * task_index
                            + 0.003 * model_index
                            - 0.00005 * anchor
                            + 0.0001 * repeat
                        )
                        null = 0.36 + 0.007 * task_index
                        row = {
                            "task_id": task_id,
                            "branch": "fixed_main",
                            "repeat": repeat,
                            "n_train": anchor,
                            "model": model,
                            "family": MODEL_FAMILIES[model],
                            "primary_loss": primary,
                            "game_equal_loss": primary + 0.001,
                            "null_loss": null,
                            "skill": 1.0 - primary / null,
                            "evidence_status": "exploratory_provisional",
                            "profile": "pilot10",
                        }
                        rows.append(row)
                        task_rows.append(row)
            frames[task_id] = pd.DataFrame(task_rows)

        metrics = pd.DataFrame(rows).sort_values(
            ["task_id", "repeat", "n_train", "model"]
        ).reset_index(drop=True)
        summary = descriptive_suite_summary(
            frames,
            require_complete_task_ids=TASKS,
        )
        summary["evidence_status"] = "exploratory_provisional"
        summary["profile"] = "pilot10"

        summary_path = root / "pilot10_suite_summary.csv"
        metrics_path = root / "pilot10_suite_summary_task_metrics.csv"
        summary_record = atomic_write_csv(summary_path, summary)
        metrics_record = atomic_write_csv(metrics_path, metrics)
        components = {}
        for task_id in TASKS:
            components[task_id] = {
                "task_id": task_id,
                "run_dir": f"../runs/{task_id}/pilot10",
                "manifest_hash": hashlib.sha256(f"{task_id}:manifest".encode()).hexdigest(),
                "manifest_file_sha256": hashlib.sha256(f"{task_id}:file".encode()).hexdigest(),
                "final_marker_sha256": hashlib.sha256(f"{task_id}:marker".encode()).hexdigest(),
                "final_receipt_sha256": hashlib.sha256(f"{task_id}:receipt".encode()).hexdigest(),
                "metrics_sha256": hashlib.sha256(f"{task_id}:metrics".encode()).hexdigest(),
            }
        receipt = {
            "schema_version": "bdb-pilot-suite-aggregate-v1",
            "profile": "pilot10",
            "evidence_status": "exploratory_provisional",
            "suite_manifest": "suite_manifest.json",
            "suite_hash": "e" * 64,
            "tasks": list(TASKS),
            "task_components": components,
            "task_count": 6,
            "primary_cells_per_task": 300,
            "combined_primary_cells": 1_800,
            "repeat_ids": list(PILOT_REPEATS),
            "common_anchors": list(COMMON_ANCHORS),
            "analysis_scope": "provisional_task_equal_descriptive_mockup_only",
            "inference": "none_no_cross_task_ci_no_confirmatory_claims",
            "model_winner_claim": "prohibited_for_pilot_suite",
            "overlap_disclosure": (
                "NFL seasons overlap across releases; BDB2024 and BDB2025 use the same 2022 games"
            ),
            "summary": summary_record.as_dict(),
            "task_metrics": metrics_record.as_dict(),
        }
        receipt_path = root / "pilot10_suite_summary.receipt.json"
        atomic_write_json(receipt_path, receipt)
        return receipt_path, summary_path, metrics_path

    def _rewrite_receipt_artifact(
        self,
        receipt_path: Path,
        field: str,
        record: object,
    ) -> None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt[field] = record.as_dict()
        atomic_write_json(receipt_path, receipt)

    def test_accepts_exact_receipt_and_generates_visibly_provisional_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, _, _ = self._fixture(root)
            aggregate = load_validated_pilot_aggregate(receipt_path)
            self.assertEqual(len(aggregate.task_metrics), 1_800)
            self.assertEqual(len(aggregate.summary), 20)

            result = generate_pilot_report(
                receipt_path,
                output_dir=root / "mockup",
                output_stem="pilot_mock",
            )
            for field in ("png", "pdf", "report", "receipt"):
                path = Path(result[field])
                self.assertTrue(path.is_file())
                self.assertTrue(path.with_name(f"{path.name}.sha256").is_file())
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("EXPLORATORY / PROVISIONAL — 10 RERUNS PER TASK", report)
            self.assertIn("does not pool task-native uncertainty", report)
            self.assertIn("no cross-task CI and no confirmatory claim", report)
            for label in (
                "Linear-Structure",
                "Boosted-Structure",
                "RelNet",
                "AttnRelNet",
                "Global Set Transformer",
            ):
                self.assertIn(label, report)
            output_receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
            self.assertEqual(
                output_receipt["schema_version"],
                "bdb-pilot10-provisional-report-v1",
            )
            self.assertEqual(
                output_receipt["model_winner_claim"],
                "prohibited_for_pilot_suite",
            )
            self.assertEqual(
                output_receipt["validated_contract"]["combined_primary_cells"],
                1_800,
            )

    def test_rejects_payload_tamper_against_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, summary_path, _ = self._fixture(root)
            summary_path.write_bytes(summary_path.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "artifact drifted"):
                load_validated_pilot_aggregate(receipt_path)

    def test_rejects_grid_tamper_even_when_all_checksums_are_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, _, metrics_path = self._fixture(root)
            metrics = pd.read_csv(metrics_path)
            metrics.loc[0, "repeat"] = 2
            record = atomic_write_csv(metrics_path, metrics)
            self._rewrite_receipt_artifact(receipt_path, "task_metrics", record)
            with self.assertRaisesRegex(ValueError, "grid mismatch|duplicate"):
                load_validated_pilot_aggregate(receipt_path)

    def test_rejects_summary_or_receipt_semantic_tamper_with_fresh_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path, summary_path, _ = self._fixture(root)
            summary = pd.read_csv(summary_path)
            summary.loc[0, "task_equal_mean_skill"] += 0.1
            record = atomic_write_csv(summary_path, summary)
            self._rewrite_receipt_artifact(receipt_path, "summary", record)
            with self.assertRaisesRegex(ValueError, "does not independently recompute"):
                load_validated_pilot_aggregate(receipt_path)

            receipt_path, _, _ = self._fixture(root / "second")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["evidence_status"] = "confirmatory_definitive"
            atomic_write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "not an admitted"):
                load_validated_pilot_aggregate(receipt_path)

    def test_cli_exposes_only_receipt_based_generator(self) -> None:
        parsed = build_parser().parse_args(
            [
                "pilot-report",
                "--aggregate-receipt",
                "aggregate.receipt.json",
                "--output-dir",
                "mockup",
            ]
        )
        self.assertIs(parsed.func, command_pilot_report)
        self.assertEqual(parsed.aggregate_receipt, "aggregate.receipt.json")
        self.assertFalse(hasattr(parsed, "task_run"))


if __name__ == "__main__":
    unittest.main()
