from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from bdb_study.cli import (
    _validated_v2_suite_component,
    build_parser,
    command_aggregate_suite,
    command_dev_cv,
    command_dev_cv_reduce,
    command_fidelity_bdb2024,
    command_freeze,
    command_plan,
    command_prepare,
    command_suite_plan,
    command_verify_determinism,
)


TASKS = (
    "bdb2021_completion",
    "bdb2022_punt_returns",
    "bdb2023_sack",
    "bdb2024_tackle",
    "bdb2025_man_zone",
    "bdb2026_trajectory",
)


class CliContractTests(unittest.TestCase):
    def test_public_commands_parse(self):
        parser = build_parser()
        for command in (
            "inventory", "import-data", "prepare", "dev-cv", "dev-cv-reduce",
            "freeze", "plan",
            "run", "status", "aggregate-task", "aggregate-suite", "suite-plan",
            "fidelity-bdb2024",
            "verify-determinism", "synthetic-smoke",
        ):
            self.assertIn(command, parser.format_help())
        self.assertNotIn("bridge", parser.format_help())

        shard = parser.parse_args(
            [
                "dev-cv",
                "--task",
                "bdb2023_sack",
                "--family",
                "relnet",
                "--fold",
                "2",
                "--candidate",
                "3",
                "--resume",
            ]
        )
        self.assertEqual(shard.fold, [2])
        self.assertEqual(shard.candidate, [3])
        self.assertTrue(shard.resume)
        set_shard = parser.parse_args(
            [
                "dev-cv",
                "--task",
                "bdb2023_sack",
                "--family",
                "set_transformer",
            ]
        )
        self.assertEqual(set_shard.family, "set_transformer")
        reducer = parser.parse_args(
            [
                "dev-cv-reduce",
                "--task",
                "bdb2023_sack",
                "--family",
                "set_transformer",
            ]
        )
        self.assertIs(reducer.func, command_dev_cv_reduce)
        self.assertEqual(reducer.family, "set_transformer")
        pilot_plan = parser.parse_args(
            [
                "plan",
                "--task",
                "bdb2023_sack",
                "--run-dir",
                "pilot-run",
                "--profile",
                "pilot10",
            ]
        )
        self.assertEqual(pilot_plan.profile, "pilot10")
        with self.assertRaisesRegex(ValueError, "require exactly one --family"):
            command_dev_cv(
                argparse.Namespace(
                    repo_root=".",
                    task="bdb2023_sack",
                    prepared_dir=None,
                    output_dir=None,
                    family=None,
                    resume=True,
                    fold=[1],
                    candidate=[0],
                )
            )

    def test_dev_cv_filters_are_checkpoint_only_and_reducer_is_fit_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "data" / "processed" / "bdb_suite" / "bdb2023_sack"
            output = root / "development" / "bdb2023_sack"
            prepared.mkdir(parents=True)
            task = object()
            spec = SimpleNamespace(
                models={
                    "linear_structure": {"family": "glm"},
                    "boosted_structure": {"family": "lightgbm"},
                    "relnet": {"family": "relnet"},
                    "attn_relnet": {"family": "attn_relnet"},
                }
            )
            provenance = {"identity": "stable"}
            design = {"task_id": "bdb2023_sack"}
            shard_result = {
                "schema_version": "bdb-development-cv-shard-v1",
                "final_receipt_emitted": False,
            }
            arguments = argparse.Namespace(
                repo_root=str(root),
                task="bdb2023_sack",
                prepared_dir=str(prepared),
                output_dir=str(output),
                family="relnet",
                resume=True,
                fold=[2],
                candidate=[3],
            )
            with (
                patch(
                    "bdb_study.manifest.build_development_provenance",
                    return_value=provenance,
                ),
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch("bdb_study.contracts.load_task_spec", return_value=spec),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch("bdb_study.design.build_task_design", return_value=design),
                patch("bdb_study.runner.TaskRuntime", return_value=object()),
                patch("bdb_study.runner.development_evaluator", return_value=object()),
                patch(
                    "bdb_study.devcv.run_development_cv_shard",
                    return_value=shard_result,
                ) as shard,
                patch("bdb_study.devcv.run_development_cv") as monolithic,
                patch("bdb_study.devcv.write_development_receipt") as writer,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                command_dev_cv(arguments)
            monolithic.assert_not_called()
            writer.assert_not_called()
            shard.assert_called_once()
            self.assertEqual(shard.call_args.kwargs["folds"], [2])
            self.assertEqual(shard.call_args.kwargs["candidate_indices"], [3])
            self.assertEqual(
                shard.call_args.kwargs["checkpoint_dir"],
                output.resolve() / ".checkpoints" / "relnet",
            )

            receipt = {
                "receipt_hash": "a" * 64,
                "selected": {"candidate_index": 3},
                "fold_scores": [{}] * 20,
            }
            reduce_arguments = argparse.Namespace(
                repo_root=str(root),
                task="bdb2023_sack",
                prepared_dir=str(prepared),
                output_dir=str(output),
                family="relnet",
            )
            with (
                patch(
                    "bdb_study.manifest.build_development_provenance",
                    return_value=provenance,
                ),
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch("bdb_study.contracts.load_task_spec", return_value=spec),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch("bdb_study.design.build_task_design", return_value=design),
                patch(
                    "bdb_study.devcv.reduce_development_checkpoints",
                    return_value=receipt,
                ) as reducer,
                patch(
                    "bdb_study.devcv.write_development_receipt",
                    return_value=output / "relnet.json",
                ) as writer,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                command_dev_cv_reduce(reduce_arguments)
            reducer.assert_called_once_with(
                task,
                design,
                "relnet",
                model_id="relnet",
                provenance=provenance,
                checkpoint_dir=output.resolve() / ".checkpoints" / "relnet",
            )
            writer.assert_called_once_with(
                receipt, output.resolve() / "relnet.json"
            )

    def test_unfiltered_dev_cv_runs_all_five_primary_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared" / "bdb2023_sack"
            output = root / "development" / "bdb2023_sack"
            prepared.mkdir(parents=True)
            task = object()
            families = (
                "glm",
                "lightgbm",
                "relnet",
                "attn_relnet",
                "set_transformer",
            )
            spec = SimpleNamespace(
                models={family: {"family": family} for family in families}
            )
            arguments = argparse.Namespace(
                repo_root=str(root),
                task="bdb2023_sack",
                prepared_dir=str(prepared),
                output_dir=str(output),
                family=None,
                resume=False,
                fold=None,
                candidate=None,
            )
            receipt = {"selected": {"candidate_index": 0}}
            with (
                patch(
                    "bdb_study.manifest.build_development_provenance",
                    return_value={"identity": "stable"},
                ),
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch("bdb_study.contracts.load_task_spec", return_value=spec),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch("bdb_study.design.build_task_design", return_value={}),
                patch("bdb_study.runner.TaskRuntime", return_value=object()),
                patch("bdb_study.runner.development_evaluator", return_value=object()),
                patch(
                    "bdb_study.devcv.run_development_cv",
                    return_value=receipt,
                ) as run_cv,
                patch(
                    "bdb_study.devcv.write_development_receipt",
                    side_effect=lambda _receipt, path: path,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                command_dev_cv(arguments)
            self.assertEqual(
                [call.args[2] for call in run_cv.call_args_list],
                list(families),
            )

    def test_prepare_verifies_canonical_source_tree_and_semantic_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "data" / "bdb2023" / "raw"
            output = root / "data" / "processed" / "bdb_suite" / "bdb2023_sack"
            raw.mkdir(parents=True)
            source = SimpleNamespace(
                destination="data/bdb2023/raw",
                source_id="bdb2023-local",
                tree_sha256="a" * 64,
            )
            spec = SimpleNamespace(
                source_tree_receipts=(source,),
                prepared_contract={"schema_version": "contract"},
            )
            task = object()
            adapter = SimpleNamespace(prepare=Mock(return_value=task))
            arguments = argparse.Namespace(
                repo_root=str(root), task="bdb2023_sack", raw_dir=str(raw),
                output_dir=str(output), max_examples=None, audit_only=False,
            )
            with (
                patch("bdb_study.adapters.get_adapter", return_value=adapter),
                patch("bdb_study.contracts.load_task_spec", return_value=spec),
                patch("bdb_study.contracts.validate_source_tree_receipts") as source_check,
                patch("bdb_study.prepared.validate_prepared_task_contract") as semantic_check,
                patch(
                    "bdb_study.prepared.write_prepared_task",
                    return_value={
                        "prepared_hash": "p" * 64,
                        "examples": 8_533,
                        "games": 122,
                    },
                ) as writer,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                command_prepare(arguments)
            source_check.assert_called_once_with(
                spec.source_tree_receipts, repo_root=root.resolve(), verify_tree=True
            )
            semantic_check.assert_called_once_with(task, spec.prepared_contract)
            writer.assert_called_once_with(
                task, output.resolve(),
                source_receipt_hashes={"bdb2023-local": "a" * 64},
            )

    def test_prepare_rejects_unstamped_raw_tree_and_canonical_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_raw = root / "data" / "bdb2023" / "raw"
            canonical_raw.mkdir(parents=True)
            spec = SimpleNamespace(
                source_tree_receipts=(
                    SimpleNamespace(
                        destination="data/bdb2023/raw",
                        source_id="source",
                        tree_sha256="a" * 64,
                    ),
                )
            )
            base = {
                "repo_root": str(root),
                "task": "bdb2023_sack",
                "audit_only": False,
            }
            wrong_raw = argparse.Namespace(
                **base, raw_dir=str(root / "other-raw"),
                output_dir=str(root / "diagnostic"), max_examples=None,
            )
            adapter = SimpleNamespace(prepare=Mock())
            with (
                patch("bdb_study.adapters.get_adapter", return_value=adapter),
                patch("bdb_study.contracts.load_task_spec", return_value=spec),
            ):
                with self.assertRaisesRegex(ValueError, "TaskSpec source tree"):
                    command_prepare(wrong_raw)
            adapter.prepare.assert_not_called()

            partial = argparse.Namespace(
                **base, raw_dir=str(canonical_raw), output_dir=None, max_examples=10,
            )
            with patch("bdb_study.adapters.get_adapter", return_value=adapter):
                with self.assertRaisesRegex(ValueError, "diagnostic only"):
                    command_prepare(partial)
            adapter.prepare.assert_not_called()

    def test_scientific_artifact_paths_fail_before_expensive_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            outside = Path(temporary) / "outside"
            cases = (
                (
                    command_dev_cv,
                    argparse.Namespace(
                        repo_root=str(root), task="bdb2023_sack",
                        prepared_dir=str(outside), output_dir=None,
                        family=None, resume=False,
                    ),
                ),
                (
                    command_freeze,
                    argparse.Namespace(
                        repo_root=str(root), task="bdb2023_sack",
                        prepared_dir=str(outside), development_dir=None,
                        output=None, verify_full_source_trees=False,
                    ),
                ),
                (
                    command_plan,
                    argparse.Namespace(
                        repo_root=str(root), task="bdb2023_sack",
                        frozen_task=str(outside / "frozen.json"), prepared_dir=None,
                        run_dir=str(outside / "run"), repeats=50,
                        cpu_workers=12, smoke=False, preflight_receipt=None,
                    ),
                ),
                (
                    command_verify_determinism,
                    argparse.Namespace(
                        repo_root=str(root), run_a=str(outside / "a"),
                        run_b=str(outside / "b"), output=str(outside / "receipt.json"),
                    ),
                ),
            )
            for function, arguments in cases:
                with self.subTest(command=function.__name__):
                    with self.assertRaisesRegex(ValueError, "beneath repo_root"):
                        function(arguments)

    def test_suite_plan_is_hash_bound_idempotent_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "suite" / "suite.json"
            import hashlib

            def task_hash(task_id: str) -> str:
                return hashlib.sha256(task_id.encode("utf-8")).hexdigest()

            def fake_component(run_dir, task_id, profile, *, repo_root):
                path = Path(run_dir).resolve()
                binding = {
                    "task_id": task_id,
                    "run_dir": str(path),
                    "manifest_hash": task_hash(task_id),
                    "manifest_file_sha256": "a" * 64,
                    "final_marker_sha256": "b" * 64,
                    "final_receipt_sha256": "c" * 64,
                    "metrics_sha256": "d" * 64,
                }
                if profile == "sensitivity20":
                    binding["primary_reference"] = {
                        "manifest_hash": task_hash(task_id)
                    }
                return SimpleNamespace(run_dir=path), binding, path / "metrics.csv"

            def fake_bdb2020(run_dir, *, repo_root):
                path = Path(run_dir).resolve()
                return {
                    "schema_version": "bdb2020-readonly-reference-v1",
                    "run_dir": str(path),
                    "manifest_hash": "e" * 64,
                    "reference_anchors": [20, 40],
                }, pd.DataFrame()

            arguments = argparse.Namespace(
                repo_root=str(root),
                task_run=[f"{task}={root / 'runs' / task}" for task in TASKS],
                sensitivity_run=[
                    f"{task}={root / 'sensitivities' / task}"
                    for task in ("bdb2024_tackle", "bdb2025_man_zone")
                ],
                bdb2020_run_dir=str(root / "runs" / "bdb2020_rushing"),
                output=str(output),
            )
            with (
                patch(
                    "bdb_study.cli._validated_v2_suite_component",
                    side_effect=fake_component,
                ),
                patch(
                    "bdb_study.cli._validated_bdb2020_reference",
                    side_effect=fake_bdb2020,
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    command_suite_plan(arguments)
                before = output.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()):
                    command_suite_plan(arguments)
                self.assertEqual(output.read_bytes(), before)
                value = json.loads(before)
                self.assertEqual(value["schema_version"], "bdb-suite-manifest-v3")
                self.assertEqual(set(value["task_runs"]), set(TASKS))
                self.assertEqual(
                    set(value["sensitivity_runs"]),
                    {"bdb2024_tackle", "bdb2025_man_zone"},
                )
                for task in TASKS:
                    self.assertEqual(value["task_runs"][task]["task_id"], task)
                    self.assertEqual(
                        value["task_runs"][task]["manifest_hash"], task_hash(task)
                    )
                self.assertEqual(len(value["suite_hash"]), 64)
                changed = argparse.Namespace(**vars(arguments))
                changed.bdb2020_run_dir = str(root / "runs" / "other-bdb2020")
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileExistsError):
                        command_suite_plan(changed)

    def test_pilot_suite_plan_is_six_task_only_and_provisionally_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "suite" / "pilot.json"

            def fake_component(run_dir, task_id, profile, *, repo_root):
                self.assertEqual(profile, "pilot10")
                path = Path(run_dir).resolve()
                binding = {
                    "task_id": task_id,
                    "run_dir": str(path),
                    "manifest_hash": task_id.ljust(64, "0")[:64],
                    "manifest_file_sha256": "a" * 64,
                    "final_marker_sha256": "b" * 64,
                    "final_receipt_sha256": "c" * 64,
                    "metrics_sha256": "d" * 64,
                }
                return SimpleNamespace(run_dir=path), binding, path / "metrics.csv"

            arguments = argparse.Namespace(
                repo_root=str(root),
                profile="pilot10",
                task_run=[f"{task}={root / 'runs' / task}" for task in TASKS],
                sensitivity_run=None,
                bdb2020_run_dir=None,
                output=str(output),
            )
            with patch(
                "bdb_study.cli._validated_v2_suite_component",
                side_effect=fake_component,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    command_suite_plan(arguments)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["schema_version"], "bdb-pilot-suite-manifest-v1")
            self.assertEqual(value["profile"], "pilot10")
            self.assertEqual(value["evidence_status"], "exploratory_provisional")
            self.assertEqual(set(value["task_runs"]), set(TASKS))
            self.assertNotIn("sensitivity_runs", value)
            self.assertNotIn("bdb2020_reference", value)

            rejected = argparse.Namespace(**vars(arguments))
            rejected.output = str(root / "suite" / "rejected.json")
            rejected.sensitivity_run = [
                f"bdb2024_tackle={root / 'runs' / 'sensitivity'}"
            ]
            with self.assertRaisesRegex(ValueError, "does not accept sensitivity"):
                command_suite_plan(rejected)

    def test_pilot_suite_aggregate_is_exact_1800_cell_provisional_mockup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "pilot.json"
            anchors = [10, 20, 40, 60, 100, 140]
            models = {
                "linear_structure": {"family": "glm"},
                "boosted_structure": {"family": "lightgbm"},
                "relnet": {"family": "relnet"},
                "attn_relnet": {"family": "attn_relnet"},
                "set_transformer": {"family": "set_transformer"},
            }
            entries = {}
            fixtures = {}
            for task_index, task_id in enumerate(TASKS):
                run_dir = root / "runs" / task_id
                run_dir.mkdir(parents=True)
                rows = []
                for repeat in range(1, 11):
                    for anchor in anchors:
                        for model_index, (model, model_entry) in enumerate(models.items()):
                            loss = 0.3 + 0.001 * task_index + 0.0001 * repeat + 0.002 * model_index
                            rows.append(
                                {
                                    "task_id": task_id,
                                    "repeat": repeat,
                                    "n_train": anchor,
                                    "model": model,
                                    "family": model_entry["family"],
                                    "primary_loss": loss,
                                    "game_equal_loss": loss + 0.001,
                                    "null_loss": 0.5,
                                }
                            )
                metrics_path = run_dir / "primary_metrics.csv"
                pd.DataFrame(rows).to_csv(metrics_path, index=False)
                binding = {
                    "task_id": task_id,
                    "run_dir": str(run_dir),
                    "manifest_hash": (str(task_index + 1) * 64)[:64],
                    "manifest_file_sha256": "a" * 64,
                    "final_marker_sha256": "b" * 64,
                    "final_receipt_sha256": "c" * 64,
                    "metrics_sha256": "d" * 64,
                }
                entries[task_id] = binding
                shared_registry = {"registry_hash": "shared"}
                fixtures[task_id] = (
                    SimpleNamespace(
                        run_dir=run_dir,
                        manifest={
                            "task_design": {
                                "anchors": anchors,
                                "game_registry": shared_registry,
                            },
                            "task_spec": {"models": models},
                        },
                    ),
                    dict(binding),
                    metrics_path,
                )
            suite = {
                "schema_version": "bdb-pilot-suite-manifest-v1",
                "profile": "pilot10",
                "evidence_status": "exploratory_provisional",
                "task_runs": entries,
            }
            import hashlib

            suite["suite_hash"] = hashlib.sha256(
                json.dumps(suite, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest_path.write_text(json.dumps(suite), encoding="utf-8")
            output = root / "pilot_summary.csv"

            def fake_component(run_dir, task_id, profile, *, repo_root):
                self.assertEqual(profile, "pilot10")
                return fixtures[task_id]

            arguments = argparse.Namespace(
                suite_manifest=str(manifest_path),
                repo_root=str(root),
                output=str(output),
            )
            with patch(
                "bdb_study.cli._validated_v2_suite_component",
                side_effect=fake_component,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    command_aggregate_suite(arguments)
            summary = pd.read_csv(output)
            combined = pd.read_csv(root / "pilot_summary_task_metrics.csv")
            receipt = json.loads(
                (root / "pilot_summary.receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary), 20)
            self.assertEqual(set(summary["tasks"]), {6})
            self.assertEqual(len(combined), 1_800)
            self.assertEqual(set(combined["repeat"]), set(range(1, 11)))
            self.assertEqual(receipt["profile"], "pilot10")
            self.assertEqual(receipt["evidence_status"], "exploratory_provisional")
            self.assertEqual(receipt["combined_primary_cells"], 1_800)
            self.assertEqual(receipt["inference"], "none_no_cross_task_ci_no_confirmatory_claims")

    def test_pilot_suite_component_rejects_nonpilot_manifest_and_receipt(self):
        task_id = "bdb2023_sack"
        full_storage = SimpleNamespace(
            manifest={
                "task_id": task_id,
                "execution": {"mode": "definitive", "profile": "full100"},
                "task_design": {"profile": "full100"},
            }
        )
        with patch("bdb_study.execution.open_run", return_value=full_storage):
            with self.assertRaisesRegex(ValueError, "not an admitted pilot10"):
                _validated_v2_suite_component(
                    "/tmp/full100",
                    task_id,
                    "pilot10",
                    repo_root="/tmp",
                )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            final = run_dir / "final"
            final.mkdir()
            (final / "primary_metrics.csv").write_text("task_id\n", encoding="utf-8")
            receipt = {
                "task_id": task_id,
                "manifest_hash": "m" * 64,
                "cells": 300,
                "primary_cells": 300,
                "ablation_cells": 0,
                "sensitivity_cells": 0,
                "primary_summary_groups": 30,
                "scientific_revalidation": (
                    "ordered_identities_splits_seeds_config_history_predictions_all_metrics"
                ),
                "evidence_status": "confirmatory_definitive",
                "analysis_scope": "pilot10_descriptive_only_no_confirmatory_inference",
                "bootstrap_draws": 0,
                "max_t_family": "none",
                "artifacts": {},
            }
            (final / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            pilot_storage = SimpleNamespace(
                run_dir=run_dir,
                manifest_hash="m" * 64,
                manifest={
                    "task_id": task_id,
                    "execution": {
                        "mode": "pilot",
                        "profile": "pilot10",
                        "evidence_status": "exploratory_provisional",
                    },
                    "task_design": {
                        "profile": "pilot10",
                        "cell_counts": {
                            "primary": 300,
                            "structural_ablation": 0,
                            "frozen_sensitivity": 0,
                            "required": 300,
                        },
                    },
                },
                validate_final=lambda _keys: True,
            )
            with (
                patch("bdb_study.execution.open_run", return_value=pilot_storage),
                patch("bdb_study.storage.cell_keys_from_design", return_value=list(range(300))),
                patch("bdb_study.storage.validate_checksum", return_value=True),
            ):
                with self.assertRaisesRegex(ValueError, "descriptive/provisional"):
                    _validated_v2_suite_component(
                        run_dir,
                        task_id,
                        "pilot10",
                        repo_root=run_dir,
                    )

    def test_fidelity_result_is_hash_bound_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fidelity.json"
            task = SimpleNamespace(
                task_id="bdb2024_tackle",
                examples=pd.DataFrame({"week": [1, 8, 9, 9]}),
                tabular=pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}),
                y=np.array([0, 1, 0, 1], dtype=np.int8),
            )
            fitted = SimpleNamespace(parameters={"n_estimators": 150})
            arguments = argparse.Namespace(
                prepared_dir=str(Path(temporary) / "prepared"),
                output=str(output),
                verbose=False,
            )
            with (
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch(
                    "bdb_study.prepared.load_prepared_receipt",
                    return_value={"prepared_hash": "d" * 64},
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.fit_executed_xgboost",
                    return_value=fitted,
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.predict_fidelity_score",
                    return_value=np.array([0.2, 0.8]),
                ),
                patch("bdb_study.fidelity.bdb2024.validate_fidelity_counts"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    command_fidelity_bdb2024(arguments)
                before = output.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()):
                    command_fidelity_bdb2024(arguments)
            value = json.loads(before)
            claimed = value.pop("result_hash")
            import hashlib

            actual = hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.assertEqual(claimed, actual)
            self.assertEqual(value["prepared_hash"], "d" * 64)
            self.assertEqual(output.read_bytes(), before)

            with (
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch(
                    "bdb_study.prepared.load_prepared_receipt",
                    return_value={"prepared_hash": "d" * 64},
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.fit_executed_xgboost",
                    return_value=fitted,
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.predict_fidelity_score",
                    return_value=np.array([0.3, 0.7]),
                ),
                patch("bdb_study.fidelity.bdb2024.validate_fidelity_counts"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileExistsError):
                        command_fidelity_bdb2024(arguments)

    def test_aggregate_suite_rejects_task_key_identity_mismatch_before_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_runs = {
                task: {
                    "task_id": task,
                    "run_dir": f"../runs/{task}",
                    "manifest_hash": "a" * 64,
                    "manifest_file_sha256": "0" * 64,
                    "final_marker_sha256": "b" * 64,
                    "final_receipt_sha256": "c" * 64,
                    "metrics_sha256": "d" * 64,
                }
                for task in TASKS
            }
            task_runs[TASKS[0]]["task_id"] = "bdb_wrong_task"
            suite = {
                "schema_version": "bdb-suite-manifest-v3",
                "task_runs": task_runs,
                "sensitivity_runs": {
                    task: {
                        **task_runs[task],
                        "primary_reference": {"manifest_hash": "a" * 64},
                    }
                    for task in ("bdb2024_tackle", "bdb2025_man_zone")
                },
                "bdb2020_reference": {"run_dir": "../runs/bdb2020_rushing"},
            }
            import hashlib

            suite["suite_hash"] = hashlib.sha256(
                json.dumps(suite, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            manifest = root / "suite.json"
            manifest.write_text(json.dumps(suite), encoding="utf-8")
            arguments = argparse.Namespace(
                suite_manifest=str(manifest),
                repo_root=str(root),
                output=None,
            )
            with patch("bdb_study.cli._validated_v2_suite_component") as opened:
                with self.assertRaisesRegex(ValueError, "key/identity mismatch"):
                    command_aggregate_suite(arguments)
            opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
