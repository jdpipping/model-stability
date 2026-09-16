from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from bdb_study.preflight import (
    BETTY_GPU_LANES,
    BETTY_WALLTIME_SECONDS,
    PreflightError,
    RUNTIME_JOB_GUARD_SECONDS,
    RUNTIME_JOB_OVERHEAD_SECONDS,
    RUNTIME_PLAN_SCHEMA_VERSION,
    RUNTIME_PLAN_V3_SCHEMA_VERSION,
    RUNTIME_SHARDING,
    RUNTIME_V3_SHARDING,
    _assert_predictions_equal,
    _canonical_runtime_queue_plans,
    _json_hash,
    _validate_runtime_plan_v3_value,
    build_runtime_plan,
    process_peak_rss_bytes,
    validate_preflight_receipt,
    validate_runtime_queue_plans,
    validate_runtime_plan,
)
from bdb_study.runtime_phases import betty_runtime_v3_resource_contract


class PreflightTests(unittest.TestCase):
    def test_v3_self_rehashed_plan_still_replays_benchmark_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "task/runtime_plan.json"
            code = [{"path": "bdb_study/example.py", "sha256": "a" * 64}]
            manifest = {
                "task_id": "bdb2023_sack",
                "task_spec_hash": "b" * 64,
                "task_spec": {},
                "prepared": {"prepared_hash": "c" * 64},
                "preflight": {"receipt_hash": "d" * 64},
                "execution": {"profile": "pilot10"},
                "provenance": {"code": code},
            }
            resource = betty_runtime_v3_resource_contract()
            value = {
                "schema_version": RUNTIME_PLAN_V3_SCHEMA_VERSION,
                "profile": "pilot10",
                "task_id": manifest["task_id"],
                "task_spec_hash": manifest["task_spec_hash"],
                "prepared_hash": manifest["prepared"]["prepared_hash"],
                "preflight_receipt_hash": manifest["preflight"]["receipt_hash"],
                "code_identity_sha256": _json_hash({"code": code}),
                "benchmark_run": "task/benchmark",
                "benchmark_manifest_hash": "e" * 64,
                "safety_factor": 1.25,
                "walltime_seconds": resource["walltime_seconds"],
                "job_guard_seconds": resource["job_guard_seconds"],
                "job_overhead_seconds": resource["job_overhead_seconds"],
                "gpu_lanes": 4,
                "cpu_lanes": 6,
                "sharding": RUNTIME_V3_SHARDING,
                "resource_contract": resource,
                "intervention_binding": None,
                "observations": [],
                "model_phase_seconds_bound": {},
                "queue_plans": {},
                "output_path": "task/runtime_plan.json",
            }
            value["runtime_plan_hash"] = _json_hash(value)
            with (
                patch(
                    "bdb_study.preflight._validate_runtime_plan_v3_sources",
                    side_effect=PreflightError("source drift"),
                ) as replay,
                self.assertRaisesRegex(PreflightError, "source drift"),
            ):
                _validate_runtime_plan_v3_value(
                    value, path, manifest, repo_root=root
                )
            replay.assert_called_once()

    def _manifest_and_receipt(self, root: Path):
        code = [{"path": "bdb_study/example.py", "size_bytes": 1, "sha256": "a" * 64}]
        environment = {
            "python": "3.12.0",
            "packages": {"numpy": "2.0.0"},
            "tensorflow_runtime": {
                "available": True,
                "devices": [
                    {"name": "/physical_device:GPU:0", "device_type": "GPU"}
                ],
            },
        }
        manifest = {
            "task_id": "bdb2023_sack",
            "task_spec": {"anchors": [10, 20, 30, 40, 50, 60]},
            "task_spec_hash": "b" * 64,
            "prepared": {"prepared_hash": "c" * 64},
            "execution": {"queues": {"cpu_tabular": {"workers": 3}}},
            "provenance": {
                "code": code,
                "dependency_lock": {"sha256": "d" * 64},
                "environment": environment,
            },
        }
        input_signature = {
            "schema_version": "bdb-representative-neural-input-v1",
            "tensors": [{"name": "player_tokens", "shape": [4, 20, 23, 24]}],
        }
        input_signature["signature_sha256"] = _json_hash(input_signature)
        head_signature = {
            "schema_version": "bdb-neural-head-loss-v1",
            "training_target": "zero_one_label",
            "loss": "mean_binary_crossentropy",
        }
        head_signature["signature_sha256"] = _json_hash(head_signature)
        payload = {
            "schema_version": "bdb-preflight-v1",
            "task_id": manifest["task_id"],
            "task_spec_hash": manifest["task_spec_hash"],
            "prepared_hash": manifest["prepared"]["prepared_hash"],
            "code_identity_sha256": _json_hash({"code": code}),
            "dependency_lock_sha256": "d" * 64,
            "environment": environment,
            "smoke_manifest_hash": "f" * 64,
            "cells": [
                {
                    "branch": "fixed_main",
                    "repeat": 1,
                    "n_train": 60,
                    "model": f"model_{family}",
                    "family": family,
                    "scientific_metrics_sha256": str(index) * 64,
                }
                for index, family in enumerate(
                    (
                        "glm",
                        "lightgbm",
                        "relnet",
                        "attn_relnet",
                        "set_transformer",
                    ),
                    start=1,
                )
            ],
            "determinism": "byte_exact_scientific_metrics_predictions_and_arrays",
            "memory": {
                "total_ram_bytes": 1000,
                "worker_peak_bytes": 250,
                "gpu_worker_peak_bytes": 300,
                "usable_fraction": 0.75,
                "worker_cap": 12,
                "recommended_cpu_workers": 3,
                "measurement": (
                    "maximum_normalized_process_ru_maxrss_across_two_"
                    "largest_anchor_smoke_runs"
                ),
            },
            "neural_matching": {
                "models": {
                    "relnet": {
                        "parameter_count": 100_000,
                        "representative_forward_flops": 200_000,
                        "representative_neural_input": input_signature,
                        "representative_neural_input_sha256": input_signature["signature_sha256"],
                        "representative_shared_neural_input": input_signature,
                        "representative_shared_neural_input_sha256": input_signature["signature_sha256"],
                        "output_head_loss_signature": head_signature,
                        "output_head_loss_signature_sha256": head_signature["signature_sha256"],
                    },
                    "attn_relnet": {
                        "parameter_count": 104_000,
                        "representative_forward_flops": 220_000,
                        "representative_neural_input": input_signature,
                        "representative_neural_input_sha256": input_signature["signature_sha256"],
                        "representative_shared_neural_input": input_signature,
                        "representative_shared_neural_input_sha256": input_signature["signature_sha256"],
                        "output_head_loss_signature": head_signature,
                        "output_head_loss_signature_sha256": head_signature["signature_sha256"],
                    },
                    "set_transformer": {
                        "parameter_count": 160_000,
                        "representative_forward_flops": 500_000,
                        "representative_neural_input": input_signature,
                        "representative_neural_input_sha256": input_signature["signature_sha256"],
                        "representative_shared_neural_input": input_signature,
                        "representative_shared_neural_input_sha256": input_signature["signature_sha256"],
                        "output_head_loss_signature": head_signature,
                        "output_head_loss_signature_sha256": head_signature["signature_sha256"],
                        "global_set_architecture": {
                            "architecture_id": "bdb_global_set_transformer_v1",
                            "inputs": "task_tokens_player_frame_masks_time_context",
                            "attention_contract": "global_set_time_attention_v1",
                            "relation_scope": "global_masked_all_player_self_attention",
                            "typed_graph_edges_consumed": False,
                            "temporal_encoder": "factorized_masked_temporal_attention",
                            "protocol_note": "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot",
                            "parameter_count": 160_000,
                            "parameter_cap": 350_000,
                            "representative_forward_flops": 500_000,
                        },
                    },
                },
                "matched_pair": ["relnet", "attn_relnet"],
                "global_set_model": "set_transformer",
                "profile_families": [
                    "glm",
                    "lightgbm",
                    "relnet",
                    "attn_relnet",
                    "set_transformer",
                ],
                "parameter_gap_fraction": 4_000 / 104_000,
                "forward_flop_gap_fraction": 20_000 / 220_000,
                "parameter_tolerance_fraction": 0.05,
                "forward_flop_tolerance_fraction": 0.15,
                "parameter_cap": 350_000,
            },
        }
        payload["receipt_hash"] = _json_hash(payload)
        path = root / "receipt.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return manifest, path

    def test_receipt_binds_code_data_environment_and_memory_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            pin = validate_preflight_receipt(path, manifest, repo_root=root)
            self.assertEqual(pin["recommended_cpu_workers"], 3)
            changed = json.loads(json.dumps(manifest))
            changed["prepared"]["prepared_hash"] = "0" * 64
            with self.assertRaisesRegex(PreflightError, "prepared_hash"):
                validate_preflight_receipt(path, changed, repo_root=root)

    def test_full50_preflight_grandfathers_exact_four_role_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            manifest["execution"]["profile"] = "full50"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt["cells"] = [
                cell
                for cell in receipt["cells"]
                if cell["family"] != "set_transformer"
            ]
            matching = receipt["neural_matching"]
            matching["models"].pop("set_transformer")
            matching["global_set_model"] = None
            matching["profile_families"] = [
                "glm",
                "lightgbm",
                "relnet",
                "attn_relnet",
            ]
            receipt["receipt_hash"] = _json_hash(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_hash"
                }
            )
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            pin = validate_preflight_receipt(path, manifest, repo_root=root)
            self.assertEqual(pin["recommended_cpu_workers"], 3)

    def test_receipt_rejects_wrong_concurrency_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            manifest["execution"]["queues"]["cpu_tabular"]["workers"] = 2
            with self.assertRaisesRegex(PreflightError, "CPU concurrency"):
                validate_preflight_receipt(path, manifest, repo_root=root)
            manifest["execution"]["queues"]["cpu_tabular"]["workers"] = 3
            value = json.loads(path.read_text(encoding="utf-8"))
            value["memory"]["worker_peak_bytes"] += 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "hash is invalid"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_receipt_rejects_self_consistent_but_wrong_memory_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["memory"]["recommended_cpu_workers"] = 2
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            manifest["execution"]["queues"]["cpu_tabular"]["workers"] = 2
            with self.assertRaisesRegex(PreflightError, "violates the memory rule"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_receipt_rejects_mismatched_neural_input_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            changed = value["neural_matching"]["models"]["attn_relnet"]
            changed_signature = dict(changed["representative_neural_input"])
            changed_signature["tensors"] = [
                {"name": "player_tokens", "shape": [4, 1, 23, 24]}
            ]
            changed_signature["signature_sha256"] = _json_hash(
                {
                    key: item
                    for key, item in changed_signature.items()
                    if key != "signature_sha256"
                }
            )
            changed["representative_neural_input"] = changed_signature
            changed["representative_neural_input_sha256"] = changed_signature[
                "signature_sha256"
            ]
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "signatures differ"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_receipt_rejects_global_set_shared_input_or_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            changed = value["neural_matching"]["models"]["set_transformer"]
            shared = dict(changed["representative_shared_neural_input"])
            shared["tensors"] = [
                {"name": "player_tokens", "shape": [4, 19, 23, 24]}
            ]
            shared["signature_sha256"] = _json_hash(
                {
                    key: item
                    for key, item in shared.items()
                    if key != "signature_sha256"
                }
            )
            changed["representative_shared_neural_input"] = shared
            changed["representative_shared_neural_input_sha256"] = shared[
                "signature_sha256"
            ]
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "shared neural"):
                validate_preflight_receipt(path, manifest, repo_root=root)

            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["neural_matching"]["models"]["set_transformer"][
                "global_set_architecture"
            ]["typed_graph_edges_consumed"] = True
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "contract drifted"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_definitive_receipt_rejects_cpu_only_tensorflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            cpu_runtime = {
                "available": True,
                "devices": [
                    {"name": "/physical_device:CPU:0", "device_type": "CPU"}
                ],
            }
            value["environment"]["tensorflow_runtime"] = cpu_runtime
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            manifest["provenance"]["environment"]["tensorflow_runtime"] = cpu_runtime
            with self.assertRaisesRegex(PreflightError, "physical GPU"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_receipt_rejects_memory_measurement_at_smaller_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = self._manifest_and_receipt(root)
            value = json.loads(path.read_text(encoding="utf-8"))
            for cell in value["cells"]:
                cell["n_train"] = 10
            value["receipt_hash"] = _json_hash(
                {key: item for key, item in value.items() if key != "receipt_hash"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "largest training anchor"):
                validate_preflight_receipt(path, manifest, repo_root=root)

    def test_exact_prediction_comparison_catches_order_dtype_and_array_drift(self):
        frame = pd.DataFrame({"id": ["a", "b"], "p": np.array([0.1, 0.2], dtype=np.float64)})
        _assert_predictions_equal(frame, frame.copy(), "frame")
        with self.assertRaises(PreflightError):
            _assert_predictions_equal(frame, frame.iloc[::-1].reset_index(drop=True), "frame")
        arrays = {"x": np.array([1.0, np.nan], dtype=np.float64)}
        _assert_predictions_equal(arrays, {"x": arrays["x"].copy()}, "arrays")
        with self.assertRaises(PreflightError):
            _assert_predictions_equal(arrays, {"x": arrays["x"].astype(np.float32)}, "arrays")

    def test_process_peak_rss_is_positive(self):
        self.assertGreater(process_peak_rss_bytes(), 0)

    def test_sensitivity_runtime_plan_binds_effective_intervention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "runtime_plan.json"
            intervention = {
                "branch": "frozen_sensitivity",
                "sensitivity_id": "include_team_identity",
                "prepared_variant": "primary",
            }
            manifest = {
                "task_id": "bdb2025_man_zone",
                "task_spec_hash": "a" * 64,
                "prepared": {"prepared_hash": "b" * 64},
                "preflight": {"receipt_hash": "c" * 64},
                "execution": {"profile": "sensitivity20"},
                "task_spec": {
                    "anchors": [10, 20, 40, 60, 80, 100],
                    "models": {
                        "linear_structure": {"family": "glm"},
                        "boosted_structure": {"family": "lightgbm"},
                        "relnet": {"family": "relnet"},
                        "attn_relnet": {"family": "attn_relnet"},
                        "set_transformer": {"family": "set_transformer"},
                    },
                    "sensitivities": [
                        {
                            "id": "include_team_identity",
                            "selection": "frozen_prespecified",
                            "execution": {
                                "prepared_variant": "primary",
                                "anchors": [20, 100],
                            },
                        }
                    ]
                },
                "provenance": {"code": []},
            }
            bounds = {
                "linear_structure": 100.0,
                "boosted_structure": 100.0,
                "relnet": 100.0,
                "attn_relnet": 100.0,
                "set_transformer": 100.0,
            }
            plan = {
                "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
                "profile": "sensitivity20",
                "task_id": manifest["task_id"],
                "task_spec_hash": manifest["task_spec_hash"],
                "prepared_hash": manifest["prepared"]["prepared_hash"],
                "preflight_receipt_hash": manifest["preflight"]["receipt_hash"],
                "code_identity_sha256": _json_hash({"code": []}),
                "benchmark_run": str(root / "benchmark"),
                "benchmark_manifest_hash": "d" * 64,
                "safety_factor": 1.25,
                "walltime_seconds": BETTY_WALLTIME_SECONDS,
                "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
                "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
                "gpu_lanes": BETTY_GPU_LANES,
                "sharding": RUNTIME_SHARDING,
                "intervention_binding": intervention,
                "observations": [],
                "model_cell_seconds_bound": bounds,
                "queue_plans": _canonical_runtime_queue_plans(
                    "sensitivity20", manifest["task_spec"], bounds
                ),
                "output_path": "runtime_plan.json",
            }
            plan["runtime_plan_hash"] = _json_hash(plan)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            binding = validate_runtime_plan(plan_path, manifest, repo_root=root)
            self.assertEqual(binding["intervention_binding"], intervention)

            plan["intervention_binding"] = None
            plan["runtime_plan_hash"] = _json_hash(
                {key: item for key, item in plan.items() if key != "runtime_plan_hash"}
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "intervention_binding"):
                validate_runtime_plan(plan_path, manifest, repo_root=root)

            plan["intervention_binding"] = intervention
            plan["queue_plans"]["gpu_neural"]["sensitivity"][0][
                "repeats_per_job"
            ] += 1
            plan["runtime_plan_hash"] = _json_hash(
                {key: item for key, item in plan.items() if key != "runtime_plan_hash"}
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "canonical exact cover"):
                validate_runtime_plan(plan_path, manifest, repo_root=root)

    def test_pilot_runtime_plan_is_primary_only_profile_bound_and_replayed(self):
        models = {
            "linear_structure": {"family": "glm"},
            "boosted_structure": {"family": "lightgbm"},
            "relnet": {"family": "relnet"},
            "attn_relnet": {"family": "attn_relnet"},
            "set_transformer": {"family": "set_transformer"},
        }
        bounds = {model: 100.0 for model in models}
        plan = {
            "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
            "profile": "pilot10",
            "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
            "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
            "sharding": RUNTIME_SHARDING,
            "model_cell_seconds_bound": bounds,
        }
        task_spec = {
            "anchors": [10, 20, 30, 40, 50, 60],
            "models": models,
            "structural_ablation": {
                "models": ["relnet", "attn_relnet"],
                "anchors": [20, 60],
            },
            "sensitivities": [],
        }
        plan["queue_plans"] = _canonical_runtime_queue_plans(
            "pilot10", task_spec, bounds
        )
        normalized = validate_runtime_queue_plans(plan, task_spec)
        self.assertEqual(set(normalized["gpu_neural"]), {"primary"})
        forged = json.loads(json.dumps(plan))
        forged["queue_plans"]["gpu_neural"]["primary"][0]["anchors"].pop()
        with self.assertRaisesRegex(PreflightError, "canonical exact cover"):
            validate_runtime_queue_plans(forged, task_spec)
        forged = json.loads(json.dumps(plan))
        forged["profile"] = "full100"
        with self.assertRaisesRegex(PreflightError, "canonical exact cover"):
            validate_runtime_queue_plans(forged, task_spec)

    def test_full_runtime_ablation_excludes_global_set_transformer(self):
        models = {
            "linear_structure": {"family": "glm"},
            "boosted_structure": {"family": "lightgbm"},
            "relnet": {"family": "relnet"},
            "attn_relnet": {"family": "attn_relnet"},
            "set_transformer": {"family": "set_transformer"},
        }
        plan = {
            "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
            "profile": "full100",
            "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
            "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
            "sharding": RUNTIME_SHARDING,
            "model_cell_seconds_bound": {model: 100.0 for model in models},
        }
        task_spec = {
            "anchors": [10, 20, 30, 40, 50, 60],
            "models": models,
            "structural_ablation": {
                "models": ["relnet", "attn_relnet"],
                "anchors": [20, 60],
            },
            "sensitivities": [],
        }
        plan["queue_plans"] = _canonical_runtime_queue_plans(
            "full100", task_spec, plan["model_cell_seconds_bound"]
        )
        normalized = validate_runtime_queue_plans(plan, task_spec)
        ablation_models = {
            model
            for shard in normalized["gpu_neural"]["ablation"]
            for model in shard["models"]
        }
        self.assertEqual(ablation_models, {"relnet", "attn_relnet"})
        self.assertNotIn("set_transformer", ablation_models)

    def test_bdb2021_gpu_timeout_case_has_three_canonical_rectangles(self):
        models = {
            "linear_structure": {"family": "glm"},
            "boosted_structure": {"family": "lightgbm"},
            "relnet": {"family": "relnet"},
            "attn_relnet": {"family": "attn_relnet"},
            "set_transformer": {"family": "set_transformer"},
        }
        task_spec = {
            "anchors": [10, 20, 40, 60, 100, 130],
            "models": models,
            "structural_ablation": {
                "models": ["relnet", "attn_relnet"],
                "anchors": [20, 130],
            },
            "sensitivities": [],
        }
        bounds = {
            "linear_structure": 10.0,
            "boosted_structure": 20.0,
            "relnet": 2006.151705,
            "attn_relnet": 2099.700607,
            "set_transformer": 590.596291,
        }
        shards = _canonical_runtime_queue_plans(
            "pilot10", task_spec, bounds
        )["gpu_neural"]["primary"]
        self.assertEqual(
            [shard["anchors"] for shard in shards],
            [[10, 20], [40, 60], [100, 130]],
        )
        self.assertTrue(
            all(
                shard["models"]
                == ["relnet", "attn_relnet", "set_transformer"]
                and shard["cells_per_repeat"] == 6
                and shard["repeats_per_job"] == 1
                and abs(shard["predicted_job_seconds"] - 9752.897206) < 1e-9
                for shard in shards
            )
        )

        impossible = dict(bounds)
        impossible["attn_relnet"] = (
            RUNTIME_JOB_GUARD_SECONDS - RUNTIME_JOB_OVERHEAD_SECONDS + 1.0
        )
        with self.assertRaisesRegex(PreflightError, "atomic anchor/model"):
            _canonical_runtime_queue_plans("pilot10", task_spec, impossible)

    def test_pilot_runtime_builder_emits_primary_only_profile_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = {
                "linear_structure": {"family": "glm", "selected_config": {}},
                "boosted_structure": {
                    "family": "lightgbm",
                    "selected_config": {},
                },
                "relnet": {
                    "family": "relnet",
                    "selected_config": {"max_epochs": 2},
                },
                "attn_relnet": {
                    "family": "attn_relnet",
                    "selected_config": {"max_epochs": 2},
                },
                "set_transformer": {
                    "family": "set_transformer",
                    "selected_config": {"max_epochs": 2},
                },
            }
            cells = [
                {
                    "branch": "fixed_main",
                    "repeat": 1,
                    "n_train": anchor,
                    "model": model,
                    "queue": (
                        "cpu_tabular"
                        if models[model]["family"] in {"glm", "lightgbm"}
                        else "gpu_neural"
                    ),
                }
                for anchor in (10, 60)
                for model in models
            ]
            manifest = {
                "task_id": "bdb2023_sack",
                "task_spec_hash": "a" * 64,
                "task_spec": {
                    "anchors": [10, 20, 30, 40, 50, 60],
                    "models": models,
                    "structural_ablation": {
                        "models": ["relnet", "attn_relnet"],
                        "anchors": [20, 60],
                    },
                    "sensitivities": [],
                },
                "prepared": {"prepared_hash": "b" * 64},
                "preflight": {"receipt_hash": "c" * 64},
                "execution": {
                    "mode": "benchmark",
                    "profile": "pilot10",
                    "benchmark": {
                        "enabled": True,
                        "cells": cells,
                        "anchors": [10, 60],
                        "intervention": None,
                    },
                },
                "provenance": {"code": []},
            }
            storage = Mock()
            storage.manifest = manifest
            storage.manifest_hash = "d" * 64
            storage.validate_cell.return_value = SimpleNamespace(is_complete=True)
            storage.load_metrics.return_value = {"elapsed_seconds": 10.0}
            storage.load_history.return_value = {
                "selector_history": {"loss": [1.0]},
                "refit_history": {"loss": [1.0]},
            }
            output = root / "runtime_plan.json"
            execution_module = ModuleType("bdb_study.execution")
            execution_module.open_run = Mock(return_value=storage)
            with patch.dict(
                sys.modules, {"bdb_study.execution": execution_module}
            ):
                plan = build_runtime_plan(
                    root / "benchmark",
                    output,
                    repo_root=root,
                    safety_factor=1.25,
                )
            self.assertEqual(plan["profile"], "pilot10")
            self.assertEqual(plan["schema_version"], RUNTIME_PLAN_SCHEMA_VERSION)
            self.assertEqual(plan["sharding"], RUNTIME_SHARDING)
            self.assertIsNone(plan["intervention_binding"])
            for queue in ("cpu_tabular", "gpu_neural"):
                self.assertEqual(set(plan["queue_plans"][queue]), {"primary"})
                self.assertTrue(plan["queue_plans"][queue]["primary"])
            validate_runtime_queue_plans(plan, manifest["task_spec"])


if __name__ == "__main__":
    unittest.main()
