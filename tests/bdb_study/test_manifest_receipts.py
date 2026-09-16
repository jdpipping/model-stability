from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from bdb_study.contracts import (
    HORIZON_SCALE_SUPPORT_REQUIREMENT,
    HORIZON_SCALE_TAIL_POLICY,
    ProtocolAmendmentError,
    TaskSpec,
    sha256_file,
    sha256_json,
    trajectory_output_head_contract,
)
from bdb_study.devcv import candidate_grid
from bdb_study.fidelity.bdb2024 import fidelity_receipt
from bdb_study.manifest import (
    ManifestError,
    _bdb2020_harmonized_amendment_binding,
    _code_receipts,
    _protocol_amendment_binding,
    build_development_provenance,
    build_shared_frozen_registry_binding,
    build_shared_prepared_registry_binding,
    freeze_task_from_development,
    model_implementation_receipt,
    validate_task_scientific_receipts,
    verify_dependency_lock,
)
from bdb_study.prepared import PreparedArtifactError


ROOT = Path(__file__).resolve().parents[2]


def frozen_spec(task_name: str) -> TaskSpec:
    path = ROOT / "configs" / "bdb_suite" / "tasks" / f"{task_name}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    for entry in value["models"].values():
        family = str(entry["family"])
        selected = dict(candidate_grid(value["outcome_type"], family)[0])
        if value["task_id"] == "bdb2026_trajectory":
            selected.update(
                {
                    "dev_horizon_scale": [0.5] * 94,
                    "dev_horizon_scale_sha256": sha256_json([0.5] * 94),
                    "dev_horizon_scale_observed_through": 94,
                    "dev_horizon_scale_support_requirement": (
                        HORIZON_SCALE_SUPPORT_REQUIREMENT
                    ),
                    "dev_horizon_scale_tail_policy": HORIZON_SCALE_TAIL_POLICY,
                }
            )
        entry["selected_config"] = selected
        entry["implementation_receipt"] = model_implementation_receipt(
            value["outcome_type"],
            family,
            value["outcome"],
            selected_config=selected,
        )
    if value["task_id"] == "bdb2024_tackle":
        value["cohort"]["fidelity_reference"] = fidelity_receipt()
    return TaskSpec.from_mapping(value)


class ModelImplementationReceiptTests(unittest.TestCase):
    def test_bdb2020_harmonized_binding_is_exact_and_task_specific(self):
        binding = _bdb2020_harmonized_amendment_binding(
            ROOT, "bdb2020_rushing_harmonized"
        )
        self.assertEqual(
            binding,
            {
                "path": (
                    "configs/bdb_suite/protocol_amendments/"
                    "20260906_bdb2020_harmonized_five_role.json"
                ),
                "sha256": (
                    "300689150325c65f12133042cd6a6459f02ec2c40174395496a5032dc4572587"
                ),
                "amendment_id": "20260906_bdb2020_harmonized_five_role",
                "evidence_status": (
                    "retrospective_harmonization_separate_from_legacy_"
                    "confirmatory_evidence"
                ),
            },
        )
        self.assertIsNone(
            _bdb2020_harmonized_amendment_binding(ROOT, "bdb2020_rushing")
        )
        self.assertIsNone(
            _bdb2020_harmonized_amendment_binding(ROOT, "bdb2021_completion")
        )

    def test_protocol_amendment_is_validated_and_bound_before_planning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path(
                "configs/bdb_suite/protocol_amendments/"
                "20260821_global_set_transformer_primary.json"
            )
            target = root / relative
            target.parent.mkdir(parents=True)
            canonical = ROOT / relative
            target.write_bytes(canonical.read_bytes())
            horizon_relative = Path(
                "configs/bdb_suite/protocol_amendments/"
                "20260821_bdb2026_horizon_scale_tail.json"
            )
            horizon_target = root / horizon_relative
            horizon_target.write_bytes((ROOT / horizon_relative).read_bytes())
            binding = _protocol_amendment_binding(root)
            self.assertEqual(
                binding["amendment_id"],
                "20260821_global_set_transformer_primary",
            )
            self.assertEqual(binding["replacement_campaign"], "pilot10_attempt3")
            self.assertEqual(
                binding["bdb2026_horizon_scale"]["replacement_campaign"],
                "pilot10_attempt5",
            )

            value = json.loads(target.read_text(encoding="utf-8"))
            value["campaign_disposition"]["pilot_outputs_created"] = 1
            target.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolAmendmentError, "evidence drifted"):
                _protocol_amendment_binding(root)

            target.write_bytes(canonical.read_bytes())
            horizon_value = json.loads(
                horizon_target.read_text(encoding="utf-8")
            )
            horizon_value["support_audit"]["development"][
                "observed_through_horizon"
            ] = 34
            horizon_target.write_text(json.dumps(horizon_value), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolAmendmentError, "evidence drifted"):
                _protocol_amendment_binding(root)

    def _joint_bound_trajectory_spec(
        self, root: Path, *, selected_target: str = "residual"
    ) -> TaskSpec:
        value = frozen_spec("bdb2026_trajectory").as_dict()
        neural_families = ("relnet", "attn_relnet", "set_transformer")
        for family in neural_families:
            value["models"][family]["selected_config"][
                "trajectory_target"
            ] = selected_target
            value["models"][family]["implementation_receipt"] = (
                model_implementation_receipt(
                    value["outcome_type"],
                    family,
                    value["outcome"],
                    selected_config=value["models"][family]["selected_config"],
                )
            )
        source_hashes = {
            "relnet": "a" * 64,
            "attn_relnet": "b" * 64,
            "set_transformer": "c" * 64,
        }
        joint = {
            "schema_version": "bdb-joint-neural-trajectory-target-v1",
            "protocol": "pooled_equal_model_weight_grouped_oof_mean_rmse_v1",
            "selected_target": selected_target,
            "source_development_receipts": {
                family: {"development_receipt_hash": receipt_hash}
                for family, receipt_hash in source_hashes.items()
            },
        }
        joint["receipt_hash"] = sha256_json(joint)
        joint_path = root / "data/development/joint_neural_trajectory_target.json"
        joint_path.parent.mkdir(parents=True)
        joint_path.write_text(
            json.dumps(joint, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        value["joint_neural_target_selection"]["receipt"] = {
            "schema_version": joint["schema_version"],
            "protocol": joint["protocol"],
            "path": joint_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(joint_path),
            "receipt_hash": joint["receipt_hash"],
            "selected_target": joint["selected_target"],
            "source_development_receipt_hashes": source_hashes,
        }
        value["output_head"] = trajectory_output_head_contract(selected_target)
        value["outcome"]["neural_training_target"] = selected_target
        for family in neural_families:
            local = {
                "schema_version": joint["schema_version"],
                "protocol": joint["protocol"],
                "joint_receipt_path": joint_path.name,
                "joint_receipt_hash": joint["receipt_hash"],
                "selected_target": joint["selected_target"],
                "source_development_receipt_hash": source_hashes[family],
                "candidate_matrix_sha256": "c" * 64,
            }
            value["models"][family]["development_receipt"] = {
                "path": f"data/development/{family}.joint_finalized.json",
                "source_receipt_hash": source_hashes[family],
                "joint_neural_target_selection": local,
            }
        return TaskSpec.from_mapping(value)

    def test_definitive_freeze_always_full_verifies_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = root / "configs/bdb_suite/tasks/task.json"
            prepared = root / "data/processed/task"
            development = root / "data/development/task"
            output = root / "configs/bdb_suite/frozen/task.json"
            draft.parent.mkdir(parents=True)
            prepared.mkdir(parents=True)
            development.mkdir(parents=True)
            draft.write_text("{}\n", encoding="utf-8")
            with patch(
                "bdb_study.manifest.load_task_spec",
                side_effect=ManifestError("source tree drift"),
            ) as loader:
                with self.assertRaisesRegex(ManifestError, "source tree drift"):
                    freeze_task_from_development(
                        draft,
                        prepared,
                        development,
                        output,
                        repo_root=root,
                        verify_full_source_trees=False,
                    )
            self.assertTrue(loader.call_args.kwargs["require_source"])

    def test_dependency_lock_honors_platform_markers_and_exact_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "configs" / "bdb_suite" / "requirements-lock.txt"
            lock.parent.mkdir(parents=True)
            lock.write_text(
                "always-on==1.2.3\n"
                "never-on==9.9.9; python_version < '0'\n",
                encoding="utf-8",
            )
            with patch(
                "bdb_study.manifest.importlib.metadata.version",
                side_effect=lambda name: {"always-on": "1.2.3"}[name],
            ):
                receipt = verify_dependency_lock(root)
            self.assertEqual(receipt["installed"], {"always-on": "1.2.3"})
            self.assertEqual(receipt["mismatches"], {})

            lock.write_text("not-exact>=1.2.3\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "invalid dependency lock"):
                verify_dependency_lock(root)

    def test_binary_architectures_and_output_heads_are_exactly_bound(self):
        spec = frozen_spec("bdb2023_sack")
        validate_task_scientific_receipts(spec)
        for entry in spec.models.values():
            receipt = entry["implementation_receipt"]
            self.assertEqual(receipt["schema_version"], "bdb-model-implementation-v2")
            self.assertEqual(
                receipt["output_head"]["type"], "positive_class_probability"
            )
            self.assertEqual(
                receipt["frozen_optimization_config"], entry["selected_config"]
            )
        transformer = spec.models["attn_relnet"]["implementation_receipt"]
        self.assertEqual(
            transformer["architecture"]["aggregation"],
            "masked_attention_weighted_edge_aggregation",
        )
        global_set = spec.models["set_transformer"]["implementation_receipt"]
        self.assertEqual(
            global_set["architecture"]["architecture_id"],
            "bdb_global_set_transformer_v1",
        )
        self.assertFalse(
            global_set["architecture"]["typed_graph_edges_consumed"]
        )

    def test_distribution_and_trajectory_heads_record_exact_shapes(self):
        distribution = frozen_spec("bdb2022_punt_returns")
        distribution_head = distribution.models["relnet"][
            "implementation_receipt"
        ]["output_head"]
        self.assertEqual(distribution_head["support"], [-20, 110])
        self.assertEqual(distribution_head["shape"], ["examples", 131])
        self.assertEqual(distribution_head["optimization_loss"], "crps")

        trajectory = frozen_spec("bdb2026_trajectory")
        trajectory_head = trajectory.models["attn_relnet"][
            "implementation_receipt"
        ]["output_head"]
        self.assertEqual(trajectory_head["shape"], ["examples", "prepared_max_horizon", 2])
        self.assertEqual(
            trajectory_head["reconstruction"],
            "constant_velocity_baseline_plus_residual",
        )

        absolute_config = dict(
            trajectory.models["attn_relnet"]["selected_config"]
        )
        absolute_config["trajectory_target"] = "absolute"
        absolute = model_implementation_receipt(
            trajectory.outcome_type,
            "attn_relnet",
            trajectory.outcome,
            selected_config=absolute_config,
        )
        self.assertEqual(
            absolute["output_head"]["type"],
            "horizon_conditioned_masked_absolute_xy_tensor",
        )
        self.assertEqual(
            absolute["output_head"]["optimization_loss"],
            "masked_absolute_coordinate_mse",
        )
        self.assertEqual(
            absolute["architecture"]["training_target"], "absolute"
        )

    def test_joint_trajectory_target_binding_is_checksum_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._joint_bound_trajectory_spec(root)
            validate_task_scientific_receipts(spec, repo_root=root)

            binding = spec.joint_neural_target_selection["receipt"]
            joint_path = root / binding["path"]
            joint_path.write_text(
                joint_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(ManifestError, "checksum or binding drifted"):
                validate_task_scientific_receipts(spec, repo_root=root)

    def test_joint_trajectory_target_requires_identical_neural_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = self._joint_bound_trajectory_spec(root).as_dict()
            changed["models"]["attn_relnet"]["selected_config"][
                "trajectory_target"
            ] = "absolute"
            changed["models"]["attn_relnet"]["implementation_receipt"] = (
                model_implementation_receipt(
                    changed["outcome_type"],
                    "attn_relnet",
                    changed["outcome"],
                    selected_config=changed["models"]["attn_relnet"][
                        "selected_config"
                    ],
                )
            )
            with self.assertRaisesRegex(ManifestError, "jointly selected target"):
                validate_task_scientific_receipts(
                    TaskSpec.from_mapping(changed), repo_root=root
                )

    def test_absolute_joint_winner_freezes_matching_head_loss_and_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._joint_bound_trajectory_spec(
                root, selected_target="absolute"
            )
            validate_task_scientific_receipts(spec, repo_root=root)
            self.assertEqual(
                spec.output_head, trajectory_output_head_contract("absolute")
            )
            self.assertEqual(spec.outcome["neural_training_target"], "absolute")
            for family in ("relnet", "attn_relnet", "set_transformer"):
                receipt = spec.models[family]["implementation_receipt"]
                self.assertEqual(receipt["architecture"]["training_target"], "absolute")
                self.assertEqual(
                    receipt["output_head"]["optimization_loss"],
                    "masked_absolute_coordinate_mse",
                )

    def test_tampered_or_wrong_capacity_receipts_fail_closed(self):
        spec = frozen_spec("bdb2023_sack")
        changed = spec.as_dict()
        changed["models"]["relnet"]["implementation_receipt"][
            "architecture"
        ]["architecture_id"] = "different"
        with self.assertRaisesRegex(ManifestError, "architecture/output-head"):
            validate_task_scientific_receipts(TaskSpec.from_mapping(changed))

        transformer_config = deepcopy(spec.models["attn_relnet"]["selected_config"])
        with self.assertRaisesRegex(ManifestError, "unsupported model family"):
            model_implementation_receipt(
                spec.outcome_type,
                "transformer",
                spec.outcome,
                selected_config=transformer_config,
            )

    def test_bdb2024_binds_executed_xgboost_reference(self):
        spec = frozen_spec("bdb2024_tackle")
        validate_task_scientific_receipts(spec)
        reference = spec.cohort["fidelity_reference"]
        self.assertEqual(reference["executed_notebook_model"]["n_estimators"], 150)
        self.assertEqual(reference["executed_notebook_model"]["reg_lambda"], 150)
        self.assertEqual(len(reference["feature_names"]), 9)

        changed = spec.as_dict()
        changed["cohort"]["fidelity_reference"]["executed_notebook_model"][
            "n_estimators"
        ] = 250
        with self.assertRaisesRegex(ManifestError, "winner-fidelity"):
            validate_task_scientific_receipts(TaskSpec.from_mapping(changed))

    def test_code_receipts_exclude_generated_frozen_and_preflight_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "bdb_study/module.py": "x = 1\n",
                "configs/bdb_suite/tasks/task.json": "{}\n",
                "configs/bdb_suite/sources/source.json": "{}\n",
                "configs/bdb_suite/frozen/task.json": "{}\n",
                "configs/bdb_suite/preflight/task.json": "{}\n",
                "configs/bdb_suite/requirements-lock.txt": "numpy==1.0\n",
            }
            for relative, content in paths.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            amendment_relative = Path(
                "configs/bdb_suite/protocol_amendments/"
                "20260821_global_set_transformer_primary.json"
            )
            amendment_target = root / amendment_relative
            amendment_target.parent.mkdir(parents=True, exist_ok=True)
            amendment_target.write_bytes((ROOT / amendment_relative).read_bytes())
            horizon_relative = Path(
                "configs/bdb_suite/protocol_amendments/"
                "20260821_bdb2026_horizon_scale_tail.json"
            )
            horizon_target = root / horizon_relative
            horizon_target.write_bytes((ROOT / horizon_relative).read_bytes())
            observed = {record["path"] for record in _code_receipts(root)}
            self.assertIn("bdb_study/module.py", observed)
            self.assertIn("configs/bdb_suite/tasks/task.json", observed)
            self.assertIn("configs/bdb_suite/sources/source.json", observed)
            self.assertIn(amendment_relative.as_posix(), observed)
            self.assertIn(horizon_relative.as_posix(), observed)
            self.assertIn("configs/bdb_suite/requirements-lock.txt", observed)
            self.assertNotIn("configs/bdb_suite/frozen/task.json", observed)
            self.assertNotIn("configs/bdb_suite/preflight/task.json", observed)

    def test_development_provenance_is_content_bound_and_has_no_git_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_id = "bdb2020_rushing_harmonized"
            draft = root / "configs" / "bdb_suite" / "tasks" / "task.json"
            prepared = root / "data" / "processed" / "task"
            draft.parent.mkdir(parents=True)
            prepared.mkdir(parents=True)
            draft.write_text(
                json.dumps({"task_id": task_id}) + "\n", encoding="utf-8"
            )
            (prepared / "receipt.json").write_text('{"prepared":true}\n', encoding="utf-8")
            deterministic = {
                "PYTHONHASHSEED": "20260817",
                "TF_DETERMINISTIC_OPS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
            }
            environment = {
                "deterministic_environment": deterministic,
                "tensorflow_runtime": {
                    "available": True,
                    "devices": [{"name": "GPU:0", "device_type": "GPU"}],
                },
            }
            with (
                patch(
                    "bdb_study.manifest.load_task_spec",
                    return_value=SimpleNamespace(
                        task_id=task_id, spec_hash="s" * 64
                    ),
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={
                        "task_id": task_id,
                        "prepared_hash": "p" * 64,
                        "semantic_receipt": {"semantic_hash": "m" * 64},
                    },
                ),
                patch(
                    "bdb_study.manifest.load_prepared_task",
                    return_value=object(),
                ),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "v" * 64},
                ) as semantic_binding,
                patch(
                    "bdb_study.manifest.game_records_from_prepared",
                    return_value=[],
                ),
                patch(
                    "bdb_study.manifest.build_game_registry",
                    return_value={"registry_hash": "r" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest._bdb2020_harmonized_amendment_binding",
                    return_value={
                        "path": "test-amendment.json",
                        "sha256": "a" * 64,
                        "amendment_id": "test-harmonized-amendment",
                        "evidence_status": "retrospective",
                    },
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64, "mismatches": {}},
                ),
                patch("bdb_study.manifest._environment_provenance", return_value=environment),
                patch.dict("os.environ", deterministic, clear=False),
            ):
                provenance = build_development_provenance(
                    draft, prepared, repo_root=root
                )
            self.assertNotIn("git", provenance)
            self.assertEqual(provenance["task_spec"]["task_spec_hash"], "s" * 64)
            self.assertEqual(provenance["prepared"]["prepared_hash"], "p" * 64)
            self.assertEqual(
                provenance["bdb2020_harmonized_amendment"]["amendment_id"],
                "test-harmonized-amendment",
            )
            # The value is recomputed from loaded arrays, not trusted from the
            # nullable semantic receipt embedded in older prepared bundles.
            self.assertEqual(provenance["prepared"]["semantic_hash"], "v" * 64)
            semantic_binding.assert_called_once()
            self.assertEqual(
                provenance["environment"]["schema_version"],
                "bdb-development-environment-v1",
            )
            self.assertEqual(
                provenance["environment"]["deterministic_environment"],
                deterministic,
            )
            self.assertNotIn("tensorflow_runtime", provenance["environment"])

    def test_development_provenance_requires_determinism_but_is_queue_neutral(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft = root / "configs" / "bdb_suite" / "tasks" / "task.json"
            prepared = root / "data" / "processed" / "task"
            draft.parent.mkdir(parents=True)
            prepared.mkdir(parents=True)
            draft.write_text("{}\n", encoding="utf-8")
            (prepared / "receipt.json").write_text("{}\n", encoding="utf-8")
            deterministic = {
                "PYTHONHASHSEED": "20260817",
                "TF_DETERMINISTIC_OPS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
            }

            def environment(devices):
                return {
                    "deterministic_environment": deterministic,
                    "tensorflow_runtime": {"available": True, "devices": devices},
                }

            common = (
                patch(
                    "bdb_study.manifest.load_task_spec",
                    return_value=SimpleNamespace(task_id="task", spec_hash="s" * 64),
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={"task_id": "task", "prepared_hash": "p" * 64},
                ),
                patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "m" * 64},
                ),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch(
                    "bdb_study.manifest.build_game_registry",
                    return_value={"registry_hash": "r" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64},
                ),
            )
            with common[0], common[1], common[2], common[3], common[4], common[5], common[6], common[7], common[8], patch(
                "bdb_study.manifest._environment_provenance",
                return_value=environment([{"device_type": "GPU"}]),
            ), patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ManifestError, "deterministic environment"):
                    build_development_provenance(draft, prepared, repo_root=root)

            common = (
                patch(
                    "bdb_study.manifest.load_task_spec",
                    return_value=SimpleNamespace(task_id="task", spec_hash="s" * 64),
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={"task_id": "task", "prepared_hash": "p" * 64},
                ),
                patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "m" * 64},
                ),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch(
                    "bdb_study.manifest.build_game_registry",
                    return_value={"registry_hash": "r" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64},
                ),
            )
            with common[0], common[1], common[2], common[3], common[4], common[5], common[6], common[7], common[8], patch(
                "bdb_study.manifest._environment_provenance",
                return_value=environment([{"device_type": "CPU"}]),
            ), patch.dict("os.environ", deterministic, clear=True):
                cpu_provenance = build_development_provenance(
                    draft, prepared, repo_root=root
                )
            common = (
                patch(
                    "bdb_study.manifest.load_task_spec",
                    return_value=SimpleNamespace(task_id="task", spec_hash="s" * 64),
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={"task_id": "task", "prepared_hash": "p" * 64},
                ),
                patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "m" * 64},
                ),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch(
                    "bdb_study.manifest.build_game_registry",
                    return_value={"registry_hash": "r" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64},
                ),
            )
            with common[0], common[1], common[2], common[3], common[4], common[5], common[6], common[7], common[8], patch(
                "bdb_study.manifest._environment_provenance",
                return_value=environment([{"device_type": "GPU"}]),
            ), patch.dict("os.environ", deterministic, clear=True):
                gpu_provenance = build_development_provenance(
                    draft, prepared, repo_root=root
                )
            self.assertEqual(cpu_provenance, gpu_provenance)
            self.assertNotIn(
                "tensorflow_runtime", cpu_provenance["environment"]
            )

    def test_shared_prepared_registry_is_required_and_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            peer_draft = root / "configs/bdb_suite/tasks/bdb2025_man_zone.json"
            peer_prepared = root / "data/processed/bdb_suite/bdb2025_man_zone"
            peer_draft.parent.mkdir(parents=True)
            peer_prepared.mkdir(parents=True)
            peer_draft.write_text("{}\n", encoding="utf-8")
            (peer_prepared / "receipt.json").write_text("{}\n", encoding="utf-8")
            registry = {"registry_hash": "r" * 64, "records": ["same"]}
            peer_spec = SimpleNamespace(
                task_id="bdb2025_man_zone", spec_hash="s" * 64
            )
            current_spec = SimpleNamespace(task_id="bdb2024_tackle")

            def patches(peer_registry):
                return (
                    patch("bdb_study.manifest.load_task_spec", return_value=peer_spec),
                    patch(
                        "bdb_study.manifest.load_prepared_receipt",
                        return_value={"prepared_hash": "p" * 64},
                    ),
                    patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                    patch(
                        "bdb_study.manifest.validate_prepared_binding",
                        return_value={"semantic_hash": "m" * 64},
                    ),
                    patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                    patch(
                        "bdb_study.manifest.build_game_registry",
                        return_value=peer_registry,
                    ),
                )

            common = patches(registry)
            with common[0], common[1], common[2], common[3], common[4], common[5]:
                receipt = build_shared_prepared_registry_binding(
                    current_spec, registry, repo_root=root
                )
            self.assertEqual(receipt["task_id"], "bdb2025_man_zone")
            self.assertEqual(receipt["game_registry_hash"], "r" * 64)

            changed = {"registry_hash": "x" * 64, "records": ["different"]}
            common = patches(changed)
            with common[0], common[1], common[2], common[3], common[4], common[5]:
                with self.assertRaisesRegex(ManifestError, "registries differ"):
                    build_shared_prepared_registry_binding(
                        current_spec, registry, repo_root=root
                    )

            # The reverse direction is also gated: 2025 cannot tune without
            # its 2024 peer prepared artifact.
            reverse_spec = SimpleNamespace(task_id="bdb2025_man_zone")
            with patch(
                "bdb_study.manifest.load_task_spec",
                side_effect=PreparedArtifactError("missing peer"),
            ):
                with self.assertRaisesRegex(ManifestError, "peer prepared artifact is required"):
                    build_shared_prepared_registry_binding(
                        reverse_spec, registry, repo_root=root
                    )

    def test_shared_frozen_registry_requires_both_peers_and_exact_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = {"registry_hash": "r" * 64, "records": ["same"]}
            for task_id, peer_id in (
                ("bdb2024_tackle", "bdb2025_man_zone"),
                ("bdb2025_man_zone", "bdb2024_tackle"),
            ):
                with self.subTest(task_id=task_id):
                    peer_path = root / "configs/bdb_suite/frozen" / f"{peer_id}.json"
                    spec = SimpleNamespace(
                        task_id=task_id, cohort={"game_registry": registry}
                    )
                    with self.assertRaisesRegex(ManifestError, "frozen peer is required"):
                        build_shared_frozen_registry_binding(spec, repo_root=root)

                    peer_path.parent.mkdir(parents=True, exist_ok=True)
                    peer_path.write_text('{"peer":true}\n', encoding="utf-8")
                    peer_spec = SimpleNamespace(
                        task_id=peer_id, spec_hash="s" * 64,
                        cohort={"game_registry": registry},
                    )
                    with (
                        patch("bdb_study.manifest.load_task_spec", return_value=peer_spec),
                        patch("bdb_study.manifest.validate_task_scientific_receipts"),
                    ):
                        receipt = build_shared_frozen_registry_binding(
                            spec, repo_root=root
                        )
                    self.assertEqual(receipt["task_id"], peer_id)

                    peer_spec.cohort = {
                        "game_registry": {
                            "registry_hash": "x" * 64,
                            "records": ["different"],
                        }
                    }
                    with (
                        patch("bdb_study.manifest.load_task_spec", return_value=peer_spec),
                        patch("bdb_study.manifest.validate_task_scientific_receipts"),
                    ):
                        with self.assertRaisesRegex(ManifestError, "registries differ"):
                            build_shared_frozen_registry_binding(spec, repo_root=root)
                    peer_path.unlink()


if __name__ == "__main__":
    unittest.main()
