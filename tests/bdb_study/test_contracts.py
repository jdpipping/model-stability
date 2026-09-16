from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bdb_study.contracts import (
    FrozenReceiptCollisionError,
    HORIZON_SCALE_SUPPORT_REQUIREMENT,
    HORIZON_SCALE_TAIL_POLICY,
    ProtocolAmendmentError,
    SOURCE_RECEIPT_SCHEMA_VERSION,
    TASK_SPEC_SCHEMA_VERSION,
    SourceReceiptError,
    TaskSpec,
    TaskSpecError,
    build_source_receipt,
    build_source_tree_receipt,
    freeze_task_spec,
    load_bdb2020_harmonized_amendment,
    load_bdb2026_horizon_scale_amendment,
    load_global_set_transformer_amendment,
    load_task_spec,
    sha256_json,
    validate_source_receipt,
    validate_source_tree_receipt,
    validate_bdb2020_harmonized_amendment,
    validate_bdb2026_horizon_scale_amendment,
    validate_global_set_transformer_amendment,
    validate_horizon_scale_config,
    validate_task_spec,
)


def prepared_contract():
    value = {
        "schema_version": "bdb-prepared-semantics-v2",
        "task_id": "bdb2023_sack",
        "outcome_type": "binary",
        "primary_metric": "brier",
        "examples": 2,
        "games": 122,
        "tabular_feature_order": ["down"],
        "channel_names": ["relative_x", "relative_y", "speed"],
        "arrays": {
            "player_tokens": {
                "axes": ["example", "time", "tracked_object", "channel"],
                "shape": [2, 1, 2, 3],
                "dtype": "float32",
            },
            "player_mask": {
                "axes": ["example", "time", "tracked_object"],
                "shape": [2, 1, 2],
                "dtype": "bool",
            },
            "frame_mask": {
                "axes": ["example", "time"],
                "shape": [2, 1],
                "dtype": "bool",
            },
            "y": {"axes": ["example"], "shape": [2], "dtype": "int8"},
            "target_values": None,
            "target_mask": None,
            "target_baseline": None,
        },
        "support": None,
        "mask_contract": {
            "player_mask": {
                "meaning": "true_iff_tracked_object_slot_is_observed",
                "observed_frame_slot_layout": "stable_per_play_slots_with_sparse_observation",
                "slot_identity": "alignment_only_never_a_model_feature",
                "masked_token_value": 0.0,
                "requires_observed_frame": True,
            },
            "frame_mask": {
                "meaning": "true_iff_time_slot_is_observed_not_padding",
                "layout": "all_observed",
            },
            "target_mask": None,
        },
        "target_contract": {
            "encoding": "zero_one",
            "examples_target": "equals_y",
            "model_output": "positive_class_probability",
        },
        "adapter_metadata": {
            "cutoff": "ball_snap",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
        },
    }
    value["semantic_hash"] = sha256_json(value)
    return value


def task_mapping(*, receipts=None, tree_receipts=None):
    return {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "task_id": "bdb2023_sack",
        "release_year": 2023,
        "outcome_type": "binary",
        "primary_metric": "brier",
        "total_games": 122,
        "development_games": 20,
        "confirmatory_games": 102,
        "outer_counts": {"train": 61, "calibration": 20, "test": 21},
        "anchors": [10, 20, 30, 40, 50, 60],
        "event_cutoff": {"event": "ball_snap", "fallback_rules": []},
        "cohort": {
            "description": "snap-observable dropbacks",
            "eligibility": ["passResult in C/I/IN/R/S"],
            "exclusions": ["augmented plays"],
        },
        "features": {
            "allowlist": ["tracking_through_snap"],
            "denylist": ["pff_role", "routeRan"],
            "channels": ["relative_x", "relative_y", "speed"],
            "coordinate_normalization": "offense moves left-to-right",
        },
        "outcome": {
            "target": "passResult == S",
            "positive_label": 1,
            "null_model": "training prevalence",
        },
        "prepared_contract": prepared_contract(),
        "history": {
            "scope": "causal_tracking_history",
            "endpoint": "ball_snap",
            "max_frames": 1,
            "padding": "none",
            "stable_player_slots": True,
            "slot_policy": "play_global_focal_independent_v1",
        },
        "graph": {
            "node_types": ["player"],
            "edge_types": ["generic"],
            "construction": "synthetic pre-snap graph",
            "masked_aggregation": "observed_nodes_and_edges_only",
        },
        "identity_policy": {
            "player_ids": "stable_slot_assignment_only",
            "player_id_feature": False,
            "team_identity_primary": False,
            "team_identity_sensitivity": False,
        },
        "model_roles": {
            "primary": [
                "linear_structure", "boosted_structure", "relnet",
                "attn_relnet", "set_transformer",
            ],
            "shared_tabular_summary": True,
            "shared_neural_inputs": [
                "tokenization", "causal_history", "player_and_frame_masks",
                "time_to_event", "context", "output_head", "loss", "tuning_grid",
                "seeds", "training_protocol",
            ],
            "relational_pair_shared_inputs": [
                "graph_edges", "relational_temporal_encoder",
            ],
            "set_transformer_structure": {
                "relation_scope": "global_masked_all_player_self_attention",
                "typed_graph_edges_consumed": False,
                "temporal_encoder": "factorized_masked_temporal_attention",
            },
            "parameter_match_tolerance_fraction": 0.05,
            "flop_match_tolerance_fraction": 0.15,
            "neural_parameter_cap": 350000,
        },
        "output_head": {
            "id": "binary_probability_v1",
            "prediction": "sack probability",
            "training_loss": "log_loss",
        },
        "uncertainty": {
            "primary_method": "out_of_game_venn_abers_calibration_v1",
            "secondary_diagnostic": "raw_score_class_conditional_conformal_sets_v1",
            "secondary_score_source": "raw_model_score",
            "field_intersection": None,
            "horizon_scale_learner": None,
            "calibration_unit": "game",
            "nominal_coverage": 0.9,
            "report": ["venn_abers_ambiguity"],
        },
        "structural_ablation": {
            "id": "collapse_typed_edges",
            "description": "synthetic edge ablation",
            "models": ["relnet", "attn_relnet"],
            "anchors": [20, 60],
            "repeats": 20,
        },
        "sensitivities": [],
        "models": {
            "linear_structure": {
                "role": "linear_structure",
                "family": "glm",
                "alpha_grid": [10.0, 10.0 / 3.0, 1.0, 1.0 / 3.0, 0.1],
            },
            "boosted_structure": {
                "role": "boosted_structure",
                "family": "lightgbm",
                "grid_id": "bdb_suite_lgbm_four_v1",
            },
            "relnet": {
                "role": "relnet",
                "family": "relnet",
                "learning_rates": [0.001, 0.0003],
                "dropouts": [0.3, 0.1],
            },
            "attn_relnet": {
                "role": "attn_relnet",
                "family": "attn_relnet",
                "learning_rates": [0.001, 0.0003],
                "dropouts": [0.3, 0.1],
            },
            "set_transformer": {
                "role": "set_transformer",
                "family": "set_transformer",
                "learning_rates": [0.001, 0.0003],
                "dropouts": [0.3, 0.1],
            },
        },
        "artifacts": {"metrics": {"format": "json"}, "predictions": {"format": "npz"}},
        "source_receipts": receipts or [],
        "source_tree_receipts": tree_receipts or [],
        "shared_game_registry_with": None,
    }


def dummy_file_receipt():
    return {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "source_id": "bdb2023_raw",
        "source_path": "/observed/source.csv",
        "destination_path": "data/bdb2023/raw/source.csv",
        "size_bytes": 10,
        "sha256": "1" * 64,
        "row_count": 1,
        "columns": ["gameId"],
    }


class TaskSpecContractTests(unittest.TestCase):
    def test_frozen_horizon_scale_binds_observed_prefix_and_constant_tail(self) -> None:
        config = {
            "dev_horizon_scale": [0.5] * 94,
            "dev_horizon_scale_sha256": sha256_json([0.5] * 94),
            "dev_horizon_scale_observed_through": 33,
            "dev_horizon_scale_support_requirement": (
                HORIZON_SCALE_SUPPORT_REQUIREMENT
            ),
            "dev_horizon_scale_tail_policy": HORIZON_SCALE_TAIL_POLICY,
        }
        validate_horizon_scale_config(config)

        for field, value in (
            ("dev_horizon_scale_observed_through", 0),
            ("dev_horizon_scale_support_requirement", "different"),
            ("dev_horizon_scale_tail_policy", "different"),
        ):
            tampered = deepcopy(config)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaises(TaskSpecError):
                validate_horizon_scale_config(tampered)

        tampered = deepcopy(config)
        tampered["dev_horizon_scale"][50:] = [0.75] * 44
        tampered["dev_horizon_scale_sha256"] = sha256_json(
            tampered["dev_horizon_scale"]
        )
        with self.assertRaises(TaskSpecError):
            validate_horizon_scale_config(tampered)

    def test_valid_task_round_trips_and_is_hash_stable(self):
        left = TaskSpec.from_mapping(task_mapping(receipts=[dummy_file_receipt()]))
        right_mapping = deepcopy(left.as_dict())
        right_mapping["outer_counts"] = dict(reversed(list(right_mapping["outer_counts"].items())))
        right = TaskSpec.from_mapping(right_mapping)
        self.assertEqual(left.spec_hash, right.spec_hash)
        self.assertEqual(validate_task_spec(left), left)

    def test_draft_may_omit_receipts_but_freeze_may_not(self):
        draft = TaskSpec.from_mapping(task_mapping(), require_source_receipts=False)
        self.assertEqual(draft.source_receipts, ())
        with self.assertRaisesRegex(TaskSpecError, "requires at least one"):
            validate_task_spec(draft)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TaskSpecError):
                freeze_task_spec(draft, Path(temporary) / "frozen.json")

    def test_scientific_invariants_are_strict(self):
        mutations = {
            "wrong total": lambda value: value.update(total_games=121),
            "future release": lambda value: value.update(release_year=2027),
            "wrong outer sum": lambda value: value["outer_counts"].update(test=20),
            "non-increasing anchors": lambda value: value.update(anchors=[10, 20, 30, 40, 60, 50]),
            "post-cutoff allowlist": lambda value: value["features"].update(
                allowlist=["pff_role"]
            ),
            "missing family": lambda value: value["models"]["attn_relnet"].update(
                family="relnet"
            ),
            "wrong metric": lambda value: value.update(primary_metric="crps"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                value = task_mapping(receipts=[dummy_file_receipt()])
                mutate(value)
                with self.assertRaises(TaskSpecError):
                    TaskSpec.from_mapping(value)

        overlap = task_mapping(receipts=[dummy_file_receipt()])
        overlap["features"]["allowlist"] = ["pff_role"]
        with self.assertRaisesRegex(TaskSpecError, "overlap"):
            TaskSpec.from_mapping(overlap)

    def test_prepared_contract_is_hash_bound_and_channel_order_is_exact(self):
        bad_hash = task_mapping(receipts=[dummy_file_receipt()])
        bad_hash["prepared_contract"]["semantic_hash"] = "0" * 64
        with self.assertRaisesRegex(TaskSpecError, "semantic_hash"):
            TaskSpec.from_mapping(bad_hash)

        reordered = task_mapping(receipts=[dummy_file_receipt()])
        reordered["prepared_contract"]["channel_names"].reverse()
        unsigned = {
            key: value
            for key, value in reordered["prepared_contract"].items()
            if key != "semantic_hash"
        }
        reordered["prepared_contract"]["semantic_hash"] = sha256_json(unsigned)
        with self.assertRaisesRegex(TaskSpecError, "features.channels order"):
            TaskSpec.from_mapping(reordered)

    def test_left_padding_policy_allows_an_all_observed_realized_mask(self):
        value = task_mapping(receipts=[dummy_file_receipt()])
        value["history"]["padding"] = "left_masked"
        parsed = TaskSpec.from_mapping(value)
        self.assertEqual(
            parsed.prepared_contract["mask_contract"]["frame_mask"]["layout"],
            "all_observed",
        )

        incompatible = task_mapping(receipts=[dummy_file_receipt()])
        incompatible["prepared_contract"]["mask_contract"]["frame_mask"][
            "layout"
        ] = "left_padded"
        unsigned = {
            key: item
            for key, item in incompatible["prepared_contract"].items()
            if key != "semantic_hash"
        }
        incompatible["prepared_contract"]["semantic_hash"] = sha256_json(unsigned)
        with self.assertRaisesRegex(TaskSpecError, "incompatible"):
            TaskSpec.from_mapping(incompatible)

    def test_declared_model_grids_and_transformer_capacity_are_exact(self):
        mutations = {
            "glm candidate order": lambda value: value["models"]["linear_structure"][
                "alpha_grid"
            ].reverse(),
            "lightgbm registry": lambda value: value["models"]["boosted_structure"].update(
                grid_id="different"
            ),
            "cnn learning-rate order": lambda value: value["models"]["relnet"][
                "learning_rates"
            ].reverse(),
            "cnn dropout order": lambda value: value["models"]["relnet"][
                "dropouts"
            ].reverse(),
            "role mismatch": lambda value: value["models"]["attn_relnet"].update(
                role="relnet"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                value = task_mapping(receipts=[dummy_file_receipt()])
                mutate(value)
                with self.assertRaises(TaskSpecError):
                    TaskSpec.from_mapping(value)

        trajectory_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "bdb_suite"
            / "tasks"
            / "bdb2026_trajectory.json"
        )
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        TaskSpec.from_mapping(trajectory)
        trajectory["models"]["linear_structure"]["alpha_grid"].reverse()
        with self.assertRaisesRegex(TaskSpecError, "alpha_grid"):
            TaskSpec.from_mapping(trajectory)

    def test_freeze_is_hash_bound_idempotent_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            destination = root / "data" / "source.csv"
            destination.parent.mkdir()
            source.write_bytes(b"x\n1\n")
            destination.write_bytes(source.read_bytes())
            receipt = build_source_receipt(
                source,
                destination,
                source_id="frozen_source",
                repo_root=root,
                row_count=1,
                columns=["x"],
            )
            spec = TaskSpec.from_mapping(task_mapping(receipts=[receipt.as_dict()]))
            path = root / "task.json"
            first = freeze_task_spec(spec, path, repo_root=root)
            second = freeze_task_spec(spec, path, repo_root=root)
            self.assertEqual(first, second)
            self.assertEqual(load_task_spec(path, repo_root=root), spec)

            changed = task_mapping(receipts=[receipt.as_dict()])
            changed["cohort"]["description"] = "different"
            with self.assertRaises(FrozenReceiptCollisionError):
                freeze_task_spec(changed, path, repo_root=root)

            payload = json.loads(path.read_text())
            payload["task_spec"]["cohort"]["description"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TaskSpecError, "hash"):
                load_task_spec(path)


class SourceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_file_receipt_verifies_copy_and_detects_tampering(self):
        source = self.root / "source.csv"
        destination = self.root / "data" / "raw" / "source.csv"
        destination.parent.mkdir(parents=True)
        source.write_bytes(b"a,b\n1,2\n")
        destination.write_bytes(source.read_bytes())
        receipt = build_source_receipt(
            source,
            destination,
            source_id="raw_file",
            repo_root=self.root,
            row_count=1,
            columns=["a", "b"],
        )
        self.assertEqual(
            validate_source_receipt(receipt, repo_root=self.root, require_source=True), receipt
        )
        destination.write_bytes(b"tampered")
        with self.assertRaises(SourceReceiptError):
            validate_source_receipt(receipt, repo_root=self.root)

    def _tree_receipt(self):
        destination = self.root / "data" / "raw"
        destination.mkdir(parents=True)
        content = b"gameId\n1\n"
        (destination / "games.csv").write_bytes(content)
        files = [
            {
                "path": "games.csv",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ]
        identity = {
            "schema_version": 1,
            "release": 2099,
            "source_id": "synthetic_tree",
            "destination": "data/raw",
            "tree_sha256": sha256_json(files),
            "file_count": 1,
            "byte_count": len(content),
            "catalog_receipts": [],
            "files": files,
        }
        payload = {
            **identity,
            "receipt_sha256": sha256_json(identity),
            "source_path_observed": "/some/source",
            "verified_at_utc": "2099-01-01T00:00:00Z",
        }
        path = self.root / "receipts" / "tree.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def test_aggregate_tree_receipt_is_compact_and_can_verify_every_file(self):
        path = self._tree_receipt()
        reference = build_source_tree_receipt(path, repo_root=self.root)
        self.assertEqual(reference.file_count, 1)
        self.assertEqual(
            validate_source_tree_receipt(reference, repo_root=self.root, verify_tree=True),
            reference,
        )
        (self.root / "data" / "raw" / "games.csv").write_bytes(b"bad")
        with self.assertRaises(SourceReceiptError):
            validate_source_tree_receipt(reference, repo_root=self.root, verify_tree=True)

    def test_task_can_use_tree_receipt_without_per_file_entries(self):
        reference = build_source_tree_receipt(self._tree_receipt(), repo_root=self.root)
        spec = TaskSpec.from_mapping(task_mapping(tree_receipts=[reference.as_dict()]))
        validate_task_spec(spec, repo_root=self.root, require_source=True)


class SchemaArtifactTests(unittest.TestCase):
    def test_bdb2020_harmonized_amendment_binds_exact_separate_evidence(self):
        path = Path(
            "configs/bdb_suite/protocol_amendments/"
            "20260906_bdb2020_harmonized_five_role.json"
        )
        amendment = load_bdb2020_harmonized_amendment(path)
        self.assertEqual(
            amendment["amendment_id"],
            "20260906_bdb2020_harmonized_five_role",
        )
        self.assertEqual(
            amendment["evidence_status"],
            "retrospective_harmonization_separate_from_legacy_confirmatory_evidence",
        )
        self.assertEqual(
            amendment["legacy_evidence_policy"]["task_id"],
            "bdb2020_rushing",
        )
        self.assertEqual(
            amendment["new_task"]["task_id"],
            "bdb2020_rushing_harmonized",
        )

        tampered = deepcopy(amendment)
        tampered["execution"]["primary_cells"] = 2_999
        with self.assertRaisesRegex(ProtocolAmendmentError, "evidence drifted"):
            validate_bdb2020_harmonized_amendment(tampered)

    def test_bdb2026_horizon_amendment_binds_exact_outcome_blind_evidence(self):
        path = Path(
            "configs/bdb_suite/protocol_amendments/"
            "20260821_bdb2026_horizon_scale_tail.json"
        )
        amendment = load_bdb2026_horizon_scale_amendment(path)
        self.assertEqual(amendment["replacement_campaign"], "pilot10_attempt5")
        self.assertEqual(
            amendment["support_audit"]["development"][
                "observed_through_horizon"
            ],
            33,
        )
        self.assertFalse(
            amendment["superseded_campaign"][
                "model_scores_or_checkpoint_metrics_inspected"
            ]
        )

        tampered = deepcopy(amendment)
        tampered["support_audit"]["development"][
            "observed_through_horizon"
        ] = 34
        with self.assertRaises(ProtocolAmendmentError):
            validate_bdb2026_horizon_scale_amendment(tampered)

    def test_bdb2026_selected_configs_validate_scale_support_and_tail(self):
        path = Path("configs/bdb_suite/tasks/bdb2026_trajectory.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        selected = {
            "dev_horizon_scale": [0.5] * 94,
            "dev_horizon_scale_sha256": sha256_json([0.5] * 94),
            "dev_horizon_scale_observed_through": 33,
            "dev_horizon_scale_support_requirement": (
                HORIZON_SCALE_SUPPORT_REQUIREMENT
            ),
            "dev_horizon_scale_tail_policy": HORIZON_SCALE_TAIL_POLICY,
        }
        for model in value["models"].values():
            model["selected_config"] = deepcopy(selected)
        TaskSpec.from_mapping(value)

        value["models"]["linear_structure"]["selected_config"][
            "dev_horizon_scale_tail_policy"
        ] = "tampered"
        with self.assertRaises(TaskSpecError):
            TaskSpec.from_mapping(value)

    def test_global_set_amendment_binds_exact_outcome_blind_attempt2_evidence(self):
        path = Path(
            "configs/bdb_suite/protocol_amendments/"
            "20260821_global_set_transformer_primary.json"
        )
        amendment = load_global_set_transformer_amendment(path)
        disposition = amendment["campaign_disposition"]
        self.assertEqual(
            disposition["attempt2_campaign_hash"],
            "96c39edb8dc84282293c70fc456a77f3e18f2956a486c84df6f1b79e20de7f03",
        )
        self.assertEqual(disposition["attempt2_terminal_job_id"], 7725228)
        self.assertEqual(disposition["attempt2_submitted_job_ids"], 710)
        self.assertEqual(
            disposition["attempt2_initially_active_cancellation_targets"], 651
        )
        self.assertEqual(disposition["pilot_outputs_created"], 0)
        self.assertFalse(
            disposition["archived_uninspected_intermediates"][
                "scores_opened_or_interpreted"
            ]
        )
        self.assertEqual(
            amendment["profile_counts_per_task"]["full50"]["primary"], 1200
        )
        tampered = deepcopy(amendment)
        tampered["campaign_disposition"]["pilot_outputs_created"] = 1
        with self.assertRaisesRegex(ProtocolAmendmentError, "evidence drifted"):
            validate_global_set_transformer_amendment(tampered)

    def test_json_schema_is_valid_json_and_tracks_runtime_version(self):
        path = Path("configs/bdb_suite/schema/task_spec.schema.json")
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], TASK_SPEC_SCHEMA_VERSION)
        self.assertIn("source_tree_receipts", schema["properties"])
        self.assertIn("prepared_contract", schema["required"])

    def test_all_release_specs_bind_explicit_prepared_semantics(self):
        from bdb_study.manifest import EXPECTED_COHORTS

        root = Path("configs/bdb_suite/tasks")
        expected_shapes = {
            "bdb2020_rushing_harmonized": [31007, 1, 23, 24],
            "bdb2021_completion": [17846, 20, 23, 24],
            "bdb2022_punt_returns": [2273, 40, 23, 24],
            "bdb2023_sack": [8533, 20, 23, 24],
            "bdb2024_tackle": [10599, 10, 23, 25],
            "bdb2025_man_zone": [9229, 20, 23, 24],
            "bdb2026_trajectory": [46045, 123, 17, 24],
        }
        for path in sorted(root.glob("*.json")):
            with self.subTest(path=path.name):
                spec = load_task_spec(path)
                contract = spec.prepared_contract
                self.assertEqual(
                    contract["arrays"]["player_tokens"]["shape"],
                    expected_shapes[spec.task_id],
                )
                self.assertEqual(contract["channel_names"], spec.features["channels"])
                self.assertTrue(contract["tabular_feature_order"])
                self.assertEqual(
                    EXPECTED_COHORTS[spec.task_id],
                    (int(contract["examples"]), int(contract["games"])),
                )


if __name__ == "__main__":
    unittest.main()
