from __future__ import annotations

from copy import deepcopy
import json
import unittest

from bdb_study.contracts import (
    SOURCE_RECEIPT_SCHEMA_VERSION,
    TASK_SPEC_SCHEMA_VERSION,
    TaskSpec,
    sha256_json,
)
from bdb_study.design import (
    BASE_SEED,
    TASK_DESIGNS,
    DesignError,
    GameRegistryError,
    SeedCollisionError,
    SeedRegistry,
    build_task_design,
    shared_split_hashes,
    stable_seed,
    validate_task_design,
)


def make_spec(task_id: str) -> TaskSpec:
    profile = TASK_DESIGNS[task_id]
    metric = {
        "binary": "brier",
        "frame_event": "brier",
        "distribution": "crps",
        "trajectory": "rmse",
    }[profile.outcome_type]
    outcome = {"target": "synthetic target", "null_model": "training-only null"}
    if profile.outcome_type in {"binary", "frame_event"}:
        outcome["positive_label"] = 1
    if profile.outcome_type == "distribution":
        outcome["support"] = [-20, 110]
    n = profile.total_games
    arrays = {
        "player_tokens": {
            "axes": ["example", "time", "tracked_object", "channel"],
            "shape": [n, 1, 2, 2],
            "dtype": "float32",
        },
        "player_mask": {
            "axes": ["example", "time", "tracked_object"],
            "shape": [n, 1, 2],
            "dtype": "bool",
        },
        "frame_mask": {
            "axes": ["example", "time"],
            "shape": [n, 1],
            "dtype": "bool",
        },
        "y": None,
        "target_values": None,
        "target_mask": None,
        "target_baseline": None,
    }
    if profile.outcome_type == "trajectory":
        arrays.update(
            {
                "target_values": {
                    "axes": ["example", "horizon", "coordinate"],
                    "shape": [n, 2, 2],
                    "dtype": "float32",
                },
                "target_mask": {
                    "axes": ["example", "horizon"],
                    "shape": [n, 2],
                    "dtype": "bool",
                },
                "target_baseline": {
                    "axes": ["example", "horizon", "coordinate"],
                    "shape": [n, 2, 2],
                    "dtype": "float32",
                },
            }
        )
        support = None
        target_contract = {
            "encoding": "absolute_xy",
            "examples_target": "nan_placeholder",
            "training_target": "target_values_minus_target_baseline",
            "model_output": "masked_xy_residual",
            "coordinate_order": ["x", "y"],
            "reconstruction": "target_baseline_plus_predicted_residual",
        }
        target_mask_contract = {
            "meaning": "true_iff_official_future_coordinate_is_observed_and_scored",
            "layout": "right_padded",
            "masked_target_value": "nan",
        }
    else:
        arrays["y"] = {
            "axes": ["example"],
            "shape": [n],
            "dtype": "int16" if profile.outcome_type == "distribution" else "int8",
        }
        target_mask_contract = None
        if profile.outcome_type == "distribution":
            support_values = [float(value) for value in range(-20, 111)]
            support = {
                "count": len(support_values),
                "minimum": -20.0,
                "maximum": 110.0,
                "values_sha256": sha256_json(support_values),
            }
            target_contract = {
                "encoding": "zero_based_support_index",
                "examples_target": "equals_support_at_y",
                "model_output": "ordered_support_probability_vector",
            }
        else:
            support = None
            target_contract = {
                "encoding": "zero_one",
                "examples_target": "equals_y",
                "model_output": "positive_class_probability",
            }
    prepared_contract = {
        "schema_version": "bdb-prepared-semantics-v2",
        "task_id": task_id,
        "outcome_type": profile.outcome_type,
        "primary_metric": metric,
        "examples": n,
        "games": n,
        "tabular_feature_order": ["pre_cutoff"],
        "channel_names": ["relative_x", "relative_y"],
        "arrays": arrays,
        "support": support,
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
            "target_mask": target_mask_contract,
        },
        "target_contract": target_contract,
        "adapter_metadata": {
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            )
        },
    }
    prepared_contract["semantic_hash"] = sha256_json(prepared_contract)
    return TaskSpec.from_mapping(
        {
            "schema_version": TASK_SPEC_SCHEMA_VERSION,
            "task_id": task_id,
            "release_year": profile.release_year,
            "outcome_type": profile.outcome_type,
            "primary_metric": metric,
            "total_games": profile.total_games,
            "development_games": profile.development_games,
            "confirmatory_games": profile.confirmatory_games,
            "outer_counts": profile.outer_counts,
            "anchors": list(profile.anchors),
            "event_cutoff": {"event": "cutoff", "fallback_rules": []},
            "cohort": {"description": "cohort", "eligibility": ["eligible"], "exclusions": []},
            "features": {
                "allowlist": ["pre_cutoff"],
                "denylist": ["post_cutoff"],
                "channels": ["relative_x", "relative_y"],
                "coordinate_normalization": "left to right",
            },
            "outcome": outcome,
            "prepared_contract": prepared_contract,
            "history": {
                "scope": "causal_tracking_history",
                "endpoint": "cutoff",
                "max_frames": 1,
                "padding": "none",
                "stable_player_slots": True,
                "slot_policy": "play_global_focal_independent_v1",
            },
            "graph": {
                "node_types": ["player"],
                "edge_types": ["generic"],
                "construction": "synthetic graph",
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
                "id": (
                    "punt_cdf_residual_v1"
                    if profile.outcome_type == "distribution"
                    else "horizon_residual_xy_v1"
                    if profile.outcome_type == "trajectory"
                    else "binary_probability_v1"
                ),
                "prediction": "synthetic prediction",
                "training_loss": (
                    "crps" if profile.outcome_type == "distribution"
                    else "masked_residual_mse" if profile.outcome_type == "trajectory"
                    else "log_loss"
                ),
            },
            "uncertainty": {
                "primary_method": (
                    "game_clustered_central_interval_padding_v1"
                    if profile.outcome_type == "distribution"
                    else "whole_path_scaled_radial_conformal_tube_v1"
                    if profile.outcome_type == "trajectory"
                    else "out_of_game_venn_abers_calibration_v1"
                ),
                "secondary_diagnostic": (
                    "raw_score_class_conditional_conformal_sets_v1"
                    if profile.outcome_type in {"binary", "frame_event"}
                    else None
                ),
                "secondary_score_source": (
                    "raw_model_score"
                    if profile.outcome_type in {"binary", "frame_event"}
                    else None
                ),
                "field_intersection": (
                    "disk_intersect_legal_field_rectangle"
                    if profile.outcome_type == "trajectory"
                    else None
                ),
                "horizon_scale_learner": (
                    {
                        "source": "five_grouped_oof_development_game_folds",
                        "fold_statistic": "horizon_radial_error_median_and_observed_path_count",
                        "pooling": "count_weighted_median_across_folds",
                        "isotonic_projection": "observed_count_weighted_nondecreasing_pava",
                        "support_requirement": "pooled_positive_counts_form_nonempty_prefix_v1",
                        "unobserved_tail_policy": "carry_forward_last_fitted_pava_level_v1",
                        "floor_yards": 0.25,
                        "horizon_count": 94,
                        "artifact_fields": [
                            "dev_horizon_scale",
                            "dev_horizon_scale_sha256",
                            "dev_horizon_scale_observed_through",
                            "dev_horizon_scale_support_requirement",
                            "dev_horizon_scale_tail_policy",
                        ],
                    }
                    if profile.outcome_type == "trajectory"
                    else None
                ),
                "calibration_unit": "game",
                "nominal_coverage": 0.9,
                "report": ["synthetic_uq"],
            },
            "structural_ablation": {
                "id": "synthetic_structure_ablation",
                "description": "synthetic ablation",
                "models": ["relnet", "attn_relnet"],
                "anchors": [20, int(max(profile.anchors))],
                "repeats": 20,
            },
            "sensitivities": [],
            "models": {
                "linear_structure": {
                    "role": "linear_structure",
                    "family": "glm",
                    "alpha_grid": (
                        [0.1, 0.01, 0.001, 0.0001, 0.00001]
                        if profile.outcome_type == "trajectory"
                        else [10.0, 10.0 / 3.0, 1.0, 1.0 / 3.0, 0.1]
                    ),
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
            "artifacts": {"metrics": {}, "predictions": {}},
            "source_receipts": [
                {
                    "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
                    "source_id": f"{task_id}_raw",
                    "source_path": "/source",
                    "destination_path": f"data/{task_id}/raw/source",
                    "size_bytes": 0,
                    "sha256": "a" * 64,
                    "row_count": None,
                    "columns": [],
                }
            ],
            "source_tree_receipts": [],
            "shared_game_registry_with": profile.shared_game_registry_with,
        }
    )


def games(task_id: str):
    profile = TASK_DESIGNS[task_id]
    seasons = (2018, 2019, 2020) if task_id == "bdb2022_punt_returns" else (2022,)
    return [
        {
            "game_id": f"game_{index:04d}",
            "season": seasons[index % len(seasons)],
            "week": 1 + index % 18,
        }
        for index in range(profile.total_games)
    ]


class LockedProfileTests(unittest.TestCase):
    def test_all_six_profiles_match_locked_counts(self):
        expected = {
            "bdb2021_completion": (253, 30, 223, (134, 44, 45)),
            "bdb2022_punt_returns": (712, 90, 622, (373, 124, 125)),
            "bdb2023_sack": (122, 20, 102, (61, 20, 21)),
            "bdb2024_tackle": (136, 30, 106, (64, 21, 21)),
            "bdb2025_man_zone": (136, 30, 106, (64, 21, 21)),
            "bdb2026_trajectory": (272, 36, 236, (142, 47, 47)),
        }
        for task_id, values in expected.items():
            profile = TASK_DESIGNS[task_id]
            self.assertEqual(
                (
                    profile.total_games,
                    profile.development_games,
                    profile.confirmatory_games,
                    (profile.train_games, profile.calibration_games, profile.test_games),
                ),
                values,
            )

    def test_wrong_locked_profile_is_rejected(self):
        value = make_spec("bdb2023_sack").as_dict()
        value["development_games"] = 21
        value["confirmatory_games"] = 101
        value["outer_counts"] = {"train": 60, "calibration": 20, "test": 21}
        with self.assertRaisesRegex(Exception, "locked design"):
            build_task_design(value, games("bdb2023_sack"), repeats=2)


class SeedTests(unittest.TestCase):
    def test_seed_is_stable_semantic_and_bounded(self):
        self.assertEqual(stable_seed(BASE_SEED, "analysis", "bootstrap"), 2_127_296_457)
        self.assertNotEqual(
            stable_seed(BASE_SEED, "task-a", 1), stable_seed(BASE_SEED, "task-b", 1)
        )

    def test_registry_rejects_collisions(self):
        registry = SeedRegistry(seed_function=lambda _base, *_parts: 7)
        registry.get("one")
        with self.assertRaises(SeedCollisionError):
            registry.get("two")


class MainDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = make_spec("bdb2023_sack")
        cls.design = build_task_design(cls.spec, games(cls.spec.task_id))

    def test_exact_development_and_confirmatory_partitions(self):
        registry = self.design["game_registry"]
        development = registry["development_game_ids"]
        confirmatory = registry["confirmatory_game_ids"]
        self.assertEqual(len(development), 20)
        self.assertEqual(len(confirmatory), 102)
        self.assertFalse(set(development) & set(confirmatory))
        folds = [fold["validation_game_ids"] for fold in registry["development_folds"]]
        self.assertEqual(sorted(sum(folds, [])), sorted(development))
        self.assertEqual({len(fold) for fold in folds}, {4})

    def test_fifty_independent_exact_nested_splits(self):
        self.assertEqual(len(self.design["split_manifests"]), 50)
        outer_hashes = set()
        development = set(self.design["game_registry"]["development_game_ids"])
        for split in self.design["split_manifests"]:
            self.assertEqual(len(split["train_game_ids"]), 61)
            self.assertEqual(len(split["calibration_game_ids"]), 20)
            self.assertEqual(len(split["test_game_ids"]), 21)
            self.assertFalse(development & set(split["train_game_ids"]))
            self.assertFalse(development & set(split["calibration_game_ids"]))
            self.assertFalse(development & set(split["test_game_ids"]))
            prior = []
            for anchor in self.spec.anchors:
                current = split["nested_train_game_ids"][str(anchor)]
                self.assertEqual(len(current), anchor)
                self.assertEqual(current[: len(prior)], prior)
                prior = current
            outer_hashes.add(split["outer_split_hash"])
        self.assertEqual(len(outer_hashes), 50)


    def test_grid_has_exactly_1500_unique_paired_cells_and_stage_seeds(self):
        cells = self.design["required_cells"]
        self.assertEqual(len(cells), 1500)
        identities = {
            (cell["repeat"], cell["n_train"], cell["model"]) for cell in cells
        }
        self.assertEqual(len(identities), 1500)
        for repeat in range(1, 51):
            repeat_cells = [cell for cell in cells if cell["repeat"] == repeat]
            self.assertEqual(len({cell["outer_split_hash"] for cell in repeat_cells}), 1)
            self.assertEqual(len({cell["nested_split_hash"] for cell in repeat_cells}), 1)
            self.assertTrue(all(len(cell["seeds"]) == 7 for cell in repeat_cells))
            self.assertEqual(
                len(
                    {
                        cell["seeds"]["uncertainty_subsample"]
                        for cell in repeat_cells
                    }
                ),
                1,
            )
            for anchor in self.spec.anchors:
                anchor_cells = [cell for cell in repeat_cells if cell["n_train"] == anchor]
                self.assertEqual(
                    len({cell["seeds"]["validation_split"] for cell in anchor_cells}),
                    1,
                )
                neural = {
                    cell["model"]: cell["seeds"]
                    for cell in anchor_cells
                    if cell["model"] in {
                        "relnet", "attn_relnet", "set_transformer"
                    }
                }
                self.assertEqual(neural["relnet"], neural["attn_relnet"])
                self.assertEqual(neural["relnet"], neural["set_transformer"])
        self.assertEqual(
            len(self.design["seed_registry"]),
            len(set(self.design["seed_registry"].values())),
        )
        validate_task_design(self.design, self.spec)

    def test_tampering_is_rejected(self):
        changed = deepcopy(self.design)
        changed["split_manifests"][0]["test_game_ids"][0] = "not-a-game"
        with self.assertRaises(DesignError):
            validate_task_design(changed, self.spec)

    def test_design_is_byte_deterministic(self):
        rebuilt = build_task_design(self.spec, list(reversed(games(self.spec.task_id))))
        self.assertEqual(rebuilt, self.design)

    def test_frozen_task_registry_is_bound_to_exact_game_ids_and_folds(self):
        frozen_mapping = self.spec.as_dict()
        frozen_mapping["cohort"]["game_registry"] = deepcopy(
            self.design["game_registry"]
        )
        frozen_spec = TaskSpec.from_mapping(frozen_mapping)
        rebuilt = build_task_design(frozen_spec, games(self.spec.task_id))
        self.assertEqual(
            rebuilt["game_registry"], frozen_mapping["cohort"]["game_registry"]
        )

        changed_games = games(self.spec.task_id)
        changed_games[0]["week"] += 1
        with self.assertRaisesRegex(DesignError, "exact registry frozen"):
            build_task_design(frozen_spec, changed_games)

        tampered_mapping = frozen_spec.as_dict()
        tampered_mapping["cohort"]["game_registry"]["development_game_ids"][0] = (
            "not-the-frozen-game"
        )
        tampered_spec = TaskSpec.from_mapping(tampered_mapping)
        with self.assertRaisesRegex(DesignError, "exact registry frozen"):
            build_task_design(tampered_spec, games(self.spec.task_id))

    def test_augmented_or_duplicate_game_ids_are_rejected(self):
        bad = games(self.spec.task_id)
        bad[0]["game_id"] += "_aug"
        with self.assertRaises(GameRegistryError):
            build_task_design(self.spec, bad, repeats=1)
        duplicate = games(self.spec.task_id)
        duplicate[1]["game_id"] = duplicate[0]["game_id"]
        with self.assertRaises(GameRegistryError):
            build_task_design(self.spec, duplicate, repeats=1)


class PilotDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = make_spec("bdb2023_sack")
        cls.pilot = build_task_design(
            cls.spec,
            games(cls.spec.task_id),
            repeats=10,
            profile="pilot10",
        )
        cls.full = build_task_design(
            cls.spec,
            games(cls.spec.task_id),
            repeats=100,
            profile="full100",
        )

    def test_pilot_is_exact_primary_only_10x6x5_grid(self):
        self.assertEqual(self.pilot["profile"], "pilot10")
        self.assertEqual(self.pilot["repeats"], 10)
        self.assertEqual(
            self.pilot["cell_counts"],
            {
                "primary": 300,
                "structural_ablation": 0,
                "frozen_sensitivity": 0,
                "required": 300,
            },
        )
        self.assertFalse(self.pilot["ablation_cells"])
        self.assertFalse(self.pilot["sensitivity_cells"])
        validate_task_design(self.pilot, self.spec, profile="pilot10")

    def test_pilot_split_cell_uq_and_analysis_seeds_are_namespaced(self):
        registry = self.pilot["seed_registry"]
        self.assertIn(
            json.dumps(
                ["task", self.spec.task_id, "pilot10", "analysis", "bootstrap"],
                separators=(",", ":"),
            ),
            registry,
        )
        self.assertTrue(
            any('"pilot10","cell","fixed_main"' in key for key in registry)
        )
        self.assertTrue(any('"pilot10","outer",1' in key for key in registry))
        first = self.pilot["primary_cells"][0]
        repeat_uq = {
            cell["seeds"]["uncertainty_subsample"]
            for cell in self.pilot["primary_cells"]
            if cell["repeat"] == first["repeat"]
        }
        self.assertEqual(len(repeat_uq), 1)

    def test_pilot_smoke_custom_grid_replays_the_same_pilot_seed_namespace(self):
        smoke = build_task_design(
            self.spec,
            games(self.spec.task_id),
            repeats=1,
            seed_profile="pilot10",
        )
        self.assertEqual(smoke["profile"], "custom")
        self.assertEqual(smoke["seed_profile"], "pilot10")
        self.assertEqual(
            smoke["split_manifests"][0], self.pilot["split_manifests"][0]
        )
        pilot_repeat_one = [
            cell for cell in self.pilot["primary_cells"] if cell["repeat"] == 1
        ]
        self.assertEqual(smoke["primary_cells"], pilot_repeat_one)
        validate_task_design(smoke, self.spec)

    def test_existing_full100_seed_namespace_is_unchanged(self):
        registry = self.full["seed_registry"]
        expected = json.dumps(
            ["task", self.spec.task_id, "analysis", "bootstrap"],
            separators=(",", ":"),
        )
        self.assertIn(expected, registry)
        self.assertFalse(any('"pilot10"' in key for key in registry))
        self.assertTrue(any('"cell","fixed_main",1' in key for key in registry))
        pilot_first = self.pilot["primary_cells"][0]["seeds"]
        full_first = self.full["primary_cells"][0]["seeds"]
        self.assertTrue(
            all(pilot_first[field] != full_first[field] for field in pilot_first)
        )
        self.assertNotEqual(
            self.pilot["split_manifests"][0]["seeds"],
            self.full["split_manifests"][0]["seeds"],
        )

    def test_grandfathered_full50_keeps_the_original_four_role_cells_and_seeds(self):
        legacy = build_task_design(
            self.spec,
            games(self.spec.task_id),
            repeats=50,
            profile="full50",
        )
        self.assertEqual(
            legacy["cell_counts"],
            {
                "primary": 1200,
                "structural_ablation": 0,
                "frozen_sensitivity": 0,
                "required": 1200,
            },
        )
        original_roles = {
            "linear_structure", "boosted_structure", "relnet", "attn_relnet"
        }
        self.assertEqual(
            {cell["model"] for cell in legacy["primary_cells"]}, original_roles
        )
        self.assertNotIn(
            "set_transformer",
            {cell["model"] for cell in legacy["primary_cells"]},
        )
        first = next(
            cell
            for cell in legacy["primary_cells"]
            if cell["repeat"] == 1
            and cell["n_train"] == 10
            and cell["model"] == "relnet"
        )
        self.assertEqual(
            first["seeds"]["fit"],
            stable_seed(
                BASE_SEED,
                "bdb-suite",
                "task",
                self.spec.task_id,
                "cell",
                "fixed_main",
                1,
                "fit",
                "shared_neural",
                10,
            ),
        )
        self.assertEqual(
            first["seeds"]["uncertainty_subsample"],
            stable_seed(
                BASE_SEED,
                "bdb-suite",
                "task",
                self.spec.task_id,
                "cell",
                "fixed_main",
                1,
                "uncertainty_subsample",
                "shared_all_models_and_anchors",
            ),
        )
        validate_task_design(legacy, self.spec, profile="full50")

    def test_profile_mismatch_and_rehashed_split_seed_tamper_fail_closed(self):
        with self.assertRaisesRegex(DesignError, "profile"):
            validate_task_design(self.pilot, self.spec, profile="full100")
        tampered = deepcopy(self.pilot)
        tampered["split_manifests"][0]["seeds"]["allocation_tie"] += 1
        unsigned = {
            key: value for key, value in tampered.items() if key != "design_hash"
        }
        tampered["design_hash"] = sha256_json(unsigned)
        with self.assertRaisesRegex(DesignError, "replay"):
            validate_task_design(tampered, self.spec, profile="pilot10")


class SharedRegistryTests(unittest.TestCase):
    def test_2024_and_2025_share_identical_game_partitions(self):
        records = games("bdb2024_tackle")
        tackles = build_task_design(make_spec("bdb2024_tackle"), records, repeats=5)
        coverage = build_task_design(make_spec("bdb2025_man_zone"), records, repeats=5)
        self.assertEqual(tackles["game_registry"], coverage["game_registry"])
        self.assertEqual(shared_split_hashes(tackles), shared_split_hashes(coverage))
        self.assertEqual(tackles["split_manifests"], coverage["split_manifests"])
        # Optimization seeds remain task-specific even though data partitions are shared.
        self.assertNotEqual(
            tackles["required_cells"][0]["seeds"],
            coverage["required_cells"][0]["seeds"],
        )


if __name__ == "__main__":
    unittest.main()
