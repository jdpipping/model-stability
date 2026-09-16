from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters.common import PreparedTask
from bdb_study.design import SeedRegistry
from bdb_study.devcv import (
    HORIZON_SCALE_SUPPORT_REQUIREMENT,
    HORIZON_SCALE_TAIL_POLICY,
    JOINT_NEURAL_TARGET_DEFAULT_PATH,
    build_joint_neural_target_receipt,
    candidate_grid,
    finalize_joint_neural_development_receipts,
    fit_development_horizon_scale,
    reduce_development_checkpoints,
    run_development_cv,
    run_development_cv_shard,
    select_candidate,
    validate_development_receipt,
    validate_joint_neural_target_receipt,
    write_joint_neural_target_receipt,
)


class DevelopmentCVTests(unittest.TestCase):
    def test_horizon_scale_is_floored_and_isotonic(self) -> None:
        scale = fit_development_horizon_scale(
            np.array([0.1, 1.0, 0.5, 2.0]), np.array([10, 10, 10, 10])
        )
        self.assertTrue(np.all(scale >= 0.25))
        self.assertTrue(np.all(np.diff(scale) >= 0.0))
        self.assertAlmostEqual(scale[1], scale[2])

    def test_horizon_scale_carries_last_pava_level_across_unobserved_tail(self) -> None:
        scale = fit_development_horizon_scale(
            np.array([0.1, 1.0, 0.5, np.nan, np.nan]),
            np.array([10, 10, 10, 0, 0]),
        )
        np.testing.assert_array_equal(scale, np.array([0.25, 0.75, 0.75, 0.75, 0.75]))

    def test_horizon_scale_rejects_noncanonical_support(self) -> None:
        invalid = (
            (np.array([np.nan, np.nan]), np.array([0, 0])),
            (np.array([np.nan, 1.0]), np.array([0, 5])),
            (np.array([1.0, np.nan, 2.0]), np.array([10, 0, 5])),
            (np.array([1.0, 2.0]), np.array([10, 0])),
            (np.array([np.nan, np.nan]), np.array([10, 0])),
            (np.array([1.0, 2.0]), np.array([10.0, 5.0])),
            (np.array([1.0, 2.0]), np.array([5, 10])),
            (np.array([1.0, -1.0]), np.array([10, 5])),
        )
        for medians, counts in invalid:
            with self.subTest(medians=medians.tolist(), counts=counts.tolist()):
                with self.assertRaises(ValueError):
                    fit_development_horizon_scale(medians, counts)

    def test_candidate_grids_are_locked(self) -> None:
        self.assertEqual([row["alpha"] for row in candidate_grid("binary", "glm")], [10, 10 / 3, 1, 1 / 3, .1])
        self.assertEqual(len(candidate_grid("binary", "lightgbm")), 4)
        neural = candidate_grid("binary", "cnn")
        self.assertEqual(len(neural), 4)
        self.assertEqual(neural[0]["dropout"], .3)
        trajectory = candidate_grid("trajectory", "relnet")
        self.assertEqual(
            {row["trajectory_target"] for row in trajectory},
            {"residual", "absolute"},
        )
        global_set = candidate_grid("binary", "set_transformer")
        self.assertEqual(global_set[0]["d_model"], 64)
        self.assertEqual(global_set[0]["layers"], 3)

    def test_trajectory_receipt_freezes_grouped_oof_94_horizon_scale(self) -> None:
        n, horizon = 12, 94
        examples = pd.DataFrame(
            {
                "example_id": [f"path-{index}" for index in range(n)],
                "game_id": np.arange(n, dtype=np.int64),
                "stratum": ["season"] * n,
                "target": [np.nan] * n,
            }
        )
        target_mask = np.zeros((n, horizon), dtype=bool)
        target_mask[:, :33] = True
        target_mask[:2, 32] = False
        task = PreparedTask(
            task_id="bdb2026_trajectory",
            outcome_type="trajectory",
            primary_metric="rmse",
            examples=examples,
            tabular=pd.DataFrame({"x": np.arange(n, dtype=float)}),
            player_tokens=np.zeros((n, 1, 1, 1), dtype=np.float32),
            player_mask=np.ones((n, 1, 1), dtype=bool),
            frame_mask=np.ones((n, 1), dtype=bool),
            channel_names=("x_rel",),
            target_values=np.zeros((n, horizon, 2), dtype=np.float32),
            target_mask=target_mask,
            target_baseline=np.zeros((n, horizon, 2), dtype=np.float32),
        )
        seeds = SeedRegistry()
        folds = []
        for fold in range(1, 6):
            games = [str(2 * (fold - 1)), str(2 * (fold - 1) + 1)]
            folds.append({"fold": fold, "validation_game_ids": games})
            seeds.get(
                "task", task.task_id, "development_cv", fold, "fit",
                "linear_structure",
            )
            seeds.get(
                "task", task.task_id, "development_cv", fold, "prediction",
                "linear_structure",
            )
        design = {
            "task_id": task.task_id,
            "design_hash": "design",
            "seed_registry": seeds.snapshot(),
            "game_registry": {
                "registry_hash": "registry",
                "development_game_ids": [str(value) for value in range(10)],
                "development_folds": folds,
            },
        }

        def evaluator(family, config, train, validation, fit_seed, prediction_seed):
            del family, train, fit_seed, prediction_seed
            # Each validation fold is a game group and contributes only its
            # OOF median/count receipt, never in-sample training residuals.
            raw = np.linspace(0.1, 2.0, horizon)
            raw[20:30] = 0.2  # exercise weighted isotonic projection
            counts = np.asarray(task.target_mask[validation], dtype=bool).sum(
                axis=0
            ).astype(int).tolist()
            medians = [
                float(raw[index]) if count > 0 else None
                for index, count in enumerate(counts)
            ]
            return (
                0.1 if config["trajectory_target"] == "absolute" else 0.2,
                {
                    "horizon_scale_median": medians,
                    "horizon_scale_count": counts,
                },
            )

        provenance = {"prepared_hash": "p" * 64}
        receipt = run_development_cv(
            task,
            design,
            "glm",
            evaluator,
            model_id="linear_structure",
            provenance=provenance,
        )
        selected = receipt["selected"]["config"]
        scale = np.asarray(selected["dev_horizon_scale"], dtype=float)
        self.assertEqual(selected["trajectory_target"], "absolute")
        self.assertEqual(scale.shape, (94,))
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale >= 0.25))
        self.assertTrue(np.all(np.diff(scale) >= 0.0))
        self.assertEqual(selected["dev_horizon_scale_observed_through"], 33)
        self.assertEqual(
            selected["dev_horizon_scale_support_requirement"],
            HORIZON_SCALE_SUPPORT_REQUIREMENT,
        )
        self.assertEqual(
            selected["dev_horizon_scale_tail_policy"], HORIZON_SCALE_TAIL_POLICY
        )
        np.testing.assert_array_equal(scale[33:], np.repeat(scale[32], 61))
        self.assertEqual(len(selected["dev_horizon_scale_sha256"]), 64)
        self.assertEqual(
            validate_development_receipt(
                receipt,
                task,
                design,
                "glm",
                model_id="linear_structure",
                expected_provenance=provenance,
            ),
            receipt,
        )

        tampered = json.loads(json.dumps(receipt))
        tampered["selected"]["config"][
            "dev_horizon_scale_observed_through"
        ] = 32
        tampered["selected"]["config_hash"] = hashlib.sha256(
            json.dumps(
                tampered["selected"]["config"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        unsigned = {key: value for key, value in tampered.items() if key != "receipt_hash"}
        tampered["receipt_hash"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(ValueError):
            validate_development_receipt(
                tampered,
                task,
                design,
                "glm",
                model_id="linear_structure",
                expected_provenance=provenance,
            )

        tampered = json.loads(json.dumps(receipt))
        tampered["fold_scores"][0]["audit"]["horizon_scale_count"][:32] = [
            1
        ] * 32
        unsigned = {
            key: value for key, value in tampered.items() if key != "receipt_hash"
        }
        tampered["receipt_hash"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "differ from the prepared fold mask"):
            validate_development_receipt(
                tampered,
                task,
                design,
                "glm",
                model_id="linear_structure",
                expected_provenance=provenance,
            )

        tampered = json.loads(json.dumps(receipt))
        tampered["fold_scores"][0]["audit"]["horizon_scale_count"][40] = 1
        tampered["fold_scores"][0]["audit"]["horizon_scale_median"][40] = 1.0
        unsigned = {key: value for key, value in tampered.items() if key != "receipt_hash"}
        tampered["receipt_hash"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(ValueError):
            validate_development_receipt(
                tampered,
                task,
                design,
                "glm",
                model_id="linear_structure",
                expected_provenance=provenance,
            )

    def test_exact_tie_retains_first_candidate(self) -> None:
        rows = []
        for candidate in range(2):
            for fold in range(1, 6):
                rows.append(
                    {
                        "candidate_index": candidate,
                        "fold": fold,
                        "config": {"alpha": 10 - candidate},
                        "config_hash": str(candidate),
                        "validation_loss": .2,
                    }
                )
        selected = select_candidate(rows)["selected"]
        self.assertEqual(selected["candidate_index"], 0)

    def _joint_trajectory_case(
        self,
        target_losses: dict[str, dict[str, float]],
        *,
        candidate_offsets: dict[str, dict[tuple[float, float], float]] | None = None,
    ):
        n, horizon = 12, 94
        examples = pd.DataFrame(
            {
                "example_id": [f"path-{index}" for index in range(n)],
                "game_id": np.arange(n, dtype=np.int64),
                "stratum": ["season"] * n,
                "target": [np.nan] * n,
            }
        )
        task = PreparedTask(
            task_id="bdb2026_trajectory",
            outcome_type="trajectory",
            primary_metric="rmse",
            examples=examples,
            tabular=pd.DataFrame({"x": np.arange(n, dtype=float)}),
            player_tokens=np.zeros((n, 1, 1, 1), dtype=np.float32),
            player_mask=np.ones((n, 1, 1), dtype=bool),
            frame_mask=np.ones((n, 1), dtype=bool),
            channel_names=("x_rel",),
            target_values=np.zeros((n, horizon, 2), dtype=np.float32),
            target_mask=np.ones((n, horizon), dtype=bool),
            target_baseline=np.zeros((n, horizon, 2), dtype=np.float32),
        )
        seeds = SeedRegistry()
        folds = []
        for fold in range(1, 6):
            games = [str(2 * (fold - 1)), str(2 * (fold - 1) + 1)]
            folds.append({"fold": fold, "validation_game_ids": games})
            for stage in ("fit", "prediction"):
                seeds.get(
                    "task",
                    task.task_id,
                    "development_cv",
                    fold,
                    stage,
                    "shared_neural",
                )
        design = {
            "task_id": task.task_id,
            "design_hash": "joint-design",
            "seed_registry": seeds.snapshot(),
            "game_registry": {
                "registry_hash": "joint-registry",
                "development_game_ids": [str(value) for value in range(10)],
                "development_folds": folds,
            },
        }
        offsets = {} if candidate_offsets is None else candidate_offsets

        def evaluator(family, config, train, validation, fit_seed, prediction_seed):
            del train, fit_seed, prediction_seed
            target = str(config["trajectory_target"])
            key = (float(config["dropout"]), float(config["learning_rate"]))
            loss = float(target_losses[family][target]) + float(
                offsets.get(family, {}).get(key, 0.0)
            )
            family_scale = {
                "relnet": 0.5,
                "attn_relnet": 1.5,
                "set_transformer": 2.5,
            }[family]
            target_scale = 0.1 if target == "residual" else 0.2
            medians = (
                family_scale
                + target_scale
                + np.linspace(0.0, 0.93, horizon, dtype=np.float64)
            )
            return loss, {
                "horizon_scale_median": medians.tolist(),
                "horizon_scale_count": [int(len(validation))] * horizon,
            }

        provenance = {"prepared_hash": "p" * 64, "code_hash": "c" * 64}
        receipts = {
            family: run_development_cv(
                task,
                design,
                family,
                evaluator,
                model_id=family,
                provenance=provenance,
            )
            for family in ("relnet", "attn_relnet", "set_transformer")
        }
        return task, design, provenance, receipts

    def test_joint_neural_target_overrides_disagreeing_independent_winners(self) -> None:
        task, design, provenance, receipts = self._joint_trajectory_case(
            {
                "relnet": {"residual": 0.8, "absolute": 2.0},
                "attn_relnet": {"residual": 3.0, "absolute": 0.7},
                "set_transformer": {"residual": 2.0, "absolute": 1.0},
            },
            candidate_offsets={
                "relnet": {
                    (0.3, 0.001): 0.3,
                    (0.3, 0.0003): 0.2,
                    (0.1, 0.001): 0.1,
                    (0.1, 0.0003): 0.0,
                },
                "attn_relnet": {
                    (0.3, 0.001): 0.0,
                    (0.3, 0.0003): 0.1,
                    (0.1, 0.001): 0.2,
                    (0.1, 0.0003): 0.3,
                },
            },
        )
        self.assertEqual(
            receipts["relnet"]["selected"]["config"]["trajectory_target"],
            "residual",
        )
        self.assertEqual(
            receipts["attn_relnet"]["selected"]["config"]["trajectory_target"],
            "absolute",
        )
        for family in ("relnet", "attn_relnet", "set_transformer"):
            self.assertEqual(
                {row["queue"] for row in receipts[family]["fold_scores"]},
                {"gpu_neural"},
            )
            self.assertEqual(
                {row["family"] for row in receipts[family]["fold_scores"]},
                {family},
            )
        self.assertEqual(
            [
                (row["fold"], row["candidate_index"], row["fit_seed"])
                for row in receipts["relnet"]["fold_scores"]
            ],
            [
                (row["fold"], row["candidate_index"], row["fit_seed"])
                for row in receipts["attn_relnet"]["fold_scores"]
            ],
        )
        self.assertEqual(
            [
                (row["fold"], row["candidate_index"], row["fit_seed"])
                for row in receipts["relnet"]["fold_scores"]
            ],
            [
                (row["fold"], row["candidate_index"], row["fit_seed"])
                for row in receipts["set_transformer"]["fold_scores"]
            ],
        )

        joint = build_joint_neural_target_receipt(
            receipts,
            task,
            design,
            expected_provenance=provenance,
        )
        self.assertEqual(joint["selected_target"], "absolute")
        self.assertEqual(
            joint["per_target"]["absolute"]["family_selections"]["relnet"][
                "selected_candidate_index"
            ],
            7,
        )
        self.assertEqual(
            joint["per_target"]["absolute"]["family_selections"]["attn_relnet"][
                "selected_candidate_index"
            ],
            1,
        )
        self.assertEqual(
            joint["per_target"]["absolute"]["family_selections"]["set_transformer"][
                "selected_candidate_index"
            ],
            1,
        )
        self.assertEqual(
            validate_joint_neural_target_receipt(
                joint,
                receipts,
                task,
                design,
                expected_provenance=provenance,
            ),
            joint,
        )

        finalized = finalize_joint_neural_development_receipts(
            receipts,
            task,
            design,
            expected_provenance=provenance,
        )
        self.assertEqual(finalized["joint_receipt"], joint)
        trio = finalized["finalized_development_receipts"]
        for family in ("relnet", "attn_relnet", "set_transformer"):
            config = trio[family]["selected"]["config"]
            self.assertEqual(config["trajectory_target"], "absolute")
            self.assertEqual(len(config["dev_horizon_scale"]), 94)
            self.assertTrue(np.all(np.diff(config["dev_horizon_scale"]) >= 0.0))
            binding = trio[family]["joint_neural_target_selection"]
            self.assertEqual(binding["selected_target"], "absolute")
            self.assertEqual(binding["joint_receipt_hash"], joint["receipt_hash"])
            self.assertEqual(
                binding["joint_receipt_path"], JOINT_NEURAL_TARGET_DEFAULT_PATH
            )
            self.assertEqual(
                validate_development_receipt(
                    trio[family],
                    task,
                    design,
                    family,
                    model_id=family,
                    expected_provenance=provenance,
                ),
                trio[family],
            )
        self.assertNotEqual(
            trio["relnet"]["selected"]["config"]["learning_rate"],
            trio["attn_relnet"]["selected"]["config"]["learning_rate"],
        )
        self.assertNotEqual(
            trio["relnet"]["selected"]["config"]["dev_horizon_scale"],
            trio["attn_relnet"]["selected"]["config"]["dev_horizon_scale"],
        )
        self.assertEqual(
            validate_joint_neural_target_receipt(
                joint,
                trio,
                task,
                design,
                expected_provenance=provenance,
            ),
            joint,
        )
        wrong_binding = {
            family: json.loads(json.dumps(receipt))
            for family, receipt in trio.items()
        }
        wrong_binding["relnet"]["joint_neural_target_selection"][
            "joint_receipt_hash"
        ] = "0" * 64
        wrong_binding["relnet"]["receipt_hash"] = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in wrong_binding["relnet"].items()
                    if key != "receipt_hash"
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_joint_neural_target_receipt(
                joint,
                wrong_binding,
                task,
                design,
                expected_provenance=provenance,
            )

    def test_joint_neural_target_exact_tie_selects_residual_and_replays(self) -> None:
        task, design, provenance, receipts = self._joint_trajectory_case(
            {
                "relnet": {"residual": 1.0, "absolute": 3.0},
                "attn_relnet": {"residual": 3.0, "absolute": 1.0},
                "set_transformer": {"residual": 2.0, "absolute": 2.0},
            }
        )
        result = finalize_joint_neural_development_receipts(
            receipts,
            task,
            design,
            expected_provenance=provenance,
            joint_receipt_path="receipts/joint-target.json",
        )
        joint = result["joint_receipt"]
        self.assertEqual(
            joint["per_target"]["residual"]["pooled_equal_weight_mean_rmse"],
            joint["per_target"]["absolute"]["pooled_equal_weight_mean_rmse"],
        )
        self.assertEqual(joint["selected_target"], "residual")
        for receipt in result["finalized_development_receipts"].values():
            self.assertEqual(
                receipt["selected"]["config"]["trajectory_target"], "residual"
            )
            self.assertEqual(receipt["selected"]["candidate_index"], 0)
            self.assertEqual(
                receipt["joint_neural_target_selection"]["joint_receipt_path"],
                "receipts/joint-target.json",
            )

        finalized_tamper = json.loads(
            json.dumps(result["finalized_development_receipts"]["relnet"])
        )
        finalized_tamper["joint_neural_target_selection"]["selected_target"] = (
            "absolute"
        )
        finalized_unsigned = {
            key: value
            for key, value in finalized_tamper.items()
            if key != "receipt_hash"
        }
        finalized_tamper["receipt_hash"] = hashlib.sha256(
            json.dumps(
                finalized_unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "selection does not replay"):
            validate_development_receipt(
                finalized_tamper,
                task,
                design,
                "relnet",
                model_id="relnet",
                expected_provenance=provenance,
            )

        tampered = json.loads(json.dumps(joint))
        tampered["per_target"]["residual"]["pooled_equal_weight_mean_rmse"] += 0.01
        unsigned = {
            key: value for key, value in tampered.items() if key != "receipt_hash"
        }
        tampered["receipt_hash"] = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not replay"):
            validate_joint_neural_target_receipt(
                tampered,
                receipts,
                task,
                design,
                expected_provenance=provenance,
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "joint.json"
            write_joint_neural_target_receipt(joint, path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), joint
            )
            write_joint_neural_target_receipt(joint, path)
            different = json.loads(json.dumps(joint))
            different["selected_target"] = "absolute"
            different_unsigned = {
                key: value
                for key, value in different.items()
                if key != "receipt_hash"
            }
            different["receipt_hash"] = hashlib.sha256(
                json.dumps(
                    different_unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            with self.assertRaises(FileExistsError):
                write_joint_neural_target_receipt(different, path)

    def _synthetic_case(self):
        n = 12
        examples = pd.DataFrame(
            {
                "example_id": [f"e{i}" for i in range(n)],
                "game_id": list(range(n)),
                "stratum": ["s"] * n,
                "target": [i % 2 for i in range(n)],
            }
        )
        task = PreparedTask(
            task_id="synthetic",
            outcome_type="binary",
            primary_metric="brier",
            examples=examples,
            tabular=pd.DataFrame({"x": np.arange(n)}),
            player_tokens=np.zeros((n, 1, 1, 1), dtype=np.float32),
            player_mask=np.ones((n, 1, 1), dtype=bool),
            frame_mask=np.ones((n, 1), dtype=bool),
            channel_names=("x_rel",),
            y=np.asarray(examples.target),
        )
        seeds = SeedRegistry()
        folds = []
        for fold in range(1, 6):
            folds.append({"fold": fold, "validation_game_ids": [str(2 * (fold - 1)), str(2 * (fold - 1) + 1)]})
            seeds.get("task", "synthetic", "development_cv", fold, "fit", "glm")
            seeds.get("task", "synthetic", "development_cv", fold, "prediction", "glm")
        design = {
            "task_id": "synthetic",
            "design_hash": "design",
            "seed_registry": seeds.snapshot(),
            "game_registry": {
                "registry_hash": "registry",
                "development_game_ids": [str(i) for i in range(10)],
                "development_folds": folds,
            },
        }
        return task, design

    def test_development_games_only_and_five_folds(self) -> None:
        task, design = self._synthetic_case()
        observed = []

        def evaluator(family, config, train, validation, fit_seed, prediction_seed):
            observed.append((set(train), set(validation)))
            return float(config["alpha"])

        receipt = run_development_cv(task, design, "glm", evaluator)
        self.assertEqual(receipt["selected"]["config"]["alpha"], .1)
        self.assertEqual(len(receipt["fold_scores"]), 25)
        self.assertTrue(all(not (train & validation) for train, validation in observed))
        self.assertTrue(all(max(train | validation) < 10 for train, validation in observed))

    def test_checkpoints_resume_without_refitting_and_bind_provenance(self) -> None:
        task, design = self._synthetic_case()
        provenance = {"prepared_hash": "p" * 64, "code_hash": "c" * 64}
        calls = []

        def evaluator(family, config, train, validation, fit_seed, prediction_seed):
            calls.append((family, config["alpha"], fit_seed, prediction_seed))
            return float(config["alpha"]), {"examples": int(len(validation))}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            first = run_development_cv(
                task,
                design,
                "glm",
                evaluator,
                provenance=provenance,
                checkpoint_dir=checkpoints,
            )
            self.assertEqual(len(calls), 25)
            self.assertEqual(first["provenance"], provenance)
            self.assertEqual(
                validate_development_receipt(
                    first,
                    task,
                    design,
                    "glm",
                    model_id="glm",
                    expected_provenance=provenance,
                ),
                first,
            )
            for changed_field in ("prepared_hash", "code_hash", "dependency_lock"):
                changed_provenance = dict(provenance)
                changed_provenance[changed_field] = "different"
                with self.subTest(changed_field=changed_field), self.assertRaisesRegex(
                    ValueError, "provenance differs"
                ):
                    validate_development_receipt(
                        first,
                        task,
                        design,
                        "glm",
                        model_id="glm",
                        expected_provenance=changed_provenance,
                    )

            tampered = json.loads(json.dumps(first))
            tampered["selected"]["candidate_index"] = 0
            unsigned = {
                key: value for key, value in tampered.items() if key != "receipt_hash"
            }
            tampered["receipt_hash"] = hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "selection does not replay"):
                validate_development_receipt(
                    tampered,
                    task,
                    design,
                    "glm",
                    model_id="glm",
                    expected_provenance=provenance,
                )

            def must_not_run(*args, **kwargs):
                raise AssertionError("a valid checkpoint was unexpectedly refit")

            second = run_development_cv(
                task,
                design,
                "glm",
                must_not_run,
                provenance=provenance,
                checkpoint_dir=checkpoints,
                resume=True,
            )
            self.assertEqual(second, first)

            checkpoint = checkpoints / "fold_01" / "candidate_00.json"
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            value["row"]["validation_loss"] = 123.0
            checkpoint.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash is invalid"):
                run_development_cv(
                    task,
                    design,
                    "glm",
                    must_not_run,
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                    resume=True,
                )

    def test_interrupted_grid_resumes_only_missing_work(self) -> None:
        task, design = self._synthetic_case()

        def loss(config):
            return float(config["alpha"])

        clean = run_development_cv(
            task,
            design,
            "glm",
            lambda family, config, *args: loss(config),
            provenance={"identity": "stable"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            initial_calls = 0

            def interrupted(family, config, *args):
                nonlocal initial_calls
                initial_calls += 1
                if initial_calls == 8:
                    raise RuntimeError("simulated interruption")
                return loss(config)

            with self.assertRaisesRegex(RuntimeError, "simulated"):
                run_development_cv(
                    task,
                    design,
                    "glm",
                    interrupted,
                    provenance={"identity": "stable"},
                    checkpoint_dir=checkpoints,
                )
            self.assertEqual(len(list(checkpoints.rglob("candidate_*.json"))), 7)
            resumed_calls = 0

            def resumed(family, config, *args):
                nonlocal resumed_calls
                resumed_calls += 1
                return loss(config)

            resumed_receipt = run_development_cv(
                task,
                design,
                "glm",
                resumed,
                provenance={"identity": "stable"},
                checkpoint_dir=checkpoints,
                resume=True,
            )
            self.assertEqual(resumed_calls, 18)
            self.assertEqual(resumed_receipt, clean)

    def test_independent_filtered_shards_reduce_to_monolithic_receipt(self) -> None:
        task, design = self._synthetic_case()
        provenance = {"identity": "sharded"}

        def evaluator(family, config, train, validation, fit_seed, prediction_seed):
            return float(config["alpha"]), {
                "queue_family": family,
                "fit_seed": int(fit_seed),
                "prediction_seed": int(prediction_seed),
                "validation_examples": int(len(validation)),
            }

        monolithic = run_development_cv(
            task,
            design,
            "glm",
            evaluator,
            provenance=provenance,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            for fold in range(1, 6):
                for candidate_index in range(5):
                    shard = run_development_cv_shard(
                        task,
                        design,
                        "glm",
                        evaluator,
                        provenance=provenance,
                        checkpoint_dir=checkpoints,
                        folds=[fold],
                        candidate_indices=[candidate_index],
                    )
                    self.assertEqual(shard["queue"], "cpu_tabular")
                    self.assertEqual(shard["family"], "glm")
                    self.assertEqual(shard["completed"], 1)
                    self.assertEqual(shard["resumed"], 0)
                    self.assertFalse(shard["final_receipt_emitted"])
                    self.assertNotIn("receipt_hash", shard)
                    self.assertEqual(len(shard["cells"]), 1)

            def must_not_fit(*args, **kwargs):
                raise AssertionError("resume unexpectedly refit a development cell")

            resumed = run_development_cv_shard(
                task,
                design,
                "glm",
                must_not_fit,
                provenance=provenance,
                checkpoint_dir=checkpoints,
                folds=[1],
                candidate_indices=[0],
                resume=True,
            )
            self.assertEqual(resumed["completed"], 0)
            self.assertEqual(resumed["resumed"], 1)
            reduced = reduce_development_checkpoints(
                task,
                design,
                "glm",
                provenance=provenance,
                checkpoint_dir=checkpoints,
            )
            self.assertEqual(reduced, monolithic)
            for row in reduced["fold_scores"]:
                self.assertEqual(row["queue"], "cpu_tabular")
                self.assertEqual(row["family"], "glm")

    def test_shard_filters_are_fail_closed_and_never_finalize_partial_grid(self) -> None:
        task, design = self._synthetic_case()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            calls = 0

            def evaluator(family, config, *args):
                nonlocal calls
                calls += 1
                return float(config["alpha"])

            partial = run_development_cv_shard(
                task,
                design,
                "glm",
                evaluator,
                checkpoint_dir=checkpoints,
                folds=[2],
                candidate_indices=[3],
            )
            self.assertEqual(calls, 1)
            self.assertEqual(partial["folds"], [2])
            self.assertEqual(partial["candidate_indices"], [3])
            self.assertFalse(partial["final_receipt_emitted"])
            with self.assertRaisesRegex(ValueError, "incomplete or noncanonical"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    checkpoint_dir=checkpoints,
                )

            for keyword, message in (
                ({"folds": [1, 1]}, "duplicates"),
                ({"candidate_indices": [0, 0]}, "duplicates"),
                ({"folds": [0]}, "outside the frozen grid"),
                ({"candidate_indices": [5]}, "outside the frozen grid"),
            ):
                with self.subTest(keyword=keyword):
                    with self.assertRaisesRegex(ValueError, message):
                        run_development_cv_shard(
                            task,
                            design,
                            "glm",
                            evaluator,
                            checkpoint_dir=checkpoints,
                            **keyword,
                        )
            with self.assertRaisesRegex(ValueError, "requires at least one"):
                run_development_cv_shard(
                    task,
                    design,
                    "glm",
                    evaluator,
                    checkpoint_dir=checkpoints,
                )
            self.assertEqual(calls, 1)

    def test_reducer_rejects_missing_duplicate_tamper_and_rehashed_identity(self) -> None:
        task, design = self._synthetic_case()
        provenance = {"identity": "stable"}

        def populate(checkpoints: Path) -> None:
            run_development_cv(
                task,
                design,
                "glm",
                lambda family, config, *args: float(config["alpha"]),
                provenance=provenance,
                checkpoint_dir=checkpoints,
            )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "missing"
            populate(checkpoints)
            (checkpoints / "fold_05" / "candidate_04.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing=.*candidate_04"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "duplicate"
            populate(checkpoints)
            source = checkpoints / "fold_01" / "candidate_00.json"
            duplicate = checkpoints / "fold_01" / "candidate_00.copy.json"
            duplicate.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "unexpected=.*copy"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "symlink"
            populate(checkpoints)
            checkpoint = checkpoints / "fold_01" / "candidate_00.json"
            target = checkpoints / "fold_01" / "candidate_01.json"
            checkpoint.unlink()
            checkpoint.symlink_to(target.name)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "tamper"
            populate(checkpoints)
            checkpoint = checkpoints / "fold_01" / "candidate_00.json"
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            value["row"]["validation_loss"] = 123.0
            checkpoint.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash is invalid"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                )

        for field, changed in (
            ("queue", "gpu_neural"),
            ("family", "lightgbm"),
            ("fit_seed", 123456789),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                checkpoints = Path(temporary) / field
                populate(checkpoints)
                checkpoint = checkpoints / "fold_01" / "candidate_00.json"
                value = json.loads(checkpoint.read_text(encoding="utf-8"))
                value["row"][field] = changed
                unsigned = {
                    key: item
                    for key, item in value.items()
                    if key != "checkpoint_hash"
                }
                value["checkpoint_hash"] = hashlib.sha256(
                    json.dumps(
                        unsigned,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                checkpoint.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity drifted"):
                    reduce_development_checkpoints(
                        task,
                        design,
                        "glm",
                        provenance=provenance,
                        checkpoint_dir=checkpoints,
                    )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "provenance"
            populate(checkpoints)
            with self.assertRaisesRegex(ValueError, "provenance drifted"):
                reduce_development_checkpoints(
                    task,
                    design,
                    "glm",
                    provenance={"identity": "different"},
                    checkpoint_dir=checkpoints,
                )

    def test_checkpoint_rejects_rehashed_extra_fields_and_provenance_drift(self) -> None:
        task, design = self._synthetic_case()
        provenance = {"identity": "stable"}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            run_development_cv(
                task,
                design,
                "glm",
                lambda family, config, *args: float(config["alpha"]),
                provenance=provenance,
                checkpoint_dir=checkpoints,
            )
            checkpoint = checkpoints / "fold_01" / "candidate_00.json"
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            value["row"]["unexpected"] = "poison"
            unsigned = {
                key: item for key, item in value.items() if key != "checkpoint_hash"
            }
            value["checkpoint_hash"] = hashlib.sha256(
                json.dumps(
                    unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            checkpoint.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row is invalid"):
                run_development_cv(
                    task,
                    design,
                    "glm",
                    lambda *args: 0.1,
                    provenance=provenance,
                    checkpoint_dir=checkpoints,
                    resume=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            run_development_cv(
                task,
                design,
                "glm",
                lambda family, config, *args: float(config["alpha"]),
                provenance=provenance,
                checkpoint_dir=checkpoints,
            )
            with self.assertRaisesRegex(ValueError, "provenance drifted"):
                run_development_cv(
                    task,
                    design,
                    "glm",
                    lambda *args: 0.1,
                    provenance={"identity": "different"},
                    checkpoint_dir=checkpoints,
                    resume=True,
                )

    def test_nonfinite_audit_is_rejected_before_checkpoint(self) -> None:
        task, design = self._synthetic_case()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary) / "checkpoints"
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                run_development_cv(
                    task,
                    design,
                    "glm",
                    lambda *args: (0.1, {"bad": float("nan")}),
                    checkpoint_dir=checkpoints,
                )
            self.assertEqual(list(checkpoints.rglob("*.json")), [])

    def test_resume_requires_checkpoint_directory(self) -> None:
        task, design = self._synthetic_case()
        with self.assertRaisesRegex(ValueError, "resume requires"):
            run_development_cv(
                task,
                design,
                "glm",
                lambda *args: 0.1,
                resume=True,
            )


if __name__ == "__main__":
    unittest.main()
