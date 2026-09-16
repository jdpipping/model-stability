from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bdb_study.adapters.common import PreparedTask
from bdb_study.aggregate import (
    ScientificValidationError,
    _paired_secondary_effects,
    _reject_unexpected_cell_directories,
    _load_sensitivity_primary_reference,
    _secondary_model_pairwise,
    _secondary_summary,
    _validate_hashed_signature,
    _validate_matched_neural_history_pairs,
    aggregate_task_run,
    validate_cell_scientifically,
)
from bdb_study.analysis import DEFAULT_MODELS
from bdb_study.metrics import skill_score
from bdb_study.models import (
    CLASSICAL_PARAMETER_COUNT_DEFINITION,
    NEURAL_PARAMETER_COUNT_DEFINITION,
    PUNT_CDF_RESIDUAL_CONTRACT,
)
from bdb_study.runner import (
    NEURAL_FINAL_PREPROCESSING_SCOPE,
    NEURAL_SELECTOR_PREPROCESSING_SCOPE,
    _binary_result,
    _distribution_result,
    _trajectory_result,
)
from bdb_study.storage import CellKey
from bdb_study.storage import atomic_write_csv


KEY = CellKey("fixed_main", 1, 1, "model")
SEEDS = {
    "fit": 11,
    "epoch_selection": 12,
    "validation": 13,
    "refit": 14,
    "prediction": 15,
    "validation_split": 16,
    "uncertainty_subsample": 17,
}
CONFIG = {"alpha": 1.0}


def prepared_task(
    outcome: str,
    targets: np.ndarray,
    game_ids: np.ndarray,
    *,
    support: tuple[float, ...] | None = None,
    target_values: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    target_baseline: np.ndarray | None = None,
) -> PreparedTask:
    n = len(game_ids)
    examples = pd.DataFrame(
        {
            "example_id": [f"example-{index:02d}" for index in range(n)],
            "game_id": game_ids,
            "stratum": ["2020w01"] * n,
            "target": targets,
        }
    )
    return PreparedTask(
        task_id=f"synthetic_{outcome}",
        outcome_type=outcome,
        primary_metric={"binary": "brier", "distribution": "crps", "trajectory": "rmse"}[outcome],
        examples=examples,
        tabular=pd.DataFrame({"feature": np.arange(n, dtype=float)}),
        player_tokens=np.zeros((n, 1, 1, 1), dtype=np.float32),
        player_mask=np.ones((n, 1, 1), dtype=bool),
        frame_mask=np.ones((n, 1), dtype=bool),
        channel_names=("x_rel",),
        y=None if outcome == "trajectory" else np.asarray(
            [int(np.flatnonzero(np.asarray(support) == value)[0]) for value in targets]
            if outcome == "distribution"
            else targets,
            dtype=int,
        ),
        support=support,
        target_values=target_values,
        target_mask=target_mask,
        target_baseline=target_baseline,
    )


class FakeStorage:
    def __init__(
        self,
        task: PreparedTask,
        metrics: dict,
        predictions: pd.DataFrame,
        history: dict,
        arrays: dict,
        artifacts: dict,
        run_dir: Path | None = None,
    ) -> None:
        split = {
            "repeat": 1,
            "nested_train_game_ids": {"1": ["1"]},
            "calibration_game_ids": ["2"],
            "test_game_ids": ["3"],
            "outer_split_hash": "outer",
            "nested_split_hash": "nested",
        }
        cell = {
            "branch": KEY.branch,
            "ablation_id": None,
            "sensitivity_id": None,
            "repeat": KEY.repeat,
            "n_train": KEY.n_train,
            "model": KEY.model,
            "queue": "cpu_tabular",
            "outer_split_hash": "outer",
            "nested_split_hash": "nested",
            "seeds": dict(SEEDS),
        }
        self.manifest = {
            "task_spec": {
                "models": {
                    KEY.model: {
                        "family": "glm",
                        "selected_config": dict(CONFIG),
                    }
                },
                "artifacts": artifacts,
            },
            "task_design": {
                "split_manifests": [split],
                "required_cells": [cell],
            },
        }
        self.metrics = metrics
        self.predictions = predictions
        self.history = history
        self.arrays = arrays
        self.run_dir = run_dir or Path(".")

    def load_metrics(self, key):
        self._check(key)
        return self.metrics

    def load_predictions(self, key):
        self._check(key)
        return self.predictions

    def load_history(self, key):
        self._check(key)
        return self.history

    def load_arrays(self, key):
        self._check(key)
        return self.arrays or None

    @staticmethod
    def _check(key):
        if key != KEY:
            raise AssertionError("unexpected test key")


class FakeAggregateStorage:
    def __init__(self, run_dir: Path, manifest: dict) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.manifest_hash = "f" * 64
        self.finalized = False

    @staticmethod
    def validate_cell(_key):
        return SimpleNamespace(is_complete=True)

    def finalize_run(self, _keys, *, final_artifacts):
        self.finalized = True
        if not final_artifacts:
            raise AssertionError("final artifacts were not supplied")


def hashed_signature(schema_version: str, **payload: object) -> dict[str, object]:
    unsigned = {"schema_version": schema_version, **payload}
    digest = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**unsigned, "signature_sha256": digest}


class FakeNeuralPairStorage:
    def __init__(self, histories: dict[str, dict[str, object]]) -> None:
        models = {
            "RelNet": {"family": "relnet"},
            "AttnRelNet": {"family": "attn_relnet"},
        }
        if "SetTransformer" in histories:
            models["SetTransformer"] = {"family": "set_transformer"}
        self.manifest = {
            "task_spec": {
                "models": models
            }
        }
        self.histories = histories

    def load_history(self, key: CellKey) -> dict[str, object]:
        return self.histories[key.model]


def bound_artifacts(
    task: PreparedTask,
    metrics: dict,
    predictions: pd.DataFrame,
    arrays: dict,
    declared: dict,
) -> FakeStorage:
    effective_config = dict(CONFIG)
    if task.outcome_type == "distribution":
        effective_config["output_contract"] = PUNT_CDF_RESIDUAL_CONTRACT
    if task.outcome_type == "trajectory":
        effective_config.update(
            {
                "trajectory_target": "residual",
                "trajectory_decoder": "shared_horizon_conditioned_v1",
            }
        )
    metrics.update(
        {
            "task_id": task.task_id,
            "branch": KEY.branch,
            "repeat": KEY.repeat,
            "n_train": KEY.n_train,
            "model": KEY.model,
            "family": "glm",
            "skill": skill_score(metrics["primary_loss"], metrics["null_loss"]),
            "n_train_examples": int((task.examples["game_id"] == 1).sum()),
            "n_calibration_examples": int((task.examples["game_id"] == 2).sum()),
            "n_test_examples": int((task.examples["game_id"] == 3).sum()),
            "outer_split_hash": "outer",
            "nested_split_hash": "nested",
            "model_config": dict(CONFIG),
            "seeds": dict(SEEDS),
            "uncertainty_subsample_seed": SEEDS["uncertainty_subsample"],
        }
    )
    history = {
        "family": "glm",
        "branch": KEY.branch,
        "ablation_id": None,
        "sensitivity_id": None,
        "parameter_count": 3,
        "parameter_count_definition": CLASSICAL_PARAMETER_COUNT_DEFINITION,
        "preprocessing": {
            "fit_scope": "selected_training_games_only",
            "team_identity": "excluded_primary",
        },
        "effective_model_config": effective_config,
        "model_config": dict(CONFIG),
        "seeds": dict(SEEDS),
        "uncertainty_subsample_seed": SEEDS["uncertainty_subsample"],
        "train_game_ids": ["1"],
        "calibration_game_ids": ["2"],
        "test_game_ids": ["3"],
    }
    return FakeStorage(task, metrics, predictions, history, arrays, declared)


class AggregateScientificValidationTests(unittest.TestCase):
    def test_hashed_neural_signature_rejects_payload_tampering(self) -> None:
        signature = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[{"name": "player_tokens", "shape": [4, 20, 23, 25]}],
        )
        _validate_hashed_signature(
            signature,
            schema_version="bdb-representative-neural-input-v1",
            label="test input signature",
        )
        tampered = deepcopy(signature)
        tampered["tensors"][0]["shape"][-1] = 24
        with self.assertRaisesRegex(ScientificValidationError, "does not match"):
            _validate_hashed_signature(
                tampered,
                schema_version="bdb-representative-neural-input-v1",
                label="test input signature",
            )

    def test_saved_neural_pair_must_match_input_head_and_complexity(self) -> None:
        input_signature = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[{"name": "player_tokens", "sha256": "a" * 64}],
        )
        head_signature = hashed_signature(
            "bdb-neural-head-loss-v1",
            head={"type": "dense_sigmoid", "shape": [1]},
            loss="mean_binary_crossentropy",
        )
        common = {
            "representative_neural_input": input_signature,
            "output_head_loss_signature": head_signature,
            "parameter_count": 100_000,
            "representative_forward_flops": 200_000,
        }
        histories = {
            "RelNet": deepcopy(common),
            "AttnRelNet": {
                **deepcopy(common),
                "parameter_count": 104_000,
                "representative_forward_flops": 225_000,
            },
        }
        keys = [
            CellKey("fixed_main", 1, 20, "RelNet"),
            CellKey("fixed_main", 1, 20, "AttnRelNet"),
        ]
        storage = FakeNeuralPairStorage(histories)
        self.assertEqual(_validate_matched_neural_history_pairs(storage, keys), 1)

        histories["AttnRelNet"]["representative_neural_input"] = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[{"name": "player_tokens", "sha256": "b" * 64}],
        )
        with self.assertRaisesRegex(ScientificValidationError, "input differ"):
            _validate_matched_neural_history_pairs(storage, keys)

    def test_saved_set_transformer_must_share_inputs_and_head_with_relational_models(self) -> None:
        relational_input = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[
                {"name": "edge_type", "sha256": "e" * 64},
                {"name": "player_tokens", "sha256": "a" * 64},
            ],
        )
        shared_input = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[{"name": "player_tokens", "sha256": "a" * 64}],
            scope="shared_tokens_history_masks_time_context_after_split_local_preprocessing",
        )
        head_signature = hashed_signature(
            "bdb-neural-head-loss-v1",
            head={"type": "dense_sigmoid", "shape": [1]},
            loss="mean_binary_crossentropy",
        )
        relational_common = {
            "representative_neural_input": relational_input,
            "representative_shared_neural_input": shared_input,
            "output_head_loss_signature": head_signature,
            "parameter_count": 100_000,
            "representative_forward_flops": 200_000,
        }
        histories = {
            "RelNet": deepcopy(relational_common),
            "AttnRelNet": {
                **deepcopy(relational_common),
                "parameter_count": 104_000,
                "representative_forward_flops": 225_000,
            },
            "SetTransformer": {
                # Its full input receipt intentionally excludes typed edges;
                # the shared receipt and head/loss receipt must still match.
                "representative_neural_input": shared_input,
                "representative_shared_neural_input": shared_input,
                "output_head_loss_signature": head_signature,
                "parameter_count": 120_000,
                "representative_forward_flops": 260_000,
                "global_set_architecture": {
                    "architecture_id": "bdb_global_set_transformer_v1",
                    "inputs": "task_tokens_player_frame_masks_time_context",
                    "attention_contract": "global_set_time_attention_v1",
                    "relation_scope": "global_masked_all_player_self_attention",
                    "typed_graph_edges_consumed": False,
                    "temporal_encoder": "factorized_masked_temporal_attention",
                    "protocol_note": (
                        "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot"
                    ),
                    "parameter_count": 120_000,
                    "parameter_cap": 350_000,
                    "representative_forward_flops": 260_000,
                },
            },
        }
        keys = [
            CellKey("fixed_main", 1, 20, "RelNet"),
            CellKey("fixed_main", 1, 20, "AttnRelNet"),
            CellKey("fixed_main", 1, 20, "SetTransformer"),
        ]
        storage = FakeNeuralPairStorage(histories)
        self.assertEqual(_validate_matched_neural_history_pairs(storage, keys), 1)
        with self.assertRaisesRegex(
            ScientificValidationError, "expected .*set_transformer"
        ):
            _validate_matched_neural_history_pairs(storage, keys[:2])

        histories["SetTransformer"]["representative_shared_neural_input"] = (
            hashed_signature(
                "bdb-representative-neural-input-v1",
                tensors=[{"name": "player_tokens", "sha256": "b" * 64}],
            )
        )
        with self.assertRaisesRegex(
            ScientificValidationError, "shared neural .* differs"
        ):
            _validate_matched_neural_history_pairs(storage, keys)

    def test_sensitivity_primary_reference_is_checksum_pinned_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "primary"
            rows = []
            for repeat in range(1, 101):
                for anchor in (10, 20, 40, 60, 100, 140):
                    for model in (
                        "linear_structure", "boosted_structure", "relnet",
                        "attn_relnet", "set_transformer",
                    ):
                        rows.append(
                            {
                                "task_id": "bdb2023_sack",
                                "branch": "fixed_main",
                                "repeat": repeat,
                                "n_train": anchor,
                                "model": model,
                                "outer_split_hash": f"outer-{repeat}",
                                "nested_split_hash": f"nested-{repeat}",
                            }
                        )
            record = atomic_write_csv(
                run_dir / "final" / "primary_metrics.csv", pd.DataFrame(rows)
            )
            manifest = {
                "task_id": "bdb2023_sack",
                "primary_reference": {
                    "run_dir": "primary",
                    "primary_metrics_sha256": record.sha256,
                },
            }
            loaded = _load_sensitivity_primary_reference(
                manifest, repo_root=root
            )
            self.assertEqual(len(loaded), 3000)
            (run_dir / "final" / "primary_metrics.csv").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaises(ScientificValidationError):
                _load_sensitivity_primary_reference(manifest, repo_root=root)

    def test_binary_recomputes_raw_calibrated_null_skill_and_exact_identity_order(self):
        games = np.asarray([1, 1, 2, 2, 2, 2, 3, 3])
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        task = prepared_task("binary", y, games)
        metrics, predictions, arrays = _binary_result(
            task,
            np.asarray([0, 1]),
            np.asarray([2, 3, 4, 5]),
            np.asarray([6, 7]),
            np.asarray([0.1, 0.8, 0.3, 0.7]),
            np.asarray([0.2, 0.9]),
            uncertainty_seed=SEEDS["uncertainty_subsample"],
        )
        storage = bound_artifacts(
            task,
            metrics,
            predictions,
            arrays,
            {
                "metrics": ["raw_brier", "game_equal_brier", "raw_log_loss", "skill"],
                "predictions": [
                    "raw_probability", "venn_abers_probability", "venn_abers_p0", "venn_abers_p1"
                ],
            },
        )
        validated = validate_cell_scientifically(storage, KEY, task)
        self.assertAlmostEqual(validated["raw_brier"], 0.025)

        reordered = deepcopy(storage)
        order = [1, 0, 2, 3, 4, 5]
        reordered.predictions = reordered.predictions.iloc[order].reset_index(drop=True)
        with self.assertRaisesRegex(ScientificValidationError, "IDs/order"):
            validate_cell_scientifically(reordered, KEY, task)

        wrong_seed = deepcopy(storage)
        wrong_seed.metrics["seeds"]["fit"] += 1
        with self.assertRaisesRegex(ScientificValidationError, "metric seeds"):
            validate_cell_scientifically(wrong_seed, KEY, task)

        wrong_parameter_definition = deepcopy(storage)
        wrong_parameter_definition.history["parameter_count_definition"] = "number of trees"
        with self.assertRaisesRegex(ScientificValidationError, "parameter_count definition"):
            validate_cell_scientifically(wrong_parameter_definition, KEY, task)

        wrong_va = deepcopy(storage)
        wrong_va.predictions.loc[
            wrong_va.predictions["partition"].eq("test"), "venn_abers_p0"
        ] = 0.0
        with self.assertRaisesRegex(ScientificValidationError, "venn_abers_p0"):
            validate_cell_scientifically(wrong_va, KEY, task)

        wrong_set = deepcopy(storage)
        test_rows = wrong_set.predictions["partition"].eq("test")
        wrong_set.predictions.loc[test_rows, "set_includes_0"] = False
        with self.assertRaisesRegex(ScientificValidationError, "label-0 conformal"):
            validate_cell_scientifically(wrong_set, KEY, task)

        wrong_uncertainty_seed = deepcopy(storage)
        wrong_uncertainty_seed.arrays["uncertainty_subsample_seed"][0] += 1
        with self.assertRaisesRegex(
            ScientificValidationError, "uncertainty-subsample seed artifact"
        ):
            validate_cell_scientifically(wrong_uncertainty_seed, KEY, task)

        wrong_identity_policy = deepcopy(storage)
        wrong_identity_policy.history["preprocessing"]["team_identity"] = (
            "included_sensitivity"
        )
        with self.assertRaisesRegex(ScientificValidationError, "team-identity policy"):
            validate_cell_scientifically(wrong_identity_policy, KEY, task)

    def test_set_transformer_cell_replays_runner_effective_config_and_neural_history(self):
        games = np.asarray([1, 1, 2, 2, 2, 2, 3, 3])
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        task = prepared_task("binary", y, games)
        metrics, predictions, arrays = _binary_result(
            task,
            np.asarray([0, 1]),
            np.asarray([2, 3, 4, 5]),
            np.asarray([6, 7]),
            np.asarray([0.1, 0.8, 0.3, 0.7]),
            np.asarray([0.2, 0.9]),
            uncertainty_seed=SEEDS["uncertainty_subsample"],
        )
        storage = bound_artifacts(
            task,
            metrics,
            predictions,
            arrays,
            {
                "metrics": ["raw_brier", "game_equal_brier", "raw_log_loss", "skill"],
                "predictions": [
                    "raw_probability",
                    "venn_abers_probability",
                    "venn_abers_p0",
                    "venn_abers_p1",
                ],
            },
        )
        signature = hashed_signature(
            "bdb-representative-neural-input-v1",
            tensors=[{"name": "player_tokens", "sha256": "a" * 64}],
        )
        head = hashed_signature(
            "bdb-neural-head-loss-v1",
            head={"type": "dense_sigmoid", "shape": [1]},
            loss="mean_binary_crossentropy",
        )
        storage.manifest["task_spec"]["models"][KEY.model].update(
            {"family": "set_transformer", "selected_config": dict(CONFIG)}
        )
        storage.manifest["task_design"]["required_cells"][0]["queue"] = "gpu_neural"
        storage.metrics["family"] = "set_transformer"
        storage.history.update(
            {
                "family": "set_transformer",
                "parameter_count": 120_000,
                "parameter_count_definition": NEURAL_PARAMETER_COUNT_DEFINITION,
                "preprocessing": {
                    "fit_scope": NEURAL_FINAL_PREPROCESSING_SCOPE,
                    "team_identity": "excluded_primary",
                    "ablation_id": None,
                    "ablation_mask_applied_to_scaler": False,
                },
                "selector_preprocessing": {
                    "fit_scope": NEURAL_SELECTOR_PREPROCESSING_SCOPE,
                    "team_identity": "excluded_primary",
                    "ablation_id": None,
                    "ablation_mask_applied_to_scaler": False,
                },
                "effective_model_config": {
                    **CONFIG,
                    "binary_loss": "log_loss",
                    "representative_time_steps": 1,
                },
                "validation_split_seed": SEEDS["validation_split"],
                "validation_game_ids": ["1"],
                "selector_fit_games": 1,
                "final_fit_games": 1,
                "final_fit_examples": 2,
                "best_epoch": 1,
                "selector_history": {"val_loss": [0.25]},
                "refit_history": {"loss": [0.3]},
                "representative_neural_input": signature,
                "representative_shared_neural_input": signature,
                "output_head_loss_signature": head,
            }
        )
        with patch(
            "bdb_study.models.grouped_validation_indices",
            return_value=(np.asarray([0]), np.asarray([1])),
        ):
            validated = validate_cell_scientifically(storage, KEY, task)
        self.assertEqual(validated["family"], "set_transformer")

    def test_distribution_recomputes_probabilities_null_crps_and_conformal_outputs(self):
        games = np.asarray([1, 1, 1, 2, 2, 2, 2, 3, 3])
        target = np.asarray([-1, 0, 1, -1, 0, 1, 0, -1, 1])
        task = prepared_task("distribution", target, games, support=(-1.0, 0.0, 1.0))
        calibration_probability = np.asarray(
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]]
        )
        test_probability = np.asarray([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
        metrics, predictions, arrays = _distribution_result(
            task,
            np.asarray([0, 1, 2]),
            np.asarray([3, 4, 5, 6]),
            np.asarray([7, 8]),
            calibration_probability,
            test_probability,
            uncertainty_seed=SEEDS["uncertainty_subsample"],
        )
        storage = bound_artifacts(
            task,
            metrics,
            predictions,
            arrays,
            {
                "metrics": ["raw_crps", "game_equal_crps", "skill", "coverage", "interval_width"],
                "predictions": ["class_probabilities", "interval_lower", "interval_upper", "per_example_crps"],
            },
        )
        validate_cell_scientifically(storage, KEY, task)
        corrupted = deepcopy(storage)
        corrupted.arrays["null_probability"] = np.asarray([1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ScientificValidationError, "null distribution"):
            validate_cell_scientifically(corrupted, KEY, task)

        wrong_registry = deepcopy(storage)
        wrong_registry.arrays["conformal_selected_calibration_indices"][0] += 1
        with self.assertRaisesRegex(ScientificValidationError, "calibration registry"):
            validate_cell_scientifically(wrong_registry, KEY, task)

    def test_trajectory_recomputes_reconstruction_pooled_game_horizon_null_and_skill(self):
        games = np.asarray([1, 2, 3, 3])
        target = np.full(4, np.nan)
        mask = np.asarray([[True, True], [True, False], [True, True], [True, False]])
        values = np.full((4, 2, 2), np.nan, dtype=np.float32)
        values[mask] = np.asarray(
            [[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
        )
        baseline = np.zeros_like(values)
        baseline[~mask] = 0.0
        task = prepared_task(
            "trajectory",
            target,
            games,
            target_values=values,
            target_mask=mask,
            target_baseline=baseline,
        )
        calibration_residual = np.zeros((1, 2, 2), dtype=float)
        test_residual = np.zeros((2, 2, 2), dtype=float)
        test_residual[0, 0] = [0.5, 1.0]
        test_residual[0, 1] = [1.0, 1.5]
        test_residual[1, 0] = [0.5, 0.5]
        metrics, predictions, arrays = _trajectory_result(
            task,
            np.asarray([1]),
            np.asarray([2, 3]),
            calibration_residual,
            test_residual,
            horizon_scale=np.ones(2, dtype=float),
            uncertainty_seed=SEEDS["uncertainty_subsample"],
        )
        storage = bound_artifacts(
            task,
            metrics,
            predictions,
            arrays,
            {
                "metrics": ["official_pooled_rmse", "game_equal_rmse", "horizon_rmse", "skill"],
                "predictions": [
                    "constant_velocity_baseline", "predicted_residual", "absolute_xy", "target_mask"
                ],
            },
        )
        validate_cell_scientifically(storage, KEY, task)
        corrupted = deepcopy(storage)
        corrupted.arrays["horizon_rmse"][0] += 0.1
        with self.assertRaisesRegex(ScientificValidationError, "horizon RMSE"):
            validate_cell_scientifically(corrupted, KEY, task)

        wrong_tube = deepcopy(storage)
        wrong_tube.arrays["path_tube_radius"][0] += 1_000.0
        with self.assertRaisesRegex(ScientificValidationError, "tube radius"):
            validate_cell_scientifically(wrong_tube, KEY, task)

    def test_declared_schema_and_unexpected_cell_directory_are_rejected(self):
        games = np.asarray([1, 1, 2, 2, 2, 2, 3, 3])
        y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        task = prepared_task("binary", y, games)
        metrics, predictions, arrays = _binary_result(
            task,
            np.asarray([0, 1]),
            np.asarray([2, 3, 4, 5]),
            np.asarray([6, 7]),
            np.asarray([0.1, 0.8, 0.3, 0.7]),
            np.asarray([0.2, 0.9]),
            uncertainty_seed=SEEDS["uncertainty_subsample"],
        )
        storage = bound_artifacts(
            task,
            metrics,
            predictions,
            arrays,
            {"metrics": ["not_saved"], "predictions": ["raw_probability"]},
        )
        with self.assertRaisesRegex(ScientificValidationError, "declared metric"):
            validate_cell_scientifically(storage, KEY, task)

        with tempfile.TemporaryDirectory() as temporary:
            storage.run_dir = Path(temporary)
            unexpected = storage.run_dir / "cells" / "fixed_main" / "repeat_001" / "n_0001" / "other"
            unexpected.mkdir(parents=True)
            with self.assertRaisesRegex(ScientificValidationError, "unexpected cell"):
                _reject_unexpected_cell_directories(storage, [KEY])

    def test_final_aggregate_integrates_distribution_bootstrap_conclusions(self):
        anchors = (10, 20, 40, 60, 100, 140)
        rows = []
        for repeat in range(1, 4):
            for model_index, model in enumerate(DEFAULT_MODELS):
                for anchor in anchors:
                    primary = (
                        0.20
                        + 0.01 * model_index
                        - 0.0001 * anchor
                        + (0.0002 + 0.0001 * model_index) * repeat
                    )
                    null = 0.30 + 0.0002 * repeat
                    rows.append(
                        {
                            "task_id": "synthetic_distribution",
                            "branch": "fixed_main",
                            "repeat": repeat,
                            "model": model,
                            "n_train": anchor,
                            "primary_loss": primary,
                            "game_equal_loss": primary + 0.001,
                            "null_loss": null,
                            "skill": 1.0 - primary / null,
                            "coverage": 0.95 + 0.001 * repeat,
                            "interval_width": (
                                8.0 + 2.0 * model_index + 0.01 * repeat
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        manifest = {
            "task_id": "synthetic_distribution",
            "task_spec": {
                "outcome_type": "distribution",
                "models": {model: {} for model in DEFAULT_MODELS},
            },
            "task_design": {
                "anchors": list(anchors),
                "repeats": 3,
                "cell_counts": {
                    "primary": len(frame),
                    "structural_ablation": 0,
                    "frozen_sensitivity": 0,
                    "required": len(frame),
                },
            },
            "analysis": {"bootstrap_seed": 413, "bootstrap_draws": 30},
            "prepared": {"path": "prepared/synthetic_distribution"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            storage = FakeAggregateStorage(Path(temporary), manifest)
            keys = list(range(len(frame)))
            with (
                patch("bdb_study.aggregate.open_run", return_value=storage),
                patch("bdb_study.aggregate.validate_task_design"),
                patch("bdb_study.aggregate.cell_keys_from_design", return_value=keys),
                patch("bdb_study.aggregate._reject_unexpected_cell_directories"),
                patch(
                    "bdb_study.aggregate._validate_matched_neural_history_pairs",
                    return_value=0,
                ),
                patch("bdb_study.aggregate.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.aggregate.validate_cell_scientifically",
                    side_effect=lambda _storage, key, _task: frame.iloc[key].to_dict(),
                ),
            ):
                receipt = aggregate_task_run(
                    temporary,
                    repo_root=temporary,
                    require_complete=True,
                )
            expected = {
                "bca_sd_intervals",
                "bca_log_sd_ratios",
                "supported_best",
                "interval_width_contrasts",
                "coverage_lower_bounds",
                "controlled_sharpness",
            }
            self.assertTrue(expected.issubset(receipt["artifacts"]))
            self.assertEqual(receipt["bootstrap_draws"], 30)
            self.assertEqual(
                receipt["distributional_inference"]["coverage_target"], 0.90
            )
            self.assertEqual(
                receipt["distributional_inference"]["coverage_family_members"], 24
            )
            self.assertTrue(storage.finalized)
            for name in expected:
                self.assertTrue((Path(temporary) / "final" / f"{name}.csv").is_file())

    def test_pilot_aggregate_is_300_cell_descriptive_only_receipt(self):
        anchors = (10, 20, 40, 60, 100, 140)
        models = (
            "linear_structure", "boosted_structure", "relnet",
            "attn_relnet", "set_transformer",
        )
        rows = []
        for repeat in range(1, 11):
            for model_index, model in enumerate(models):
                for anchor in anchors:
                    loss = 0.2 + 0.01 * model_index + 0.0001 * repeat
                    rows.append(
                        {
                            "task_id": "synthetic_distribution",
                            "branch": "fixed_main",
                            "repeat": repeat,
                            "model": model,
                            "n_train": anchor,
                            "primary_loss": loss,
                            "game_equal_loss": loss + 0.001,
                            "null_loss": 0.3,
                            "coverage": 0.9 + repeat / 10_000,
                            "interval_width": 8.0 + model_index,
                        }
                    )
        frame = pd.DataFrame(rows)
        manifest = {
            "task_id": "synthetic_distribution",
            "task_spec": {
                "outcome_type": "distribution",
                "models": {model: {} for model in models},
            },
            "task_design": {
                "profile": "pilot10",
                "anchors": list(anchors),
                "repeats": 10,
                "cell_counts": {
                    "primary": 300,
                    "structural_ablation": 0,
                    "frozen_sensitivity": 0,
                    "required": 300,
                },
            },
            "execution": {
                "profile": "pilot10",
                "mode": "pilot",
                "evidence_status": "exploratory_provisional",
            },
            "analysis": {"bootstrap_seed": 91, "bootstrap_draws": 0},
            "prepared": {"path": "prepared/synthetic_distribution"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            storage = FakeAggregateStorage(Path(temporary), manifest)
            keys = list(range(len(frame)))
            with (
                patch("bdb_study.aggregate.open_run", return_value=storage),
                patch("bdb_study.aggregate.validate_task_design"),
                patch("bdb_study.aggregate.cell_keys_from_design", return_value=keys),
                patch("bdb_study.aggregate._reject_unexpected_cell_directories"),
                patch(
                    "bdb_study.aggregate._validate_matched_neural_history_pairs",
                    return_value=120,
                ),
                patch("bdb_study.aggregate.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.aggregate.validate_cell_scientifically",
                    side_effect=lambda _storage, key, _task: frame.iloc[key].to_dict(),
                ),
            ):
                receipt = aggregate_task_run(
                    temporary,
                    repo_root=temporary,
                    require_complete=True,
                )
            self.assertEqual(receipt["cells"], 300)
            self.assertEqual(receipt["evidence_status"], "exploratory_provisional")
            self.assertEqual(receipt["bootstrap_draws"], 0)
            self.assertEqual(receipt["max_t_family"], "none")
            self.assertEqual(receipt["multiplicity_scope"], "none_pilot_descriptive")
            forbidden = {
                "paired_contrasts",
                "supported_best",
                "bca_sd_intervals",
                "bca_log_sd_ratios",
                "interval_width_contrasts",
                "coverage_lower_bounds",
                "controlled_sharpness",
            }
            self.assertFalse(forbidden & set(receipt["artifacts"]))
            self.assertTrue(storage.finalized)

    def test_structural_ablation_is_summarized_separately_and_paired_to_main(self):
        primary_rows = []
        ablation_rows = []
        for repeat in range(1, 21):
            for model, offset in (("relnet", 0.0), ("attn_relnet", -0.02)):
                for anchor in (20, 60):
                    base = 0.30 + offset + repeat / 10_000 + anchor / 100_000
                    identity = {
                        "task_id": "synthetic_binary",
                        "repeat": repeat,
                        "model": model,
                        "n_train": anchor,
                        "game_equal_loss": base,
                        "skill": 0.1,
                    }
                    primary_rows.append(
                        {**identity, "branch": "fixed_main", "primary_loss": base}
                    )
                    ablation_rows.append(
                        {
                            **identity,
                            "branch": "structural_ablation",
                            "ablation_id": "remove_edges",
                            "primary_loss": base + 0.05,
                            "game_equal_loss": base + 0.05,
                        }
                    )
        primary = pd.DataFrame(primary_rows)
        ablation = pd.DataFrame(ablation_rows)
        summary = _secondary_summary(ablation, identity_column="ablation_id")
        effects = _paired_secondary_effects(
            primary, ablation, identity_column="ablation_id"
        )
        pairwise = _secondary_model_pairwise(
            ablation, identity_column="ablation_id"
        )
        primary_effects = effects.loc[effects["metric"].eq("primary_loss")]
        self.assertEqual(len(summary.loc[summary["metric"].eq("primary_loss")]), 4)
        self.assertTrue(np.allclose(primary_effects["mean_difference"], 0.05))
        primary_pairwise = pairwise.loc[pairwise["metric"].eq("primary_loss")]
        self.assertTrue(np.allclose(primary_pairwise["mean_difference"], 0.02))
        self.assertTrue(
            primary_pairwise["inferential_family"].eq(
                "secondary_descriptive_only"
            ).all()
        )

    def test_frozen_sensitivity_reports_all_ten_model_pairs(self):
        models = (
            "linear_structure", "boosted_structure", "relnet",
            "attn_relnet", "set_transformer",
        )
        rows = []
        for repeat in range(1, 21):
            for model_index, model in enumerate(models):
                rows.append(
                    {
                        "task_id": "synthetic_binary",
                        "branch": "frozen_sensitivity",
                        "sensitivity_id": "include_identity",
                        "repeat": repeat,
                        "model": model,
                        "n_train": 20,
                        "primary_loss": 0.2 + model_index / 100 + repeat / 10000,
                    }
                )
        pairwise = _secondary_model_pairwise(
            pd.DataFrame(rows),
            identity_column="sensitivity_id",
            expected_models=models,
            all_pairs=True,
        )
        self.assertEqual(len(pairwise), 10)
        self.assertEqual(
            set(zip(pairwise["model_left"], pairwise["model_right"])),
            {
                (left, right)
                for index, left in enumerate(models)
                for right in models[index + 1 :]
            },
        )


if __name__ == "__main__":
    unittest.main()
