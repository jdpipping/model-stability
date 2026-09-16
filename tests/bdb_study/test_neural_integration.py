from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bdb_study.adapters.common import PreparedTask
from bdb_study.models import NeuralFit, build_cnn, grouped_validation_indices
from bdb_study.runner import (
    TaskRuntime,
    _binary_result,
    _distribution_result,
    _fit_and_predict,
    _training_target,
    _trajectory_result,
)


CHANNELS = (
    "x_rel",
    "y_rel",
    "vx",
    "vy",
    "speed",
    "acceleration",
    "dir_sin",
    "dir_cos",
    "orientation_sin",
    "orientation_cos",
    "offense",
    "defense",
    "football",
    "focal",
)


def _tracking(n_examples: int, *, seed: int = 19) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tokens = np.zeros((n_examples, 2, 3, len(CHANNELS)), dtype=np.float32)
    tokens[..., :10] = rng.normal(size=(n_examples, 2, 3, 10))
    tokens[:, :, 0, CHANNELS.index("offense")] = 1.0
    tokens[:, :, 1, CHANNELS.index("defense")] = 1.0
    tokens[:, :, 2, CHANNELS.index("football")] = 1.0
    tokens[:, :, 0, CHANNELS.index("focal")] = 1.0
    return (
        tokens,
        np.ones(tokens.shape[:3], dtype=bool),
        np.ones(tokens.shape[:2], dtype=bool),
    )


def _binary_task(n_games: int = 12) -> PreparedTask:
    tokens, player_mask, frame_mask = _tracking(n_games)
    games = np.arange(100, 100 + n_games, dtype=np.int64)
    strata = np.where(np.arange(n_games) < n_games // 2, "2018", "2019")
    labels = np.arange(n_games, dtype=int) % 2
    examples = pd.DataFrame(
        {
            "example_id": [f"example-{game}" for game in games],
            "game_id": games,
            "stratum": strata,
            "target": labels,
        }
    )
    tabular = pd.DataFrame(
        {
            "numeric_context": np.arange(n_games, dtype=float),
            "categorical_context": [f"formation-{index}" for index in range(n_games)],
        }
    )
    return PreparedTask(
        task_id="synthetic_binary",
        outcome_type="binary",
        primary_metric="brier",
        examples=examples,
        tabular=tabular,
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CHANNELS,
        y=labels,
    )


class NeuralIntegrationTests(unittest.TestCase):
    def test_binary_game_equal_diagnostics_use_games_as_equal_units(self) -> None:
        tokens, player_mask, frame_mask = _tracking(10)
        games = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 4], dtype=np.int64)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1], dtype=int)
        task = PreparedTask(
            task_id="synthetic_binary",
            outcome_type="binary",
            primary_metric="brier",
            examples=pd.DataFrame(
                {
                    "example_id": [f"binary-{index}" for index in range(10)],
                    "game_id": games,
                    "stratum": ["season"] * 10,
                    "target": labels,
                }
            ),
            tabular=pd.DataFrame({"x": np.arange(10)}),
            player_tokens=tokens,
            player_mask=player_mask,
            frame_mask=frame_mask,
            channel_names=CHANNELS,
            y=labels,
        )
        test_probability = np.array([0.1, 0.2, 0.7, 0.9])
        metrics, predictions, arrays = _binary_result(
            task,
            np.array([0, 1]),
            np.array([2, 3, 4, 5]),
            np.array([6, 7, 8, 9]),
            np.array([0.2, 0.8, 0.3, 0.7]),
            test_probability,
            uncertainty_seed=321,
        )
        contribution = -(
            labels[6:] * np.log(test_probability)
            + (1 - labels[6:]) * np.log(1 - test_probability)
        )
        expected = (float(np.mean(contribution[:3])) + float(contribution[3])) / 2
        self.assertAlmostEqual(metrics["game_equal_raw_log_loss"], expected)
        for name in (
            "game_equal_calibrated_log_loss",
            "game_equal_calibration_bias",
            "game_equal_rms_reliability",
            "game_equal_mean_entropy",
            "game_equal_mean_va_imprecision",
            "game_equal_p90_va_imprecision",
            "game_equal_label0_set_coverage",
            "game_equal_label1_set_coverage",
            "game_equal_set_singleton_rate",
            "game_equal_set_doubleton_rate",
            "game_equal_set_empty_rate",
        ):
            self.assertTrue(np.isfinite(metrics[name]), name)
        self.assertEqual(metrics["uncertainty_subsample_seed"], 321)
        np.testing.assert_array_equal(arrays["uncertainty_subsample_seed"], [321])
        self.assertEqual(
            int((predictions["partition"] == "test").sum()), len(test_probability)
        )

    def test_validation_is_game_grouped_stratified_and_seed_stable(self) -> None:
        games = np.repeat(np.arange(20), 2)
        strata = np.repeat(np.array(["2018"] * 12 + ["2019"] * 8), 2)
        fit_a, validation_a = grouped_validation_indices(games, 91, strata=strata)
        fit_b, validation_b = grouped_validation_indices(games, 91, strata=strata)
        np.testing.assert_array_equal(fit_a, fit_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        fit_games = set(games[fit_a])
        validation_games = set(games[validation_a])
        self.assertFalse(fit_games & validation_games)
        self.assertEqual(len(validation_games), 4)
        self.assertEqual(
            {stratum: len(set(games[validation_a][strata[validation_a] == stratum])) for stratum in ("2018", "2019")},
            {"2018": 2, "2019": 2},
        )
        with self.assertRaisesRegex(ValueError, "fraction"):
            grouped_validation_indices(games, 91, fraction=1.0, strata=strata)

    def test_selector_and_refit_preprocessors_have_distinct_fit_scopes(self) -> None:
        task = _binary_task()
        all_index = np.arange(len(task.examples))
        local_fit, local_validation = grouped_validation_indices(
            task.game_ids, 213, strata=task.strata
        )
        # Make the held-out selector games visibly influential. They must not
        # change selector statistics, but must enter the all-n refit statistics.
        task.tabular.loc[local_validation, "numeric_context"] = 10_000.0
        task.player_tokens[
            local_validation, :, :, CHANNELS.index("x_rel")
        ] = 50.0
        runtime = TaskRuntime(task, queue="gpu_neural")
        selector, selector_audit = runtime.neural_inputs(
            "transformer",
            local_fit,
            local_validation,
            fit_scope="epoch_selector_training_games_only",
        )
        final, final_audit = runtime.neural_inputs(
            "transformer",
            all_index,
            fit_scope="all_selected_training_games",
        )
        self.assertEqual(
            selector[0].input_shapes["global_context"],
            final[0].input_shapes["global_context"],
        )
        self.assertEqual(
            selector_audit["fit_scope"], "epoch_selector_training_games_only"
        )
        self.assertEqual(final_audit["fit_scope"], "all_selected_training_games")
        self.assertNotEqual(
            selector_audit["global_context"]["numeric_means"],
            final_audit["global_context"]["numeric_means"],
        )
        self.assertNotEqual(selector_audit["means"], final_audit["means"])

        cnn, cnn_audit = runtime.neural_inputs(
            "cnn",
            local_fit,
            local_validation,
            fit_scope="epoch_selector_training_games_only",
        )
        self.assertFalse(hasattr(cnn[1], "raster"))
        batch = cnn[1].batch([0])
        self.assertEqual(batch["raster"].shape, (1, 2, 60, 27, 8))
        self.assertEqual(cnn_audit["representation"], "two_yard_spatial_raster")

    def test_runner_passes_exact_scopes_and_records_actual_neural_seeds(self) -> None:
        task = _binary_task()
        runtime = TaskRuntime(task, queue="gpu_neural")
        train_index = np.arange(10)
        prediction_index = np.arange(10, 12)
        local_fit, local_validation = grouped_validation_indices(
            task.game_ids[train_index], 701, strata=task.strata[train_index]
        )
        captured: dict[str, object] = {}

        def fake_fit(
            family,
            selector_train_inputs,
            selector_train_y,
            selector_validation_inputs,
            selector_validation_y,
            final_inputs,
            final_y,
            **kwargs,
        ):
            captured.update(
                {
                    "family": family,
                    "selector_train": selector_train_inputs.example_indices.copy(),
                    "selector_validation": selector_validation_inputs.example_indices.copy(),
                    "final": final_inputs.example_indices.copy(),
                    "selection_seed": kwargs["selection_seed"],
                    "refit_seed": kwargs["refit_seed"],
                }
            )
            return NeuralFit(
                family=family,
                model=object(),
                best_epoch=1,
                selector_history={"loss": [0.3], "val_loss": [0.4]},
                refit_history={"loss": [0.2]},
                parameter_count=123,
                validation_game_ids=kwargs["validation_game_ids"],
                validation_split_seed=kwargs["validation_split_seed"],
            )

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))
        )
        with (
            patch("bdb_study.runner.fit_neural_explicit_preprocessing", side_effect=fake_fit),
            patch(
                "bdb_study.runner.predict_neural",
                side_effect=lambda model, inputs, batch_size: np.full(len(inputs), 0.25),
            ),
            patch.dict(sys.modules, {"tensorflow": fake_tensorflow}),
        ):
            predictions, history = _fit_and_predict(
                runtime,
                "transformer",
                {"batch_size": 4},
                train_index,
                (prediction_index,),
                fit_seed=601,
                validation_split_seed=701,
                selection_seed=801,
                refit_seed=901,
            )

        np.testing.assert_array_equal(
            captured["selector_train"], train_index[local_fit]
        )
        np.testing.assert_array_equal(
            captured["selector_validation"], train_index[local_validation]
        )
        np.testing.assert_array_equal(captured["final"], train_index)
        self.assertEqual(captured["selection_seed"], 801)
        self.assertEqual(captured["refit_seed"], 901)
        self.assertEqual(history["validation_split_seed"], 701)
        self.assertEqual(
            history["training_seeds"],
            {"epoch_selection": 801, "refit": 901, "validation_split": 701},
        )
        self.assertEqual(
            history["selector_preprocessing"]["fit_scope"],
            "epoch_selector_training_games_only",
        )
        self.assertEqual(
            history["preprocessing"]["fit_scope"], "all_selected_training_games"
        )
        self.assertEqual(history["selector_fit_games"], 8)
        self.assertEqual(history["final_fit_games"], 10)
        np.testing.assert_array_equal(predictions[0], np.full(2, 0.25))

    def test_runner_records_equal_relational_input_and_head_loss_signatures(self) -> None:
        task = _binary_task()
        runtime = TaskRuntime(task, queue="gpu_neural")
        train_index = np.arange(10)
        prediction_index = np.arange(10, 12)

        def fake_fit(family, *args, **kwargs):
            return NeuralFit(
                family=family,
                model=object(),
                best_epoch=1,
                selector_history={"loss": [0.3], "val_loss": [0.4]},
                refit_history={"loss": [0.2]},
                parameter_count=123,
                validation_game_ids=kwargs["validation_game_ids"],
                validation_split_seed=kwargs["validation_split_seed"],
                fairness_metadata=(
                    {"representative_forward_flops": 456}
                    if family in {"relnet", "attn_relnet"}
                    else {}
                ),
                complexity_metadata=(
                    {
                        "representative_forward_flops": 789,
                        "architecture_id": "bdb_global_set_transformer_v1",
                    }
                    if family == "set_transformer"
                    else {}
                ),
            )

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))
        )
        histories = []
        with (
            patch("bdb_study.runner.fit_neural_explicit_preprocessing", side_effect=fake_fit),
            patch(
                "bdb_study.runner.predict_neural",
                side_effect=lambda model, inputs, batch_size: np.full(len(inputs), 0.25),
            ),
            patch.dict(sys.modules, {"tensorflow": fake_tensorflow}),
        ):
            for family in ("relnet", "attn_relnet", "set_transformer"):
                _, history = _fit_and_predict(
                    runtime,
                    family,
                    {"batch_size": 4, "d_model": 48, "edge_dim": 8, "edge_hidden": 64},
                    train_index,
                    (prediction_index,),
                    fit_seed=601,
                    validation_split_seed=701,
                    selection_seed=801,
                    refit_seed=901,
                )
                histories.append(history)
        self.assertEqual(
            histories[0]["representative_neural_input"],
            histories[1]["representative_neural_input"],
        )
        self.assertNotEqual(
            histories[0]["representative_neural_input"],
            histories[2]["representative_neural_input"],
        )
        self.assertEqual(
            histories[0]["representative_shared_neural_input"],
            histories[1]["representative_shared_neural_input"],
        )
        self.assertEqual(
            histories[0]["representative_shared_neural_input"],
            histories[2]["representative_shared_neural_input"],
        )
        self.assertEqual(
            histories[0]["output_head_loss_signature"],
            histories[1]["output_head_loss_signature"],
        )
        self.assertEqual(
            histories[0]["output_head_loss_signature"],
            histories[2]["output_head_loss_signature"],
        )
        self.assertEqual(
            histories[2]["global_set_architecture"]["architecture_id"],
            "bdb_global_set_transformer_v1",
        )
        self.assertEqual(
            histories[0]["output_head_loss_signature"]["loss"],
            "mean_binary_crossentropy",
        )

    def test_runner_absolute_trajectory_binds_effective_target_and_head_signature(self) -> None:
        n_examples, horizon = 8, 3
        tokens, player_mask, frame_mask = _tracking(n_examples)
        games = np.arange(n_examples, dtype=np.int64)
        baseline = np.full((n_examples, horizon, 2), 10.0, dtype=np.float32)
        absolute = baseline + np.arange(
            n_examples * horizon * 2, dtype=np.float32
        ).reshape(n_examples, horizon, 2) / 100.0
        task = PreparedTask(
            task_id="bdb2026_trajectory",
            outcome_type="trajectory",
            primary_metric="rmse",
            examples=pd.DataFrame(
                {
                    "example_id": [f"path-{game}" for game in games],
                    "game_id": games,
                    "stratum": ["2023"] * n_examples,
                    "target": [np.nan] * n_examples,
                }
            ),
            tabular=pd.DataFrame({"context": np.arange(n_examples, dtype=float)}),
            player_tokens=tokens,
            player_mask=player_mask,
            frame_mask=frame_mask,
            channel_names=CHANNELS,
            target_values=absolute,
            target_mask=np.ones((n_examples, horizon), dtype=bool),
            target_baseline=baseline,
        )
        runtime = TaskRuntime(task, queue="gpu_neural")
        train_index = np.arange(6)
        prediction_index = np.arange(6, 8)
        captured: dict[str, object] = {}

        def fake_fit(family, *args, **kwargs):
            captured["final_y"] = np.asarray(args[5]).copy()
            captured["config"] = dict(kwargs["config"])
            return NeuralFit(
                family=family,
                model=object(),
                best_epoch=1,
                selector_history={"loss": [0.3], "val_loss": [0.4]},
                refit_history={"loss": [0.2]},
                parameter_count=123,
                validation_game_ids=kwargs["validation_game_ids"],
                validation_split_seed=kwargs["validation_split_seed"],
                fairness_metadata={"representative_forward_flops": 456},
            )

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda: None))
        )
        with (
            patch(
                "bdb_study.runner.fit_neural_explicit_preprocessing",
                side_effect=fake_fit,
            ),
            patch(
                "bdb_study.runner.predict_neural",
                side_effect=lambda model, inputs, batch_size: np.full(
                    (len(inputs), horizon, 2), 12.0, dtype=np.float32
                ),
            ),
            patch.dict(sys.modules, {"tensorflow": fake_tensorflow}),
        ):
            predictions, history = _fit_and_predict(
                runtime,
                "relnet",
                {
                    "batch_size": 4,
                    "d_model": 48,
                    "edge_dim": 8,
                    "edge_hidden": 64,
                    "trajectory_target": "absolute",
                },
                train_index,
                (prediction_index,),
                fit_seed=601,
                validation_split_seed=701,
                selection_seed=801,
                refit_seed=901,
            )

        self.assertEqual(captured["config"]["trajectory_target"], "absolute")
        np.testing.assert_array_equal(captured["final_y"], absolute[train_index])
        self.assertEqual(
            history["effective_model_config"]["trajectory_target"], "absolute"
        )
        self.assertEqual(
            history["output_head_loss_signature"]["training_target"],
            "masked_absolute_xy",
        )
        np.testing.assert_array_equal(
            predictions[0],
            np.full((len(prediction_index), horizon, 2), 2.0, dtype=np.float32),
        )

    def test_trajectory_padding_is_masked_and_scientific_arrays_keep_float64(self) -> None:
        tokens, player_mask, frame_mask = _tracking(6)
        games = np.arange(6, dtype=np.int64)
        mask = np.array(
            [[True, True, False], [True, False, False]] * 3, dtype=bool
        )
        baseline = np.zeros((6, 3, 2), dtype=np.float32)
        target = np.arange(36, dtype=np.float32).reshape(6, 3, 2) / 10.0
        target[~mask] = np.nan
        task = PreparedTask(
            task_id="synthetic_trajectory",
            outcome_type="trajectory",
            primary_metric="rmse",
            examples=pd.DataFrame(
                {
                    "example_id": [f"trajectory-{game}" for game in games],
                    "game_id": games,
                    "stratum": ["2023"] * 6,
                    "target": [np.nan] * 6,
                }
            ),
            tabular=pd.DataFrame({"context": np.arange(6, dtype=float)}),
            player_tokens=tokens,
            player_mask=player_mask,
            frame_mask=frame_mask,
            channel_names=CHANNELS,
            target_values=target,
            target_mask=mask,
            target_baseline=baseline,
        )
        residual, training_mask = _training_target(task, np.arange(6))
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertTrue(np.all(residual[~training_mask] == 0.0))

        calibration_index = np.array([0, 1])
        test_index = np.array([2, 3, 4, 5])
        calibration_residual = np.zeros((2, 3, 2), dtype=np.float64)
        test_residual = np.full((4, 3, 2), 0.123456789123, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "frozen development horizon scale"):
            _trajectory_result(
                task,
                calibration_index,
                test_index,
                calibration_residual,
                test_residual,
                horizon_scale=None,
                uncertainty_seed=12345,
            )
        metrics, _, arrays = _trajectory_result(
            task,
            calibration_index,
            test_index,
            calibration_residual,
            test_residual,
            horizon_scale=np.ones(3, dtype=float),
            uncertainty_seed=12345,
        )
        self.assertTrue(np.isfinite(metrics["official_pooled_rmse"]))
        self.assertEqual(arrays["test_prediction"].dtype, np.float64)
        self.assertEqual(arrays["test_predicted_residual"].dtype, np.float64)
        self.assertEqual(arrays["horizon_rmse"].dtype, np.float64)

    def test_distribution_artifacts_preserve_metric_precision(self) -> None:
        tokens, player_mask, frame_mask = _tracking(6)
        labels = np.array([0, 1, 1, 2, 0, 2], dtype=int)
        task = PreparedTask(
            task_id="synthetic_distribution",
            outcome_type="distribution",
            primary_metric="crps",
            examples=pd.DataFrame(
                {
                    "example_id": [f"distribution-{index}" for index in range(6)],
                    "game_id": np.arange(6, dtype=np.int64),
                    "stratum": ["2019"] * 6,
                    "target": labels,
                }
            ),
            tabular=pd.DataFrame({"context": np.arange(6, dtype=float)}),
            player_tokens=tokens,
            player_mask=player_mask,
            frame_mask=frame_mask,
            channel_names=CHANNELS,
            y=labels,
            support=(0.0, 1.0, 2.0),
        )
        calibration = np.array(
            [[0.123456789123, 0.333333333333, 0.543209877544],
             [0.400000000001, 0.299999999999, 0.3]],
            dtype=np.float64,
        )
        test = np.array(
            [[0.234567891234, 0.345678912345, 0.419753196421],
             [0.111111111111, 0.222222222222, 0.666666666667]],
            dtype=np.float64,
        )
        _, _, arrays = _distribution_result(
            task,
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4, 5]),
            calibration,
            test,
            uncertainty_seed=12345,
        )
        self.assertEqual(arrays["calibration_probabilities"].dtype, np.float64)
        self.assertEqual(arrays["test_probabilities"].dtype, np.float64)
        np.testing.assert_array_equal(arrays["calibration_probabilities"], calibration)
        np.testing.assert_array_equal(arrays["test_probabilities"], test)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") is not None,
        "TensorFlow is optional in CPU-only test environments",
    )
    def test_cnn_padding_mask_makes_leading_padding_invariant(self) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(314)
        model = build_cnn(
            (None, 6, 4, 8),
            outcome_type="binary",
            n_outputs=1,
            max_horizon=1,
            dropout=0.0,
        )
        rng = np.random.default_rng(2718)
        valid = rng.normal(size=(1, 2, 6, 4, 8)).astype(np.float32)
        padded = np.concatenate(
            [np.zeros((1, 3, 6, 4, 8), dtype=np.float32), valid], axis=1
        )
        short_prediction = model.predict(
            {"raster": valid, "frame_mask": np.ones((1, 2), dtype=bool)},
            verbose=0,
        )
        padded_prediction = model.predict(
            {
                "raster": padded,
                "frame_mask": np.asarray([[False, False, False, True, True]]),
            },
            verbose=0,
        )
        np.testing.assert_allclose(short_prediction, padded_prediction, atol=1e-7)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") is not None,
        "TensorFlow is optional in CPU-only test environments",
    )
    def test_one_epoch_lazy_cnn_and_global_set_transformer_smoke(self) -> None:
        task = _binary_task(n_games=10)
        runtime = TaskRuntime(task, queue="gpu_neural")
        config = {
            "learning_rate": 1e-3,
            "dropout": 0.1,
            "max_epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "deterministic": True,
            "d_model": 8,
            "heads": 2,
            "ff_dim": 16,
            "layers": 2,
        }
        for family in ("cnn", "set_transformer"):
            with self.subTest(family=family):
                predictions, history = _fit_and_predict(
                    runtime,
                    family,
                    config,
                    np.arange(8),
                    (np.arange(8, 10),),
                    fit_seed=11,
                    validation_split_seed=12,
                    selection_seed=13,
                    refit_seed=14,
                )
                self.assertEqual(predictions[0].shape, (2,))
                self.assertTrue(np.all(np.isfinite(predictions[0])))
                self.assertEqual(history["best_epoch"], 1)
                self.assertEqual(history["selector_fit_games"], 6)
                self.assertEqual(history["final_fit_games"], 8)
                self.assertGreater(history["parameter_count"], 0)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") is not None,
        "TensorFlow is optional in CPU-only test environments",
    )
    def test_same_seed_lazy_primary_neural_fits_are_byte_exact(self) -> None:
        """Exercise the exact lazy selector/refit path that attempt 7 exposed."""

        task = _binary_task(n_games=10)
        runtime = TaskRuntime(task, queue="gpu_neural")
        config = {
            "learning_rate": 1e-3,
            "dropout": 0.3,
            "max_epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "deterministic": True,
            "d_model": 8,
            "edge_dim": 4,
            "edge_hidden": 8,
            "heads": 2,
            "ff_dim": 16,
            "layers": 2,
        }
        arguments = {
            "train_index": np.arange(8),
            "prediction_indices": (np.arange(8, 10),),
            "fit_seed": 101,
            "validation_split_seed": 102,
            "selection_seed": 103,
            "refit_seed": 104,
        }
        for family in ("relnet", "attn_relnet", "set_transformer"):
            with self.subTest(family=family):
                first_predictions, first_history = _fit_and_predict(
                    runtime, family, config, **arguments
                )
                second_predictions, second_history = _fit_and_predict(
                    runtime, family, config, **arguments
                )
                np.testing.assert_array_equal(
                    first_predictions[0], second_predictions[0]
                )
                self.assertEqual(
                    first_history["selector_history"],
                    second_history["selector_history"],
                )
                self.assertEqual(
                    first_history["refit_history"],
                    second_history["refit_history"],
                )
                self.assertEqual(
                    first_history["training_seeds"],
                    second_history["training_seeds"],
                )


if __name__ == "__main__":
    unittest.main()
