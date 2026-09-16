from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from bdb_study.models import (
    PUNT_CDF_RESIDUAL_CONTRACT,
    bounded_pava,
    build_relational_network,
    build_transformer,
    classical_parameter_count,
    fit_glm,
    fit_lightgbm,
    relational_fairness_metadata,
    transformer_complexity_metadata,
    validate_relational_pair_fairness,
    neural_head_loss_signature,
    _keras_batch_dataset,
    _make_neural_model,
    _neural_loss,
)
from bdb_study.representations import LazyNeuralInputs, MaskedTokenScaler


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.x = rng.normal(size=(40, 5)).astype(np.float32)
        self.y_binary = (self.x[:, 0] > 0).astype(int)

    def test_import_does_not_initialize_tensorflow(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys,bdb_study.models; print(int('tensorflow' in sys.modules))"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0")

    def test_glm_binary_and_distribution_predictions(self) -> None:
        binary = fit_glm(
            self.x,
            self.y_binary,
            outcome_type="binary",
            config={"alpha": 0.1, "epochs": 2, "batch_size": 8},
            seed=7,
        )
        probability = binary.predict(self.x)
        self.assertEqual(probability.shape, (40,))
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))
        self.assertEqual(
            classical_parameter_count(binary),
            binary.models[0].coef_.size + binary.models[0].intercept_.size,
        )

        y_distribution = np.arange(40) % 4
        distribution = fit_glm(
            self.x,
            y_distribution,
            outcome_type="distribution",
            n_outputs=4,
            config={"alpha": 0.1, "epochs": 2, "batch_size": 8},
            seed=8,
        )
        predicted = distribution.predict(self.x)
        self.assertEqual(predicted.shape, (40, 4))
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0)

    def test_lightgbm_is_single_threaded(self) -> None:
        fitted = fit_lightgbm(
            self.x,
            self.y_binary,
            outcome_type="binary",
            config={"n_estimators": 3, "min_child_samples": 2},
            seed=9,
        )
        self.assertEqual(fitted.models[0].get_params()["n_jobs"], 1)
        self.assertEqual(fitted.predict(self.x).shape, (40,))
        tree_count = int(fitted.models[0].booster_.num_trees())
        self.assertGreaterEqual(classical_parameter_count(fitted), tree_count)
        self.assertNotEqual(
            classical_parameter_count(fitted),
            tree_count,
            "a tree count must not be mislabeled as a parameter count",
        )

    def test_punt_cdf_residual_is_full_valid_pmf_with_one_model(self) -> None:
        labels = np.arange(len(self.x)) % 5
        config = {"alpha": 0.1, "output_contract": PUNT_CDF_RESIDUAL_CONTRACT}
        linear = fit_glm(
            self.x, labels, outcome_type="distribution", n_outputs=5,
            config=config, seed=17,
        )
        self.assertEqual(len(linear.models), 1)
        probability = linear.predict(self.x)
        self.assertEqual(probability.shape, (len(self.x), 5))
        self.assertTrue(np.all(probability >= 0.0))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)

        boosted = fit_lightgbm(
            self.x, labels, outcome_type="distribution", n_outputs=5,
            config={
                "output_contract": PUNT_CDF_RESIDUAL_CONTRACT,
                "n_estimators": 3,
                "min_child_samples": 2,
            },
            seed=18,
        )
        self.assertEqual(len(boosted.models), 1)
        self.assertTrue(boosted.config["threshold_query"])
        self.assertEqual(boosted.models[0].booster_.num_trees(), 3)
        np.testing.assert_allclose(boosted.predict(self.x).sum(axis=1), 1.0)
        projected = bounded_pava(np.array([[0.8, 0.2, 1.4, -0.1]]))
        self.assertTrue(np.all(np.diff(projected, axis=1) >= 0.0))
        self.assertTrue(np.all((projected >= 0.0) & (projected <= 1.0)))

    def test_trajectory_tabular_fit_is_shared_across_horizons(self) -> None:
        target = np.zeros((len(self.x), 7, 2), dtype=np.float32)
        target[..., 0] = self.x[:, :1] + np.arange(7)[None, :] / 10
        mask = np.ones(target.shape[:2], dtype=bool)
        fitted = fit_glm(
            self.x, target, outcome_type="trajectory", config={"alpha": 0.1},
            seed=19, target_mask=mask,
        )
        self.assertEqual(len(fitted.models), 2)
        self.assertEqual(fitted.predict(self.x).shape, target.shape)

    def test_matched_relational_architectures_have_exact_parameter_match(self) -> None:
        common = dict(
            outcome_type="binary", n_outputs=1, max_horizon=1, dropout=0.3,
            context_dim=5, output_config={"binary_loss": "log_loss"},
        )
        relnet = build_relational_network((None, 6, 24), aggregation="fixed", **common)
        attention = build_relational_network((None, 6, 24), aggregation="attention", **common)
        self.assertEqual(relnet.count_params(), attention.count_params())
        shapes = {"player_tokens": (20, 6, 24), "global_context": (5,)}
        first = relational_fairness_metadata("relnet", shapes, {}, relnet.count_params())
        second = relational_fairness_metadata("attn_relnet", shapes, {}, attention.count_params())
        receipt = validate_relational_pair_fairness(first, second)
        self.assertEqual(receipt["parameter_gap_fraction"], 0.0)
        self.assertLess(receipt["flop_gap_fraction"], 0.15)

    def test_variable_history_sources_train_all_primary_neural_families(self) -> None:
        """Keras must not freeze T=6 before a later batch needs T=20."""

        import tensorflow as tf

        examples, time_steps, players = 129, 20, 4
        names = (
            "x_rel", "y_rel", "vx", "vy", "speed", "acceleration",
            "offense", "defense", "football", "focal", "position_qb",
            "position_wr", "position_te", "position_rb", "position_ol",
            "position_dl", "position_lb", "position_db",
        )
        tokens = np.zeros(
            (examples, time_steps, players, len(names)), dtype=np.float32
        )
        frame_mask = np.zeros((examples, time_steps), dtype=bool)
        # With batch_size=64, the first two Keras batches see only six-frame
        # examples. The final batch contains the first twenty-frame example.
        frame_mask[:128, -6:] = True
        frame_mask[128, :] = True
        player_mask = np.repeat(frame_mask[:, :, None], players, axis=2)
        tokens[:, :, 0, names.index("offense")] = 1.0
        tokens[:, :, 0, names.index("focal")] = 1.0
        tokens[:, :, 0, names.index("position_qb")] = 1.0
        tokens[:, :, 1, names.index("offense")] = 1.0
        tokens[:, :, 1, names.index("position_ol")] = 1.0
        tokens[:, :, 2, names.index("defense")] = 1.0
        tokens[:, :, 2, names.index("position_dl")] = 1.0
        tokens[:, :, 3, names.index("offense")] = 1.0
        tokens[:, :, 3, names.index("position_wr")] = 1.0
        scaler = MaskedTokenScaler(names).fit(tokens, player_mask)
        context = np.zeros((examples, 1), dtype=np.float32)
        target = (np.arange(examples) % 2).astype(np.float32)[:, None]

        for family in ("relnet", "attn_relnet", "set_transformer"):
            with self.subTest(family=family):
                source = LazyNeuralInputs(
                    family,
                    tokens,
                    player_mask,
                    frame_mask,
                    context,
                    np.arange(examples),
                    names,
                    token_scaler=scaler,
                    task_id="bdb2023_sack",
                )
                widths = [
                    source.batch(np.arange(start, stop))["player_tokens"].shape[1]
                    for start, stop in ((0, 64), (64, 128), (128, 129))
                ]
                self.assertEqual(widths, [20, 20, 20])
                config = {
                    "dropout": 0.0,
                    "d_model": 8,
                    "edge_dim": 4,
                    "edge_hidden": 8,
                    "heads": 2,
                    "ff_dim": 16,
                    "layers": 2,
                    "binary_loss": "log_loss",
                }
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(20260821)
                model = _make_neural_model(
                    family,
                    source,
                    outcome_type="binary",
                    n_outputs=1,
                    max_horizon=1,
                    config=config,
                )
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=_neural_loss("binary", config),
                    jit_compile=False,
                )
                history = model.fit(
                    _keras_batch_dataset(
                        source,
                        targets=target,
                        batch_size=64,
                        shuffle=False,
                        seed=20260821,
                    ),
                    epochs=1,
                    verbose=0,
                ).history
                self.assertTrue(np.all(np.isfinite(history["loss"])))

    def test_global_set_transformer_has_temporal_input_and_frozen_identity(self) -> None:
        model = build_transformer(
            (None, 6, 24),
            outcome_type="binary",
            n_outputs=1,
            max_horizon=1,
            dropout=0.3,
            context_dim=5,
            output_config={"binary_loss": "log_loss"},
        )
        input_names = {tensor.name.split(":", 1)[0] for tensor in model.inputs}
        self.assertEqual(
            input_names,
            {
                "player_tokens",
                "player_mask",
                "frame_mask",
                "time_to_event",
                "global_context",
            },
        )
        self.assertNotIn("edge_type", input_names)
        metadata = transformer_complexity_metadata(
            "set_transformer",
            {
                "player_tokens": (20, 6, 24),
                "global_context": (5,),
            },
            {"d_model": 64, "heads": 2, "ff_dim": 256, "layers": 3},
            model.count_params(),
        )
        self.assertEqual(
            metadata["architecture_id"], "bdb_global_set_transformer_v1"
        )
        self.assertIs(metadata["typed_graph_edges_consumed"], False)
        self.assertLessEqual(metadata["parameter_count"], 350_000)
        self.assertGreater(metadata["representative_forward_flops"], 0)

    def test_global_set_transformer_is_player_permutation_and_padding_invariant_but_uses_time(self) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(20260821)
        model = build_transformer(
            (None, 4, 6),
            outcome_type="binary",
            n_outputs=1,
            max_horizon=1,
            dropout=0.0,
            d_model=16,
            heads=2,
            ff_dim=32,
            layers_count=2,
        )
        rng = np.random.default_rng(12)
        tokens = rng.normal(size=(1, 3, 4, 6)).astype(np.float32)
        player_mask = np.ones((1, 3, 4), dtype=bool)
        frame_mask = np.ones((1, 3), dtype=bool)
        time_to_event = np.asarray([[[-0.2], [-0.1], [0.0]]], dtype=np.float32)

        def predict(
            local_tokens: np.ndarray,
            local_player_mask: np.ndarray,
            local_frame_mask: np.ndarray,
            local_time: np.ndarray,
        ) -> np.ndarray:
            return np.asarray(
                model(
                    {
                        "player_tokens": local_tokens,
                        "player_mask": local_player_mask,
                        "frame_mask": local_frame_mask,
                        "time_to_event": local_time,
                    },
                    training=False,
                )
            )

        baseline = predict(tokens, player_mask, frame_mask, time_to_event)
        permutation = np.asarray([2, 0, 3, 1])
        permuted = predict(
            tokens[:, :, permutation],
            player_mask[:, :, permutation],
            frame_mask,
            time_to_event,
        )
        np.testing.assert_allclose(baseline, permuted, atol=1e-6, rtol=0.0)

        padded_tokens = np.pad(tokens, ((0, 0), (2, 0), (0, 0), (0, 0)))
        padded_player_mask = np.pad(
            player_mask, ((0, 0), (2, 0), (0, 0)), constant_values=False
        )
        padded_frame_mask = np.asarray([[False, False, True, True, True]])
        padded_time = np.asarray(
            [[[0.0], [0.0], [-0.2], [-0.1], [0.0]]], dtype=np.float32
        )
        padded = predict(
            padded_tokens, padded_player_mask, padded_frame_mask, padded_time
        )
        np.testing.assert_allclose(baseline, padded, atol=1e-6, rtol=0.0)

        no_time_order = predict(
            tokens, player_mask, frame_mask, np.zeros_like(time_to_event)
        )
        self.assertGreater(float(np.max(np.abs(baseline - no_time_order))), 1e-7)

    def test_v2_binary_log_loss_does_not_change_legacy_brier(self) -> None:
        truth = np.array([[1.0], [0.0]], dtype=np.float32)
        prediction = np.array([[0.8], [0.2]], dtype=np.float32)
        legacy = float(_neural_loss("binary")(truth, prediction).numpy())
        matched = float(
            _neural_loss("binary", {"binary_loss": "log_loss"})(truth, prediction).numpy()
        )
        self.assertAlmostEqual(legacy, 0.04, places=6)
        self.assertAlmostEqual(matched, -np.log(0.8), places=6)

        relnet = neural_head_loss_signature(
            "binary", n_outputs=1, max_horizon=1,
            config={"binary_loss": "log_loss"},
        )
        attention = neural_head_loss_signature(
            "binary", n_outputs=1, max_horizon=1,
            config={"binary_loss": "log_loss"},
        )
        self.assertEqual(relnet, attention)
        legacy_signature = neural_head_loss_signature(
            "binary", n_outputs=1, max_horizon=1, config={}
        )
        self.assertNotEqual(
            relnet["signature_sha256"], legacy_signature["signature_sha256"]
        )

    def test_head_loss_signature_binds_ordered_baseline_and_trajectory_target(self) -> None:
        ordered = neural_head_loss_signature(
            "distribution",
            n_outputs=4,
            max_horizon=1,
            config={
                "output_contract": PUNT_CDF_RESIDUAL_CONTRACT,
                "cdf_baseline": [0.2, 0.5, 0.8, 1.0],
            },
        )
        changed = neural_head_loss_signature(
            "distribution",
            n_outputs=4,
            max_horizon=1,
            config={
                "output_contract": PUNT_CDF_RESIDUAL_CONTRACT,
                "cdf_baseline": [0.1, 0.5, 0.8, 1.0],
            },
        )
        self.assertNotEqual(ordered["signature_sha256"], changed["signature_sha256"])
        residual = neural_head_loss_signature(
            "trajectory", n_outputs=1, max_horizon=94,
            config={"trajectory_target": "residual"},
        )
        absolute = neural_head_loss_signature(
            "trajectory", n_outputs=1, max_horizon=94,
            config={"trajectory_target": "absolute"},
        )
        self.assertNotEqual(residual["signature_sha256"], absolute["signature_sha256"])


if __name__ == "__main__":
    unittest.main()
