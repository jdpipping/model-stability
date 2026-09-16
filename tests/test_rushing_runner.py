"""Tests for the prospective rushing-study fitting and scheduling layers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from rushing_study.data import StudyData, load_study_data
from rushing_study.design import SeedRegistry
from rushing_study.execution import expected_cell_keys
from rushing_study.metrics import evaluate_cell
from rushing_study.models import MODEL_IDS, NeuralFit, fit_ovr_logistic, sensitivity_candidates
from rushing_study.runner import _fit_primary, _fit_sensitivity, _neural_inner_indices


def _register(registry: SeedRegistry, *parts):
    registry.get(*parts)


class RushingRunnerTests(unittest.TestCase):
    def test_locked_logistic_configuration_is_valid_in_pinned_sklearn(self):
        x = np.arange(48, dtype=float).reshape(12, 4)
        y = np.arange(12) % 3
        model = fit_ovr_logistic(
            x,
            y,
            {"alpha": 1 / 3, "eta0": 0.01, "epochs": 1, "batch_size": 4},
            seed=17,
        )
        self.assertEqual(model.predict_proba(x).shape, (12, 80))

    def test_data_loader_drops_augmented_rows_and_attaches_games(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            spatial = np.zeros((3, 11, 10, 10), dtype=np.float32)
            player_set = np.zeros((3, 22, 10), dtype=np.float32)
            np.save(processed / "train_x.npy", spatial)
            np.save(processed / "train_x_set.npy", player_set)
            pd.DataFrame(
                {
                    "PlayId": ["10", "10_aug", "20"],
                    "YardIndexClipped": [99, 99, 100],
                }
            ).to_pickle(processed / "train_y.pkl")
            raw_path = root / "train.csv"
            pd.DataFrame(
                {
                    "GameId": [1, 1, 2, 2],
                    "PlayId": [10, 10, 20, 20],
                    "Season": [2017, 2017, 2018, 2018],
                }
            ).to_csv(raw_path, index=False)
            config = {
                "data": {"processed_dir": str(processed), "raw_train_csv": str(raw_path)},
            }
            loaded = load_study_data(config, mmap_mode=None)

        self.assertEqual(len(loaded.y), 2)
        self.assertEqual(loaded.metadata["play_id"].tolist(), ["10", "20"])
        self.assertFalse(loaded.metadata["play_id"].str.endswith("_aug").any())
        self.assertEqual(loaded.tabular.shape, (2, 40))

    def test_prediction_artifacts_recompute_play_and_game_metrics(self):
        rng = np.random.default_rng(4)
        cal = rng.dirichlet(np.ones(80), size=12)
        test = rng.dirichlet(np.ones(80), size=8)
        cal_y = rng.integers(0, 80, size=12)
        test_y = rng.integers(0, 80, size=8)
        cal_meta = pd.DataFrame(
            {"game_id": [f"c{i // 3}" for i in range(12)], "play_id": range(12), "season": 2017}
        )
        test_meta = pd.DataFrame(
            {"game_id": ["a"] * 2 + ["b"] * 6, "play_id": range(8), "season": 2018}
        )
        result = evaluate_cell(cal, test, cal_y, test_y, cal_meta, test_meta, alpha=0.1, local_k=5)

        self.assertAlmostEqual(result.metrics["crps"], result.test["crps_contribution"].mean())
        self.assertAlmostEqual(result.metrics["coverage"], result.test["covered"].mean())
        self.assertAlmostEqual(result.metrics["mean_width"], result.test["width"].mean())
        game_equal = result.test.groupby("game_id")["crps_contribution"].mean().mean()
        self.assertAlmostEqual(result.metrics["crps_game_equal"], game_equal)
        self.assertEqual(result.arrays["test_proba"].shape, (8, 80))
        self.assertEqual(result.arrays["test_proba"].dtype, np.float64)

    def test_neural_inner_games_partition_all_selected_rows(self):
        metadata = pd.DataFrame(
            {
                "game_id": [str(i) for i in range(1, 6) for _ in range(2)],
                "play_id": [str(i) for i in range(10)],
                "season": [2017] * 10,
            }
        )
        data = StudyData(
            spatial=np.zeros((10, 11, 10, 10), dtype=np.float32),
            player_set=np.zeros((10, 22, 10), dtype=np.float32),
            tabular=np.zeros((10, 40), dtype=np.float32),
            y=np.zeros(10, dtype=int),
            metadata=metadata,
        )
        fit, validation = _neural_inner_indices(
            data,
            np.arange(10),
            {"neural_fit_games": ["1", "2", "3", "4"], "neural_validation_games": ["5"]},
        )
        self.assertEqual(len(fit), 8)
        self.assertEqual(len(validation), 2)
        self.assertFalse(set(fit) & set(validation))

    def test_primary_logistic_uses_every_anchor_row_and_never_tunes(self):
        games = [str(i) for i in range(20)]
        metadata = pd.DataFrame({"game_id": games, "play_id": games, "season": [2017] * 20})
        data = StudyData(
            spatial=np.zeros((20, 11, 10, 10), dtype=np.float32),
            player_set=np.zeros((20, 22, 10), dtype=np.float32),
            tabular=np.zeros((20, 40), dtype=np.float32),
            y=np.arange(20) % 2,
            metadata=metadata,
        )
        config = {
            "models": {"ridge_sgd_l2": {"alpha": 1 / 3, "epochs": 50, "batch_size": 64}},
            "execution": {},
        }
        registry = SeedRegistry()
        _register(registry, "cell", 1, "fit", "ridge_sgd_l2", 20)
        manifest = {
            "config": config,
            "seed_registry": registry.snapshot(),
        }
        split = {"anchors": {"20": {"train_games": games}}}

        class FakeModel:
            coef_ = np.zeros((80, 40))
            intercept_ = np.zeros(80)

        captured = {}

        def fake_fit(x, y, model_config, seed):
            captured["rows"] = len(x)
            captured["seed"] = seed
            return FakeModel()

        with mock.patch("rushing_study.runner.fit_ovr_logistic", side_effect=fake_fit), mock.patch(
            "rushing_study.runner.sensitivity_candidates", side_effect=AssertionError("main tuned")
        ):
            _fit_primary(data, manifest, split, 1, 20, "ridge_sgd_l2")
        self.assertEqual(captured["rows"], 20)

    def test_primary_neural_selects_epoch_by_game_then_refits_on_all_n_games(self):
        games = [str(i) for i in range(20)]
        metadata = pd.DataFrame({"game_id": games, "play_id": games, "season": [2017] * 20})
        data = StudyData(
            spatial=np.zeros((20, 11, 10, 10), dtype=np.float32),
            player_set=np.zeros((20, 22, 10), dtype=np.float32),
            tabular=np.zeros((20, 40), dtype=np.float32),
            y=np.arange(20) % 3,
            metadata=metadata,
        )
        registry = SeedRegistry()
        _register(registry, "cell", 1, "epoch_selection", "zoo_cnn", 20)
        _register(registry, "cell", 1, "refit", "zoo_cnn", 20)
        manifest = {
            "config": {
                "models": {"zoo_cnn": {"frozen": True, "dropout": 0.3, "expected_parameters": 123}},
                "neural_training": {
                    "learning_rate": 1e-3,
                    "batch_size": 64,
                    "max_epochs": 50,
                    "early_stopping_patience": 10,
                },
            },
            "seed_registry": registry.snapshot(),
        }
        split = {
            "anchors": {
                "20": {
                    "train_games": games,
                    "neural_fit_games": games[:16],
                    "neural_validation_games": games[16:],
                }
            }
        }
        captured = {}

        def fake_neural(model_id, x, y, fit_indices, validation_indices, config, selection_seed, refit_seed):
            captured.update(
                n_rows=len(x),
                n_labels=len(y),
                fit_games=set(fit_indices.tolist()),
                validation_games=set(validation_indices.tolist()),
            )
            return NeuralFit(object(), 3, {"val_loss": [2.0, 1.0, 1.5]}, {"loss": [2.0, 1.5, 1.0]}, 123)

        with mock.patch(
            "rushing_study.runner.fit_neural_select_and_refit", side_effect=fake_neural
        ):
            _fit_primary(data, manifest, split, 1, 20, "zoo_cnn")

        self.assertEqual(captured["n_rows"], 20)
        self.assertEqual(captured["n_labels"], 20)
        self.assertEqual(len(captured["fit_games"]), 16)
        self.assertEqual(len(captured["validation_games"]), 4)
        self.assertFalse(captured["fit_games"] & captured["validation_games"])

    def test_sensitivity_fit_and_selection_never_access_calibration_or_test_rows(self):
        games = [str(i) for i in range(23)]
        features = np.zeros((23, 40), dtype=np.float32)
        features[:, 0] = np.arange(23)
        data = StudyData(
            spatial=np.zeros((23, 11, 10, 10), dtype=np.float32),
            player_set=np.zeros((23, 22, 10), dtype=np.float32),
            tabular=features,
            y=np.arange(23) % 4,
            metadata=pd.DataFrame(
                {"game_id": games, "play_id": games, "season": [2017] * 23}
            ),
        )
        registry = SeedRegistry()
        for stage in ("fit", "epoch_selection", "refit", "prediction_rebuild"):
            _register(registry, "cell", 1, stage, "ridge_sgd_l2", 20)
        manifest = {"config": {"models": {}}, "seed_registry": registry.snapshot()}
        split = {
            "anchors": {"20": {"train_games": games[:20]}},
            "tune_games": [games[20]],
            "calibration_games": [games[21]],
            "test_games": [games[22]],
        }

        fit_rows = []
        prediction_rows = []

        class FakeModel:
            coef_ = np.zeros((80, 40))
            intercept_ = np.zeros(80)

        def fake_fit(x, y, config, seed):
            fit_rows.append(set(x[:, 0].astype(int).tolist()))
            return FakeModel()

        def fake_predict(model, x):
            prediction_rows.append(set(x[:, 0].astype(int).tolist()))
            return np.full((len(x), 80), 1 / 80, dtype=float)

        with mock.patch(
            "rushing_study.runner.sensitivity_candidates",
            return_value=[{"alpha": 1 / 3}, {"alpha": 1.0}],
        ), mock.patch("rushing_study.runner.fit_ovr_logistic", side_effect=fake_fit), mock.patch(
            "rushing_study.runner.predict_classical", side_effect=fake_predict
        ):
            _fit_sensitivity(data, manifest, split, 1, 20, "ridge_sgd_l2")

        self.assertTrue(all(rows == set(range(20)) for rows in fit_rows))
        self.assertTrue(all(rows == {20} for rows in prediction_rows))
        self.assertNotIn(21, set().union(*fit_rows, *prediction_rows))
        self.assertNotIn(22, set().union(*fit_rows, *prediction_rows))

    def test_sensitivity_grids_cover_all_four_models_with_main_first(self):
        config = json.loads(Path("configs/rushing_confirmatory_v1.json").read_text())
        expected = {"ridge_sgd_l2": 5, "lightgbm_multiclass": 4, "zoo_cnn": 4, "set_transformer": 4}
        for model, count in expected.items():
            candidates = sensitivity_candidates(config, model)
            self.assertEqual(len(candidates), count)
            frozen = config["models"][model]
            if model == "ridge_sgd_l2":
                self.assertEqual(candidates[0]["alpha"], frozen["alpha"])
            if model in {"zoo_cnn", "set_transformer"}:
                self.assertEqual(candidates[0]["learning_rate"], 1e-3)
                self.assertEqual(candidates[0]["dropout"], 0.3)

    def test_locked_grid_sizes(self):
        config = json.loads(Path("configs/rushing_confirmatory_v1.json").read_text())
        manifest = {"config": config}
        self.assertEqual(len(expected_cell_keys(manifest, "main")), 1200)
        self.assertEqual(len(expected_cell_keys(manifest, "sensitivity")), 240)
        self.assertEqual(
            len(expected_cell_keys(manifest, "sensitivity", sensitivity_extended=True)), 600
        )
        full_manifest = json.loads(json.dumps(manifest))
        full_manifest["config"]["execution"]["confirmatory_repeats"] = 100
        self.assertEqual(len(expected_cell_keys(full_manifest, "main")), 2400)
        self.assertEqual(len(expected_cell_keys(full_manifest, "sensitivity")), 240)
        self.assertEqual(
            len(
                expected_cell_keys(
                    full_manifest, "sensitivity", sensitivity_extended=True
                )
            ),
            600,
        )
        self.assertEqual(tuple(MODEL_IDS), tuple(config["models"]))


if __name__ == "__main__":
    unittest.main()
