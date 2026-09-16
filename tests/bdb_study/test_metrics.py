from __future__ import annotations

import unittest

import numpy as np

from bdb_study.metrics import (
    bernoulli_entropy,
    brier_contributions,
    calibration_bias,
    crps_contributions,
    fit_venn_abers,
    game_equal_mean,
    hierarchical_distribution_intervals,
    hierarchical_subsample_indices,
    label_conditional_prediction_sets,
    pathwise_conformal_tube,
    prevalence_null,
    rms_reliability_error,
    skill_score,
    trajectory_rmse,
)


class MetricTests(unittest.TestCase):
    def test_binary_and_grouped_metrics(self) -> None:
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.8, 0.7, 0.2])
        contributions = brier_contributions(y, p)
        self.assertAlmostEqual(float(contributions.mean()), 0.045)
        self.assertAlmostEqual(game_equal_mean(contributions, np.array([1, 1, 2, 2])), 0.045)
        self.assertAlmostEqual(prevalence_null(y), 0.5)
        self.assertAlmostEqual(skill_score(0.25, 0.5), 0.5)

    def test_crps_is_zero_for_perfect_ordered_distribution(self) -> None:
        probability = np.eye(4)
        contribution = crps_contributions(np.arange(4), probability)
        np.testing.assert_allclose(contribution, 0.0)

    def test_venn_abers_retains_raw_and_calibrated_point_probabilities(self) -> None:
        calibration_p = np.array([0.05, 0.15, 0.20, 0.40, 0.55, 0.70, 0.80, 0.95])
        calibration_y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        raw = np.array([0.25, 0.65])
        result = fit_venn_abers(calibration_p, calibration_y, raw)
        np.testing.assert_allclose(result.raw_probability, raw)
        self.assertEqual(result.calibrated_probability.shape, (2,))
        self.assertTrue(np.all(result.lower <= result.upper))
        self.assertTrue(np.all(result.imprecision >= 0.0))

    def test_trajectory_rmse_uses_pooled_coordinates(self) -> None:
        truth = np.zeros((2, 2, 2), dtype=float)
        prediction = np.ones_like(truth)
        mask = np.array([[True, True], [True, False]])
        self.assertAlmostEqual(trajectory_rmse(truth, prediction, mask), 1.0)

    def test_hierarchical_subsample_is_one_per_game_and_deterministic(self) -> None:
        games = np.array(["b", "a", "b", "a", "c"])
        first = hierarchical_subsample_indices(games, seed=17)
        second = hierarchical_subsample_indices(games, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(games[first]), {"a", "b", "c"})
        self.assertEqual(len(first), 3)

    def test_hierarchical_distribution_interval_uses_game_registry(self) -> None:
        cal = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.6, 0.2],
            ]
        )
        result = hierarchical_distribution_intervals(
            cal,
            np.array([0, 1, 2, 1]),
            np.array(["a", "a", "b", "b"]),
            np.array([[0.2, 0.6, 0.2]]),
            seed=9,
        )
        self.assertEqual(result["n_calibration_games"], 2)
        self.assertEqual(len(result["selected_calibration_indices"]), 2)
        self.assertLessEqual(int(result["lower"][0]), int(result["upper"][0]))

    def test_binary_reliability_entropy_and_label_conditional_sets(self) -> None:
        cal_p = np.array([0.05, 0.20, 0.75, 0.90, 0.10, 0.80])
        cal_y = np.array([0, 0, 1, 1, 0, 1])
        games = np.array(["a", "b", "a", "b", "c", "c"])
        test_p = np.array([0.1, 0.9, 0.4, 0.6])
        test_y = np.array([0, 1, 0, 1])
        sets = label_conditional_prediction_sets(
            cal_p, cal_y, games, test_p, seed=3
        )
        self.assertEqual(sets["included"].shape, (4, 2))
        self.assertTrue(np.all((sets["set_size"] >= 0) & (sets["set_size"] <= 2)))
        reliability, boundaries = rms_reliability_error(cal_p, test_p, test_y)
        self.assertGreaterEqual(reliability, 0.0)
        self.assertTrue(np.all(np.diff(boundaries) > 0.0))
        self.assertAlmostEqual(calibration_bias(test_y, test_p), 0.0)
        self.assertTrue(np.all(bernoulli_entropy(test_p) >= 0.0))

    def test_pathwise_tube_covers_complete_paths_and_uses_one_path_per_game(self) -> None:
        cal_y = np.zeros((4, 3, 2), dtype=float)
        cal_prediction = np.zeros_like(cal_y)
        cal_prediction[:, :, 0] = np.array([0.1, 0.2, 0.3, 0.4])[:, None]
        mask = np.ones((4, 3), dtype=bool)
        test_y = np.zeros((2, 3, 2), dtype=float)
        test_prediction = np.zeros_like(test_y)
        test_prediction[0, :, 0] = 0.1
        test_prediction[1, :, 0] = 2.0
        result = pathwise_conformal_tube(
            cal_y,
            cal_prediction,
            mask,
            np.array(["a", "a", "b", "b"]),
            test_y,
            test_prediction,
            np.ones((2, 3), dtype=bool),
            np.ones(3),
            alpha=0.5,
            seed=4,
        )
        self.assertEqual(result["n_calibration_games"], 2)
        np.testing.assert_array_equal(result["path_covered"], [True, False])
        self.assertTrue(np.all(result["radius"] > 0.0))
        np.testing.assert_allclose(result["field_bounds"], [0.0, 120.0, 0.0, 160.0 / 3.0])
        self.assertEqual(
            result["region_contract"], "disk_intersect_legal_field_rectangle"
        )
        outside = cal_y.copy()
        outside[0, 0, 1] = 60.0
        with self.assertRaisesRegex(ValueError, "outside legal field"):
            pathwise_conformal_tube(
                outside,
                cal_prediction,
                mask,
                np.array(["a", "a", "b", "b"]),
                test_y,
                test_prediction,
                np.ones((2, 3), dtype=bool),
                np.ones(3),
                alpha=0.5,
                seed=4,
            )

    def test_pathwise_tube_uses_constant_completed_scale_at_valid_horizon_94(self) -> None:
        scale = np.concatenate([np.linspace(0.25, 1.0, 33), np.ones(61)])
        cal_y = np.zeros((2, 94, 2), dtype=float)
        cal_prediction = np.zeros_like(cal_y)
        cal_prediction[..., 0] = 0.5 * scale[None, :]
        cal_mask = np.ones((2, 94), dtype=bool)
        test_y = np.zeros((2, 94, 2), dtype=float)
        test_prediction = np.zeros_like(test_y)
        test_mask = np.zeros((2, 94), dtype=bool)
        test_mask[0, :34] = True
        test_mask[1, :] = True
        # A huge miss on a padded horizon is ignored; the same kind of miss at
        # a valid H94 participates in the whole-path coverage event.
        test_prediction[0, 80, 0] = 100.0
        test_prediction[1, 93, 0] = 2.0
        result = pathwise_conformal_tube(
            cal_y,
            cal_prediction,
            cal_mask,
            np.array(["a", "b"]),
            test_y,
            test_prediction,
            test_mask,
            scale,
            alpha=0.5,
            seed=7,
        )
        np.testing.assert_array_equal(result["path_covered"], [True, False])
        np.testing.assert_array_equal(result["horizon_scale"][33:], np.ones(61))


if __name__ == "__main__":
    unittest.main()
