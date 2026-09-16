"""Tests for the locked rushing-study statistical aggregation layer."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zoo-analysis-mpl"))

from rushing_study.analysis import (  # noqa: E402
    ANCHOR_SIZES,
    FULL_MAIN_REPEAT_IDS,
    MAIN_MODELS,
    MAIN_REPEAT_IDS,
    SENSITIVITY_REPEAT_IDS,
    _bootstrap_max_t,
    bootstrap_cell_sd_intervals,
    bootstrap_log_sd_ratios,
    controlled_sharpness_table,
    coverage_qualification,
    endpoint_changes,
    flatten_sensitivity_candidate_scores,
    paired_model_contrasts,
    paired_size_contrasts,
    plot_metric_ribbons,
    repeat_block_bootstrap_indices,
    run_aggregation,
    run_sensitivity_aggregation,
    summarize_metrics,
    summarize_metrics_wide,
    summarize_paired_contrasts,
    summarize_tuning_sensitivity,
    supported_best_declarations,
    validate_main_grid,
)


def make_main_metrics(
    repeat_ids: tuple[int, ...] = MAIN_REPEAT_IDS,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_crps = {
        MAIN_MODELS[0]: 0.0140,
        MAIN_MODELS[1]: 0.0135,
        MAIN_MODELS[2]: 0.0130,
        MAIN_MODELS[3]: 0.0132,
    }
    model_width = {
        MAIN_MODELS[0]: 21.0,
        MAIN_MODELS[1]: 14.0,
        MAIN_MODELS[2]: 15.0,
        MAIN_MODELS[3]: 16.0,
    }
    for repeat in repeat_ids:
        centered = repeat - np.mean(repeat_ids)
        for model_index, model in enumerate(MAIN_MODELS):
            for size in ANCHOR_SIZES:
                crps = (
                    model_crps[model]
                    - 0.000001 * size * (model_index + 1)
                    + centered * 0.00001 * (model_index + 1)
                )
                width = (
                    model_width[model]
                    - 0.002 * size * (model_index + 1)
                    + centered * 0.01 * (model_index + 1)
                )
                coverage_center = 0.885 if model == MAIN_MODELS[3] else 0.925
                coverage = coverage_center + centered * 0.0001
                rows.append(
                    {
                        "repeat": repeat,
                        "repeat_seed": 10_000 + repeat,
                        "model": model,
                        "n_train": size,
                        "crps": crps,
                        "mean_width": width,
                        "coverage": coverage,
                        "crps_game_equal": crps + 0.0002,
                        "mean_width_game_equal": width + 0.25,
                        "coverage_game_equal": coverage - 0.001,
                    }
                )
    return pd.DataFrame(rows)


def make_full_manifest(bootstrap_seed: int = 1234, repeats: int = 50) -> dict:
    return {
        "config": {
            "base_seed": 20260817,
            "models": {model: {"frozen": True} for model in MAIN_MODELS},
            "splits": {"nested_train_anchors": list(ANCHOR_SIZES)},
            "execution": {"confirmatory_repeats": repeats},
            "analysis": {
                "bootstrap_draws": 10_000,
                "bootstrap_seed_key": ["analysis", "bootstrap"],
                "coverage_target": 0.90,
                "metric_columns": ["crps", "coverage", "mean_width"],
                "coverage_metrics": ["coverage"],
                "plot_metrics": ["crps", "coverage"],
            },
            "sensitivity": {
                "stage1": {
                    "repeats": 20,
                    "repeat_ids": list(SENSITIVITY_REPEAT_IDS),
                    "anchors": [20, 160, 360],
                },
                "extension": {
                    "target_total_repeats": 50,
                    "additional_repeat_ids": list(range(21, 51)),
                },
            },
        },
        "seed_registry": {'["analysis","bootstrap"]': bootstrap_seed},
        "manifest_hash": "synthetic-test-manifest",
    }


def make_candidate_score_cells(*, extended: bool = False) -> list[dict]:
    repeat_ids = MAIN_REPEAT_IDS if extended else SENSITIVITY_REPEAT_IDS
    rows: list[dict] = []
    for repeat in repeat_ids:
        for size in (20, 160, 360):
            for model in MAIN_MODELS:
                count = 5 if model == "ridge_sgd_l2" else 4
                candidates = []
                for index in range(count):
                    config = {"model": model, "candidate": index}
                    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
                    score = 0.01 + index * 0.001
                    if index == 1:
                        score = 0.01  # exact tie retains frozen candidate zero
                    candidates.append(
                        {
                            "candidate_index": index,
                            "is_frozen_main": index == 0,
                            "config": config,
                            "config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
                            "tune_crps": score,
                            "elapsed_seconds": 0.1 + index,
                            "fit_seed": 1000 + repeat,
                            "refit_seed": None if model in MAIN_MODELS[:2] else 2000 + repeat,
                        }
                    )
                rows.append(
                    {
                        "branch": "sensitivity",
                        "repeat": repeat,
                        "n_train": size,
                        "model": model,
                        "candidate_scores": candidates,
                        "selected_candidate_index": 0,
                        "selected_config": json.dumps(candidates[0]["config"], sort_keys=True),
                    }
                )
    return rows


class RushingAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metrics = make_main_metrics()

    def test_exact_main_grid_validation(self):
        validated = validate_main_grid(self.metrics)
        self.assertEqual(len(validated), 50 * 4 * 6)

        missing = self.metrics.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "exact requested Cartesian product"):
            validate_main_grid(missing)

        duplicate = pd.concat([self.metrics, self.metrics.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_main_grid(duplicate)

        wrong_size = self.metrics.copy()
        wrong_size.loc[0, "n_train"] = 21
        with self.assertRaisesRegex(ValueError, "exact requested Cartesian product"):
            validate_main_grid(wrong_size)

    def test_summary_contains_all_distribution_statistics_and_scopes(self):
        summary = summarize_metrics(self.metrics)
        expected_stats = {"mean", "sd", "mcse", "median", "iqr", "q10", "q90"}
        self.assertTrue(expected_stats.issubset(summary.columns))
        self.assertIn("game_equal", set(summary["aggregation"]))
        self.assertIn("play", set(summary["aggregation"]))
        display_names = dict(summary[["model", "model_display_name"]].drop_duplicates().to_numpy())
        self.assertEqual(display_names["ridge_sgd_l2"], "L2 one-vs-rest logistic")
        self.assertEqual(display_names["lightgbm_multiclass"], "LightGBM")

        row = summary[
            (summary["model"] == MAIN_MODELS[0])
            & (summary["n_train"] == 20)
            & (summary["metric"] == "crps")
        ].iloc[0]
        source = self.metrics[
            (self.metrics["model"] == MAIN_MODELS[0]) & (self.metrics["n_train"] == 20)
        ]["crps"].to_numpy()
        self.assertAlmostEqual(row["mean"], float(source.mean()))
        self.assertAlmostEqual(row["sd"], float(source.std(ddof=1)))
        self.assertAlmostEqual(row["mcse"], float(source.std(ddof=1) / np.sqrt(50)))
        self.assertAlmostEqual(row["iqr"], float(np.quantile(source, 0.75) - np.quantile(source, 0.25)))
        wide = summarize_metrics_wide(summary)
        self.assertEqual(len(wide), 24)
        self.assertIn("crps_mean", wide.columns)
        self.assertIn("coverage_game_equal_q90", wide.columns)
        self.assertIn("model_display_name", wide.columns)

    def test_paired_model_and_size_contrasts_and_endpoint_changes(self):
        model_rows = paired_model_contrasts(
            self.metrics,
            metric_columns=["crps"],
            model_pairs=[(MAIN_MODELS[0], MAIN_MODELS[1])],
        )
        self.assertEqual(len(model_rows), 50 * 6)
        source = self.metrics.set_index(["repeat", "model", "n_train"])
        first = model_rows.iloc[0]
        expected = source.loc[(first["repeat"], MAIN_MODELS[0], first["n_train"]), "crps"] - source.loc[
            (first["repeat"], MAIN_MODELS[1], first["n_train"]), "crps"
        ]
        self.assertAlmostEqual(first["value"], expected)

        size_rows = paired_size_contrasts(
            self.metrics,
            metric_columns=["crps"],
            size_pairs=[(20, 40)],
        )
        self.assertEqual(len(size_rows), 50 * 4)
        endpoint_rows = endpoint_changes(self.metrics, metric_columns=["crps"])
        self.assertEqual(len(endpoint_rows), 50 * 4)
        one = endpoint_rows.iloc[0]
        expected_change = source.loc[(one["repeat"], one["model"], 360), "crps"] - source.loc[
            (one["repeat"], one["model"], 20), "crps"
        ]
        self.assertAlmostEqual(one["value"], expected_change)

    def test_repeat_block_bootstrap_is_deterministic_and_defaults_to_10000(self):
        first = repeat_block_bootstrap_indices(seed=77)
        second = repeat_block_bootstrap_indices(seed=77)
        third = repeat_block_bootstrap_indices(seed=78)
        self.assertEqual(first.shape, (10_000, 50))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))
        self.assertTrue(np.all((first >= 0) & (first < 50)))

    def test_max_t_retains_nonzero_displacement_when_bootstrap_se_is_zero(self):
        # The observed repeat vector is nonconstant, but these deliberately
        # degenerate draws resample only a zero-valued repeat.  Its resampled
        # mean differs from the observed mean while its within-draw SE is zero.
        matrix = np.zeros((20, 1), dtype=np.float64)
        matrix[-1, 0] = 1.0
        degenerate_draws = np.zeros((20, 20), dtype=np.int64)

        critical = _bootstrap_max_t(
            matrix,
            degenerate_draws,
            confidence=0.95,
            two_sided=True,
        )

        self.assertTrue(np.isfinite(critical))
        self.assertGreater(critical, 1_000_000.0)

    def test_supported_best_requires_simultaneous_support_against_all_three(self):
        target = MAIN_MODELS[2]
        rows = []
        for model_a_index, model_a in enumerate(MAIN_MODELS):
            for model_b in MAIN_MODELS[model_a_index + 1 :]:
                if model_a == target:
                    lower, upper, mean = -2.0, -0.5, -1.0
                elif model_b == target:
                    lower, upper, mean = 0.5, 2.0, 1.0
                else:
                    lower, upper, mean = -1.0, 1.0, 0.0
                rows.append(
                    {
                        "metric": "crps",
                        "n_train": 20,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean": mean,
                        "simultaneous_lower": lower,
                        "simultaneous_upper": upper,
                    }
                )
        declarations = supported_best_declarations(pd.DataFrame(rows))
        supported = declarations[declarations["supported_best"]]
        self.assertEqual(supported["model"].tolist(), [target])
        self.assertEqual(int(supported.iloc[0]["comparators_supported"]), 3)
        self.assertFalse(bool(declarations[declarations["model"] != target]["supported_best"].any()))

    def test_controlled_sharpness_requires_both_coverage_lower_bounds(self):
        summary = summarize_metrics(
            self.metrics,
            metric_columns=["mean_width", "coverage"],
        )
        contrast_values = paired_model_contrasts(
            self.metrics,
            metric_columns=["mean_width"],
        )
        contrasts = summarize_paired_contrasts(
            contrast_values,
            id_columns=["metric", "n_train", "model_a", "model_b", "contrast"],
            repeat_ids=MAIN_REPEAT_IDS,
            n_bootstrap=300,
            seed=55,
        )
        coverage = coverage_qualification(
            self.metrics,
            coverage_metrics=["coverage"],
            n_bootstrap=300,
            seed=55,
        )
        controlled = controlled_sharpness_table(summary, coverage, contrasts)
        labelled = controlled[controlled["sharper_at_controlled_coverage"]]
        self.assertFalse(labelled.empty)
        self.assertTrue(bool(labelled["both_coverage_qualified"].all()))
        self.assertTrue(bool((labelled["coverage_lcb_a"] >= labelled["coverage_target"]).all()))
        self.assertTrue(bool((labelled["coverage_lcb_b"] >= labelled["coverage_target"]).all()))
        involving_undercovered = controlled[
            (controlled["model_a"] == MAIN_MODELS[3]) | (controlled["model_b"] == MAIN_MODELS[3])
        ]
        self.assertFalse(bool(involving_undercovered["sharper_at_controlled_coverage"].any()))
        self.assertEqual(set(involving_undercovered["sharpness_status"]), {"coverage_not_controlled"})

    def test_max_t_inference_is_paired_deterministic_and_simultaneous(self):
        contrast_rows = paired_model_contrasts(
            self.metrics,
            metric_columns=["crps"],
            model_pairs=[(MAIN_MODELS[0], MAIN_MODELS[1]), (MAIN_MODELS[2], MAIN_MODELS[1])],
        )
        first = summarize_paired_contrasts(
            contrast_rows,
            id_columns=["metric", "n_train", "model_a", "model_b", "contrast"],
            repeat_ids=MAIN_REPEAT_IDS,
            n_bootstrap=500,
            seed=123,
        )
        second = summarize_paired_contrasts(
            contrast_rows,
            id_columns=["metric", "n_train", "model_a", "model_b", "contrast"],
            repeat_ids=MAIN_REPEAT_IDS,
            n_bootstrap=500,
            seed=123,
        )
        pd.testing.assert_frame_equal(first, second)
        simultaneous_width = first["simultaneous_upper"] - first["simultaneous_lower"]
        pointwise_width = first["pointwise_upper"] - first["pointwise_lower"]
        self.assertTrue(np.all(simultaneous_width >= pointwise_width - 1e-15))

    def test_coverage_qualification_and_stability_intervals(self):
        coverage = coverage_qualification(
            self.metrics,
            coverage_metrics=["coverage", "coverage_game_equal"],
            n_bootstrap=500,
            seed=99,
        )
        low = coverage[coverage["model"] == MAIN_MODELS[3]]
        high = coverage[coverage["model"] != MAIN_MODELS[3]]
        self.assertFalse(bool(low["qualified"].any()))
        self.assertTrue(bool(high["qualified"].all()))

        sd_intervals = bootstrap_cell_sd_intervals(
            self.metrics,
            metric_columns=["crps"],
            n_bootstrap=300,
            seed=13,
        )
        self.assertEqual(len(sd_intervals), 4 * 6)
        self.assertTrue(np.all(sd_intervals["bootstrap_lower"] <= sd_intervals["sd"]))
        self.assertTrue(np.all(sd_intervals["sd"] <= sd_intervals["bootstrap_upper"]))
        self.assertEqual(set(sd_intervals["interval_method"]), {"bca_repeat_block"})
        self.assertTrue(np.isfinite(sd_intervals[["bootstrap_lower", "bootstrap_upper"]]).all().all())
        self.assertTrue(np.all(sd_intervals["bootstrap_lower"] <= sd_intervals["bootstrap_upper"]))

        ratios = bootstrap_log_sd_ratios(
            self.metrics,
            metric_columns=["crps"],
            model_pairs=[(MAIN_MODELS[1], MAIN_MODELS[0])],
            n_bootstrap=300,
            seed=13,
        )
        self.assertEqual(len(ratios), 6)
        self.assertTrue(np.all(np.isfinite(ratios["log_sd_ratio"])))
        self.assertTrue(np.all(ratios["ratio_bootstrap_lower"] > 0.0))
        self.assertEqual(set(ratios["interval_method"]), {"bca_repeat_block"})
        self.assertTrue(np.isfinite(ratios[["log_bootstrap_lower", "log_bootstrap_upper"]]).all().all())
        self.assertTrue(np.all(ratios["log_bootstrap_lower"] <= ratios["log_bootstrap_upper"]))

    def test_plot_uses_summary_ribbons(self):
        summary = summarize_metrics(self.metrics, metric_columns=["crps", "coverage"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.png"
            figure, axes = plot_metric_ribbons(summary, metrics=["crps", "coverage"], output_path=path)
            self.assertTrue(path.exists())
            self.assertEqual(len(axes), 2)
            self.assertGreater(len(axes[0].collections), 0)  # fill_between ribbon
            labels = [text.get_text() for text in axes[0].get_legend().get_texts()]
            self.assertIn("L2 one-vs-rest logistic", labels)
            self.assertIn("LightGBM", labels)
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_sensitivity_summary_triggers_extension(self):
        fixed = self.metrics.copy()
        tuned = fixed[
            fixed["repeat"].isin(SENSITIVITY_REPEAT_IDS)
            & fixed["n_train"].isin([20, 160, 360])
        ].copy()
        tuned["selected_config"] = "candidate_a"
        tuned.loc[tuned["model"] == MAIN_MODELS[0], "crps"] -= 0.01
        result = summarize_tuning_sensitivity(
            fixed,
            tuned,
            metric_columns=["crps", "coverage"],
            n_bootstrap=300,
            seed=41,
        )
        self.assertTrue(result["extension_trigger"])
        self.assertTrue(bool(result["summary"]["significant_shift"].any()))
        self.assertTrue(bool(result["rankings"]["ranking_changed"].any()))
        self.assertEqual(int(result["configuration_frequencies"]["count"].sum()), 20 * 4 * 3)

    def test_significant_common_tuning_shift_does_not_trigger_extension(self):
        fixed = self.metrics[
            self.metrics["repeat"].isin(SENSITIVITY_REPEAT_IDS)
            & self.metrics["n_train"].isin([20, 160, 360])
        ].copy()
        tuned = fixed.copy()
        tuned["crps"] -= 0.001
        result = summarize_tuning_sensitivity(
            fixed,
            tuned,
            metric_columns=["crps", "coverage", "mean_width"],
            n_bootstrap=200,
            seed=42,
        )
        self.assertTrue(bool(result["summary"]["significant_shift"].any()))
        self.assertFalse(result["extension_trigger"])
        self.assertFalse(bool(result["rankings"]["ranking_changed"].any()))
        self.assertFalse(bool(result["conclusions"]["conclusion_changed"].any()))

    def test_inconclusive_to_supported_sensitivity_change_does_not_trigger(self):
        fixed_rows = []
        tuned_rows = []
        means = dict(zip(MAIN_MODELS, [0.01, 0.02, 0.03, 0.04]))
        amplitudes = dict(zip(MAIN_MODELS, [0.08, -0.06, 0.04, -0.02]))
        for repeat in SENSITIVITY_REPEAT_IDS:
            sign = -1.0 if repeat % 2 else 1.0
            for size in (20, 160, 360):
                for model in MAIN_MODELS:
                    common = {
                        "repeat": repeat,
                        "model": model,
                        "n_train": size,
                        "coverage": 0.95,
                    }
                    fixed_rows.append({**common, "crps": means[model] + sign * amplitudes[model]})
                    tuned_rows.append({**common, "crps": means[model]})
        result = summarize_tuning_sensitivity(
            pd.DataFrame(fixed_rows),
            pd.DataFrame(tuned_rows),
            metric_columns=["crps", "coverage"],
            n_bootstrap=300,
            seed=77,
        )
        self.assertFalse(bool(result["rankings"]["ranking_changed"].any()))
        self.assertTrue(bool(result["conclusions"]["conclusion_state_changed"].any()))
        self.assertFalse(bool(result["conclusions"]["conclusion_changed"].any()))
        self.assertFalse(result["extension_trigger"])

    def test_model_b_supported_to_inconclusive_triggers_regardless_of_orientation(self):
        fixed_rows = []
        tuned_rows = []
        # Descending means make model B the supported winner for every stored
        # A-minus-B pair.  Symmetric repeat noise preserves the same mean
        # ranking while making the tuned pairwise conclusions inconclusive.
        means = dict(zip(MAIN_MODELS, [0.04, 0.03, 0.02, 0.01]))
        amplitudes = dict(zip(MAIN_MODELS, [0.08, -0.06, 0.04, -0.02]))
        for repeat in SENSITIVITY_REPEAT_IDS:
            sign = -1.0 if repeat % 2 else 1.0
            for size in (20, 160, 360):
                for model in MAIN_MODELS:
                    common = {
                        "repeat": repeat,
                        "model": model,
                        "n_train": size,
                        "coverage": 0.95,
                    }
                    fixed_rows.append({**common, "crps": means[model]})
                    tuned_rows.append(
                        {**common, "crps": means[model] + sign * amplitudes[model]}
                    )
        result = summarize_tuning_sensitivity(
            pd.DataFrame(fixed_rows),
            pd.DataFrame(tuned_rows),
            metric_columns=["crps", "coverage"],
            n_bootstrap=300,
            seed=78,
        )
        changed = result["conclusions"].loc[
            result["conclusions"]["conclusion_changed"]
        ]
        self.assertFalse(bool(result["rankings"]["ranking_changed"].any()))
        self.assertFalse(changed.empty)
        self.assertTrue((changed["fixed_conclusion"] == "opposite").all())
        self.assertTrue((changed["tuned_conclusion"] == "inconclusive").all())
        self.assertTrue(result["extension_trigger"])

    def test_run_aggregation_accepts_record_dicts_and_writes_artifacts(self):
        manifest = make_full_manifest(bootstrap_seed=1234)
        records = self.metrics.to_dict(orient="records")
        with tempfile.TemporaryDirectory() as directory:
            result = run_aggregation(records, manifest, directory, bootstrap_draws=100)
            expected = {
                "summary",
                "model_contrasts",
                "adjacent_size_contrasts",
                "endpoint_changes",
                "coverage_qualification",
                "supported_best_crps",
                "controlled_sharpness",
                "sd_intervals",
                "log_sd_ratios",
            }
            self.assertTrue(expected.issubset(result))
            self.assertEqual(len(result["summary_wide"]), 24)
            for path in result["output_paths"].values():
                self.assertTrue(Path(path).exists(), path)
            with Path(result["output_paths"]["manifest"]).open() as handle:
                saved = json.load(handle)
            self.assertEqual(saved["bootstrap_draws"], 100)
            self.assertEqual(saved["bootstrap_seed"], 1234)
            self.assertEqual(saved["bootstrap_unit"], "whole repeat vector")
            self.assertIn("BCa repeat-block", saved["stability_interval_method"])

    def test_run_aggregation_accepts_exact_100_repeat_full_grid(self):
        metrics = make_main_metrics(FULL_MAIN_REPEAT_IDS)
        manifest = make_full_manifest(bootstrap_seed=5678, repeats=100)
        with tempfile.TemporaryDirectory() as directory:
            result = run_aggregation(
                metrics,
                manifest,
                directory,
                bootstrap_draws=50,
            )
        self.assertEqual(len(result["summary_wide"]), 24)
        self.assertEqual(result["manifest"]["repeat_ids"], list(FULL_MAIN_REPEAT_IDS))

    def test_sensitivity_entrypoint_writes_stage1_and_trigger_files(self):
        fixed = self.metrics.copy()
        tuned = fixed[
            fixed["repeat"].isin(SENSITIVITY_REPEAT_IDS)
            & fixed["n_train"].isin([20, 160, 360])
        ].copy()
        tuned["selected_config"] = "candidate_a"
        tuned.loc[tuned["model"] == MAIN_MODELS[0], "crps"] -= 0.01
        with tempfile.TemporaryDirectory() as directory:
            result = run_sensitivity_aggregation(
                fixed,
                tuned,
                make_full_manifest(bootstrap_seed=9123),
                directory,
                bootstrap_draws=100,
            )
            self.assertTrue(result["extension_trigger"])
            self.assertEqual(result["trigger_manifest"]["additional_repeat_ids"], list(range(21, 51)))
            self.assertEqual(result["trigger_manifest"]["sensitivity_anchors"], [20, 160, 360])
            for path in result["output_paths"].values():
                self.assertTrue(Path(path).exists(), path)
            trigger_path = Path(result["output_paths"]["extension_trigger"])
            trigger = json.loads(trigger_path.read_text())
            self.assertEqual(trigger["extension_scope"], "whole sensitivity branch at the same three anchors")

    def test_sensitivity_entrypoint_supports_exact_all_50_extension(self):
        fixed = self.metrics.copy()
        tuned = fixed[fixed["n_train"].isin([20, 160, 360])].copy()
        tuned["selected_config"] = "candidate_a"
        tuned.loc[tuned["model"] == MAIN_MODELS[0], "crps"] -= 0.01
        with tempfile.TemporaryDirectory() as directory:
            result = run_sensitivity_aggregation(
                fixed,
                tuned,
                make_full_manifest(bootstrap_seed=9123),
                directory,
                bootstrap_draws=100,
                extended=True,
            )
            self.assertTrue((result["summary"]["n_repeats"] == 50).all())
            self.assertEqual(result["trigger_manifest"]["analysis_stage"], "all_50_extension")
            self.assertEqual(result["trigger_manifest"]["repeats_analyzed"], list(MAIN_REPEAT_IDS))
            self.assertTrue(result["trigger_manifest"]["extension_complete"])
            self.assertEqual(result["trigger_manifest"]["additional_repeat_ids"], [])

            incomplete = tuned[tuned["repeat"] <= 20].copy()
            with self.assertRaisesRegex(ValueError, "exact requested Cartesian product"):
                run_sensitivity_aggregation(
                    fixed,
                    incomplete,
                    make_full_manifest(bootstrap_seed=9123),
                    Path(directory) / "incomplete",
                    bootstrap_draws=20,
                    extended=True,
                )

    def test_candidate_score_flattener_validates_grid_counts_indices_and_ties(self):
        stage1_rows = make_candidate_score_cells()
        flattened = flatten_sensitivity_candidate_scores(stage1_rows)
        self.assertEqual(len(flattened), 20 * 3 * (5 + 4 + 4 + 4))
        self.assertEqual(
            flattened.groupby(["repeat", "n_train", "model"])["is_selected"].sum().unique().tolist(),
            [1],
        )
        self.assertTrue(bool(flattened[flattened["candidate_index"] == 0]["is_selected"].all()))
        self.assertEqual(
            set(flattened["tie_break_rule"]),
            {"lowest_ordered_candidate_on_exact_tie"},
        )
        self.assertEqual(set(flattened["tie_tolerance"]), {0.0})

        near_tie = make_candidate_score_cells()
        near_tie[0]["candidate_scores"][1]["tune_crps"] = 0.01 - 0.5e-12
        near_tie[0]["selected_candidate_index"] = 1
        near_tie[0]["selected_config"] = json.dumps(
            near_tie[0]["candidate_scores"][1]["config"], sort_keys=True
        )
        exact = flatten_sensitivity_candidate_scores(near_tie)
        first_cell = exact[
            (exact["repeat"] == 1)
            & (exact["n_train"] == 20)
            & (exact["model"] == MAIN_MODELS[0])
        ]
        self.assertEqual(first_cell.loc[first_cell["is_selected"], "candidate_index"].tolist(), [1])
        with self.assertRaisesRegex(ValueError, "exactly 0.0"):
            flatten_sensitivity_candidate_scores(stage1_rows, tie_tolerance=1e-12)

        extended = flatten_sensitivity_candidate_scores(
            make_candidate_score_cells(extended=True),
            extended=True,
        )
        self.assertEqual(len(extended), 50 * 3 * (5 + 4 + 4 + 4))

        bad_count = make_candidate_score_cells()
        bad_count[0]["candidate_scores"] = bad_count[0]["candidate_scores"][:-1]
        with self.assertRaisesRegex(ValueError, "must have 5 candidates"):
            flatten_sensitivity_candidate_scores(bad_count)

        bad_index = make_candidate_score_cells()
        bad_index[0]["candidate_scores"][1]["candidate_index"] = 3
        with self.assertRaisesRegex(ValueError, "indices"):
            flatten_sensitivity_candidate_scores(bad_index)

        bad_frozen = make_candidate_score_cells()
        bad_frozen[0]["candidate_scores"][0]["is_frozen_main"] = False
        with self.assertRaisesRegex(ValueError, "is_frozen_main"):
            flatten_sensitivity_candidate_scores(bad_frozen)

        bad_selected = make_candidate_score_cells()
        bad_selected[0]["selected_candidate_index"] = 1
        with self.assertRaisesRegex(ValueError, "tie replay"):
            flatten_sensitivity_candidate_scores(bad_selected)


if __name__ == "__main__":
    unittest.main()
