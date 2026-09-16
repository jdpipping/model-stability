from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.analysis import (
    CANONICAL_CONTRAST_COUNT,
    DEFAULT_MODELS,
    STRUCTURE_MODELS,
    bca_stability_intervals,
    controlled_sharpness_conclusions,
    descriptive_suite_summary,
    pilot_task_analysis_tables,
    paired_difference_stability,
    paired_max_t_intervals,
    simultaneous_coverage_lower_bounds,
    stability_ribbons,
    supported_best_conclusions,
    summarize_task,
    task_analysis_tables,
    training_size_changes,
    training_size_difference_in_changes,
    validate_task_grid,
    write_task_analysis,
)


ANCHORS = (10, 20, 40, 60, 100, 140)


def _task_frame(repeats: int = 8) -> pd.DataFrame:
    offsets = {"glm": 0.020, "lightgbm": 0.010, "cnn": 0.005, "transformer": 0.000}
    slopes = {"glm": 0.0007, "lightgbm": -0.0003, "cnn": 0.0002, "transformer": -0.0006}
    rows = []
    for repeat in range(1, repeats + 1):
        for model in DEFAULT_MODELS:
            for anchor in ANCHORS:
                primary = 0.24 - 0.00025 * anchor + offsets[model] + slopes[model] * repeat
                null = 0.31 - 0.00005 * anchor + 0.0001 * repeat
                rows.append(
                    {
                        "task_id": "synthetic_binary",
                        "repeat": repeat,
                        "model": model,
                        "n_train": anchor,
                        "primary_loss": primary,
                        "game_equal_loss": primary + 0.002 + 0.00005 * repeat,
                        "null_loss": null,
                        "skill": 1.0 - primary / null,
                        "raw_brier": primary,
                        "game_equal_brier": primary + 0.002 + 0.00005 * repeat,
                        "calibrated_brier": primary - 0.001,
                        "coverage": 0.87 + 0.0002 * anchor + 0.0005 * repeat,
                        "interval_width": 18.0 - 0.02 * anchor + offsets[model],
                    }
                )
    return pd.DataFrame(rows)


def _prospective_task_frame(repeats: int = 8) -> pd.DataFrame:
    rows = []
    for repeat in range(1, repeats + 1):
        for model_index, model in enumerate(STRUCTURE_MODELS):
            for anchor in ANCHORS:
                primary = (
                    0.24 - 0.00025 * anchor + 0.005 * model_index
                    + 0.0001 * (model_index + 1) * repeat
                )
                null = 0.31 - 0.00005 * anchor + 0.0001 * repeat
                rows.append(
                    {
                        "task_id": "synthetic_prospective_binary",
                        "repeat": repeat,
                        "model": model,
                        "n_train": anchor,
                        "primary_loss": primary,
                        "game_equal_loss": primary + 0.002,
                        "null_loss": null,
                        "coverage": 0.93 + 0.0001 * repeat,
                        "interval_width": 12.0 + model_index,
                    }
                )
    return pd.DataFrame(rows)


class TaskAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = _task_frame()

    def test_exact_grid_and_core_summary_statistics(self) -> None:
        supplied = self.metrics.copy()
        supplied["skill"] = -999.0
        validated = validate_task_grid(
            supplied,
            task_id="synthetic_binary",
            anchors=ANCHORS,
            repeats=8,
        )
        first = validated.iloc[0]
        self.assertAlmostEqual(
            first["skill"], 1.0 - first["primary_loss"] / first["null_loss"]
        )

        summary = summarize_task(validated)
        self.assertEqual(len(summary), 24)
        for metric in ("primary_loss", "game_equal_loss", "null_loss", "skill"):
            for statistic in ("mean", "sd", "mcse", "median", "iqr", "q10", "q90"):
                self.assertIn(f"{metric}_{statistic}", summary.columns)
        np.testing.assert_allclose(summary["raw_loss_sd"], summary["primary_loss_sd"])
        self.assertEqual(set(summary["repeats"]), {8})

    def test_validate_grid_rejects_noncanonical_or_incomplete_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "six strictly increasing anchors"):
            validate_task_grid(
                self.metrics[self.metrics["n_train"] != 140],
                task_id="synthetic_binary",
                anchors=ANCHORS[:-1],
                repeats=8,
            )
        with self.assertRaisesRegex(ValueError, "task grid mismatch"):
            validate_task_grid(
                self.metrics.iloc[:-1],
                task_id="synthetic_binary",
                anchors=ANCHORS,
                repeats=8,
            )

    def test_legacy_max_t_is_exact_36_member_repeat_block_family(self) -> None:
        result = paired_max_t_intervals(self.metrics, draws=250, seed=1729)
        self.assertEqual(len(result), 36)
        self.assertEqual(result.groupby("n_train").size().to_dict(), {anchor: 6 for anchor in ANCHORS})
        self.assertEqual(set(result["family_contrasts"]), {36})
        self.assertEqual(set(result["bootstrap_unit"]), {"complete_repeat_4x6_vector"})
        self.assertEqual(result["max_t_critical"].nunique(), 1)

        shuffled = self.metrics.sample(frac=1.0, random_state=9).reset_index(drop=True)
        repeated = paired_max_t_intervals(shuffled, draws=250, seed=1729)
        pd.testing.assert_frame_equal(result, repeated)

    def test_prospective_max_t_is_exact_60_member_repeat_block_family(self) -> None:
        metrics = _prospective_task_frame()
        result = paired_max_t_intervals(metrics, draws=100, seed=1729)
        self.assertEqual(CANONICAL_CONTRAST_COUNT, 60)
        self.assertEqual(len(result), 60)
        self.assertEqual(
            result.groupby("n_train").size().to_dict(),
            {anchor: 10 for anchor in ANCHORS},
        )
        self.assertEqual(set(result["family_contrasts"]), {60})
        self.assertEqual(
            set(result["bootstrap_unit"]), {"complete_repeat_5x6_vector"}
        )

    def test_max_t_rejects_incomplete_blocks_and_cross_task_pooling(self) -> None:
        with self.assertRaisesRegex(ValueError, "repeat blocks are incomplete"):
            paired_max_t_intervals(self.metrics.iloc[:-1], draws=10, seed=1)
        mixed = self.metrics.copy()
        mixed.loc[mixed.index[0], "task_id"] = "another_task"
        with self.assertRaisesRegex(ValueError, "exactly one task ID"):
            paired_max_t_intervals(mixed, draws=10, seed=1)

    def test_bca_stability_uses_one_deterministic_complete_repeat_registry(self) -> None:
        sd_intervals, ratios = bca_stability_intervals(
            self.metrics,
            draws=120,
            seed=20260817,
        )
        self.assertEqual(len(sd_intervals), 24)
        self.assertEqual(len(ratios), 36)
        self.assertEqual(set(sd_intervals["interval_method"]), {"BCa"})
        self.assertEqual(set(ratios["interval_method"]), {"BCa"})
        self.assertEqual(set(sd_intervals["bootstrap_unit"]), {"complete_repeat_4x6_vector"})
        self.assertEqual(sd_intervals["bootstrap_index_sha256"].nunique(), 1)
        self.assertEqual(ratios["bootstrap_index_sha256"].nunique(), 1)
        self.assertEqual(
            sd_intervals["bootstrap_index_sha256"].iloc[0],
            ratios["bootstrap_index_sha256"].iloc[0],
        )
        row = ratios[
            (ratios["n_train"] == 20)
            & (ratios["model_left"] == "glm")
            & (ratios["model_right"] == "lightgbm")
        ].iloc[0]
        pivot = self.metrics[self.metrics["n_train"] == 20].pivot(
            index="repeat", columns="model", values="primary_loss"
        )
        expected = np.log(
            pivot["glm"].std(ddof=1) / pivot["lightgbm"].std(ddof=1)
        )
        self.assertAlmostEqual(row["log_sd_ratio"], expected)
        self.assertAlmostEqual(row["sd_ratio"], np.exp(expected))

        shuffled = self.metrics.sample(frac=1.0, random_state=41)
        repeated_sd, repeated_ratios = bca_stability_intervals(
            shuffled,
            draws=120,
            seed=20260817,
        )
        pd.testing.assert_frame_equal(sd_intervals, repeated_sd)
        pd.testing.assert_frame_equal(ratios, repeated_ratios)

    def test_supported_best_requires_all_three_simultaneous_contrasts(self) -> None:
        frame = self.metrics.copy()
        rank = {"transformer": 0.0, "cnn": 0.05, "lightgbm": 0.10, "glm": 0.15}
        frame["primary_loss"] = [
            0.10 + rank[model] + 0.001 * repeat + 0.00001 * anchor
            for repeat, model, anchor in zip(
                frame["repeat"], frame["model"], frame["n_train"]
            )
        ]
        contrasts = paired_max_t_intervals(frame, draws=80, seed=71)
        conclusions = supported_best_conclusions(contrasts)
        supported = conclusions[conclusions["supported_best"]]
        self.assertEqual(len(supported), 6)
        self.assertEqual(set(supported["model"]), {"transformer"})
        self.assertTrue((supported["comparators_supported"] == 3).all())

        weakened = contrasts.copy()
        row = (
            weakened["n_train"].eq(20)
            & weakened["model_left"].eq("cnn")
            & weakened["model_right"].eq("transformer")
        )
        weakened.loc[row, ["simultaneous_lower", "simultaneous_upper"]] = [-0.1, 0.1]
        weakened_conclusions = supported_best_conclusions(weakened)
        transformer = weakened_conclusions[
            weakened_conclusions["n_train"].eq(20)
            & weakened_conclusions["model"].eq("transformer")
        ].iloc[0]
        self.assertFalse(transformer["supported_best"])
        self.assertEqual(transformer["conclusion"], "no_supported_best_claim")

    def test_distribution_coverage_gate_is_one_sided_simultaneous_and_required(self) -> None:
        frame = self.metrics.copy()
        frame["coverage"] = 0.955 + 0.0002 * frame["repeat"]
        width_rank = {"transformer": 0.0, "cnn": 2.0, "lightgbm": 4.0, "glm": 6.0}
        frame["interval_width"] = [
            10.0 + width_rank[model] + 0.01 * repeat
            for repeat, model in zip(frame["repeat"], frame["model"])
        ]
        coverage = simultaneous_coverage_lower_bounds(
            frame,
            draws=100,
            seed=91,
        )
        widths = paired_max_t_intervals(
            frame,
            value="interval_width",
            draws=100,
            seed=91,
        )
        conclusions = controlled_sharpness_conclusions(widths, coverage)
        self.assertEqual(len(coverage), 24)
        self.assertEqual(set(coverage["sidedness"]), {"one_sided_lower"})
        self.assertTrue(coverage["coverage_controlled"].all())
        supported = conclusions[conclusions["sharper_at_controlled_coverage"]]
        self.assertEqual(len(supported), 36)
        transformer_pairs = supported[
            supported["model_left"].eq("transformer")
            | supported["model_right"].eq("transformer")
        ]
        self.assertEqual(len(transformer_pairs), 18)
        self.assertEqual(set(transformer_pairs["sharper_model"]), {"transformer"})

        failed_coverage = coverage.copy()
        mask = failed_coverage["model"].eq("transformer")
        failed_coverage.loc[mask, "simultaneous_lower"] = 0.89
        failed_coverage.loc[mask, "coverage_controlled"] = False
        gated = controlled_sharpness_conclusions(widths, failed_coverage)
        transformer = gated[
            gated["model_left"].eq("transformer")
            | gated["model_right"].eq("transformer")
        ]
        self.assertFalse(transformer["sharper_at_controlled_coverage"].any())
        self.assertEqual(
            set(transformer["conclusion"]),
            {"coverage_not_controlled"},
        )

    def test_paired_difference_sd_is_computed_within_repeat(self) -> None:
        result = paired_difference_stability(self.metrics)
        self.assertEqual(len(result), 36)
        row = result[
            (result["n_train"] == 20)
            & (result["model_left"] == "glm")
            & (result["model_right"] == "lightgbm")
        ].iloc[0]
        pivot = self.metrics[self.metrics["n_train"] == 20].pivot(
            index="repeat", columns="model", values="primary_loss"
        )
        expected = (pivot["glm"] - pivot["lightgbm"]).std(ddof=1)
        self.assertAlmostEqual(row["difference_sd"], expected)
        differences = (pivot["glm"] - pivot["lightgbm"]).to_numpy()
        self.assertAlmostEqual(row["difference_q10"], np.quantile(differences, 0.10))
        self.assertAlmostEqual(row["difference_q90"], np.quantile(differences, 0.90))

    def test_ribbons_use_mean_and_q10_q90_not_min_max(self) -> None:
        ribbons = stability_ribbons(self.metrics, values=("primary_loss",))
        self.assertEqual(len(ribbons), 24)
        self.assertEqual(set(ribbons["ribbon_definition"]), {"between_repeat_q10_q90"})
        self.assertFalse(ribbons["inferential_interval"].any())
        self.assertNotIn("minimum", ribbons.columns)
        self.assertNotIn("maximum", ribbons.columns)
        group = self.metrics[
            (self.metrics["model"] == "glm") & (self.metrics["n_train"] == 20)
        ]["primary_loss"]
        row = ribbons[
            (ribbons["model"] == "glm") & (ribbons["n_train"] == 20)
        ].iloc[0]
        self.assertAlmostEqual(row["mean"], group.mean())
        self.assertAlmostEqual(row["ribbon_lower"], group.quantile(0.10))
        self.assertAlmostEqual(row["ribbon_upper"], group.quantile(0.90))

    def test_twenty_to_largest_changes_are_paired_by_repeat(self) -> None:
        changes = training_size_changes(
            self.metrics,
            values=("primary_loss", "coverage"),
        )
        self.assertEqual(len(changes), 8)
        self.assertEqual(set(changes["from_n_train"]), {20})
        self.assertEqual(set(changes["to_n_train"]), {140})
        row = changes[
            (changes["model"] == "cnn") & (changes["metric"] == "primary_loss")
        ].iloc[0]
        subset = self.metrics[self.metrics["model"] == "cnn"].pivot(
            index="repeat", columns="n_train", values="primary_loss"
        )
        expected = subset[140] - subset[20]
        self.assertAlmostEqual(row["change_mean"], expected.mean())
        self.assertAlmostEqual(row["change_sd"], expected.std(ddof=1))

        differences = training_size_difference_in_changes(
            self.metrics,
            values=("primary_loss",),
        )
        self.assertEqual(len(differences), 6)
        row = differences[
            (differences["model_left"] == "glm")
            & (differences["model_right"] == "lightgbm")
        ].iloc[0]
        left = self.metrics[self.metrics["model"] == "glm"].pivot(
            index="repeat", columns="n_train", values="primary_loss"
        )
        right = self.metrics[self.metrics["model"] == "lightgbm"].pivot(
            index="repeat", columns="n_train", values="primary_loss"
        )
        expected_difference = (left[140] - left[20]) - (
            right[140] - right[20]
        )
        self.assertAlmostEqual(
            row["difference_in_change_mean"], expected_difference.mean()
        )
        self.assertAlmostEqual(
            row["difference_in_change_q10"],
            expected_difference.quantile(0.10),
        )
        self.assertEqual(
            row["multiplicity_scope"],
            "secondary_descriptive_not_in_primary_max_t_family",
        )

    def test_suite_summary_is_task_equal_and_has_no_cross_task_ci(self) -> None:
        first = pd.DataFrame(
            {
                "task_id": ["task_a"] * 2,
                "repeat": [1, 2],
                "model": ["glm", "glm"],
                "family": ["linear", "linear"],
                "n_train": [20, 20],
                "skill": [0.0, 0.0],
            }
        )
        second = pd.DataFrame(
            {
                "task_id": ["task_b"] * 8,
                "repeat": list(range(1, 9)),
                "model": ["glm"] * 8,
                "family": ["linear"] * 8,
                "n_train": [20] * 8,
                "skill": [1.0] * 8,
            }
        )
        result = descriptive_suite_summary({"task_a": first, "task_b": second})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["task_equal_mean_skill"], 0.5)
        self.assertEqual(result.iloc[0]["tasks"], 2)
        self.assertEqual(result.iloc[0]["repeat_pairing"], "none_across_releases")
        self.assertEqual(
            result.iloc[0]["inference"],
            "descriptive_fixed_dependent_tasks_no_cross_task_ci",
        )
        self.assertFalse(any("confidence" in column or column.endswith("_se") for column in result.columns))

    def test_definitive_suite_accepts_bdb2020_only_at_completed_overlap(self) -> None:
        frames = {}
        for index in range(7):
            task_id = f"task_{index}"
            frame = _task_frame(repeats=2)
            frame["task_id"] = task_id
            frame["family"] = frame["model"]
            frames[task_id] = frame
        frames["task_0"] = frames["task_0"].loc[
            frames["task_0"]["n_train"].isin([20, 40])
        ].copy()
        result = descriptive_suite_summary(
            frames,
            require_complete_task_ids=tuple(sorted(frames)),
            partial_task_anchors={"task_0": (20, 40)},
        )
        self.assertEqual(len(result), 16)
        self.assertEqual(set(result.loc[result["n_train"].isin([20, 40]), "tasks"]), {7})
        self.assertEqual(set(result.loc[result["n_train"].isin([10, 60]), "tasks"]), {6})

        incomplete = {key: value.copy() for key, value in frames.items()}
        incomplete["task_1"] = incomplete["task_1"].loc[
            ~(
                incomplete["task_1"]["model"].eq("cnn")
                & incomplete["task_1"]["n_train"].eq(40)
            )
        ]
        with self.assertRaisesRegex(ValueError, "exact declared"):
            descriptive_suite_summary(
                incomplete,
                require_complete_task_ids=tuple(sorted(incomplete)),
                partial_task_anchors={"task_0": (20, 40)},
            )

    def test_task_products_are_distribution_specific_not_venn_abers_intervals(self) -> None:
        binary_tables = task_analysis_tables(
            self.metrics,
            bootstrap_seed=29,
            bootstrap_draws=40,
            outcome_type="binary",
        )
        self.assertNotIn("coverage_lower_bounds", binary_tables)
        self.assertNotIn("interval_width_contrasts", binary_tables)
        self.assertNotIn("controlled_sharpness", binary_tables)
        # Binary uncertainty is now part of the procedural-stability axis.
        # This synthetic frame exposes calibrated Brier in addition to the
        # three common metrics, yielding four complete model-anchor families.
        self.assertEqual(len(binary_tables["bca_sd_intervals"]), 96)
        self.assertEqual(len(binary_tables["bca_log_sd_ratios"]), 144)
        self.assertFalse(
            any(column.startswith("coverage_") for column in binary_tables["summary"])
        )
        self.assertFalse(
            any(column.startswith("interval_width_") for column in binary_tables["summary"])
        )

        distribution_tables = task_analysis_tables(
            self.metrics,
            bootstrap_seed=29,
            bootstrap_draws=40,
            outcome_type="distribution",
        )
        self.assertIn("coverage_lower_bounds", distribution_tables)
        self.assertIn("interval_width_contrasts", distribution_tables)
        self.assertIn("controlled_sharpness", distribution_tables)
        self.assertEqual(len(distribution_tables["bca_sd_intervals"]), 120)
        self.assertEqual(len(distribution_tables["bca_log_sd_ratios"]), 180)
        hashes = {
            table["bootstrap_index_sha256"].iloc[0]
            for table in (
                distribution_tables["paired_contrasts"],
                distribution_tables["bca_sd_intervals"],
                distribution_tables["bca_log_sd_ratios"],
                distribution_tables["interval_width_contrasts"],
                distribution_tables["coverage_lower_bounds"],
            )
        }
        self.assertEqual(len(hashes), 1)

    def test_pilot_products_are_exactly_descriptive_and_never_emit_inference(self) -> None:
        pilot = _task_frame(repeats=10)
        tables = pilot_task_analysis_tables(pilot, outcome_type="binary")
        self.assertEqual(
            set(tables),
            {
                "summary",
                "paired_stability",
                "stability_ribbons",
                "training_size_changes",
                "training_size_difference_in_changes",
            },
        )
        self.assertEqual(len(tables["summary"]), 24)
        self.assertEqual(set(tables["summary"]["repeats"]), {10})
        for table in tables.values():
            self.assertEqual(
                set(table["evidence_status"]), {"exploratory_provisional"}
            )
            self.assertEqual(
                set(table["inference_scope"]),
                {"descriptive_only_no_confirmatory_inference"},
            )
            self.assertFalse(
                any(
                    column in table
                    for column in (
                        "simultaneous_lower",
                        "simultaneous_upper",
                        "bootstrap_index_sha256",
                        "supported_best",
                    )
                )
            )

    def test_writer_materializes_all_analysis_products_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_task_analysis(
                self.metrics,
                directory,
                bootstrap_seed=17,
                bootstrap_draws=50,
                outcome_type="binary",
            )
            self.assertEqual(
                set(artifacts),
                {
                    "summary",
                    "paired_contrasts",
                    "paired_stability",
                    "stability_ribbons",
                    "training_size_changes",
                    "training_size_difference_in_changes",
                    "bca_sd_intervals",
                    "bca_log_sd_ratios",
                    "supported_best",
                },
            )
            for relative in artifacts.values():
                self.assertTrue((Path(directory) / relative).is_file())
            receipt = json.loads((Path(directory) / "analysis_receipt.json").read_text())
            self.assertEqual(receipt["max_t_family_contrasts"], 36)
            self.assertEqual(receipt["cross_task_inference"], "none")
            self.assertEqual(receipt["bca_stability"]["confidence_level"], 0.95)
            self.assertNotIn("distributional_inference", receipt)


if __name__ == "__main__":
    unittest.main()
