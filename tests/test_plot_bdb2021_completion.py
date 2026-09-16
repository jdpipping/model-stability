from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "plots" / "plot_bdb2021_completion.py"
SPEC = importlib.util.spec_from_file_location("plot_bdb2021_completion", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot import {MODULE_PATH}")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def _fixture() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(PLOT.MODELS):
        for repeat in PLOT.REPEATS:
            for anchor in PLOT.ANCHORS:
                raw = 0.24 - 0.0001 * anchor + 0.001 * model_index + repeat * 1e-6
                skill = 0.02 + 0.0001 * anchor - 0.001 * model_index
                rows.append(
                    {
                        "task_id": PLOT.TASK_ID,
                        "branch": "fixed_main",
                        "ablation_id": np.nan,
                        "sensitivity_id": np.nan,
                        "model": model,
                        "repeat": repeat,
                        "n_train": anchor,
                        "primary_loss": raw,
                        "raw_brier": raw,
                        "calibrated_brier": raw - 0.001,
                        "null_loss": raw / (1.0 - skill),
                        "skill": skill,
                    }
                )
    return pd.DataFrame(rows)


class BDB2021CompletionPlotTests(unittest.TestCase):
    def test_loads_exact_full100_primary_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _fixture().to_csv(path, index=False)
            loaded = PLOT.load_primary_metrics(path)
        self.assertEqual(len(loaded), 3_000)
        self.assertEqual(tuple(sorted(loaded["repeat"].unique())), PLOT.REPEATS)
        self.assertEqual(tuple(sorted(loaded["n_train"].unique())), PLOT.ANCHORS)
        np.testing.assert_allclose(
            loaded["brier_reduction_pct"].to_numpy(),
            100.0 * loaded["skill"].to_numpy(),
        )

    def test_rejects_missing_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _fixture().iloc[:-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exact 3,000-cell grid"):
                PLOT.load_primary_metrics(path)

    def test_rejects_primary_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            frame = _fixture()
            frame.loc[0, "primary_loss"] += 1e-8
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "byte-equivalent raw_brier"):
                PLOT.load_primary_metrics(path)

    def test_rejects_null_reduction_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            frame = _fixture()
            frame.loc[0, "null_loss"] += 1e-4
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "reduction relative to null_loss"):
                PLOT.load_primary_metrics(path)

    def test_writes_both_figures_as_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "primary_metrics.csv"
            _fixture().to_csv(path, index=False)
            outputs = PLOT.plot_all(path, root / "figures", prefix="fixture")
            self.assertEqual(len(outputs), 4)
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(outputs[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(outputs[1].read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
