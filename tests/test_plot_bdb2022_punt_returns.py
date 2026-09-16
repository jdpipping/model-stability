from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "plots" / "plot_bdb2022_punt_returns.py"
SPEC = importlib.util.spec_from_file_location("plot_bdb2022_punt_returns", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot import {MODULE_PATH}")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def _fixture() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(PLOT.MODELS):
        for repeat in PLOT.REPEATS:
            for anchor in PLOT.ANCHORS:
                raw = 0.050 - 0.00003 * anchor + 0.001 * model_index + repeat * 1e-7
                skill = 0.02 + 0.00002 * anchor - 0.001 * model_index
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
                        "raw_crps": raw,
                        "game_equal_crps": raw + 0.0002,
                        "coverage": 0.91 + 0.002 * model_index,
                        "interval_width": 30.0 + model_index - 0.005 * anchor,
                        "null_loss": raw / (1.0 - skill),
                        "skill": skill,
                    }
                )
    return pd.DataFrame(rows)


def _write_source(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


class BDB2022PuntReturnsPlotTests(unittest.TestCase):
    def test_loads_exact_checksum_valid_full100_primary_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _write_source(path, _fixture())
            loaded = PLOT.load_primary_metrics(path)
        self.assertEqual(len(loaded), 3_000)
        self.assertEqual(tuple(sorted(loaded["repeat"].unique())), PLOT.REPEATS)
        self.assertEqual(tuple(sorted(loaded["n_train"].unique())), PLOT.ANCHORS)
        np.testing.assert_allclose(
            loaded["skill_pct"].to_numpy(), 100.0 * loaded["skill"].to_numpy()
        )

    def test_rejects_missing_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _write_source(path, _fixture().iloc[:-1])
            with self.assertRaisesRegex(ValueError, "exact 3,000-cell primary grid"):
                PLOT.load_primary_metrics(path)

    def test_rejects_checksum_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _write_source(path, _fixture())
            with path.open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                PLOT.load_primary_metrics(path)

    def test_rejects_invalid_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            frame = _fixture()
            frame.loc[0, "coverage"] = 1.01
            _write_source(path, frame)
            with self.assertRaisesRegex(ValueError, "coverage outside"):
                PLOT.load_primary_metrics(path)

    def test_writes_all_three_views_as_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "primary_metrics.csv"
            _write_source(path, _fixture())
            outputs = PLOT.plot_all(path, root / "figures", prefix="fixture")
            self.assertEqual(len(outputs), 6)
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1_000)
            self.assertEqual(outputs[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(outputs[1].read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
