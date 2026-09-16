from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "plots" / "plot_bdb2023_sack.py"
SPEC = importlib.util.spec_from_file_location("plot_bdb2023_sack", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot import {MODULE_PATH}")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def _fixture() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(PLOT.MODELS):
        for repeat in PLOT.REPEATS:
            for anchor in PLOT.ANCHORS:
                raw = 0.064 - 0.00002 * anchor + 0.0005 * model_index + repeat * 1e-7
                skill = 0.01 + 0.0001 * anchor - 0.002 * model_index
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
                        "calibrated_brier": raw - 0.0002,
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


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    assert fields == [hashlib.sha256(path.read_bytes()).hexdigest(), path.name]


class BDB2023SackPlotTests(unittest.TestCase):
    def test_loads_exact_checksum_valid_full100_primary_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            _write_source(path, _fixture())
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

    def test_rejects_primary_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            frame = _fixture()
            frame.loc[0, "primary_loss"] += 1e-8
            _write_source(path, frame)
            with self.assertRaisesRegex(ValueError, "exactly raw_brier"):
                PLOT.load_primary_metrics(path)

    def test_rejects_null_reduction_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "primary_metrics.csv"
            frame = _fixture()
            frame.loc[0, "null_loss"] += 1e-4
            _write_source(path, frame)
            with self.assertRaisesRegex(ValueError, "reduction relative to null_loss"):
                PLOT.load_primary_metrics(path)

    def test_writes_single_three_panel_figure_with_checksum_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "primary_metrics.csv"
            _write_source(path, _fixture())
            outputs = PLOT.plot_all(path, root / "figures", prefix="fixture")
            self.assertEqual(len(outputs), 4)
            png, pdf, png_sidecar, pdf_sidecar = outputs
            self.assertEqual(png.name, "fixture_learning_curves.png")
            self.assertEqual(pdf.name, "fixture_learning_curves.pdf")
            self.assertEqual(png_sidecar.name, png.name + ".sha256")
            self.assertEqual(pdf_sidecar.name, pdf.name + ".sha256")
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 50)
            self.assertGreater(png.stat().st_size, 1_000)
            self.assertGreater(pdf.stat().st_size, 1_000)
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
            _verify_sidecar(png)
            _verify_sidecar(pdf)


if __name__ == "__main__":
    unittest.main()
