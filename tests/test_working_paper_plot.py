from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "plots" / "plot_rushing_confirmatory.py"
SPEC = importlib.util.spec_from_file_location("plot_rushing_confirmatory", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot import {MODULE_PATH}")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def _write_checksummed(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


class PublicationPlotBindingTests(unittest.TestCase):
    def _fixture(self, run_dir: Path) -> Path:
        manifest_hash = "a" * 64
        _write_checksummed(
            run_dir / "manifest.json",
            json.dumps({"manifest_hash": manifest_hash}).encode("utf-8"),
        )
        metrics_path = run_dir / "final" / "metrics.csv"
        metrics_digest = _write_checksummed(metrics_path, b"model,repeat,n_train,crps\n")
        analysis = {
            "models": list(PLOT.MODELS),
            "sizes": list(PLOT.ANCHORS),
            "repeat_ids": list(PLOT.REPEATS),
            "bootstrap_unit": "whole repeat vector",
        }
        _write_checksummed(
            run_dir / "final" / "main" / "analysis_manifest.json",
            json.dumps(analysis).encode("utf-8"),
        )
        cells = [
            {
                "key": {
                    "branch": "main",
                    "repeat": repeat,
                    "n_train": anchor,
                    "model": model,
                },
                "path": f"cells/main/{repeat}/{anchor}/{model}",
                "marker_sha256": "b" * 64,
            }
            for repeat in PLOT.REPEATS
            for anchor in PLOT.ANCHORS
            for model in PLOT.MODELS
        ]
        marker = {
            "schema_version": 1,
            "manifest_hash": manifest_hash,
            "cell_count": len(cells),
            "cells": cells,
            "final_artifacts": [
                {"file": "final/metrics.csv", "sha256": metrics_digest}
            ],
        }
        _write_checksummed(
            run_dir / "_SUCCESS", json.dumps(marker).encode("utf-8")
        )
        return metrics_path

    def test_accepts_exact_checksummed_full100_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._fixture(run_dir)
            PLOT._validate_final_aggregate_binding(run_dir)

    def test_rejects_shaped_metrics_after_payload_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics_path = self._fixture(run_dir)
            metrics_path.write_bytes(metrics_path.read_bytes() + b"tampered,row\n")
            with self.assertRaisesRegex(ValueError, "artifact drifted"):
                PLOT._validate_final_aggregate_binding(run_dir)

    def test_rejects_final_marker_after_grid_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._fixture(run_dir)
            marker_path = run_dir / "_SUCCESS"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["cells"].pop()
            marker["cell_count"] -= 1
            _write_checksummed(marker_path, json.dumps(marker).encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "exact full100 main grid"):
                PLOT._validate_final_aggregate_binding(run_dir)


if __name__ == "__main__":
    unittest.main()
