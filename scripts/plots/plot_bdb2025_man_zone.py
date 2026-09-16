#!/usr/bin/env python3
"""Plot checksum-validated BDB2025 Man-versus-Zone full100 learning curves."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "scripts" / "plots" / "plot_bdb2023_sack.py"
SPEC = importlib.util.spec_from_file_location("plot_bdb2023_sack_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

TASK_ID = "bdb2025_man_zone"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "bdb" / "full100_serial_bdb2025_attempt1" / "plots"
)


def load_primary_metrics(path: str | Path) -> pd.DataFrame:
    """Validate the exact checksum-bound 3,000-cell BDB2025 primary grid."""

    original_task_id = BASE.TASK_ID
    BASE.TASK_ID = TASK_ID
    try:
        return BASE.load_primary_metrics(path)
    finally:
        BASE.TASK_ID = original_task_id


def plot_learning_curves(data: pd.DataFrame, output_stem: Path):
    """Render the established three-panel full100 sweep plot."""

    BASE._set_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 8.3), sharex=True)
    fig.subplots_adjust(
        left=0.135,
        right=0.985,
        bottom=0.085,
        top=0.805,
        hspace=0.36,
    )
    fig.suptitle(
        "BDB2025 pre-snap Man-versus-Zone confirmatory learning curves",
        x=0.5,
        y=0.978,
        fontsize=13.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.946,
        (
            "Confirmatory full100 profile (R = 100): lines are means; "
            "bands are whole-repeat 10th--90th percentiles"
        ),
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    fig.legend(
        handles=BASE._legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.913),
        ncol=3,
        columnspacing=1.55,
        handlelength=2.8,
        handletextpad=0.65,
        fontsize=8.4,
    )

    specifications = (
        (
            "raw_brier",
            "(A) Raw test Brier score (confirmatory primary)",
            "Brier score (lower is better)",
        ),
        (
            "calibrated_brier",
            "(B) Venn--Abers calibrated test Brier score (secondary)",
            "Calibrated Brier (lower is better)",
        ),
        (
            "brier_reduction_pct",
            "(C) Brier reduction relative to base-rate null",
            "Brier reduction vs base-rate null (%)",
        ),
    )
    for ax, (metric, title, ylabel) in zip(axes, specifications, strict=True):
        BASE._draw_ribbons(ax, data, metric)
        ax.set_title(title, loc="left", fontweight="semibold", pad=5)
        ax.set_ylabel(ylabel)

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.1f%%"))
    axes[2].axhline(
        0.0,
        color="#555555",
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        zorder=0,
    )
    axes[2].annotate(
        "base-rate null parity",
        xy=(BASE.ANCHORS[-1], 0.0),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    axes[-1].set_xlabel("Training games")
    return BASE._save_figure(
        fig,
        output_stem,
        "BDB2025 Man-versus-Zone 100-repeat confirmatory learning curves",
    )


def plot_all(input_path: str | Path, output_dir: str | Path):
    data = load_primary_metrics(input_path)
    destination = Path(output_dir)
    return plot_learning_curves(
        data,
        destination / "bdb2025_man_zone_full100_learning_curves",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in plot_all(args.input.resolve(), args.output_dir.resolve()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
