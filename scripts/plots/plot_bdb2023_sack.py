#!/usr/bin/env python3
"""Plot checksum-validated BDB2023 sack full100 learning curves.

The input is the aggregate's exact 3,000-cell ``primary_metrics.csv`` and its
adjacent SHA-256 sidecar.  The output is the established BDB2020-style
three-panel mean/q10--q90 ribbon figure, written as PNG and PDF with SHA-256
sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "figures"
TASK_ID = "bdb2023_sack"
ANCHORS = (10, 20, 30, 40, 50, 60)
REPEATS = tuple(range(1, 101))

# Okabe-Ito colors plus a muted purple.  Markers and line styles preserve
# model identity in grayscale and match the established BDB2020 figure.
MODELS: Mapping[str, Mapping[str, object]] = {
    "linear_structure": {
        "label": "Linear-Structure",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "boosted_structure": {
        "label": "Boosted-Structure",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
    },
    "relnet": {
        "label": "RelNet",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
    },
    "attn_relnet": {
        "label": "AttnRelNet",
        "color": "#D55E00",
        "marker": "v",
        "linestyle": (0, (5, 2, 1, 2)),
    },
    "set_transformer": {
        "label": "Global Set Transformer",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _write_sidecar(path: Path) -> Path:
    sidecar = _sidecar_path(path)
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _verify_sidecar(path: Path) -> None:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        raise ValueError(f"missing checksum sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"malformed checksum sidecar: {sidecar}")
    if _sha256(path) != fields[0]:
        raise ValueError(f"checksum mismatch: {path}")


def load_primary_metrics(path: str | Path) -> pd.DataFrame:
    """Fail closed unless ``path`` is the exact BDB2023 full100 primary grid."""

    source = Path(path).resolve()
    _verify_sidecar(source)
    data = pd.read_csv(source)
    required = {
        "task_id",
        "branch",
        "ablation_id",
        "sensitivity_id",
        "model",
        "repeat",
        "n_train",
        "primary_loss",
        "raw_brier",
        "calibrated_brier",
        "null_loss",
        "skill",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    if set(data["task_id"].astype(str)) != {TASK_ID}:
        raise ValueError(f"{source} is not exclusively {TASK_ID}")
    if set(data["branch"].astype(str)) != {"fixed_main"}:
        raise ValueError(f"{source} is not exclusively the fixed_main branch")
    if data["ablation_id"].notna().any() or data["sensitivity_id"].notna().any():
        raise ValueError(f"{source} contains non-primary rows")
    if set(data["model"].astype(str)) != set(MODELS):
        raise ValueError(f"{source} model registry does not match BDB2023 full100")
    if tuple(sorted(data["n_train"].astype(int).unique())) != ANCHORS:
        raise ValueError(f"{source} training-size anchors are not exact")
    if tuple(sorted(data["repeat"].astype(int).unique())) != REPEATS:
        raise ValueError(f"{source} repeat IDs are not exactly 1--100")
    if data.duplicated(["model", "repeat", "n_train"]).any():
        raise ValueError(f"{source} contains duplicate model-repeat-anchor rows")

    expected_keys = {
        (model, repeat, anchor)
        for model in MODELS
        for repeat in REPEATS
        for anchor in ANCHORS
    }
    observed_keys = set(
        zip(
            data["model"].astype(str),
            data["repeat"].astype(int),
            data["n_train"].astype(int),
            strict=True,
        )
    )
    if observed_keys != expected_keys or len(data) != 3_000:
        raise ValueError(f"{source} is not the exact 3,000-cell primary grid")

    for metric in (
        "primary_loss",
        "raw_brier",
        "calibrated_brier",
        "null_loss",
        "skill",
    ):
        values = data[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains non-finite {metric}")
    for metric in ("primary_loss", "raw_brier", "calibrated_brier", "null_loss"):
        if not data[metric].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{source} contains out-of-range {metric}")
    if not (data["null_loss"] > 0.0).all():
        raise ValueError(f"{source} contains non-positive null_loss")
    if not np.array_equal(
        data["primary_loss"].to_numpy(dtype=float),
        data["raw_brier"].to_numpy(dtype=float),
    ):
        raise ValueError(f"{source} primary_loss is not exactly raw_brier")
    expected_skill = 1.0 - (
        data["raw_brier"].to_numpy(dtype=float)
        / data["null_loss"].to_numpy(dtype=float)
    )
    if not np.allclose(
        data["skill"].to_numpy(dtype=float),
        expected_skill,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(f"{source} skill is not the reduction relative to null_loss")

    result = data.copy()
    result["brier_reduction_pct"] = 100.0 * expected_skill
    return result.sort_values(["model", "repeat", "n_train"]).reset_index(drop=True)


def _set_style() -> None:
    """Apply the established BDB2020 publication style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=str(style["color"]),
            marker=str(style["marker"]),
            linestyle=style["linestyle"],
            linewidth=2.1,
            markersize=5.0,
            label=str(style["label"]),
        )
        for style in MODELS.values()
    ]


def _draw_ribbons(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    for model, style in MODELS.items():
        grouped = data.loc[data["model"] == model].groupby("n_train", sort=True)[
            metric
        ]
        means = grouped.mean().reindex(ANCHORS)
        q10 = grouped.quantile(0.10).reindex(ANCHORS)
        q90 = grouped.quantile(0.90).reindex(ANCHORS)
        if means.isna().any() or q10.isna().any() or q90.isna().any():
            raise ValueError(f"{model}/{metric} ribbon statistics are incomplete")
        ax.fill_between(
            ANCHORS,
            q10.to_numpy(dtype=float),
            q90.to_numpy(dtype=float),
            color=str(style["color"]),
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            ANCHORS,
            means.to_numpy(dtype=float),
            color=str(style["color"]),
            linestyle=style["linestyle"],
            linewidth=2.2,
            marker=str(style["marker"]),
            markersize=5.1,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )
    ax.set_xticks(ANCHORS)
    ax.margins(x=0.015, y=0.10)
    ax.spines[["top", "right"]].set_visible(False)


def _save_figure(
    fig: plt.Figure, output_stem: Path, title: str
) -> tuple[Path, Path, Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "Matplotlib", "Title": title},
    )
    fig.savefig(
        pdf_path,
        metadata={
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "Title": title,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    png_sidecar = _write_sidecar(png_path)
    pdf_sidecar = _write_sidecar(pdf_path)
    return png_path, pdf_path, png_sidecar, pdf_sidecar


def plot_learning_curves(
    data: pd.DataFrame, output_stem: Path
) -> tuple[Path, Path, Path, Path]:
    """Plot BDB2020-style means and whole-repeat q10--q90 ribbons."""

    _set_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 8.3), sharex=True)
    fig.subplots_adjust(
        left=0.135,
        right=0.985,
        bottom=0.085,
        top=0.805,
        hspace=0.36,
    )
    fig.suptitle(
        "BDB2023 sack probability confirmatory learning curves",
        x=0.5,
        y=0.978,
        fontsize=13.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.946,
        "Confirmatory full100 profile (R = 100): lines are means; bands are whole-repeat 10th--90th percentiles",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    fig.legend(
        handles=_legend_handles(),
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
        _draw_ribbons(ax, data, metric)
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
        xy=(ANCHORS[-1], 0.0),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    axes[-1].set_xlabel("Training games")
    return _save_figure(
        fig,
        output_stem,
        "BDB2023 sack 100-repeat confirmatory learning curves",
    )


def plot_all(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str = "bdb2023_sack_full100",
) -> tuple[Path, Path, Path, Path]:
    data = load_primary_metrics(input_path)
    destination = Path(output_dir)
    return plot_learning_curves(data, destination / f"{prefix}_learning_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to checksum-validated BDB2023 primary_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        default="bdb2023_sack_full100",
        help="Output filename prefix",
    )
    args = parser.parse_args()
    for path in plot_all(
        args.input.resolve(), args.output_dir.resolve(), prefix=str(args.prefix)
    ):
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
