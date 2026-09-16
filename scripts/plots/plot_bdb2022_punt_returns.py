#!/usr/bin/env python3
"""Plot checksum-validated BDB2022 punt-return full100 learning curves."""

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
TASK_ID = "bdb2022_punt_returns"
ANCHORS = (10, 20, 40, 60, 160, 360)
REPEATS = tuple(range(1, 101))
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


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing checksum sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"malformed checksum sidecar: {sidecar}")
    if _sha256(path) != fields[0]:
        raise ValueError(f"checksum mismatch: {path}")


def load_primary_metrics(path: str | Path) -> pd.DataFrame:
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
        "raw_crps",
        "game_equal_crps",
        "coverage",
        "interval_width",
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
        raise ValueError(f"{source} model registry drifted")
    if tuple(sorted(data["n_train"].astype(int).unique())) != ANCHORS:
        raise ValueError(f"{source} training anchors drifted")
    if tuple(sorted(data["repeat"].astype(int).unique())) != REPEATS:
        raise ValueError(f"{source} repeat IDs are not exactly 1--100")
    if data.duplicated(["model", "repeat", "n_train"]).any():
        raise ValueError(f"{source} contains duplicate grid cells")
    expected = {
        (model, repeat, anchor)
        for model in MODELS
        for repeat in REPEATS
        for anchor in ANCHORS
    }
    observed = set(
        zip(
            data["model"].astype(str),
            data["repeat"].astype(int),
            data["n_train"].astype(int),
            strict=True,
        )
    )
    if len(data) != 3000 or observed != expected:
        raise ValueError(f"{source} is not the exact 3,000-cell primary grid")
    for metric in (
        "primary_loss",
        "raw_crps",
        "game_equal_crps",
        "coverage",
        "interval_width",
        "null_loss",
        "skill",
    ):
        if not np.isfinite(data[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"{source} contains non-finite {metric}")
    coverage = data["coverage"].to_numpy(dtype=float)
    if ((coverage < 0.0) | (coverage > 1.0)).any():
        raise ValueError(f"{source} contains coverage outside [0, 1]")
    if (data["interval_width"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(f"{source} contains negative interval_width")
    if not np.array_equal(
        data["primary_loss"].to_numpy(dtype=float),
        data["raw_crps"].to_numpy(dtype=float),
    ):
        raise ValueError("primary_loss is not exactly raw_crps")
    expected_skill = 1.0 - (
        data["raw_crps"].to_numpy(dtype=float)
        / data["null_loss"].to_numpy(dtype=float)
    )
    if not np.allclose(
        data["skill"].to_numpy(dtype=float), expected_skill, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("skill is not the reduction relative to null_loss")
    result = data.copy()
    result["skill_pct"] = 100.0 * expected_skill
    return result.sort_values(["model", "repeat", "n_train"]).reset_index(drop=True)


def _set_style() -> None:
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
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _handles() -> list[Line2D]:
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


def _save(fig: plt.Figure, stem: Path, title: str) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, metadata={"Software": "Matplotlib", "Title": title})
    fig.savefig(
        pdf,
        metadata={
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "Title": title,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def plot_spaghetti(data: pd.DataFrame, stem: Path) -> tuple[Path, Path]:
    _set_style()
    fig, axes_array = plt.subplots(3, 2, figsize=(9.2, 9.0), sharex=True, sharey=True)
    axes = list(axes_array.ravel())
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.885, hspace=0.30, wspace=0.14)
    fig.suptitle("BDB2022 punt returns: repeat-level learning curves", y=0.978, fontsize=13.0, fontweight="semibold")
    fig.text(0.5, 0.946, "Confirmatory full100 profile (R = 100): thin lines are complete repeats; thick lines are across-repeat means", ha="center", va="top", fontsize=9.2, color="#444444")
    values = data["raw_crps"].to_numpy(dtype=float)
    padding = max(0.001, 0.05 * float(np.ptp(values)))
    for panel, (model, style) in enumerate(MODELS.items()):
        ax = axes[panel]
        matrix = data.loc[data.model == model].pivot(index="repeat", columns="n_train", values="raw_crps").reindex(index=REPEATS, columns=ANCHORS).to_numpy(dtype=float)
        if matrix.shape != (100, 6) or not np.isfinite(matrix).all():
            raise ValueError(f"{model} trajectory matrix is incomplete")
        for row in matrix:
            ax.plot(ANCHORS, row, color=str(style["color"]), linewidth=0.55, alpha=0.105, zorder=1)
        ax.plot(ANCHORS, matrix.mean(axis=0), color=str(style["color"]), linestyle=style["linestyle"], linewidth=2.4, marker=str(style["marker"]), markersize=5.0, markeredgecolor="white", markeredgewidth=0.65, zorder=3)
        ax.set_title(f"({chr(ord('A') + panel)}) {style['label']}", loc="left", fontweight="semibold", pad=5)
        ax.set_xticks(ANCHORS)
        ax.set_ylim(float(values.min() - padding), float(values.max() + padding))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].axis("off")
    axes[-2].tick_params(axis="x", labelrotation=45)
    for label in axes[-2].get_xticklabels():
        label.set_horizontalalignment("right")
    fig.supxlabel("Training games", y=0.035)
    fig.supylabel("Raw test CRPS (lower is better)", x=0.025)
    return _save(fig, stem, "BDB2022 punt-return full100 spaghetti learning curves")


def _ribbons(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    for model, style in MODELS.items():
        grouped = data.loc[data.model == model].groupby("n_train", sort=True)[metric]
        mean = grouped.mean().reindex(ANCHORS)
        q10 = grouped.quantile(0.10).reindex(ANCHORS)
        q90 = grouped.quantile(0.90).reindex(ANCHORS)
        if mean.isna().any() or q10.isna().any() or q90.isna().any():
            raise ValueError(f"{model}/{metric} ribbon is incomplete")
        ax.fill_between(ANCHORS, q10, q90, color=str(style["color"]), alpha=0.14, linewidth=0, zorder=1)
        ax.plot(ANCHORS, mean, color=str(style["color"]), linestyle=style["linestyle"], linewidth=2.2, marker=str(style["marker"]), markersize=5.1, markeredgecolor="white", markeredgewidth=0.65, zorder=3)
    ax.set_xticks(ANCHORS)
    ax.margins(x=0.015, y=0.10)
    ax.spines[["top", "right"]].set_visible(False)


def plot_learning_curves(data: pd.DataFrame, stem: Path) -> tuple[Path, Path]:
    _set_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 8.3), sharex=True)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.085, top=0.805, hspace=0.36)
    fig.suptitle("BDB2022 punt-return confirmatory learning curves", y=0.978, fontsize=13.0, fontweight="semibold")
    fig.text(0.5, 0.946, "Confirmatory full100 profile (R = 100): lines are means; bands are whole-repeat 10th–90th percentiles", ha="center", va="top", fontsize=9.2, color="#444444")
    fig.legend(handles=_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.913), ncol=3, columnspacing=1.55, handlelength=2.8, handletextpad=0.65, fontsize=8.4)
    specs = (
        ("raw_crps", "(A) Raw test CRPS (confirmatory primary)", "CRPS (lower is better)"),
        ("game_equal_crps", "(B) Game-equal test CRPS (secondary)", "Game-equal CRPS"),
        ("skill_pct", "(C) CRPS reduction relative to empirical null", "CRPS reduction vs null (%)"),
    )
    for ax, (metric, title, ylabel) in zip(axes, specs, strict=True):
        _ribbons(ax, data, metric)
        ax.set_title(title, loc="left", fontweight="semibold", pad=5)
        ax.set_ylabel(ylabel)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.1f%%"))
    axes[2].axhline(0.0, color="#555555", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
    axes[2].annotate("empirical-null parity", xy=(ANCHORS[-1], 0.0), xytext=(-3, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8.0, color="#555555")
    axes[-1].tick_params(axis="x", labelrotation=30)
    for label in axes[-1].get_xticklabels():
        label.set_horizontalalignment("right")
    axes[-1].set_xlabel("Training games")
    return _save(fig, stem, "BDB2022 punt-return full100 confirmatory learning curves")


def plot_uncertainty_diagnostics(
    data: pd.DataFrame, stem: Path
) -> tuple[Path, Path]:
    """Render the BDB2020-style width, coverage, and primary-loss view."""

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
        "BDB2022 punt-return conformal diagnostics",
        y=0.978,
        fontsize=13.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.946,
        "Confirmatory full100 profile (R = 100): lines are means; bands are whole-repeat 10th–90th percentiles",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    fig.legend(
        handles=_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.913),
        ncol=3,
        columnspacing=1.55,
        handlelength=2.8,
        handletextpad=0.65,
        fontsize=8.4,
    )
    specs = (
        (
            "interval_width",
            "(A) Mean 90% conformal interval width",
            "Interval width (return yards)",
        ),
        ("coverage", "(B) Empirical coverage", "Coverage"),
        (
            "raw_crps",
            "(C) Raw test CRPS (confirmatory primary)",
            "CRPS (lower is better)",
        ),
    )
    for ax, (metric, title, ylabel) in zip(axes, specs, strict=True):
        _ribbons(ax, data, metric)
        ax.set_title(title, loc="left", fontweight="semibold", pad=5)
        ax.set_ylabel(ylabel)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[1].axhline(
        0.90,
        color="#555555",
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        zorder=0,
    )
    axes[1].annotate(
        "90% target",
        xy=(ANCHORS[-1], 0.90),
        xytext=(-3, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    axes[-1].tick_params(axis="x", labelrotation=30)
    for label in axes[-1].get_xticklabels():
        label.set_horizontalalignment("right")
    axes[-1].set_xlabel("Training games")
    return _save(fig, stem, "BDB2022 punt-return full100 conformal diagnostics")


def plot_all(
    source: str | Path, output_dir: str | Path, *, prefix: str
) -> tuple[Path, ...]:
    data = load_primary_metrics(source)
    destination = Path(output_dir)
    return (
        *plot_spaghetti(data, destination / f"{prefix}_spaghetti"),
        *plot_learning_curves(data, destination / f"{prefix}_learning_curves"),
        *plot_uncertainty_diagnostics(
            data, destination / f"{prefix}_uncertainty_diagnostics"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="bdb2022_punt_returns_full100")
    args = parser.parse_args()
    outputs = plot_all(args.input, args.output_dir, prefix=args.prefix)
    for path in outputs:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
