"""Render checksum-traceable plots from a completed BDB final directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


MODEL_ORDER = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
MODEL_LABELS = {
    "linear_structure": "Linear structure",
    "boosted_structure": "Boosted structure",
    "relnet": "RelNet",
    "attn_relnet": "Attn-RelNet",
    "set_transformer": "Set Transformer",
}
MODEL_COLORS = {
    "linear_structure": "#4C78A8",
    "boosted_structure": "#F58518",
    "relnet": "#54A24B",
    "attn_relnet": "#E45756",
    "set_transformer": "#B279A2",
}


def _require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _read_tables(final_dir: Path) -> dict[str, pd.DataFrame]:
    names = (
        "stability_ribbons",
        "paired_contrasts",
        "supported_best",
        "structural_ablation_effects",
    )
    tables = {name: pd.read_csv(final_dir / f"{name}.csv") for name in names}
    _require_columns(
        tables["stability_ribbons"],
        {"model", "n_train", "metric", "mean", "ribbon_lower", "ribbon_upper"},
        "stability ribbons",
    )
    _require_columns(
        tables["paired_contrasts"],
        {
            "n_train",
            "model_left",
            "model_right",
            "mean_difference",
            "simultaneous_lower",
            "simultaneous_upper",
        },
        "paired contrasts",
    )
    _require_columns(
        tables["supported_best"],
        {"n_train", "model", "supported_best"},
        "supported-best table",
    )
    _require_columns(
        tables["structural_ablation_effects"],
        {
            "model",
            "n_train",
            "metric",
            "mean_difference",
            "difference_q10",
            "difference_q90",
        },
        "structural-ablation effects",
    )
    return tables


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBFC",
            "axes.edgecolor": "#B8BDC7",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E3E6EB",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def _plot_ribbons(
    ax: plt.Axes,
    ribbons: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    metric_rows = ribbons[ribbons["metric"] == metric]
    for model in MODEL_ORDER:
        rows = metric_rows[metric_rows["model"] == model].sort_values("n_train")
        if rows.empty:
            continue
        x = rows["n_train"].to_numpy(dtype=float)
        mean = rows["mean"].to_numpy(dtype=float)
        lower = rows["ribbon_lower"].to_numpy(dtype=float)
        upper = rows["ribbon_upper"].to_numpy(dtype=float)
        color = MODEL_COLORS[model]
        ax.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
            label=MODEL_LABELS[model],
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Training seasons")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(metric_rows["n_train"].unique()))


def _plot_contrasts(ax: plt.Axes, contrasts: pd.DataFrame, *, anchor: int) -> None:
    rows = contrasts[contrasts["n_train"] == anchor].copy()
    rows["label"] = rows.apply(
        lambda row: (
            f"{MODEL_LABELS.get(row['model_left'], row['model_left'])} − "
            f"{MODEL_LABELS.get(row['model_right'], row['model_right'])}"
        ),
        axis=1,
    )
    rows = rows.sort_values("mean_difference").reset_index(drop=True)
    y = np.arange(len(rows))
    mean = rows["mean_difference"].to_numpy(dtype=float)
    lower = rows["simultaneous_lower"].to_numpy(dtype=float)
    upper = rows["simultaneous_upper"].to_numpy(dtype=float)
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack((mean - lower, upper - mean)),
        fmt="o",
        color="#394B59",
        ecolor="#8A98A6",
        elinewidth=1.4,
        capsize=2.5,
    )
    ax.axvline(0, color="#C44E52", linewidth=1.2, linestyle="--")
    ax.set_yticks(y, rows["label"])
    ax.set_xlabel("Mean primary-loss difference (left − right)")
    ax.set_title(
        f"Anchor {anchor}: simultaneous 95% paired contrasts",
        loc="left",
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)


def _plot_ablation(
    ax: plt.Axes,
    effects: pd.DataFrame,
    *,
    metric: str,
    title: str,
) -> None:
    rows = effects[effects["metric"] == metric].copy()
    rows = rows.sort_values(["model", "n_train"]).reset_index(drop=True)
    labels = [
        f"{MODEL_LABELS.get(row.model, row.model)}, n={int(row.n_train)}"
        for row in rows.itertuples()
    ]
    y = np.arange(len(rows))
    mean = rows["mean_difference"].to_numpy(dtype=float)
    lower = rows["difference_q10"].to_numpy(dtype=float)
    upper = rows["difference_q90"].to_numpy(dtype=float)
    colors = [MODEL_COLORS.get(model, "#394B59") for model in rows["model"]]
    for index in range(len(rows)):
        ax.errorbar(
            mean[index],
            y[index],
            xerr=np.array(
                [[mean[index] - lower[index]], [upper[index] - mean[index]]]
            ),
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            elinewidth=1.6,
            capsize=3,
        )
    ax.axvline(0, color="#6B7280", linewidth=1.2, linestyle="--")
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value * 1_000:.2f}"))
    ax.set_xlabel("Collapsed typed edges − fixed main (×10⁻³)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "zoo-bdb"})
    plt.close(fig)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def render(final_dir: Path, output_dir: Path) -> list[Path]:
    tables = _read_tables(final_dir)
    _style()
    ribbons = tables["stability_ribbons"]
    contrasts = tables["paired_contrasts"]
    supported = tables["supported_best"]
    ablations = tables["structural_ablation_effects"]
    anchor = int(contrasts["n_train"].max())
    supported_count = int(supported["supported_best"].astype(bool).sum())

    outputs: list[Path] = []
    for metric, title, ylabel, filename in (
        (
            "primary_loss",
            "Primary loss across training sizes",
            "Primary loss (lower is better)",
            "primary_loss_by_training_size.png",
        ),
        (
            "skill",
            "Null-relative skill across training sizes",
            "Skill (higher is better)",
            "skill_by_training_size.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        _plot_ribbons(ax, ribbons, metric=metric, title=title, ylabel=ylabel)
        ax.legend(ncol=3, loc="best")
        fig.text(
            0.01,
            0.01,
            "Lines are means; shaded bands are between-repeat q10–q90 ribbons (100 repeats).",
            color="#5F6B7A",
            fontsize=8.5,
        )
        path = output_dir / filename
        _save(fig, path)
        outputs.append(path)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    _plot_contrasts(ax, contrasts, anchor=anchor)
    fig.text(
        0.01,
        0.01,
        f"Familywise simultaneous intervals; supported-best declarations: {supported_count}.",
        color="#5F6B7A",
        fontsize=8.5,
    )
    path = output_dir / "anchor60_simultaneous_contrasts.png"
    _save(fig, path)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    _plot_ablation(
        axes[0],
        ablations,
        metric="primary_loss",
        title="Typed-edge collapse: primary loss",
    )
    _plot_ablation(
        axes[1],
        ablations,
        metric="calibrated_brier",
        title="Typed-edge collapse: calibrated Brier",
    )
    fig.suptitle(
        "BDB2023 structural ablation (secondary descriptive analysis)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.text(
        0.01,
        0.01,
        "Points are within-repeat mean differences; bars are q10–q90. Zero means no observed change.",
        color="#5F6B7A",
        fontsize=8.5,
    )
    path = output_dir / "structural_ablation_effects.png"
    _save(fig, path)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    _plot_ribbons(
        axes[0, 0],
        ribbons,
        metric="primary_loss",
        title="A. Primary loss",
        ylabel="Primary loss (lower is better)",
    )
    _plot_ribbons(
        axes[0, 1],
        ribbons,
        metric="skill",
        title="B. Null-relative skill",
        ylabel="Skill (higher is better)",
    )
    _plot_contrasts(axes[1, 0], contrasts, anchor=anchor)
    _plot_ablation(
        axes[1, 1],
        ablations,
        metric="primary_loss",
        title="D. Typed-edge collapse",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "BDB2023 Sack — full100 confirmatory results",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.35)
    fig.text(
        0.02,
        0.01,
        (
            "Ribbons and ablation bars are descriptive q10–q90 ranges. "
            f"Simultaneous supported-best declarations: {supported_count}."
        ),
        color="#5F6B7A",
        fontsize=9,
    )
    path = output_dir / "bdb2023_results_dashboard.png"
    _save(fig, path)
    outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    for path in render(arguments.final_dir.resolve(), arguments.output_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
