#!/usr/bin/env python3
"""Build the publication figure for the 100-repeat rushing study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "results" / "rushing" / "full100-confirmatory-20260820"
ANCHORS = (20, 40, 80, 160, 240, 360)
REPEATS = tuple(range(1, 101))

# Okabe-Ito colors plus a muted purple. Markers and line styles preserve the
# model identities in grayscale and at abstract-column scale.
MODELS = {
    "ridge_sgd_l2": {
        "label": "L2 one-vs-rest logistic",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "lightgbm_multiclass": {
        "label": "LightGBM",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
    },
    "zoo_cnn": {
        "label": "Zoo CNN",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
    },
    "set_transformer": {
        "label": "Set Transformer",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum_sidecar(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        digest, filename = sidecar.read_text(encoding="ascii").strip().split(
            maxsplit=1
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"missing or malformed checksum sidecar: {sidecar}") from exc
    if filename != path.name or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"invalid checksum sidecar: {sidecar}")
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError(f"checksummed publication artifact drifted: {path}")
    return digest


def _validate_final_aggregate_binding(run_dir: Path) -> None:
    """Require the copied Betty final marker and its exact metrics binding."""

    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "final" / "metrics.csv"
    analysis_path = run_dir / "final" / "main" / "analysis_manifest.json"
    success_path = run_dir / "_SUCCESS"
    _validate_checksum_sidecar(manifest_path)
    metrics_sha256 = _validate_checksum_sidecar(metrics_path)
    _validate_checksum_sidecar(analysis_path)
    _validate_checksum_sidecar(success_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(success_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("publication run lacks a readable final binding") from exc
    if not all(isinstance(value, dict) for value in (manifest, marker, analysis)):
        raise ValueError("publication final binding must contain JSON objects")
    if marker.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("publication final marker belongs to a different manifest")
    if marker.get("schema_version") != 1 or marker.get("cell_count") != len(
        marker.get("cells", ())
    ):
        raise ValueError("publication final marker has an invalid storage contract")
    expected_main = {
        (repeat, anchor, model)
        for repeat in REPEATS
        for anchor in ANCHORS
        for model in MODELS
    }
    try:
        observed_main = {
            (
                int(entry["key"]["repeat"]),
                int(entry["key"]["n_train"]),
                str(entry["key"]["model"]),
            )
            for entry in marker["cells"]
            if entry["key"]["branch"] == "main"
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("publication final marker cell index is malformed") from exc
    if observed_main != expected_main:
        raise ValueError("publication final marker does not bind the exact full100 main grid")
    indexed = {
        str(record.get("file")): str(record.get("sha256"))
        for record in marker.get("final_artifacts", [])
        if isinstance(record, dict)
    }
    if indexed.get("final/metrics.csv") != metrics_sha256:
        raise ValueError("publication metrics are not indexed by finalization")
    if (
        tuple(analysis.get("models", ())) != tuple(MODELS)
        or tuple(analysis.get("sizes", ())) != ANCHORS
        or tuple(analysis.get("repeat_ids", ())) != REPEATS
        or analysis.get("bootstrap_unit") != "whole repeat vector"
    ):
        raise ValueError("publication analysis manifest differs from the full100 contract")


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
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load(run_dir: Path) -> pd.DataFrame:
    _validate_final_aggregate_binding(run_dir)
    path = run_dir / "final" / "metrics.csv"
    data = pd.read_csv(path)
    required = {"model", "repeat", "n_train", "crps", "coverage", "mean_width"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    data = data.loc[
        data["model"].isin(MODELS) & data["n_train"].isin(ANCHORS)
    ].copy()
    expected_rows = len(MODELS) * len(ANCHORS) * len(REPEATS)
    if len(data) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(data)}")
    if data.duplicated(["model", "repeat", "n_train"]).any():
        raise ValueError(f"{path}: duplicate model-repeat-size rows")
    if set(data["model"]) != set(MODELS):
        raise ValueError(f"{path}: model registry does not match plotting registry")
    if tuple(sorted(data["n_train"].unique())) != ANCHORS:
        raise ValueError(f"{path}: training-size anchors do not match")
    if tuple(sorted(data["repeat"].unique())) != REPEATS:
        raise ValueError(f"{path}: repeat IDs are not exactly 1--100")
    for metric in ("crps", "coverage", "mean_width"):
        if not np.isfinite(data[metric]).all():
            raise ValueError(f"{path}: non-finite {metric} values")
    return data.sort_values(["model", "repeat", "n_train"]).reset_index(drop=True)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            markersize=5.0,
            label=style["label"],
        )
        for style in MODELS.values()
    ]


def _draw_ribbons(ax: plt.Axes, data: pd.DataFrame, metric: str) -> None:
    for model, style in MODELS.items():
        model_data = data.loc[data["model"] == model]
        grouped = model_data.groupby("n_train", sort=True)[metric]
        means = grouped.mean().reindex(ANCHORS)
        q10 = grouped.quantile(0.10).reindex(ANCHORS)
        q90 = grouped.quantile(0.90).reindex(ANCHORS)
        ax.fill_between(
            ANCHORS,
            q10.to_numpy(),
            q90.to_numpy(),
            color=style["color"],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            ANCHORS,
            means.to_numpy(),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            marker=style["marker"],
            markersize=5.1,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )
    ax.set_xticks(ANCHORS)
    ax.margins(x=0.015, y=0.10)
    ax.spines[["top", "right"]].set_visible(False)


def plot(run_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    _set_style()
    data = _load(run_dir)
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 8.15), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.085, top=0.825, hspace=0.34)
    fig.suptitle(
        "Rushing-yards confirmatory learning curves",
        x=0.5,
        y=0.975,
        fontsize=13.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.943,
        "Confirmatory full100 profile (R = 100): lines are means; bands are whole-rerun 10th--90th percentiles",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.916),
        ncol=2,
        columnspacing=2.0,
        handlelength=2.8,
        handletextpad=0.7,
        fontsize=8.7,
    )

    specifications = (
        ("mean_width", "(A) Mean conformal interval width", "Width (yards; upper index - lower index)"),
        ("coverage", "(B) Empirical coverage", "Coverage"),
        ("crps", "(C) Test CRPS", "CRPS (lower is better)"),
    )
    for ax, (metric, title, ylabel) in zip(axes, specifications, strict=True):
        _draw_ribbons(ax, data, metric)
        ax.set_title(title, loc="left", fontweight="semibold", pad=5)
        ax.set_ylabel(ylabel)

    axes[1].axhline(0.90, color="#555555", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
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
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[-1].set_xlabel("Training games")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    title = "BDB2020 rushing-yards 100-repeat confirmatory learning curves"
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
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    output_stem = args.output_stem or (
        args.run_dir / "final" / "main" / "rushing_confirmatory_100_learning_curves"
    )
    png_path, pdf_path = plot(args.run_dir.resolve(), output_stem.resolve())
    print(f"wrote {png_path.relative_to(ROOT)}")
    print(f"wrote {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
