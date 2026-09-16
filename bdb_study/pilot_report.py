"""Checksum-gated figures and report for the six-task ``pilot10`` aggregate.

This module is deliberately narrower than the definitive suite analysis.  It
accepts only a finalized ``bdb-pilot-suite-aggregate-v1`` receipt and the two
CSV artifacts bound by that receipt.  It then independently revalidates the
complete 1,800-cell grid and recomputes the task-equal descriptive summary
before producing any visible result.

The generated material is an implementation-pilot mock-up.  It never computes
cross-task uncertainty, hypothesis tests, supported-best decisions, or a
model-winner declaration.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from .analysis import descriptive_suite_summary, validate_task_grid
from .storage import ArtifactRecord, atomic_write_json, register_existing_artifact


PILOT_PROFILE = "pilot10"
PILOT_REPEATS = tuple(range(1, 11))
COMMON_ANCHORS = (10, 20, 40, 60)
TASKS = (
    "bdb2021_completion",
    "bdb2022_punt_returns",
    "bdb2023_sack",
    "bdb2024_tackle",
    "bdb2025_man_zone",
    "bdb2026_trajectory",
)
TASK_ANCHORS = {
    "bdb2021_completion": (10, 20, 40, 60, 100, 130),
    "bdb2022_punt_returns": (10, 20, 40, 60, 160, 360),
    "bdb2023_sack": (10, 20, 30, 40, 50, 60),
    "bdb2024_tackle": (10, 20, 30, 40, 50, 60),
    "bdb2025_man_zone": (10, 20, 30, 40, 50, 60),
    "bdb2026_trajectory": (10, 20, 40, 60, 100, 140),
}
TASK_LABELS = {
    "bdb2021_completion": "2021 Completion — Brier skill",
    "bdb2022_punt_returns": "2022 Punt returns — CRPS skill",
    "bdb2023_sack": "2023 Sacks — Brier skill",
    "bdb2024_tackle": "2024 Tackles — case-control Brier skill",
    "bdb2025_man_zone": "2025 Man/zone — Brier skill",
    "bdb2026_trajectory": "2026 Trajectories — RMSE skill",
}
MODELS = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
MODEL_FAMILIES = {
    "linear_structure": "glm",
    "boosted_structure": "lightgbm",
    "relnet": "relnet",
    "attn_relnet": "attn_relnet",
    "set_transformer": "set_transformer",
}

# The same Okabe-Ito palette, redundant markers, and line styles used by the
# May pilot and the BDB2020 full100 publication figure.
MODEL_STYLES: dict[str, dict[str, Any]] = {
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
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
    "set_transformer": {
        "label": "Global Set Transformer",
        "color": "#000000",
        "marker": "P",
        "linestyle": (0, (3, 1, 1, 1)),
    },
}

RECEIPT_FIELDS = {
    "schema_version",
    "profile",
    "evidence_status",
    "suite_manifest",
    "suite_hash",
    "tasks",
    "task_components",
    "task_count",
    "primary_cells_per_task",
    "combined_primary_cells",
    "repeat_ids",
    "common_anchors",
    "analysis_scope",
    "inference",
    "model_winner_claim",
    "overlap_disclosure",
    "summary",
    "task_metrics",
}
COMPONENT_FIELDS = {
    "task_id",
    "run_dir",
    "manifest_hash",
    "manifest_file_sha256",
    "final_marker_sha256",
    "final_receipt_sha256",
    "metrics_sha256",
}
ARTIFACT_FIELDS = {"file", "sha256", "format"}
SUMMARY_COLUMNS = (
    "family",
    "n_train",
    "tasks",
    "task_ids",
    "task_equal_mean_skill",
    "task_median_skill",
    "task_min_skill",
    "task_max_skill",
    "task_equal_weighting",
    "repeat_pairing",
    "inference",
    "evidence_status",
    "profile",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class ValidatedPilotAggregate:
    """In-memory view of one fully revalidated provisional aggregate."""

    receipt_path: Path
    receipt_sha256: str
    receipt: dict[str, Any]
    summary_path: Path
    summary_sha256: str
    task_metrics_path: Path
    task_metrics_sha256: str
    summary: pd.DataFrame
    task_metrics: pd.DataFrame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_checksum(path: Path, *, expected: str | None = None) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        parts = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"missing or unreadable checksum sidecar: {sidecar}") from exc
    if (
        len(parts) != 2
        or not _SHA256.fullmatch(parts[0])
        or parts[1] != path.name
    ):
        raise ValueError(f"invalid checksum sidecar: {sidecar}")
    if not path.is_file():
        raise ValueError(f"checksummed aggregate artifact is missing: {path}")
    actual = _sha256_file(path)
    if actual != parts[0] or (expected is not None and actual != expected):
        raise ValueError(f"checksummed aggregate artifact drifted: {path}")
    return actual


def _strict_csv(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError(f"aggregate CSV is unreadable: {path}") from exc
    if not header or any(not value for value in header) or len(header) != len(set(header)):
        raise ValueError(f"aggregate CSV has an empty or duplicate column name: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"aggregate CSV cannot be parsed: {path}") from exc


def _artifact_from_receipt(
    receipt_path: Path,
    raw: Any,
    *,
    expected_filename: str,
) -> tuple[Path, str]:
    if not isinstance(raw, Mapping) or set(raw) != ARTIFACT_FIELDS:
        raise ValueError(f"aggregate receipt artifact pin is malformed: {expected_filename}")
    filename = raw.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or filename != expected_filename
        or raw.get("format") != "csv"
    ):
        raise ValueError(f"aggregate receipt artifact path is invalid: {expected_filename}")
    digest = _require_sha256(raw.get("sha256"), f"{filename} receipt hash")
    path = receipt_path.parent / filename
    return path, _validate_checksum(path, expected=digest)


def _validate_receipt_structure(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_FIELDS:
        raise ValueError("pilot aggregate receipt fields differ from the frozen v1 schema")
    if (
        receipt.get("schema_version") != "bdb-pilot-suite-aggregate-v1"
        or receipt.get("profile") != PILOT_PROFILE
        or receipt.get("evidence_status") != "exploratory_provisional"
        or receipt.get("analysis_scope")
        != "provisional_task_equal_descriptive_mockup_only"
        or receipt.get("inference")
        != "none_no_cross_task_ci_no_confirmatory_claims"
        or receipt.get("model_winner_claim") != "prohibited_for_pilot_suite"
    ):
        raise ValueError("receipt is not an admitted exploratory/provisional pilot10 aggregate")
    if (
        receipt.get("task_count") != 6
        or receipt.get("primary_cells_per_task") != 300
        or receipt.get("combined_primary_cells") != 1_800
        or receipt.get("repeat_ids") != list(PILOT_REPEATS)
        or receipt.get("common_anchors") != list(COMMON_ANCHORS)
        or receipt.get("tasks") != list(TASKS)
    ):
        raise ValueError("pilot aggregate receipt count/grid contract is invalid")
    if not isinstance(receipt.get("suite_manifest"), str) or not receipt["suite_manifest"]:
        raise ValueError("pilot aggregate receipt lacks its source suite-manifest path")
    _require_sha256(receipt.get("suite_hash"), "pilot suite hash")
    expected_overlap = (
        "NFL seasons overlap across releases; BDB2024 and BDB2025 use the same 2022 games"
    )
    if receipt.get("overlap_disclosure") != expected_overlap:
        raise ValueError("pilot aggregate receipt lost its cross-task overlap disclosure")

    components = receipt.get("task_components")
    if not isinstance(components, dict) or tuple(sorted(components)) != TASKS:
        raise ValueError("pilot aggregate receipt does not bind exactly the six tasks")
    for task_id in TASKS:
        component = components[task_id]
        if (
            not isinstance(component, dict)
            or set(component) != COMPONENT_FIELDS
            or component.get("task_id") != task_id
            or not isinstance(component.get("run_dir"), str)
            or not component.get("run_dir")
        ):
            raise ValueError(f"pilot task component pin is malformed: {task_id}")
        for field in COMPONENT_FIELDS - {"task_id", "run_dir"}:
            _require_sha256(component.get(field), f"{task_id}.{field}")
    return receipt


def _validate_integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.floor(values)):
        raise ValueError(f"pilot task metrics column {column} must contain finite integers")
    return values.astype(np.int64)


def _validate_task_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    required = {
        "task_id",
        "branch",
        "repeat",
        "n_train",
        "model",
        "family",
        "primary_loss",
        "game_equal_loss",
        "null_loss",
        "skill",
        "evidence_status",
        "profile",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"pilot combined task metrics lack columns {sorted(missing)}")
    if len(metrics) != 1_800:
        raise ValueError("pilot combined task metrics must contain exactly 1,800 cells")
    if set(metrics["task_id"].astype(str)) != set(TASKS):
        raise ValueError("pilot combined task metrics contain the wrong task registry")
    if set(metrics["branch"].astype(str)) != {"fixed_main"}:
        raise ValueError("pilot combined task metrics contain a non-primary branch")
    if set(metrics["profile"].astype(str)) != {PILOT_PROFILE}:
        raise ValueError("pilot combined task metrics lost the pilot10 profile label")
    if set(metrics["evidence_status"].astype(str)) != {"exploratory_provisional"}:
        raise ValueError("pilot combined task metrics lost the provisional evidence label")
    if set(metrics["model"].astype(str)) != set(MODELS):
        raise ValueError("pilot combined task metrics contain the wrong model registry")

    normalized = metrics.copy()
    normalized["repeat"] = _validate_integer_column(normalized, "repeat")
    normalized["n_train"] = _validate_integer_column(normalized, "n_train")
    normalized["task_id"] = normalized["task_id"].astype(str)
    normalized["model"] = normalized["model"].astype(str)
    normalized["family"] = normalized["family"].astype(str)
    expected_families = normalized["model"].map(MODEL_FAMILIES)
    if expected_families.isna().any() or not np.array_equal(
        normalized["family"].to_numpy(), expected_families.to_numpy()
    ):
        raise ValueError("pilot model-to-family identities were altered")

    supplied_skill = pd.to_numeric(normalized["skill"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(supplied_skill)):
        raise ValueError("pilot combined task metrics contain non-finite skill values")

    frames: dict[str, pd.DataFrame] = {}
    for task_id in TASKS:
        task_frame = normalized.loc[normalized["task_id"].eq(task_id)].copy()
        validated = validate_task_grid(
            task_frame,
            task_id=task_id,
            anchors=TASK_ANCHORS[task_id],
            repeats=len(PILOT_REPEATS),
            models=MODELS,
        )
        if len(validated) != 300:
            raise ValueError(f"pilot task {task_id} lacks its exact 300-cell grid")
        supplied = pd.to_numeric(
            task_frame.sort_values(["repeat", "n_train", "model"])["skill"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        recomputed = validated["skill"].to_numpy(dtype=np.float64)
        if not np.allclose(supplied, recomputed, rtol=2e-13, atol=2e-15):
            raise ValueError(f"pilot task {task_id} skill does not recompute from raw losses")
        frames[task_id] = validated

    combined = pd.concat(
        [frames[task_id] for task_id in TASKS], ignore_index=True
    ).sort_values(["task_id", "repeat", "n_train", "model"], kind="stable")
    return combined.reset_index(drop=True), frames


def _validate_summary(
    summary: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if tuple(summary.columns) != SUMMARY_COLUMNS:
        raise ValueError("pilot suite summary columns differ from the frozen descriptive schema")
    expected = descriptive_suite_summary(
        dict(frames), require_complete_task_ids=TASKS
    )
    expected["evidence_status"] = "exploratory_provisional"
    expected["profile"] = PILOT_PROFILE
    observed = summary.sort_values(["family", "n_train"], kind="stable").reset_index(
        drop=True
    )
    expected = expected.sort_values(["family", "n_train"], kind="stable").reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=2e-13,
            atol=2e-15,
        )
    except AssertionError as exc:
        raise ValueError(
            "pilot suite summary does not independently recompute from its 1,800 cells"
        ) from exc
    if len(observed) != 20 or set(observed["tasks"].astype(int)) != {6}:
        raise ValueError("pilot suite summary is not the exact five-role/common-anchor grid")
    return observed


def load_validated_pilot_aggregate(
    receipt_path: str | Path,
) -> ValidatedPilotAggregate:
    """Load and independently validate one pilot aggregate, failing closed."""

    path = Path(receipt_path).resolve()
    receipt_digest = _validate_checksum(path)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("pilot aggregate receipt is not readable JSON") from exc
    receipt = _validate_receipt_structure(receipt)
    suffix = ".receipt.json"
    if not path.name.endswith(suffix) or len(path.name) == len(suffix):
        raise ValueError("pilot aggregate receipt must end in .receipt.json")
    base = path.name[: -len(suffix)]
    summary_path, summary_digest = _artifact_from_receipt(
        path,
        receipt["summary"],
        expected_filename=f"{base}.csv",
    )
    task_metrics_path, task_metrics_digest = _artifact_from_receipt(
        path,
        receipt["task_metrics"],
        expected_filename=f"{base}_task_metrics.csv",
    )
    metrics, frames = _validate_task_metrics(_strict_csv(task_metrics_path))
    summary = _validate_summary(_strict_csv(summary_path), frames)
    return ValidatedPilotAggregate(
        receipt_path=path,
        receipt_sha256=receipt_digest,
        receipt=receipt,
        summary_path=summary_path,
        summary_sha256=summary_digest,
        task_metrics_path=task_metrics_path,
        task_metrics_sha256=task_metrics_digest,
        summary=summary,
        task_metrics=metrics,
    )


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
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
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            markersize=5.0,
            label=style["label"],
        )
        for style in MODEL_STYLES.values()
    ]


def _draw_task_panel(ax: plt.Axes, task_id: str, metrics: pd.DataFrame, letter: str) -> None:
    anchors = TASK_ANCHORS[task_id]
    task = metrics.loc[metrics["task_id"].eq(task_id)]
    for model in MODELS:
        style = MODEL_STYLES[model]
        model_rows = task.loc[task["model"].eq(model)]
        for repeat in PILOT_REPEATS:
            path = model_rows.loc[model_rows["repeat"].eq(repeat)].sort_values("n_train")
            ax.plot(
                path["n_train"],
                path["skill"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.75,
                marker=style["marker"],
                markersize=2.1,
                markeredgewidth=0,
                alpha=0.18,
                zorder=1,
            )
        means = model_rows.groupby("n_train", sort=True)["skill"].mean().reindex(anchors)
        ax.plot(
            anchors,
            means.to_numpy(dtype=np.float64),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.15,
            marker=style["marker"],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
    ax.axhline(0.0, color="#666666", linestyle=(0, (3, 3)), linewidth=0.85, zorder=0)
    ax.set_title(f"({letter}) {TASK_LABELS[task_id]}", loc="left", fontweight="semibold", pad=5)
    ax.set_xticks(anchors)
    if task_id == "bdb2022_punt_returns":
        # The two smallest anchors are close on this task's much wider
        # 10--360 linear scale. Rotation preserves all six literal anchors
        # without implying equal spacing or changing the learning-curve axis.
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    ax.set_xlabel("Training games")
    ax.set_ylabel("Skill vs task null (higher is better)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.margins(x=0.02, y=0.12)
    ax.spines[["top", "right"]].set_visible(False)


def _temporary_output(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _save_figure(fig: plt.Figure, path: Path, *, title: str) -> ArtifactRecord:
    temporary = _temporary_output(path)
    try:
        if path.suffix == ".png":
            fig.savefig(
                temporary,
                dpi=300,
                metadata={"Software": "Matplotlib", "Title": title},
            )
        elif path.suffix == ".pdf":
            fig.savefig(
                temporary,
                metadata={
                    "Creator": "Matplotlib",
                    "Producer": "Matplotlib",
                    "Title": title,
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        else:  # pragma: no cover - private caller contract
            raise ValueError(f"unsupported figure format: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return register_existing_artifact(path, path.suffix.lstrip("."))


def _write_text_artifact(path: Path, text: str) -> ArtifactRecord:
    temporary = _temporary_output(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return register_existing_artifact(path, "markdown")


def _summary_markdown(summary: pd.DataFrame) -> str:
    role_labels = {
        "glm": "Linear-Structure",
        "lightgbm": "Boosted-Structure",
        "relnet": "RelNet",
        "attn_relnet": "AttnRelNet",
        "set_transformer": "Global Set Transformer",
    }
    pivot = summary.pivot(
        index="n_train", columns="family", values="task_equal_mean_skill"
    ).reindex(index=COMMON_ANCHORS, columns=tuple(role_labels))
    lines = [
        "| Training games | " + " | ".join(role_labels[value] for value in pivot.columns) + " |",
        "| ---: | " + " | ".join("---:" for _ in pivot.columns) + " |",
    ]
    for anchor, row in pivot.iterrows():
        lines.append(
            f"| {int(anchor)} | "
            + " | ".join(f"{float(row[value]):.4f}" for value in pivot.columns)
            + " |"
        )
    return "\n".join(lines)


def _build_markdown(
    aggregate: ValidatedPilotAggregate,
    *,
    png_name: str,
) -> str:
    task_rows = []
    for task_id in TASKS:
        task_rows.append(
            f"| {TASK_LABELS[task_id].split(' — ')[0]} | "
            f"{TASK_LABELS[task_id].split(' — ')[1]} | "
            f"{', '.join(str(value) for value in TASK_ANCHORS[task_id])} | 300 |"
        )
    return f"""# Six-task BDB pilot10 mock-up — exploratory/provisional

> **EXPLORATORY / PROVISIONAL — 10 RERUNS PER TASK.** This is an implementation-pilot artifact, not confirmatory evidence. It contains no hypothesis test, cross-task uncertainty interval, supported-best decision, or model-winner claim.

The checksum-gated aggregate contains exactly 1,800 primary cells: six tasks × ten reruns × six anchors × five frozen model roles. Every cell and the 5×4 common-anchor descriptive summary were revalidated before this report was written.

![Six independent task learning-curve panels]({png_name})

Thin paths are the ten complete pilot reruns; thick paths are within-task means. Each panel uses the task's own null-relative skill score, and its vertical scale is interpreted within that task. The figure does not pool task-native uncertainty outputs across releases.

## Validated task panels

| Task | Displayed score | Training-game anchors | Cells |
| --- | --- | --- | ---: |
{chr(10).join(task_rows)}

## Task-equal common-anchor description

The table below reproduces the receipt-bound descriptive reduction: one ten-rerun mean per task is computed first, then the six task means are averaged at each common anchor. The six releases are not independent replications; seasons overlap, and BDB2024 and BDB2025 use the same 2022 games. These values have no cross-task confidence interval and are not used to declare a winner.

{_summary_markdown(aggregate.summary)}

## Provenance and permitted interpretation

- Aggregate schema: `bdb-pilot-suite-aggregate-v1`
- Evidence status: `exploratory_provisional`
- Profile: `pilot10` (repeat IDs 1–10; pilot seed namespace disjoint from `full100`)
- Aggregate receipt SHA-256: `{aggregate.receipt_sha256}`
- Suite hash: `{aggregate.receipt['suite_hash']}`
- Summary CSV SHA-256: `{aggregate.summary_sha256}`
- Combined task-metrics CSV SHA-256: `{aggregate.task_metrics_sha256}`
- Analysis scope: descriptive implementation mock-up only
- Inferential scope: none; no cross-task CI and no confirmatory claim
- Model-winner claim: prohibited for this pilot suite

This report may be used to inspect execution, runtime, artifact validity, and the qualitative shape of the provisional learning curves. Scientific choices remain frozen, and these ten draws are not imported into the later `full100` evidence.
"""


def generate_pilot_report(
    receipt_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_stem: str = "bdb2021_2026_pilot10_provisional",
) -> dict[str, Any]:
    """Validate a pilot aggregate and emit its visibly provisional mock-up."""

    if not _SAFE_STEM.fullmatch(str(output_stem)):
        raise ValueError("output_stem must be a path-safe filename stem")
    aggregate = load_validated_pilot_aggregate(receipt_path)
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else aggregate.receipt_path.parent / "mockup"
    )
    destination.mkdir(parents=True, exist_ok=True)

    _set_style()
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.4))
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.835,
        hspace=0.43,
        wspace=0.25,
    )
    fig.suptitle(
        "BDB 2021–2026 structure pilot: six task-specific learning curves",
        x=0.5,
        y=0.977,
        fontsize=14.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.945,
        "EXPLORATORY / PROVISIONAL • pilot10 • R = 10 per task",
        ha="center",
        va="top",
        fontsize=10.3,
        color="#A33A2B",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.921,
        "Faint paths are complete reruns; thick lines are within-task means",
        ha="center",
        va="top",
        fontsize=9.2,
        color="#444444",
    )
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.902),
        ncol=5,
        columnspacing=1.8,
        handlelength=2.8,
        handletextpad=0.65,
        fontsize=8.8,
    )
    for letter, task_id, ax in zip("ABCDEF", TASKS, axes.flat, strict=True):
        _draw_task_panel(ax, task_id, aggregate.task_metrics, letter)
    fig.text(
        0.5,
        0.027,
        "Implementation pilot only — no pooled cross-task uncertainty, inferential comparison, or model-winner claim.",
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#555555",
        style="italic",
    )

    title = "BDB2021–2026 exploratory/provisional pilot10 learning curves"
    png_path = destination / f"{output_stem}.png"
    pdf_path = destination / f"{output_stem}.pdf"
    png_record = _save_figure(fig, png_path, title=title)
    pdf_record = _save_figure(fig, pdf_path, title=title)
    plt.close(fig)

    markdown_path = destination / f"{output_stem}.md"
    markdown_record = _write_text_artifact(
        markdown_path,
        _build_markdown(aggregate, png_name=png_path.name),
    )
    output_records = {
        "png": png_record.as_dict(),
        "pdf": pdf_record.as_dict(),
        "markdown": markdown_record.as_dict(),
    }
    report_receipt: dict[str, Any] = {
        "schema_version": "bdb-pilot10-provisional-report-v1",
        "profile": PILOT_PROFILE,
        "evidence_status": "exploratory_provisional",
        "label": "EXPLORATORY / PROVISIONAL — 10 RERUNS PER TASK",
        "aggregate_receipt": {
            "file": str(aggregate.receipt_path),
            "sha256": aggregate.receipt_sha256,
            "format": "json",
        },
        "suite_hash": aggregate.receipt["suite_hash"],
        "validated_inputs": {
            "summary": aggregate.receipt["summary"],
            "task_metrics": aggregate.receipt["task_metrics"],
        },
        "validated_contract": {
            "tasks": list(TASKS),
            "task_count": 6,
            "repeats_per_task": 10,
            "anchors_per_task": 6,
            "model_roles": list(MODELS),
            "primary_cells_per_task": 300,
            "combined_primary_cells": 1_800,
        },
        "analysis_scope": "provisional_task_equal_descriptive_mockup_only",
        "inference": "none_no_cross_task_ci_no_confirmatory_claims",
        "model_winner_claim": "prohibited_for_pilot_suite",
        "outputs": output_records,
    }
    report_receipt["report_hash"] = hashlib.sha256(
        json.dumps(report_receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt_record = atomic_write_json(
        destination / f"{output_stem}_report.receipt.json",
        report_receipt,
    )
    return {
        "schema_version": report_receipt["schema_version"],
        "evidence_status": report_receipt["evidence_status"],
        "png": str(png_path),
        "pdf": str(pdf_path),
        "report": str(markdown_path),
        "receipt": str(destination / receipt_record.file),
        "report_hash": report_receipt["report_hash"],
    }
