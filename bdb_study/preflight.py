"""Deterministic-smoke, memory, and runtime receipts for definitive BDB runs.

A definitive protocol-v2 manifest is allowed to exist only after two
independent smoke run directories reproduce the same five largest-anchor
cells.  The receipt also freezes the observed worker peak used by the
RAM-aware CPU concurrency rule. Endpoint benchmarks additionally cap every
Betty job at 3h20 within the four-hour allocation. Timing is deliberately
excluded from the scientific comparison and identity hash.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import sha256_file
from .runtime_phases import (
    BETTY_ATTEMPT9_CPU_LANES as BETTY_V4_CPU_LANES,
    BETTY_CPU_LANES as BETTY_V3_CPU_LANES,
    BETTY_MIG45_LANES as BETTY_V4_GPU_LANES,
    BETTY_MIG90_LANES as BETTY_V3_GPU_LANES,
    NEURAL_PHASES,
    RUNTIME_PLAN_V3_SCHEMA_VERSION,
    RUNTIME_PLAN_V4_SCHEMA_VERSION,
    RUNTIME_V3_SHARDING,
    RUNTIME_V4_SHARDING,
    RuntimePhasePlanError,
    betty_runtime_v3_resource_contract,
    betty_runtime_v4_resource_contract,
    canonical_neural_phase_shards,
    validate_neural_phase_shards,
)
from .storage import CellKey, immutable_write_json, recommended_cpu_workers


PREFLIGHT_SCHEMA_VERSION = "bdb-preflight-v1"
RUNTIME_PLAN_SCHEMA_VERSION = "bdb-runtime-plan-v2"
RUNTIME_JOB_GUARD_SECONDS = 3 * 60 * 60 + 20 * 60
RUNTIME_JOB_OVERHEAD_SECONDS = 6 * 60
BETTY_WALLTIME_SECONDS = 4 * 60 * 60
BETTY_GPU_LANES = 4
RUNTIME_SHARDING = "rectangular_anchor_model_shards_round_robin_lanes_v2"
PRIMARY_MODEL_ROLES = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
CPU_MODEL_ROLES = PRIMARY_MODEL_ROLES[:2]
GPU_MODEL_ROLES = PRIMARY_MODEL_ROLES[2:]
PRIMARY_NEURAL_FAMILIES = ("relnet", "attn_relnet", "set_transformer")
PRIMARY_FAMILIES = ("glm", "lightgbm", *PRIMARY_NEURAL_FAMILIES)
FULL50_NEURAL_FAMILIES = ("relnet", "attn_relnet")
FULL50_FAMILIES = ("glm", "lightgbm", *FULL50_NEURAL_FAMILIES)
NONSCIENTIFIC_METRIC_FIELDS = {"elapsed_seconds", "process_peak_rss_bytes"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """A smoke pair is incomplete, non-deterministic, or provenance-drifted."""


def _profile_families(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile = str(manifest.get("execution", {}).get("profile", ""))
    if profile == "full50":
        return FULL50_FAMILIES, FULL50_NEURAL_FAMILIES
    return PRIMARY_FAMILIES, PRIMARY_NEURAL_FAMILIES


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _total_ram_bytes() -> int:
    """Return physical RAM without adding a suite dependency."""

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    raise PreflightError("could not determine physical RAM for the concurrency receipt")


def process_peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss`` to bytes on macOS and Linux."""

    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value <= 0:
        raise PreflightError("process peak RSS is unavailable")
    # Darwin reports bytes; Linux and most other Unix implementations report
    # KiB.  Keep this explicit in the preflight receipt.
    return value if platform.system() == "Darwin" else value * 1024


def _scientific_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in metrics.items()
        if str(key) not in NONSCIENTIFIC_METRIC_FIELDS
    }


def _require_gpu_runtime(environment: Mapping[str, Any]) -> None:
    runtime = environment.get("tensorflow_runtime")
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    has_gpu = isinstance(devices, list) and any(
        isinstance(device, Mapping)
        and str(device.get("device_type", "")).upper() == "GPU"
        for device in devices
    )
    if not isinstance(runtime, Mapping) or runtime.get("available") is not True or not has_gpu:
        raise PreflightError(
            "definitive preflight requires TensorFlow with a physical GPU device"
        )


def _assert_json_equal(left: Any, right: Any, label: str) -> None:
    if json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=True) != json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=True
    ):
        raise PreflightError(f"deterministic smoke mismatch in {label}")


def _assert_predictions_equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(
                left.reset_index(drop=True),
                right.reset_index(drop=True),
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise PreflightError(f"deterministic smoke mismatch in {label}: {exc}") from exc
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise PreflightError(f"deterministic smoke array names differ in {label}")
        for name in sorted(left):
            one = np.asarray(left[name])
            two = np.asarray(right[name])
            if one.dtype != two.dtype or one.shape != two.shape or not np.array_equal(
                one, two, equal_nan=True
            ):
                raise PreflightError(f"deterministic smoke mismatch in {label}.{name}")
        return
    raise PreflightError(f"deterministic smoke artifact types differ in {label}")


def _validated_signed_signature(value: Any, label: str) -> dict[str, Any]:
    """Validate a self-hashed neural input or head/loss signature."""

    if not isinstance(value, Mapping):
        raise PreflightError(f"neural smoke {label} is absent or invalid")
    normalized = json.loads(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    expected_schema = {
        "representative_neural_input": "bdb-representative-neural-input-v1",
        "representative_shared_neural_input": "bdb-representative-neural-input-v1",
        "output_head_loss_signature": "bdb-neural-head-loss-v1",
    }.get(label.rsplit(".", 1)[-1])
    if expected_schema is None or normalized.get("schema_version") != expected_schema:
        raise PreflightError(f"neural smoke {label} has an unsupported schema")
    digest = normalized.get("signature_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise PreflightError(f"neural smoke {label} has no valid signature hash")
    unsigned = {key: item for key, item in normalized.items() if key != "signature_sha256"}
    if digest != _json_hash(unsigned):
        raise PreflightError(f"neural smoke {label} signature hash is invalid")
    return normalized


def _validated_global_set_architecture(value: Any) -> dict[str, Any]:
    """Validate the explicit non-2020 Global Set Transformer identity."""

    if not isinstance(value, Mapping):
        raise PreflightError("Set Transformer architecture receipt is absent")
    normalized = dict(value)
    expected_fields = {
        "architecture_id",
        "inputs",
        "attention_contract",
        "relation_scope",
        "typed_graph_edges_consumed",
        "temporal_encoder",
        "protocol_note",
        "parameter_count",
        "parameter_cap",
        "representative_forward_flops",
    }
    if set(normalized) != expected_fields:
        raise PreflightError("Set Transformer architecture receipt fields differ")
    if (
        normalized.get("architecture_id") != "bdb_global_set_transformer_v1"
        or normalized.get("inputs")
        != "task_tokens_player_frame_masks_time_context"
        or normalized.get("attention_contract") != "global_set_time_attention_v1"
        or normalized.get("relation_scope")
        != "global_masked_all_player_self_attention"
        or normalized.get("typed_graph_edges_consumed") is not False
        or normalized.get("temporal_encoder")
        != "factorized_masked_temporal_attention"
        or normalized.get("protocol_note")
        != "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot"
        or normalized.get("parameter_cap") != 350_000
    ):
        raise PreflightError("Set Transformer architecture contract drifted")
    for field in ("parameter_count", "representative_forward_flops"):
        item = normalized.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise PreflightError(f"Set Transformer {field} is invalid")
    if int(normalized["parameter_count"]) > 350_000:
        raise PreflightError("Set Transformer exceeds the 350000-parameter cap")
    return normalized


def _smoke_cells(manifest: Mapping[str, Any]) -> list[CellKey]:
    smoke = manifest.get("execution", {}).get("smoke")
    if not isinstance(smoke, Mapping) or smoke.get("enabled") is not True:
        raise PreflightError("preflight requires a manifest-declared smoke run")
    cells = smoke.get("cells")
    expected_families, _ = _profile_families(manifest)
    if not isinstance(cells, list) or len(cells) != len(expected_families):
        raise PreflightError(
            "smoke manifest must declare its exact profile-bound model panel"
        )
    keys = [
        CellKey(
            str(record["branch"]), int(record["repeat"]),
            int(record["n_train"]), str(record["model"]),
        )
        for record in cells
    ]
    families = {
        manifest["task_spec"]["models"][key.model]["family"] for key in keys
    }
    if families != set(expected_families):
        raise PreflightError(
            "smoke cells must contain the profile-bound canonical families"
        )
    if {key.repeat for key in keys} != {1} or len({key.n_train for key in keys}) != 1:
        raise PreflightError("smoke cells must share repeat 1 and one training anchor")
    anchors = manifest.get("task_spec", {}).get("anchors", [])
    try:
        largest_anchor = max(int(value) for value in anchors)
    except (TypeError, ValueError) as exc:
        raise PreflightError("smoke manifest has no valid task anchors") from exc
    if {key.n_train for key in keys} != {largest_anchor}:
        raise PreflightError(
            "smoke cells must use the largest training anchor for memory preflight"
        )
    return sorted(keys, key=lambda key: key.model)


def build_preflight_receipt(
    run_a: str | Path,
    run_b: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    total_ram_bytes: int | None = None,
) -> dict[str, Any]:
    """Compare two independent smoke runs and atomically freeze an attestation."""

    # Import lazily so merely inspecting this module does not initialize model
    # or storage runtimes.
    from .execution import open_run

    # Smoke GPU workers already enforce the exact TensorFlow device receipt.
    # Reduction runs on the target 12-core CPU queue so RAM-derived worker
    # concurrency is measured against that allocation rather than a DGX host.
    first = open_run(
        run_a, repo_root=repo_root, verify_tensorflow_runtime=False
    )
    second = open_run(
        run_b, repo_root=repo_root, verify_tensorflow_runtime=False
    )
    if first.manifest_hash != second.manifest_hash:
        raise PreflightError("smoke run manifests differ")
    keys = _smoke_cells(first.manifest)
    expected_families, expected_neural_families = _profile_families(
        first.manifest
    )
    cpu_peaks: list[int] = []
    gpu_peaks: list[int] = []
    neural_complexity: dict[str, dict[str, Any]] = {}
    cell_records: list[dict[str, Any]] = []
    for key in keys:
        validation_a = first.validate_cell(key)
        validation_b = second.validate_cell(key)
        if not validation_a.is_complete or not validation_b.is_complete:
            raise PreflightError(f"smoke cell is incomplete: {key.relative_dir}")
        metrics_a = first.load_metrics(key)
        metrics_b = second.load_metrics(key)
        _assert_json_equal(
            _scientific_metrics(metrics_a), _scientific_metrics(metrics_b),
            f"{key.relative_dir}.metrics",
        )
        _assert_predictions_equal(
            first.load_predictions(key), second.load_predictions(key),
            f"{key.relative_dir}.predictions",
        )
        arrays_a = first.load_arrays(key)
        arrays_b = second.load_arrays(key)
        if (arrays_a is None) != (arrays_b is None):
            raise PreflightError(f"smoke arrays differ for {key.relative_dir}")
        if arrays_a is not None:
            _assert_predictions_equal(arrays_a, arrays_b, f"{key.relative_dir}.arrays")
        family = first.manifest["task_spec"]["models"][key.model]["family"]
        if family in expected_neural_families:
            history_a = first.load_history(key)
            history_b = second.load_history(key)
            if not isinstance(history_a, Mapping) or not isinstance(history_b, Mapping):
                raise PreflightError(f"neural smoke cell lacks history: {key.relative_dir}")
            complexity: dict[str, Any] = {}
            for field in ("parameter_count", "representative_forward_flops"):
                left = history_a.get(field)
                right = history_b.get(field)
                if (
                    not isinstance(left, int)
                    or isinstance(left, bool)
                    or left <= 0
                    or right != left
                ):
                    raise PreflightError(
                        f"neural smoke {field} is absent, invalid, or non-deterministic"
                    )
                complexity[field] = int(left)
            for field in (
                "representative_neural_input",
                "representative_shared_neural_input",
                "output_head_loss_signature",
            ):
                left = _validated_signed_signature(history_a.get(field), field)
                right = _validated_signed_signature(history_b.get(field), field)
                _assert_json_equal(left, right, f"{key.relative_dir}.{field}")
                complexity[field] = left
                complexity[f"{field}_sha256"] = left["signature_sha256"]
            if family == "set_transformer":
                architecture_a = _validated_global_set_architecture(
                    history_a.get("global_set_architecture")
                )
                architecture_b = _validated_global_set_architecture(
                    history_b.get("global_set_architecture")
                )
                _assert_json_equal(
                    architecture_a,
                    architecture_b,
                    f"{key.relative_dir}.global_set_architecture",
                )
                if (
                    architecture_a["parameter_count"]
                    != complexity["parameter_count"]
                    or architecture_a["representative_forward_flops"]
                    != complexity["representative_forward_flops"]
                ):
                    raise PreflightError(
                        "Set Transformer architecture/count receipt is inconsistent"
                    )
                complexity["global_set_architecture"] = architecture_a
            neural_complexity[family] = complexity
        for metrics in (metrics_a, metrics_b):
            peak = metrics.get("process_peak_rss_bytes")
            if not isinstance(peak, (int, float)) or isinstance(peak, bool) or peak <= 0:
                raise PreflightError(f"smoke cell lacks a peak-RSS measurement: {key.relative_dir}")
            (cpu_peaks if family in {"glm", "lightgbm"} else gpu_peaks).append(int(peak))
        cell_records.append(
            {
                "branch": key.branch,
                "repeat": key.repeat,
                "n_train": key.n_train,
                "model": key.model,
                "family": family,
                "scientific_metrics_sha256": _json_hash(_scientific_metrics(metrics_a)),
            }
        )
    total_ram = _total_ram_bytes() if total_ram_bytes is None else int(total_ram_bytes)
    if not cpu_peaks or not gpu_peaks:
        raise PreflightError("smoke receipt lacks one of the two execution queues")
    if set(neural_complexity) != set(expected_neural_families):
        raise PreflightError(
            "smoke receipt lacks its profile-bound primary neural families"
        )
    relnet = neural_complexity["relnet"]
    attention = neural_complexity["attn_relnet"]
    for field in (
        "representative_neural_input",
        "representative_neural_input_sha256",
    ):
        if relnet[field] != attention[field]:
            raise PreflightError(
                "RelNet and AttnRelNet differ in their matched neural "
                f"{field} contract"
            )
    for field in (
        "representative_shared_neural_input",
        "representative_shared_neural_input_sha256",
        "output_head_loss_signature",
        "output_head_loss_signature_sha256",
    ):
        values = [
            neural_complexity[family][field]
            for family in expected_neural_families
        ]
        if any(value != values[0] for value in values[1:]):
            raise PreflightError(
                "primary neural families differ in their shared input/head-loss "
                f"{field} contract"
            )
    parameter_gap = abs(relnet["parameter_count"] - attention["parameter_count"]) / max(
        relnet["parameter_count"], attention["parameter_count"]
    )
    flop_gap = abs(
        relnet["representative_forward_flops"]
        - attention["representative_forward_flops"]
    ) / max(
        relnet["representative_forward_flops"],
        attention["representative_forward_flops"],
    )
    if max(
        record["parameter_count"] for record in neural_complexity.values()
    ) > 350_000:
        raise PreflightError("a neural model exceeds the 350000-parameter cap")
    if parameter_gap > 0.05:
        raise PreflightError("RelNet and AttnRelNet parameter counts differ by more than 5%")
    if flop_gap > 0.15:
        raise PreflightError("RelNet and AttnRelNet forward FLOPs differ by more than 15%")
    worker_peak = max(cpu_peaks)
    recommended = recommended_cpu_workers(total_ram, worker_peak)
    provenance = first.manifest["provenance"]
    _require_gpu_runtime(provenance["environment"])
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "task_id": first.manifest["task_id"],
        "task_spec_hash": first.manifest["task_spec_hash"],
        "prepared_hash": first.manifest["prepared"]["prepared_hash"],
        "code_identity_sha256": _json_hash({"code": provenance["code"]}),
        "dependency_lock_sha256": provenance["dependency_lock"]["sha256"],
        "environment": provenance["environment"],
        "smoke_manifest_hash": first.manifest_hash,
        "cells": cell_records,
        "determinism": "byte_exact_scientific_metrics_predictions_and_arrays",
        "memory": {
            "total_ram_bytes": total_ram,
            "worker_peak_bytes": worker_peak,
            "gpu_worker_peak_bytes": max(gpu_peaks),
            "usable_fraction": 0.75,
            "worker_cap": 12,
            "recommended_cpu_workers": recommended,
            "measurement": (
                "maximum_normalized_process_ru_maxrss_across_two_"
                "largest_anchor_smoke_runs"
            ),
        },
        "neural_matching": {
            "models": neural_complexity,
            "matched_pair": ["relnet", "attn_relnet"],
            "global_set_model": (
                "set_transformer"
                if "set_transformer" in expected_neural_families
                else None
            ),
            "profile_families": list(expected_families),
            "parameter_gap_fraction": parameter_gap,
            "forward_flop_gap_fraction": flop_gap,
            "parameter_tolerance_fraction": 0.05,
            "forward_flop_tolerance_fraction": 0.15,
            "parameter_cap": 350_000,
        },
    }
    payload["receipt_hash"] = _json_hash(payload)
    target = immutable_write_json(output_path, payload)
    return payload


def validate_preflight_receipt(
    receipt_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind a definitive manifest to the exact tested code/data/environment."""

    path = Path(receipt_path).resolve()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not read preflight receipt {path}: {exc}") from exc
    if receipt.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise PreflightError("unsupported preflight receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if receipt.get("receipt_hash") != _json_hash(unsigned):
        raise PreflightError("preflight receipt hash is invalid")
    if receipt.get("determinism") != (
        "byte_exact_scientific_metrics_predictions_and_arrays"
    ):
        raise PreflightError("preflight determinism claim is missing or unsupported")
    smoke_hash = receipt.get("smoke_manifest_hash")
    if not isinstance(smoke_hash, str) or _SHA256.fullmatch(smoke_hash) is None:
        raise PreflightError("preflight smoke-manifest hash is invalid")
    cells = receipt.get("cells")
    expected_families, expected_neural_families = _profile_families(manifest)
    if not isinstance(cells, list) or len(cells) != len(expected_families):
        raise PreflightError(
            "preflight must contain its exact profile-bound smoke cells"
        )
    identities: set[tuple[str, int, int, str]] = set()
    families: set[str] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise PreflightError("preflight smoke-cell entry is invalid")
        try:
            identity = (
                str(cell["branch"]), int(cell["repeat"]),
                int(cell["n_train"]), str(cell["model"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError("preflight smoke-cell identity is invalid") from exc
        family = cell.get("family")
        metric_hash = cell.get("scientific_metrics_sha256")
        if (
            family not in set(expected_families)
            or not isinstance(metric_hash, str)
            or _SHA256.fullmatch(metric_hash) is None
        ):
            raise PreflightError("preflight smoke-cell provenance is invalid")
        identities.add(identity)
        families.add(str(family))
    if (
        len(identities) != len(expected_families)
        or families != set(expected_families)
    ):
        raise PreflightError(
            "preflight smoke cells do not cover the profile-bound families exactly"
        )
    if {identity[1] for identity in identities} != {1} or len(
        {identity[2] for identity in identities}
    ) != 1:
        raise PreflightError("preflight smoke cells do not share repeat 1 and one anchor")
    anchors = manifest.get("task_spec", {}).get("anchors", [])
    try:
        largest_anchor = max(int(value) for value in anchors)
    except (TypeError, ValueError) as exc:
        raise PreflightError("definitive manifest has no valid task anchors") from exc
    if {identity[2] for identity in identities} != {largest_anchor}:
        raise PreflightError(
            "preflight smoke cells did not measure the largest training anchor"
        )
    provenance = manifest.get("provenance", {})
    expected = {
        "task_id": manifest.get("task_id"),
        "task_spec_hash": manifest.get("task_spec_hash"),
        "prepared_hash": manifest.get("prepared", {}).get("prepared_hash"),
        "code_identity_sha256": _json_hash({"code": provenance.get("code")}),
        "dependency_lock_sha256": provenance.get("dependency_lock", {}).get("sha256"),
        "environment": provenance.get("environment"),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise PreflightError(f"preflight receipt provenance drift: {field}")
    _require_gpu_runtime(receipt.get("environment", {}))
    memory = receipt.get("memory", {})
    integer_fields = (
        "total_ram_bytes", "worker_peak_bytes", "gpu_worker_peak_bytes",
        "worker_cap", "recommended_cpu_workers",
    )
    if not isinstance(memory, Mapping):
        raise PreflightError("preflight memory receipt is missing")
    for field in integer_fields:
        value = memory.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PreflightError(f"preflight memory field {field} is invalid")
    if memory.get("usable_fraction") != 0.75 or memory.get("worker_cap") != 12:
        raise PreflightError("preflight memory rule differs from the locked 75%/12-worker rule")
    expected_workers = recommended_cpu_workers(
        int(memory["total_ram_bytes"]),
        int(memory["worker_peak_bytes"]),
        cap=int(memory["worker_cap"]),
        usable_fraction=float(memory["usable_fraction"]),
    )
    if memory.get("recommended_cpu_workers") != expected_workers:
        raise PreflightError("preflight recommended CPU concurrency violates the memory rule")
    if memory.get("measurement") != (
        "maximum_normalized_process_ru_maxrss_across_two_largest_anchor_smoke_runs"
    ):
        raise PreflightError("preflight memory measurement is unsupported")
    matching = receipt.get("neural_matching")
    if not isinstance(matching, Mapping):
        raise PreflightError("preflight neural-matching receipt is missing")
    if (
        matching.get("parameter_tolerance_fraction") != 0.05
        or matching.get("forward_flop_tolerance_fraction") != 0.15
        or matching.get("parameter_cap") != 350_000
        or matching.get("matched_pair") != ["relnet", "attn_relnet"]
        or matching.get("global_set_model")
        != (
            "set_transformer"
            if "set_transformer" in expected_neural_families
            else None
        )
        or matching.get("profile_families") != list(expected_families)
    ):
        raise PreflightError("preflight neural-matching thresholds drifted")
    models = matching.get("models")
    if not isinstance(models, Mapping) or set(models) != set(
        expected_neural_families
    ):
        raise PreflightError("preflight neural-matching models are invalid")
    try:
        relnet = models["relnet"]
        attention = models["attn_relnet"]
        set_transformer = models.get("set_transformer")
        for model_name in expected_neural_families:
            record = models[model_name]
            if not isinstance(record, Mapping):
                raise TypeError(model_name)
            for field in (
                "representative_neural_input",
                "representative_shared_neural_input",
                "output_head_loss_signature",
            ):
                signature = _validated_signed_signature(
                    record.get(field), f"{model_name}.{field}"
                )
                if record.get(f"{field}_sha256") != signature["signature_sha256"]:
                    raise PreflightError(
                        f"preflight {model_name} {field} hash binding is invalid"
                    )
            parameter_count = int(record["parameter_count"])
            forward_flops = int(record["representative_forward_flops"])
            if parameter_count <= 0 or parameter_count > 350_000 or forward_flops <= 0:
                raise PreflightError(
                    f"preflight {model_name} complexity is invalid"
                )
        if set_transformer is not None:
            architecture = _validated_global_set_architecture(
                set_transformer.get("global_set_architecture")
            )
            if (
                architecture["parameter_count"]
                != set_transformer["parameter_count"]
                or architecture["representative_forward_flops"]
                != set_transformer["representative_forward_flops"]
            ):
                raise PreflightError(
                    "preflight Set Transformer architecture/count binding is invalid"
                )
        for field in (
            "representative_neural_input",
            "representative_neural_input_sha256",
        ):
            if relnet.get(field) != attention.get(field):
                raise PreflightError(
                    "preflight matched neural input/head-loss signatures differ"
                )
        for field in (
            "representative_shared_neural_input",
            "representative_shared_neural_input_sha256",
            "output_head_loss_signature",
            "output_head_loss_signature_sha256",
        ):
            values = [
                models[family].get(field) for family in expected_neural_families
            ]
            if any(value != values[0] for value in values[1:]):
                raise PreflightError(
                    "preflight shared neural input/head-loss signatures differ"
                )
        parameter_gap = abs(
            int(relnet["parameter_count"]) - int(attention["parameter_count"])
        ) / max(int(relnet["parameter_count"]), int(attention["parameter_count"]))
        flop_gap = abs(
            int(relnet["representative_forward_flops"])
            - int(attention["representative_forward_flops"])
        ) / max(
            int(relnet["representative_forward_flops"]),
            int(attention["representative_forward_flops"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise PreflightError("preflight neural-matching counts are invalid") from exc
    if (
        max(
            int(models[family]["parameter_count"])
            for family in expected_neural_families
        )
        > 350_000
        or parameter_gap > 0.05
        or flop_gap > 0.15
        or matching.get("parameter_gap_fraction") != parameter_gap
        or matching.get("forward_flop_gap_fraction") != flop_gap
    ):
        raise PreflightError("preflight neural-matching assertion is invalid")
    workers = manifest.get("execution", {}).get("queues", {}).get("cpu_tabular", {}).get("workers")
    if workers != memory.get("recommended_cpu_workers"):
        raise PreflightError(
            "definitive CPU concurrency differs from the smoke memory rule: "
            f"manifest={workers}, receipt={memory.get('recommended_cpu_workers')}"
        )
    root = Path(repo_root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError("preflight receipt must be stored inside the repository") from exc
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "receipt_hash": receipt["receipt_hash"],
        "smoke_manifest_hash": receipt["smoke_manifest_hash"],
        "recommended_cpu_workers": int(memory["recommended_cpu_workers"]),
    }


def _observed_training_epochs(history: Mapping[str, Any]) -> int:
    total = 0
    for field in ("selector_history", "refit_history"):
        value = history.get(field)
        if isinstance(value, Mapping):
            losses = value.get("loss")
            if isinstance(losses, list):
                total += len(losses)
    return max(1, total)


def _strictly_increasing_anchors(value: Any, *, label: str) -> list[int]:
    """Normalize one frozen anchor registry without changing its order."""

    if not isinstance(value, list) or not value:
        raise PreflightError(f"{label} anchors are missing")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PreflightError(f"{label} anchors must be integers")
    anchors = [int(item) for item in value]
    if any(item <= 0 for item in anchors) or anchors != sorted(set(anchors)):
        raise PreflightError(
            f"{label} anchors must be unique, positive, and strictly increasing"
        )
    return anchors


def _runtime_phase_registry(
    profile: str, task_spec: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, list[Any]]]]:
    """Return the only admissible queue/phase anchor-model rectangles.

    This registry is deliberately derived from the immutable TaskSpec rather
    than trusted from a runtime plan.  Both plan construction and admission
    use it, so an omitted, duplicated, reordered, or cross-phase cell fails
    closed before Slurm sees the run.
    """

    models = task_spec.get("models")
    expected_families = dict(
        zip(
            PRIMARY_MODEL_ROLES,
            ("glm", "lightgbm", *PRIMARY_NEURAL_FAMILIES),
            strict=True,
        )
    )
    if (
        not isinstance(models, Mapping)
        or set(models) != set(PRIMARY_MODEL_ROLES)
        or any(
            not isinstance(models[model_id], Mapping)
            or models[model_id].get("family") != expected_families[model_id]
            for model_id in PRIMARY_MODEL_ROLES
        )
    ):
        raise PreflightError("runtime plan TaskSpec lacks the five model roles")

    if profile in {"pilot10", "full100"}:
        anchors = _strictly_increasing_anchors(
            task_spec.get("anchors"), label=profile
        )
        if len(anchors) != 6:
            raise PreflightError(f"{profile} runtime plan requires six TaskSpec anchors")
        phases: dict[str, dict[str, dict[str, list[Any]]]] = {
            "cpu_tabular": {
                "primary": {
                    "anchors": anchors,
                    "models": list(CPU_MODEL_ROLES),
                }
            },
            "gpu_neural": {
                "primary": {
                    "anchors": anchors,
                    "models": list(GPU_MODEL_ROLES),
                }
            },
        }
        if profile == "full100":
            ablation = task_spec.get("structural_ablation")
            if not isinstance(ablation, Mapping):
                raise PreflightError("runtime plan lacks its structural ablation")
            ablation_anchors = _strictly_increasing_anchors(
                ablation.get("anchors"), label="structural-ablation"
            )
            if len(ablation_anchors) != 2:
                raise PreflightError(
                    "runtime plan requires two structural-ablation anchors"
                )
            if not set(ablation_anchors).issubset(anchors):
                raise PreflightError(
                    "runtime plan structural-ablation anchors escape the primary grid"
                )
            if ablation.get("models") != list(GPU_MODEL_ROLES[:2]):
                raise PreflightError(
                    "runtime plan structural ablation must remain the relational pair"
                )
            phases["gpu_neural"]["ablation"] = {
                "anchors": ablation_anchors,
                "models": list(GPU_MODEL_ROLES[:2]),
            }
        return phases

    if profile != "sensitivity20":
        raise PreflightError("runtime plan has an unsupported execution profile")
    frozen = [
        item
        for item in task_spec.get("sensitivities", [])
        if isinstance(item, Mapping)
        and item.get("selection") == "frozen_prespecified"
    ]
    if (
        len(frozen) != 1
        or not isinstance(frozen[0].get("execution"), Mapping)
    ):
        raise PreflightError(
            "sensitivity runtime plan requires one two-anchor intervention"
        )
    anchors = _strictly_increasing_anchors(
        frozen[0]["execution"].get("anchors"), label="sensitivity"
    )
    if len(anchors) != 2:
        raise PreflightError(
            "sensitivity runtime plan requires one two-anchor intervention"
        )
    primary_anchors = _strictly_increasing_anchors(
        task_spec.get("anchors"), label="sensitivity TaskSpec"
    )
    if not set(anchors).issubset(primary_anchors):
        raise PreflightError(
            "sensitivity runtime anchors escape the primary TaskSpec grid"
        )
    return {
        "cpu_tabular": {
            "sensitivity": {
                "anchors": anchors,
                "models": list(CPU_MODEL_ROLES),
            }
        },
        "gpu_neural": {
            "sensitivity": {
                "anchors": anchors,
                "models": list(GPU_MODEL_ROLES),
            }
        },
    }


def _canonical_runtime_phase_shards(
    *,
    queue: str,
    phase: str,
    anchors: Sequence[int],
    models: Sequence[str],
    model_bounds: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Find the canonical minimum rectangular exact cover for one phase.

    Rectangles use contiguous anchor blocks and canonically ordered model
    subsets.  The exact-cover objective first minimizes the number of shards,
    then minimizes model cuts (wide/all-model rectangles win), and finally
    applies a stable anchor/model ordering.  The phase grids contain at most
    18 cells, so an exhaustive memoized cover is small and makes the
    minimality claim independently replayable instead of heuristic.
    """

    ordered_anchors = [int(value) for value in anchors]
    ordered_models = [str(value) for value in models]
    if not ordered_anchors or not ordered_models:
        raise PreflightError(f"runtime phase is empty: {queue}/{phase}")
    if len(set(ordered_anchors)) != len(ordered_anchors):
        raise PreflightError(f"runtime phase repeats an anchor: {queue}/{phase}")
    if len(set(ordered_models)) != len(ordered_models):
        raise PreflightError(f"runtime phase repeats a model: {queue}/{phase}")
    usable_seconds = RUNTIME_JOB_GUARD_SECONDS - RUNTIME_JOB_OVERHEAD_SECONDS
    for model_id in ordered_models:
        bound = model_bounds.get(model_id)
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
            or float(bound) <= 0.0
        ):
            raise PreflightError(f"runtime plan model bound is invalid: {model_id}")
        if float(bound) > usable_seconds:
            raise PreflightError(
                "one atomic anchor/model cell exceeds the 3h20 runtime guard: "
                f"{queue}/{phase}/{model_id}"
            )

    n_anchors = len(ordered_anchors)
    n_models = len(ordered_models)
    full_mask = (1 << (n_anchors * n_models)) - 1
    candidates: list[dict[str, Any]] = []
    for first_anchor in range(n_anchors):
        for last_anchor in range(first_anchor, n_anchors):
            anchor_indices = tuple(range(first_anchor, last_anchor + 1))
            for model_mask in range(1, 1 << n_models):
                model_indices = tuple(
                    index
                    for index in range(n_models)
                    if model_mask & (1 << index)
                )
                seconds = len(anchor_indices) * sum(
                    float(model_bounds[ordered_models[index]])
                    for index in model_indices
                )
                if seconds > usable_seconds:
                    continue
                cell_mask = 0
                for anchor_index in anchor_indices:
                    for model_index in model_indices:
                        cell_mask |= 1 << (anchor_index * n_models + model_index)
                candidates.append(
                    {
                        "anchor_indices": anchor_indices,
                        "model_indices": model_indices,
                        "seconds": seconds,
                        "cell_mask": cell_mask,
                    }
                )

    by_cell: list[list[int]] = [[] for _ in range(n_anchors * n_models)]
    for candidate_index, candidate in enumerate(candidates):
        for cell_index in range(n_anchors * n_models):
            if int(candidate["cell_mask"]) & (1 << cell_index):
                by_cell[cell_index].append(candidate_index)

    def candidate_order(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        anchor_indices = tuple(candidate["anchor_indices"])
        model_indices = tuple(candidate["model_indices"])
        return (
            anchor_indices[0],
            model_indices[0],
            -len(anchor_indices),
            -len(model_indices),
            anchor_indices,
            model_indices,
        )

    def solution_key(indices: Sequence[int]) -> tuple[Any, ...]:
        ordered = sorted(indices, key=lambda index: candidate_order(candidates[index]))
        return (
            len(ordered),
            # For one anchor, every rectangle touching it is one model group;
            # groups minus one is the exact number of model cuts.  Summing
            # rectangle anchor-heights and subtracting the fixed anchor count
            # therefore implements the stated all-models-together tie-break.
            sum(
                len(candidates[index]["anchor_indices"])
                for index in ordered
            )
            - n_anchors,
            tuple(candidate_order(candidates[index]) for index in ordered),
        )

    @lru_cache(maxsize=None)
    def best_cover(covered: int) -> tuple[int, ...] | None:
        if covered == full_mask:
            return ()
        remaining = full_mask ^ covered
        first_cell_bit = remaining & -remaining
        first_cell = first_cell_bit.bit_length() - 1
        best: tuple[int, ...] | None = None
        for candidate_index in by_cell[first_cell]:
            candidate_mask = int(candidates[candidate_index]["cell_mask"])
            if candidate_mask & covered:
                continue
            suffix = best_cover(covered | candidate_mask)
            if suffix is None:
                continue
            proposal = (candidate_index, *suffix)
            if best is None or solution_key(proposal) < solution_key(best):
                best = proposal
        return best

    selected = best_cover(0)
    if selected is None:
        raise PreflightError(f"runtime phase has no safe exact cover: {queue}/{phase}")
    selected = tuple(
        sorted(selected, key=lambda index: candidate_order(candidates[index]))
    )
    shards: list[dict[str, Any]] = []
    for shard_index, candidate_index in enumerate(selected):
        candidate = candidates[candidate_index]
        shard_anchors = [
            ordered_anchors[index] for index in candidate["anchor_indices"]
        ]
        shard_models = [
            ordered_models[index] for index in candidate["model_indices"]
        ]
        seconds = float(candidate["seconds"])
        repeats_per_job = int(usable_seconds // seconds)
        if repeats_per_job < 1:
            raise PreflightError(
                f"runtime phase produced an unsafe atomic shard: {queue}/{phase}"
            )
        predicted = RUNTIME_JOB_OVERHEAD_SECONDS + repeats_per_job * seconds
        if predicted > RUNTIME_JOB_GUARD_SECONDS:
            raise PreflightError(
                f"runtime phase produced an unsafe predicted job: {queue}/{phase}"
            )
        shards.append(
            {
                "shard_id": f"{queue}-{phase}-{shard_index:03d}",
                "anchors": shard_anchors,
                "models": shard_models,
                "cells_per_repeat": len(shard_anchors) * len(shard_models),
                "seconds_per_repeat_bound": seconds,
                "repeats_per_job": repeats_per_job,
                "predicted_job_seconds": predicted,
            }
        )
    return shards


def _canonical_runtime_queue_plans(
    profile: str,
    task_spec: Mapping[str, Any],
    model_bounds: Mapping[str, float],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    registry = _runtime_phase_registry(profile, task_spec)
    return {
        queue: {
            phase: _canonical_runtime_phase_shards(
                queue=queue,
                phase=phase,
                anchors=record["anchors"],
                models=record["models"],
                model_bounds=model_bounds,
            )
            for phase, record in phases.items()
        }
        for queue, phases in registry.items()
    }


def build_runtime_plan(
    benchmark_run: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Freeze endpoint benchmarks into a four-lane, four-hour-safe plan."""

    if isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float)):
        raise PreflightError("runtime safety_factor must be numeric")
    if float(safety_factor) < 1.0:
        raise PreflightError("runtime safety_factor must be at least 1")
    from .execution import open_run

    root = Path(repo_root).resolve()
    # The exact GPU runtime is already frozen by preflight. Runtime reduction
    # is a queue-neutral read-only operation and runs on a 12-core CPU node.
    storage = open_run(
        benchmark_run, repo_root=root, verify_tensorflow_runtime=False
    )
    manifest = storage.manifest
    benchmark = manifest.get("execution", {}).get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("enabled") is not True:
        raise PreflightError("runtime planning requires a benchmark-mode run")
    if manifest.get("execution", {}).get("mode") != "benchmark":
        raise PreflightError("runtime planning source is not a benchmark manifest")
    target_profile = str(manifest.get("execution", {}).get("profile", ""))
    if target_profile not in {"pilot10", "full100", "sensitivity20"}:
        raise PreflightError(
            "runtime planning supports only pilot10, full100, and sensitivity20 profiles"
        )
    cells = benchmark.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise PreflightError("benchmark manifest must project exactly ten cells")
    anchors = sorted(int(value) for value in benchmark.get("anchors", []))
    expected_benchmark_anchors = (
        sorted(
            int(value)
            for item in manifest["task_spec"].get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
            for value in item["execution"]["anchors"]
        )
        if target_profile == "sensitivity20"
        else sorted(
            [
                int(min(manifest["task_spec"]["anchors"])),
                int(max(manifest["task_spec"]["anchors"])),
            ]
        )
    )
    if anchors != expected_benchmark_anchors:
        raise PreflightError("benchmark did not measure the smallest and largest anchors")
    intervention_binding = benchmark.get("intervention")
    if target_profile == "sensitivity20":
        frozen = [
            item
            for item in manifest["task_spec"].get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
        ]
        expected_intervention = (
            {
                "branch": "frozen_sensitivity",
                "sensitivity_id": str(frozen[0]["id"]),
                "prepared_variant": str(
                    frozen[0]["execution"]["prepared_variant"]
                ),
            }
            if len(frozen) == 1
            else None
        )
        if intervention_binding != expected_intervention:
            raise PreflightError(
                "sensitivity runtime benchmark did not bind the effective intervention"
            )
    elif intervention_binding is not None:
        raise PreflightError(
            f"{target_profile} runtime benchmark cannot bind a sensitivity"
        )

    observations: list[dict[str, Any]] = []
    bounds_by_model: dict[str, float] = {}
    for record in cells:
        if target_profile == "sensitivity20" and (
            record.get("branch") != "frozen_sensitivity"
            or record.get("sensitivity_id")
            != intervention_binding["sensitivity_id"]
        ):
            raise PreflightError(
                "sensitivity runtime benchmark cell does not execute its intervention"
            )
        key = CellKey(
            str(record["branch"]), int(record["repeat"]),
            int(record["n_train"]), str(record["model"]),
        )
        if not storage.validate_cell(key).is_complete:
            raise PreflightError(f"benchmark cell is incomplete: {key.relative_dir}")
        metrics = storage.load_metrics(key)
        history = storage.load_history(key)
        elapsed = metrics.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not np.isfinite(float(elapsed))
            or float(elapsed) <= 0
        ):
            raise PreflightError(f"benchmark cell lacks elapsed time: {key.relative_dir}")
        model_entry = manifest["task_spec"]["models"][key.model]
        family = str(model_entry["family"])
        observed_epochs: int | None = None
        max_epoch_equivalent: int | None = None
        epoch_multiplier = 1.0
        if family in set(PRIMARY_NEURAL_FAMILIES):
            if not isinstance(history, Mapping):
                raise PreflightError(f"benchmark neural history is invalid: {key.relative_dir}")
            observed_epochs = _observed_training_epochs(history)
            config = model_entry.get("selected_config")
            if not isinstance(config, Mapping):
                raise PreflightError("benchmark neural model has no selected config")
            max_epochs = int(config.get("max_epochs", 0))
            if max_epochs <= 0:
                raise PreflightError("benchmark neural model has no positive max_epochs")
            max_epoch_equivalent = 2 * max_epochs
            epoch_multiplier = max(1.0, max_epoch_equivalent / observed_epochs)
        bounded = float(elapsed) * epoch_multiplier * float(safety_factor)
        bounds_by_model[key.model] = max(bounds_by_model.get(key.model, 0.0), bounded)
        observations.append(
            {
                "branch": key.branch,
                "repeat": key.repeat,
                "n_train": key.n_train,
                "model": key.model,
                "family": family,
                "queue": str(record["queue"]),
                "elapsed_seconds": float(elapsed),
                "observed_training_epochs": observed_epochs,
                "max_epoch_equivalent": max_epoch_equivalent,
                "bounded_seconds": bounded,
                "metrics_sha256": _json_hash(dict(metrics)),
            }
        )
    if set(bounds_by_model) != set(manifest["task_spec"]["models"]):
        raise PreflightError("benchmark did not cover every primary model")

    queue_plans = _canonical_runtime_queue_plans(
        target_profile, manifest["task_spec"], bounds_by_model
    )

    try:
        output_relative = Path(output_path).resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError("runtime plan must be stored inside repo_root") from exc
    provenance = manifest["provenance"]
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_PLAN_SCHEMA_VERSION,
        "profile": target_profile,
        "task_id": manifest["task_id"],
        "task_spec_hash": manifest["task_spec_hash"],
        "prepared_hash": manifest["prepared"]["prepared_hash"],
        "preflight_receipt_hash": manifest["preflight"]["receipt_hash"],
        "code_identity_sha256": _json_hash({"code": provenance["code"]}),
        "benchmark_run": str(Path(benchmark_run).resolve()),
        "benchmark_manifest_hash": storage.manifest_hash,
        "safety_factor": float(safety_factor),
        "walltime_seconds": BETTY_WALLTIME_SECONDS,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "gpu_lanes": BETTY_GPU_LANES,
        "sharding": RUNTIME_SHARDING,
        "intervention_binding": intervention_binding,
        "observations": observations,
        "model_cell_seconds_bound": bounds_by_model,
        "queue_plans": queue_plans,
        "output_path": output_relative,
    }
    payload["runtime_plan_hash"] = _json_hash(payload)
    immutable_write_json(output_path, payload)
    return payload


def _bounded_phase_seconds(
    elapsed: Any,
    observed_epochs: Any,
    max_epochs: Any,
    safety_factor: float,
    *,
    label: str,
) -> float:
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
        or isinstance(observed_epochs, bool)
        or not isinstance(observed_epochs, int)
        or observed_epochs < 1
        or isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs < 1
    ):
        raise PreflightError(f"runtime neural phase timing is invalid: {label}")
    return (
        float(elapsed)
        * max(1.0, float(max_epochs) / float(observed_epochs))
        * float(safety_factor)
    )


def _benchmark_intervention_binding(
    manifest: Mapping[str, Any], target_profile: str
) -> tuple[list[int], dict[str, Any] | None]:
    benchmark = manifest.get("execution", {}).get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("enabled") is not True:
        raise PreflightError("runtime planning requires a benchmark-mode run")
    anchors = sorted(int(value) for value in benchmark.get("anchors", []))
    expected = (
        sorted(
            int(value)
            for item in manifest["task_spec"].get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
            for value in item["execution"]["anchors"]
        )
        if target_profile == "sensitivity20"
        else sorted(
            [
                int(min(manifest["task_spec"]["anchors"])),
                int(max(manifest["task_spec"]["anchors"])),
            ]
        )
    )
    if anchors != expected:
        raise PreflightError("benchmark did not measure the smallest and largest anchors")
    intervention = benchmark.get("intervention")
    if target_profile == "sensitivity20":
        frozen = [
            item
            for item in manifest["task_spec"].get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
        ]
        expected_intervention = (
            {
                "branch": "frozen_sensitivity",
                "sensitivity_id": str(frozen[0]["id"]),
                "prepared_variant": str(frozen[0]["execution"]["prepared_variant"]),
            }
            if len(frozen) == 1
            else None
        )
        if intervention != expected_intervention:
            raise PreflightError(
                "sensitivity runtime benchmark did not bind the effective intervention"
            )
    elif intervention is not None:
        raise PreflightError(
            f"{target_profile} runtime benchmark cannot bind a sensitivity"
        )
    return anchors, None if intervention is None else dict(intervention)


def _canonical_runtime_v3_queue_plans(
    profile: str,
    task_spec: Mapping[str, Any],
    model_phase_bounds: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    registry = _runtime_phase_registry(profile, task_spec)
    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        "cpu_tabular": {},
        "gpu_neural": {},
    }
    cpu_bounds = {
        model: float(model_phase_bounds[model]["monolithic"])
        for model in CPU_MODEL_ROLES
    }
    for phase, record in registry["cpu_tabular"].items():
        result["cpu_tabular"][phase] = _canonical_runtime_phase_shards(
            queue="cpu_tabular",
            phase=phase,
            anchors=record["anchors"],
            models=record["models"],
            model_bounds=cpu_bounds,
        )
    neural_bounds = {
        model: {
            neural_phase: float(model_phase_bounds[model][neural_phase])
            for neural_phase in NEURAL_PHASES
        }
        for model in GPU_MODEL_ROLES
    }
    try:
        for phase, record in registry["gpu_neural"].items():
            result["gpu_neural"][phase] = canonical_neural_phase_shards(
                experimental_phase=phase,
                anchors=record["anchors"],
                models=record["models"],
                model_phase_seconds_bound=neural_bounds,
            )
    except RuntimePhasePlanError as exc:
        raise PreflightError(str(exc)) from exc
    return result


def build_runtime_plan_v3(
    benchmark_run: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    safety_factor: float = 1.25,
    _schema_version: str = RUNTIME_PLAN_V3_SCHEMA_VERSION,
    _sharding: str = RUNTIME_V3_SHARDING,
    _resource_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze split selector/refit endpoints into the four-MIG90 plan."""

    if isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float)):
        raise PreflightError("runtime safety_factor must be numeric")
    if float(safety_factor) < 1.0:
        raise PreflightError("runtime safety_factor must be at least 1")
    from .execution import (
        _cell_record,
        _effective_task_spec,
        _execution_design,
        load_neural_selector_receipt,
        neural_selector_receipt_path,
        open_run,
    )
    from .prepared import load_prepared_task
    from .runner import TaskRuntime, validate_neural_selector_for_cell

    root = Path(repo_root).resolve()
    storage = open_run(
        benchmark_run, repo_root=root, verify_tensorflow_runtime=False
    )
    manifest = storage.manifest
    benchmark = manifest.get("execution", {}).get("benchmark")
    if (
        manifest.get("execution", {}).get("mode") != "benchmark"
        or not isinstance(benchmark, Mapping)
        or benchmark.get("enabled") is not True
    ):
        raise PreflightError("runtime planning source is not a benchmark manifest")
    target_profile = str(manifest.get("execution", {}).get("profile", ""))
    if target_profile not in {"pilot10", "full100", "sensitivity20"}:
        raise PreflightError("runtime planning has an unsupported profile")
    cells = benchmark.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise PreflightError("benchmark manifest must project exactly ten cells")
    _, intervention_binding = _benchmark_intervention_binding(
        manifest, target_profile
    )

    prepared = load_prepared_task(
        root / str(manifest["prepared"]["path"]),
        mmap_mode="r",
        verify_files=True,
    )
    gpu_runtime = TaskRuntime(prepared, queue="gpu_neural")
    effective_spec = _effective_task_spec(manifest)
    execution_design = _execution_design(manifest)

    observations: list[dict[str, Any]] = []
    phase_bounds: dict[str, dict[str, float]] = {}
    for record in cells:
        key = CellKey(
            str(record["branch"]),
            int(record["repeat"]),
            int(record["n_train"]),
            str(record["model"]),
        )
        if not storage.validate_cell(key).is_complete:
            raise PreflightError(f"benchmark cell is incomplete: {key.relative_dir}")
        metrics = storage.load_metrics(key)
        elapsed = metrics.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise PreflightError(f"benchmark cell lacks elapsed time: {key.relative_dir}")
        model_entry = manifest["task_spec"]["models"][key.model]
        family = str(model_entry["family"])
        base = {
            "branch": key.branch,
            "repeat": key.repeat,
            "n_train": key.n_train,
            "model": key.model,
            "family": family,
            "queue": str(record["queue"]),
            "metrics_sha256": _json_hash(dict(metrics)),
        }
        if family not in set(PRIMARY_NEURAL_FAMILIES):
            bounded = float(elapsed) * float(safety_factor)
            phase_bounds.setdefault(key.model, {})["monolithic"] = max(
                phase_bounds.get(key.model, {}).get("monolithic", 0.0), bounded
            )
            observations.append(
                {
                    **base,
                    "elapsed_seconds": float(elapsed),
                    "bounded_seconds": bounded,
                }
            )
            continue

        cell_record = _cell_record(execution_design, key)
        selector_path = neural_selector_receipt_path(benchmark_run, key)
        try:
            selector = load_neural_selector_receipt(
                benchmark_run,
                storage,
                key,
                cell_record,
            )
            validate_neural_selector_for_cell(
                selector["selector_core"],
                gpu_runtime,
                effective_spec,
                execution_design,
                repeat=key.repeat,
                n_train=key.n_train,
                model_id=key.model,
                branch=key.branch,
                ablation_id=cell_record.get("ablation_id"),
                sensitivity_id=cell_record.get("sensitivity_id"),
            )
        except Exception as exc:
            raise PreflightError(
                f"benchmark selector receipt failed: {key.relative_dir}: {exc}"
            ) from exc
        config = model_entry.get("selected_config")
        if not isinstance(config, Mapping):
            raise PreflightError("benchmark neural model has no selected config")
        max_epochs = int(config.get("max_epochs", 0))
        selector_bounded = _bounded_phase_seconds(
            selector["selector_elapsed_seconds"],
            selector["selector_observed_epochs"],
            max_epochs,
            float(safety_factor),
            label=f"{key.model}/selector",
        )
        refit_bounded = _bounded_phase_seconds(
            elapsed,
            selector["refit_epochs"],
            max_epochs,
            float(safety_factor),
            label=f"{key.model}/refit",
        )
        prior = phase_bounds.setdefault(key.model, {})
        prior["selector"] = max(prior.get("selector", 0.0), selector_bounded)
        prior["refit"] = max(prior.get("refit", 0.0), refit_bounded)
        observations.append(
            {
                **base,
                "selector_elapsed_seconds": float(
                    selector["selector_elapsed_seconds"]
                ),
                "selector_observed_epochs": int(
                    selector["selector_observed_epochs"]
                ),
                "selector_bounded_seconds": selector_bounded,
                "refit_elapsed_seconds": float(elapsed),
                "refit_epochs": int(selector["refit_epochs"]),
                "refit_bounded_seconds": refit_bounded,
                "max_epochs": max_epochs,
                "selector_receipt_hash": str(selector["receipt_hash"]),
                "selector_receipt_sha256": sha256_file(selector_path),
            }
        )
    expected_bound_shapes = {
        **{model: {"monolithic"} for model in CPU_MODEL_ROLES},
        **{model: set(NEURAL_PHASES) for model in GPU_MODEL_ROLES},
    }
    if set(phase_bounds) != set(expected_bound_shapes) or any(
        set(phase_bounds[model]) != fields
        for model, fields in expected_bound_shapes.items()
    ):
        raise PreflightError("benchmark did not cover every model phase")
    queue_plans = _canonical_runtime_v3_queue_plans(
        target_profile, manifest["task_spec"], phase_bounds
    )
    resource_contract = (
        betty_runtime_v3_resource_contract()
        if _resource_contract is None
        else dict(_resource_contract)
    )
    gpu_lanes = resource_contract.get("gpu", {}).get("lanes")
    cpu_lanes = resource_contract.get("cpu", {}).get("lanes")
    if (
        isinstance(gpu_lanes, bool)
        or not isinstance(gpu_lanes, int)
        or gpu_lanes < 1
        or isinstance(cpu_lanes, bool)
        or not isinstance(cpu_lanes, int)
        or cpu_lanes < 1
    ):
        raise PreflightError("phased runtime resource lane registry is invalid")
    try:
        output_relative = Path(output_path).resolve().relative_to(root).as_posix()
        benchmark_relative = storage.run_dir.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError(
            "runtime plan and benchmark must be stored inside repo_root"
        ) from exc
    provenance = manifest["provenance"]
    payload: dict[str, Any] = {
        "schema_version": _schema_version,
        "profile": target_profile,
        "task_id": manifest["task_id"],
        "task_spec_hash": manifest["task_spec_hash"],
        "prepared_hash": manifest["prepared"]["prepared_hash"],
        "preflight_receipt_hash": manifest["preflight"]["receipt_hash"],
        "code_identity_sha256": _json_hash({"code": provenance["code"]}),
        "benchmark_run": benchmark_relative,
        "benchmark_manifest_hash": storage.manifest_hash,
        "safety_factor": float(safety_factor),
        "walltime_seconds": BETTY_WALLTIME_SECONDS,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "gpu_lanes": gpu_lanes,
        "cpu_lanes": cpu_lanes,
        "sharding": _sharding,
        "resource_contract": resource_contract,
        "intervention_binding": intervention_binding,
        "observations": observations,
        "model_phase_seconds_bound": phase_bounds,
        "queue_plans": queue_plans,
        "output_path": output_relative,
    }
    payload["runtime_plan_hash"] = _json_hash(payload)
    immutable_write_json(output_path, payload)
    return payload


def build_runtime_plan_v4(
    benchmark_run: str | Path,
    output_path: str | Path,
    *,
    determinism_probe: Mapping[str, Any],
    repo_root: str | Path = ".",
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Freeze attempt9 timings under the probe-bound MIG45/CPU8 layout."""

    # Read only the task identity here; ``build_runtime_plan_v3`` immediately
    # reopens and fully validates this checksummed benchmark manifest.
    root = Path(repo_root).resolve()
    benchmark_path = Path(benchmark_run)
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    try:
        benchmark_path.resolve().relative_to(root)
        benchmark_manifest = json.loads(
            (benchmark_path.resolve() / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PreflightError("v4 runtime benchmark manifest is unreadable") from exc
    task_id = str(benchmark_manifest.get("task_id", ""))
    resource = betty_runtime_v4_resource_contract(
        determinism_probe, repo_root=root, task_id=task_id
    )
    return build_runtime_plan_v3(
        benchmark_run,
        output_path,
        repo_root=repo_root,
        safety_factor=safety_factor,
        _schema_version=RUNTIME_PLAN_V4_SCHEMA_VERSION,
        _sharding=RUNTIME_V4_SHARDING,
        _resource_contract=resource,
    )


def _validate_runtime_plan_v3_sources(
    value: Mapping[str, Any], *, repo_root: str | Path
) -> None:
    """Replay every timing observation against its immutable benchmark bytes.

    The runtime-plan hash alone is not an admission proof: a caller could
    otherwise rewrite both the observed timings and the embedded hash.  This
    verifier reopens the checksum-bound ten-cell benchmark, validates every
    final cell and neural selector receipt, and requires byte-derived timing
    identities before any shard duration can reach Slurm.
    """

    root = Path(repo_root).resolve()
    raw_run = value.get("benchmark_run")
    if not isinstance(raw_run, str) or not raw_run:
        raise PreflightError("phased runtime benchmark path is missing")
    relative = Path(raw_run)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(
            "phased runtime benchmark path must be repository-relative"
        )
    benchmark_run = (root / relative).resolve()
    try:
        observed_relative = benchmark_run.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError("phased runtime benchmark escapes repo_root") from exc
    if observed_relative != raw_run:
        raise PreflightError("phased runtime benchmark path is not canonical")

    from .execution import (
        _cell_record,
        _effective_task_spec,
        _execution_design,
        load_neural_selector_receipt,
        neural_selector_receipt_path,
        open_run,
    )
    from .prepared import load_prepared_task
    from .runner import TaskRuntime, validate_neural_selector_for_cell

    try:
        storage = open_run(
            benchmark_run, repo_root=root, verify_tensorflow_runtime=False
        )
    except Exception as exc:
        raise PreflightError(
            f"phased runtime benchmark failed verification: {exc}"
        ) from exc
    manifest = storage.manifest
    benchmark = manifest.get("execution", {}).get("benchmark")
    cells = benchmark.get("cells") if isinstance(benchmark, Mapping) else None
    if (
        manifest.get("execution", {}).get("mode") != "benchmark"
        or not isinstance(cells, list)
        or len(cells) != 10
        or storage.manifest_hash != value.get("benchmark_manifest_hash")
        or manifest.get("execution", {}).get("profile") != value.get("profile")
        or manifest.get("task_id") != value.get("task_id")
        or manifest.get("task_spec_hash") != value.get("task_spec_hash")
        or manifest.get("prepared", {}).get("prepared_hash")
        != value.get("prepared_hash")
        or manifest.get("preflight", {}).get("receipt_hash")
        != value.get("preflight_receipt_hash")
    ):
        raise PreflightError(
            "phased runtime benchmark identity differs from the plan"
        )
    _, intervention = _benchmark_intervention_binding(
        manifest, str(value.get("profile", ""))
    )
    if intervention != value.get("intervention_binding"):
        raise PreflightError(
            "phased runtime benchmark intervention differs from the plan"
        )

    records: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for record in cells:
        if not isinstance(record, Mapping):
            raise PreflightError("phased runtime benchmark cell is invalid")
        try:
            identity = (
                str(record["branch"]),
                int(record["repeat"]),
                int(record["n_train"]),
                str(record["model"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError(
                "phased runtime benchmark cell identity is invalid"
            ) from exc
        if identity in records:
            raise PreflightError("phased runtime benchmark repeats a cell")
        records[identity] = record

    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != 10:
        raise PreflightError(
            "phased runtime plan must bind ten endpoint observations"
        )
    execution_design = _execution_design(manifest)
    effective_spec = _effective_task_spec(manifest)
    gpu_runtime: Any | None = None
    seen: set[tuple[str, int, int, str]] = set()
    common_fields = {
        "branch",
        "repeat",
        "n_train",
        "model",
        "family",
        "queue",
        "metrics_sha256",
    }
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise PreflightError("phased runtime observation is invalid")
        try:
            identity = (
                str(observation["branch"]),
                int(observation["repeat"]),
                int(observation["n_train"]),
                str(observation["model"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError(
                "phased runtime observation identity is invalid"
            ) from exc
        model = identity[3]
        neural = model in GPU_MODEL_ROLES
        expected_fields = common_fields | (
            {
                "selector_elapsed_seconds",
                "selector_observed_epochs",
                "selector_bounded_seconds",
                "refit_elapsed_seconds",
                "refit_epochs",
                "refit_bounded_seconds",
                "max_epochs",
                "selector_receipt_hash",
                "selector_receipt_sha256",
            }
            if neural
            else {"elapsed_seconds", "bounded_seconds"}
        )
        record = records.get(identity)
        if (
            set(observation) != expected_fields
            or record is None
            or identity in seen
        ):
            raise PreflightError(
                "phased runtime observation does not exactly cover its benchmark"
            )
        seen.add(identity)
        key = CellKey(*identity)
        if not storage.validate_cell(key).is_complete:
            raise PreflightError(
                f"phased runtime benchmark cell is incomplete: {key.relative_dir}"
            )
        metrics = storage.load_metrics(key)
        elapsed = metrics.get("elapsed_seconds")
        model_entry = manifest.get("task_spec", {}).get("models", {}).get(model)
        if (
            not isinstance(model_entry, Mapping)
            or observation.get("family") != model_entry.get("family")
            or observation.get("queue") != record.get("queue")
            or observation.get("metrics_sha256") != _json_hash(dict(metrics))
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise PreflightError(
                f"phased runtime benchmark bytes differ: {key.relative_dir}"
            )
        if not neural:
            if float(observation.get("elapsed_seconds", -1.0)) != float(elapsed):
                raise PreflightError(
                    f"tabular runtime observation differs: {key.relative_dir}"
                )
            continue

        cell_record = _cell_record(execution_design, key)
        try:
            selector = load_neural_selector_receipt(
                benchmark_run, storage, key, cell_record
            )
            if gpu_runtime is None:
                prepared = load_prepared_task(
                    root / str(manifest["prepared"]["path"]),
                    mmap_mode="r",
                    verify_files=True,
                )
                gpu_runtime = TaskRuntime(prepared, queue="gpu_neural")
            validate_neural_selector_for_cell(
                selector["selector_core"],
                gpu_runtime,
                effective_spec,
                execution_design,
                repeat=key.repeat,
                n_train=key.n_train,
                model_id=key.model,
                branch=key.branch,
                ablation_id=cell_record.get("ablation_id"),
                sensitivity_id=cell_record.get("sensitivity_id"),
            )
        except Exception as exc:
            raise PreflightError(
                f"phased selector receipt failed replay: {key.relative_dir}: {exc}"
            ) from exc
        selector_path = neural_selector_receipt_path(benchmark_run, key)
        config = model_entry.get("selected_config")
        if (
            not isinstance(config, Mapping)
            or observation.get("selector_receipt_hash")
            != selector.get("receipt_hash")
            or observation.get("selector_receipt_sha256")
            != sha256_file(selector_path)
            or float(observation.get("selector_elapsed_seconds", -1.0))
            != float(selector.get("selector_elapsed_seconds", -2.0))
            or observation.get("selector_observed_epochs")
            != selector.get("selector_observed_epochs")
            or observation.get("refit_epochs") != selector.get("refit_epochs")
            or observation.get("max_epochs") != int(config.get("max_epochs", 0))
            or float(observation.get("refit_elapsed_seconds", -1.0))
            != float(elapsed)
        ):
            raise PreflightError(
                f"phased selector/refit observation differs: {key.relative_dir}"
            )
    if seen != set(records):
        raise PreflightError(
            "phased runtime observations do not cover the benchmark exactly"
        )


def validate_runtime_queue_plans_v3(
    runtime_plan: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Regenerate the exact v3 CPU rectangles and neural cell chains."""

    schema = runtime_plan.get("schema_version")
    if schema == RUNTIME_PLAN_V3_SCHEMA_VERSION:
        sharding = RUNTIME_V3_SHARDING
        resource = betty_runtime_v3_resource_contract()
        gpu_lanes = BETTY_V3_GPU_LANES
        cpu_lanes = BETTY_V3_CPU_LANES
    elif schema == RUNTIME_PLAN_V4_SCHEMA_VERSION:
        sharding = RUNTIME_V4_SHARDING
        embedded = runtime_plan.get("resource_contract")
        probe = (
            embedded.get("determinism_probe")
            if isinstance(embedded, Mapping)
            else None
        )
        if not isinstance(probe, Mapping):
            raise PreflightError("v4 phased runtime plan lacks its green probe")
        try:
            resource = betty_runtime_v4_resource_contract(
                probe, task_id=str(runtime_plan.get("task_id", ""))
            )
        except RuntimePhasePlanError as exc:
            raise PreflightError(str(exc)) from exc
        gpu_lanes = BETTY_V4_GPU_LANES
        cpu_lanes = BETTY_V4_CPU_LANES
    else:
        raise PreflightError("unsupported phased runtime plan")
    if (
        runtime_plan.get("sharding") != sharding
        or runtime_plan.get("resource_contract") != resource
        or runtime_plan.get("gpu_lanes") != gpu_lanes
        or runtime_plan.get("cpu_lanes") != cpu_lanes
    ):
        raise PreflightError("phased runtime resource/capacity contract drifted")
    bounds = runtime_plan.get("model_phase_seconds_bound")
    expected_shapes = {
        **{model: {"monolithic"} for model in CPU_MODEL_ROLES},
        **{model: set(NEURAL_PHASES) for model in GPU_MODEL_ROLES},
    }
    if not isinstance(bounds, Mapping) or set(bounds) != set(expected_shapes):
        raise PreflightError("phased runtime model timing registry differs")
    normalized: dict[str, dict[str, float]] = {}
    for model, fields in expected_shapes.items():
        raw = bounds[model]
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise PreflightError(f"phased runtime timing fields differ: {model}")
        normalized[model] = {}
        for phase in fields:
            value = raw[phase]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise PreflightError(
                    f"phased runtime model bound is invalid: {model}/{phase}"
                )
            normalized[model][phase] = float(value)
    expected = _canonical_runtime_v3_queue_plans(
        str(runtime_plan.get("profile", "")), task_spec, normalized
    )
    observed = runtime_plan.get("queue_plans")
    if observed != expected:
        raise PreflightError(
            "phased runtime queue shards are not the canonical exact cover"
        )
    # Replay neural shards independently so the exact-cover error is explicit.
    registry = _runtime_phase_registry(str(runtime_plan.get("profile", "")), task_spec)
    neural_bounds = {
        model: normalized[model] for model in GPU_MODEL_ROLES
    }
    try:
        for phase, record in registry["gpu_neural"].items():
            validate_neural_phase_shards(
                observed["gpu_neural"][phase],
                experimental_phase=phase,
                anchors=record["anchors"],
                models=record["models"],
                model_phase_seconds_bound=neural_bounds,
            )
    except RuntimePhasePlanError as exc:
        raise PreflightError(str(exc)) from exc
    return expected


def validate_runtime_queue_plans_v4(
    runtime_plan: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Regenerate the exact attempt9 CPU8/MIG45 phased execution cover."""

    if runtime_plan.get("schema_version") != RUNTIME_PLAN_V4_SCHEMA_VERSION:
        raise PreflightError("runtime-plan v4 schema differs")
    return validate_runtime_queue_plans_v3(runtime_plan, task_spec)


def validate_runtime_queue_plans(
    runtime_plan: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Regenerate and replay the canonical rectangular phase-shard cover."""

    profile = str(runtime_plan.get("profile", ""))
    if profile not in {"pilot10", "full100", "sensitivity20"}:
        raise PreflightError("runtime plan has an unsupported execution profile")
    if runtime_plan.get("schema_version") != RUNTIME_PLAN_SCHEMA_VERSION:
        raise PreflightError("unsupported runtime plan")
    if (
        runtime_plan.get("job_guard_seconds") != RUNTIME_JOB_GUARD_SECONDS
        or runtime_plan.get("job_overhead_seconds")
        != RUNTIME_JOB_OVERHEAD_SECONDS
        or runtime_plan.get("sharding") != RUNTIME_SHARDING
    ):
        raise PreflightError("runtime plan startup/dispatch or sharding guard drifted")
    models = task_spec.get("models")
    bounds = runtime_plan.get("model_cell_seconds_bound")
    if (
        not isinstance(models, Mapping)
        or not isinstance(bounds, Mapping)
        or set(bounds) != set(PRIMARY_MODEL_ROLES)
        or set(models) != set(PRIMARY_MODEL_ROLES)
    ):
        raise PreflightError("runtime plan model timing registry differs")
    normalized_bounds: dict[str, float] = {}
    for model_id in PRIMARY_MODEL_ROLES:
        value = bounds[model_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise PreflightError(
                f"runtime plan model bound is invalid: {model_id}"
            )
        normalized_bounds[str(model_id)] = float(value)
    expected = _canonical_runtime_queue_plans(
        profile, task_spec, normalized_bounds
    )
    observed = runtime_plan.get("queue_plans")
    if observed != expected:
        raise PreflightError(
            "runtime plan queue/phase shards are not the canonical exact cover"
        )
    return expected


def _validate_runtime_plan_v3_value(
    value: Mapping[str, Any],
    path: Path,
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "profile",
        "task_id",
        "task_spec_hash",
        "prepared_hash",
        "preflight_receipt_hash",
        "code_identity_sha256",
        "benchmark_run",
        "benchmark_manifest_hash",
        "safety_factor",
        "walltime_seconds",
        "job_guard_seconds",
        "job_overhead_seconds",
        "gpu_lanes",
        "cpu_lanes",
        "sharding",
        "resource_contract",
        "intervention_binding",
        "observations",
        "model_phase_seconds_bound",
        "queue_plans",
        "output_path",
        "runtime_plan_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise PreflightError("phased runtime plan root schema drifted")
    unsigned = {key: item for key, item in value.items() if key != "runtime_plan_hash"}
    if value.get("runtime_plan_hash") != _json_hash(unsigned):
        raise PreflightError("phased runtime-plan hash is invalid")
    provenance = manifest.get("provenance", {})
    manifest_profile = str(manifest.get("execution", {}).get("profile", ""))
    expected_intervention: dict[str, Any] | None = None
    if manifest_profile == "sensitivity20":
        frozen = [
            item
            for item in manifest.get("task_spec", {}).get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
        ]
        if len(frozen) != 1 or not isinstance(frozen[0].get("execution"), Mapping):
            raise PreflightError(
                "sensitivity runtime-plan validation lacks one frozen intervention"
            )
        expected_intervention = {
            "branch": "frozen_sensitivity",
            "sensitivity_id": str(frozen[0]["id"]),
            "prepared_variant": str(frozen[0]["execution"]["prepared_variant"]),
        }
    schema = value.get("schema_version")
    if schema == RUNTIME_PLAN_V3_SCHEMA_VERSION:
        resource = betty_runtime_v3_resource_contract()
        gpu_lanes = BETTY_V3_GPU_LANES
        cpu_lanes = BETTY_V3_CPU_LANES
        sharding = RUNTIME_V3_SHARDING
    elif schema == RUNTIME_PLAN_V4_SCHEMA_VERSION:
        embedded = value.get("resource_contract")
        probe = (
            embedded.get("determinism_probe")
            if isinstance(embedded, Mapping)
            else None
        )
        if not isinstance(probe, Mapping):
            raise PreflightError("runtime-plan v4 lacks its green probe binding")
        try:
            resource = betty_runtime_v4_resource_contract(
                probe,
                repo_root=repo_root,
                task_id=str(manifest.get("task_id", "")),
            )
        except RuntimePhasePlanError as exc:
            raise PreflightError(str(exc)) from exc
        gpu_lanes = BETTY_V4_GPU_LANES
        cpu_lanes = BETTY_V4_CPU_LANES
        sharding = RUNTIME_V4_SHARDING
    else:
        raise PreflightError("unsupported phased runtime plan")
    expected = {
        "schema_version": schema,
        "profile": manifest_profile,
        "task_id": manifest.get("task_id"),
        "task_spec_hash": manifest.get("task_spec_hash"),
        "prepared_hash": manifest.get("prepared", {}).get("prepared_hash"),
        "preflight_receipt_hash": manifest.get("preflight", {}).get("receipt_hash"),
        "code_identity_sha256": _json_hash({"code": provenance.get("code")}),
        "walltime_seconds": resource["walltime_seconds"],
        "job_guard_seconds": resource["job_guard_seconds"],
        "job_overhead_seconds": resource["job_overhead_seconds"],
        "gpu_lanes": gpu_lanes,
        "cpu_lanes": cpu_lanes,
        "sharding": sharding,
        "resource_contract": resource,
        "intervention_binding": expected_intervention,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise PreflightError(
                f"phased runtime-plan provenance/resource drift: {field}"
            )
    safety = value.get("safety_factor")
    if (
        isinstance(safety, bool)
        or not isinstance(safety, (int, float))
        or not math.isfinite(float(safety))
        or float(safety) < 1.0
    ):
        raise PreflightError("phased runtime-plan safety factor is invalid")
    _validate_runtime_plan_v3_sources(value, repo_root=repo_root)
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != 10:
        raise PreflightError("phased runtime-plan must bind ten endpoint observations")
    observed_bounds: dict[str, dict[str, float]] = {}
    identities: set[tuple[str, int]] = set()
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise PreflightError("phased runtime observation is invalid")
        model = str(observation.get("model", ""))
        n_train = observation.get("n_train")
        if (
            model not in PRIMARY_MODEL_ROLES
            or isinstance(n_train, bool)
            or not isinstance(n_train, int)
            or (model, n_train) in identities
            or not _SHA256.fullmatch(str(observation.get("metrics_sha256", "")))
        ):
            raise PreflightError("phased runtime observation identity drifted")
        identities.add((model, n_train))
        if model in CPU_MODEL_ROLES:
            bounded = observation.get("bounded_seconds")
            elapsed = observation.get("elapsed_seconds")
            if (
                isinstance(bounded, bool)
                or not isinstance(bounded, (int, float))
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or float(bounded) != float(elapsed) * float(safety)
            ):
                raise PreflightError("tabular runtime observation bound drifted")
            observed_bounds.setdefault(model, {})["monolithic"] = max(
                observed_bounds.get(model, {}).get("monolithic", 0.0),
                float(bounded),
            )
        else:
            if (
                not _SHA256.fullmatch(
                    str(observation.get("selector_receipt_hash", ""))
                )
                or not _SHA256.fullmatch(
                    str(observation.get("selector_receipt_sha256", ""))
                )
            ):
                raise PreflightError("selector runtime receipt binding drifted")
            selector_bound = _bounded_phase_seconds(
                observation.get("selector_elapsed_seconds"),
                observation.get("selector_observed_epochs"),
                observation.get("max_epochs"),
                float(safety),
                label=f"{model}/selector",
            )
            refit_bound = _bounded_phase_seconds(
                observation.get("refit_elapsed_seconds"),
                observation.get("refit_epochs"),
                observation.get("max_epochs"),
                float(safety),
                label=f"{model}/refit",
            )
            if (
                float(observation.get("selector_bounded_seconds", -1.0))
                != selector_bound
                or float(observation.get("refit_bounded_seconds", -1.0))
                != refit_bound
            ):
                raise PreflightError("neural runtime observation bound drifted")
            prior = observed_bounds.setdefault(model, {})
            prior["selector"] = max(prior.get("selector", 0.0), selector_bound)
            prior["refit"] = max(prior.get("refit", 0.0), refit_bound)
    if value.get("model_phase_seconds_bound") != observed_bounds:
        raise PreflightError("phased runtime model bounds differ from observations")
    plans = validate_runtime_queue_plans_v3(value, manifest.get("task_spec", {}))
    root = Path(repo_root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError("runtime plan must be stored inside the repository") from exc
    if value.get("output_path") != relative:
        raise PreflightError("runtime plan output_path differs from its immutable location")
    return {
        "schema_version": schema,
        "path": relative,
        "sha256": sha256_file(path),
        "runtime_plan_hash": value["runtime_plan_hash"],
        "benchmark_manifest_hash": value["benchmark_manifest_hash"],
        "gpu_lanes": gpu_lanes,
        "cpu_lanes": cpu_lanes,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "sharding": sharding,
        "resource_contract": resource,
        "intervention_binding": value["intervention_binding"],
        "queue_plans": plans,
    }


def validate_runtime_plan(
    runtime_plan_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate and bind a profile-specific runtime plan to its run manifest."""

    path = Path(runtime_plan_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not read runtime plan {path}: {exc}") from exc
    if isinstance(value, Mapping) and value.get("schema_version") in {
        RUNTIME_PLAN_V3_SCHEMA_VERSION,
        RUNTIME_PLAN_V4_SCHEMA_VERSION,
    }:
        return _validate_runtime_plan_v3_value(
            value, path, manifest, repo_root=repo_root
        )
    expected_fields = {
        "schema_version",
        "profile",
        "task_id",
        "task_spec_hash",
        "prepared_hash",
        "preflight_receipt_hash",
        "code_identity_sha256",
        "benchmark_run",
        "benchmark_manifest_hash",
        "safety_factor",
        "walltime_seconds",
        "job_guard_seconds",
        "job_overhead_seconds",
        "gpu_lanes",
        "sharding",
        "intervention_binding",
        "observations",
        "model_cell_seconds_bound",
        "queue_plans",
        "output_path",
        "runtime_plan_hash",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise PreflightError("runtime plan root schema drifted")
    if value.get("schema_version") != RUNTIME_PLAN_SCHEMA_VERSION:
        raise PreflightError("unsupported runtime plan")
    unsigned = {key: item for key, item in value.items() if key != "runtime_plan_hash"}
    if value.get("runtime_plan_hash") != _json_hash(unsigned):
        raise PreflightError("runtime-plan hash is invalid")
    provenance = manifest.get("provenance", {})
    manifest_profile = manifest.get("execution", {}).get("profile")
    expected_intervention: dict[str, Any] | None = None
    if manifest_profile == "sensitivity20":
        frozen = [
            item
            for item in manifest.get("task_spec", {}).get("sensitivities", [])
            if item.get("selection") == "frozen_prespecified"
        ]
        if len(frozen) != 1 or not isinstance(frozen[0].get("execution"), Mapping):
            raise PreflightError(
                "sensitivity runtime-plan validation lacks one frozen intervention"
            )
        expected_intervention = {
            "branch": "frozen_sensitivity",
            "sensitivity_id": str(frozen[0]["id"]),
            "prepared_variant": str(frozen[0]["execution"]["prepared_variant"]),
        }
    expected = {
        "profile": manifest.get("execution", {}).get("profile"),
        "task_id": manifest.get("task_id"),
        "task_spec_hash": manifest.get("task_spec_hash"),
        "prepared_hash": manifest.get("prepared", {}).get("prepared_hash"),
        "preflight_receipt_hash": manifest.get("preflight", {}).get("receipt_hash"),
        "code_identity_sha256": _json_hash({"code": provenance.get("code")}),
        "walltime_seconds": BETTY_WALLTIME_SECONDS,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "gpu_lanes": BETTY_GPU_LANES,
        "sharding": RUNTIME_SHARDING,
        "intervention_binding": expected_intervention,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise PreflightError(f"runtime-plan provenance or guard drift: {field}")
    plans = validate_runtime_queue_plans(value, manifest.get("task_spec", {}))
    root = Path(repo_root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreflightError("runtime plan must be stored inside the repository") from exc
    if value.get("output_path") != relative:
        raise PreflightError("runtime plan output_path differs from its immutable location")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "runtime_plan_hash": value["runtime_plan_hash"],
        "benchmark_manifest_hash": value["benchmark_manifest_hash"],
        "gpu_lanes": BETTY_GPU_LANES,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "intervention_binding": value["intervention_binding"],
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "queue_plans": plans,
    }
