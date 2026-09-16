"""Operational phase-sharding contracts for long Betty neural cells.

This module contains no scientific selection logic.  It turns independently
measured selector and refit timings into an exact, replayable execution cover
that keeps every Slurm allocation below the fixed Betty guard.  Scientific
cell identities, seeds, configurations, and final artifacts are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attempt9_contract import ATTEMPT9_OPERATIONAL_SOURCE_BINDINGS
from scripts.betty.paths import resolve_record_path


RUNTIME_PLAN_V3_SCHEMA_VERSION = "bdb-runtime-plan-v3"
RUNTIME_V3_SHARDING = "atomic_anchor_model_selector_refit_lpt_lanes_v3"
RUNTIME_V3_RESOURCE_SCHEMA_VERSION = "bdb-betty-resource-capacity-v1"
RUNTIME_PLAN_V4_SCHEMA_VERSION = "bdb-runtime-plan-v4"
RUNTIME_V4_SHARDING = "atomic_anchor_model_selector_refit_lpt_lanes_v4"
RUNTIME_V4_RESOURCE_SCHEMA_VERSION = "bdb-betty-resource-capacity-v2"

BETTY_WALLTIME_SECONDS = 4 * 60 * 60
RUNTIME_JOB_GUARD_SECONDS = 3 * 60 * 60 + 20 * 60
RUNTIME_JOB_OVERHEAD_SECONDS = 6 * 60

# Read-only Betty measurements on 2026-08-22 established that MIG90 has the
# highest aggregate throughput under *this account's* four-GPU MaxTRESPA cap.
# The QoS-wide group limit is sixteen GPUs, but it is not available to one
# account.  The winning probe allocated fourteen CPUs per MIG90; four of
# those jobs plus six twelve-core CPU jobs use the exact 128-CPU ceiling.
BETTY_MIG90_LANES = 4
BETTY_MIG90_CPUS_PER_JOB = 14
BETTY_CPU_LANES = 6
BETTY_CPU_CPUS_PER_JOB = 12
BETTY_ACCOUNT_CPU_LIMIT = 128
BETTY_ACCOUNT_GPU_LIMIT = 4
BETTY_QOS_GPU_GROUP_LIMIT = 16
BETTY_CAPACITY_EVIDENCE_PATH = (
    "scripts/betty/capacity/2026-08-22_fastest_lane.json"
)
BETTY_CAPACITY_EVIDENCE_SHA256 = (
    "9af681abac8a2c461a996ec3bfff7c1e2bba37b9356cb7d4a32b665fa3aabe47"
)

# Attempt 9 preserves the four-GPU account ceiling but deliberately rejects
# the faster MIG90 class after two independent campaign preflights exposed
# shape-dependent neural nondeterminism there.  A checksum-valid paired
# BDB2021 probe is required by ``betty_runtime_v4_resource_contract`` before
# this operational fallback can be admitted.  Four six-core MIG45 jobs plus
# eight twelve-core CPU jobs peak at 120 cores, leaving eight cores of account
# headroom without reducing either queue's model-level parallelism.
BETTY_MIG45_LANES = 4
BETTY_MIG45_CPUS_PER_JOB = 6
BETTY_ATTEMPT9_CPU_LANES = 8
BETTY_ATTEMPT9_PEAK_CPUS = 120
BETTY_ATTEMPT9_SPARE_CPUS = 8
BETTY_DETERMINISM_PROBE_BINDING_SCHEMA_VERSION = (
    "bdb-betty-gpu-determinism-probe-binding-v2"
)
BETTY_DETERMINISM_PROBE_BINDING_PATH = (
    "scripts/betty/probes/attempt9_mig45_current_pair/binding.json"
)
BETTY_DETERMINISM_PROBE_BINDING_SHA256 = (
    "67368fe7b33ffbd46249791e64031a78354726774b9ac22c76c410cb73865ccb"
)
BETTY_DETERMINISM_PROBE_BINDING_HASH = (
    "3e81c1a177d85465524ab75765cf3e9d22a2f868b11d60e4139fb468eeb3a6a8"
)
BETTY_DETERMINISM_PROBE_COMPARISON_SHA256 = (
    "5ab3df14ec6fdbd7b1b3e7510190d63ea413f6f3a830c1efba1a477a2901a79b"
)
BETTY_DETERMINISM_PROBE_HASH = (
    "ad790e2cb3929ea5b535f69fcc93dd5d1e9ec77bc237489ad8f990c253b2efe1"
)
BDB2020_HARMONIZED_TASK_ID = "bdb2020_rushing_harmonized"
BDB2020_HARMONIZED_AMENDMENT_PATH = (
    "configs/bdb_suite/protocol_amendments/"
    "20260906_bdb2020_harmonized_five_role.json"
)
BDB2020_HARMONIZED_AMENDMENT_SHA256 = (
    "300689150325c65f12133042cd6a6459f02ec2c40174395496a5032dc4572587"
)

NEURAL_PHASES = ("selector", "refit")


class RuntimePhasePlanError(RuntimeError):
    """A phase timing or exact execution cover is unsafe or malformed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_betty_capacity_evidence(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Replay the checksum-bound measurement behind the v3 lane choice.

    This is an operational admission gate, not a scientific input.  It makes
    the resource contract fail closed if its source receipt, probe programs,
    probe wrappers, or selected launch wrappers differ from the measured
    decision recorded on 2026-08-22.
    """

    root = Path(repo_root).resolve()
    path = (root / BETTY_CAPACITY_EVIDENCE_PATH).resolve()
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimePhasePlanError(
            "Betty capacity evidence escapes the repository"
        ) from exc
    if path.is_symlink() or sidecar.is_symlink():
        raise RuntimePhasePlanError("Betty capacity evidence cannot be a symlink")
    try:
        payload = path.read_bytes()
        fields = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimePhasePlanError(
            "Betty capacity evidence/checksum is unreadable"
        ) from exc
    payload_sha = hashlib.sha256(payload).hexdigest()
    if (
        len(fields) != 2
        or fields[1] != path.name
        or fields[0] != payload_sha
        or payload_sha != BETTY_CAPACITY_EVIDENCE_SHA256
        or not isinstance(value, Mapping)
        or value.get("schema_version") != "bdb-betty-capacity-probe-v1"
        or value.get("decision_id")
        != "20260822_betty_fastest_lane_for_pilot10_attempt7"
    ):
        raise RuntimePhasePlanError("Betty capacity evidence identity drifted")

    limits = value.get("scheduler_limits")
    decision = value.get("decision")
    probes = value.get("gpu_probes")
    if not isinstance(limits, Mapping) or not isinstance(decision, Mapping):
        raise RuntimePhasePlanError("Betty capacity evidence fields are missing")
    try:
        gpu_limit = int(
            limits["gpu_qos"]["max_tres_per_account"]["gres/gpu"]
        )
        group_limit = int(limits["gpu_qos"]["qos_group_tres"]["gres/gpu"])
        cpu_limit = int(limits["cpu_qos"]["max_tres_per_account"]["cpu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimePhasePlanError(
            "Betty capacity scheduler limits are malformed"
        ) from exc
    expected_decision = {
        "gpu_resource_class": "mig90",
        "gpu_lanes": BETTY_MIG90_LANES,
        "gpu_cpus_per_lane": BETTY_MIG90_CPUS_PER_JOB,
        "gpu_lane_cpus": BETTY_MIG90_LANES * BETTY_MIG90_CPUS_PER_JOB,
        "cpu_resource_class": "genoa-std-mem",
        "cpu_lanes": BETTY_CPU_LANES,
        "cpu_cpus_per_lane": BETTY_CPU_CPUS_PER_JOB,
        "cpu_lane_cpus": BETTY_CPU_LANES * BETTY_CPU_CPUS_PER_JOB,
        "peak_scheduled_cpus": BETTY_ACCOUNT_CPU_LIMIT,
        "account_cpu_limit": BETTY_ACCOUNT_CPU_LIMIT,
        "account_gpu_limit": BETTY_ACCOUNT_GPU_LIMIT,
        "spare_cpus_at_peak": 0,
        "spare_gpus_at_peak": 0,
        "wait_for_reset": False,
    }
    if (
        gpu_limit != BETTY_ACCOUNT_GPU_LIMIT
        or group_limit != BETTY_QOS_GPU_GROUP_LIMIT
        or cpu_limit != BETTY_ACCOUNT_CPU_LIMIT
        or any(decision.get(key) != expected for key, expected in expected_decision.items())
        or not isinstance(probes, list)
        or len(probes) != 3
    ):
        raise RuntimePhasePlanError("Betty capacity arithmetic drifted")

    throughputs: dict[str, float] = {}
    source_bindings: list[Mapping[str, Any]] = []
    benchmark_source = value.get("probe_workload", {}).get("benchmark_source")
    if isinstance(benchmark_source, Mapping):
        source_bindings.append(benchmark_source)
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise RuntimePhasePlanError("Betty capacity probe record is malformed")
        resource = str(probe.get("resource_class", ""))
        total = probe.get("total_model_seconds")
        throughput = probe.get("four_lane_workloads_per_second")
        model_seconds = probe.get("model_seconds")
        wrapper = probe.get("wrapper")
        if (
            resource not in {"mig45", "mig90", "full_b200"}
            or isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not isinstance(model_seconds, Mapping)
            or not math.isclose(
                float(total),
                sum(float(item) for item in model_seconds.values()),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or isinstance(throughput, bool)
            or not isinstance(throughput, (int, float))
            or not math.isclose(
                float(throughput), 4.0 / float(total), rel_tol=1e-15
            )
            or not isinstance(wrapper, Mapping)
        ):
            raise RuntimePhasePlanError("Betty capacity probe arithmetic drifted")
        throughputs[resource] = float(throughput)
        source_bindings.append(wrapper)
    if max(throughputs, key=throughputs.get) != "mig90":
        raise RuntimePhasePlanError("Betty capacity evidence no longer selects MIG90")

    wrappers = decision.get("execution_wrappers")
    if not isinstance(wrappers, Mapping) or set(wrappers) != {"gpu", "cpu"}:
        raise RuntimePhasePlanError("Betty selected wrapper binding is missing")
    source_bindings.extend(
        record for record in wrappers.values() if isinstance(record, Mapping)
    )
    if len(source_bindings) != 6:
        raise RuntimePhasePlanError("Betty capacity source registry is incomplete")
    for binding in source_bindings:
        relative = binding.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimePhasePlanError("Betty capacity source path is invalid")
        try:
            source = resolve_record_path(root, relative)
        except ValueError as exc:
            raise RuntimePhasePlanError(
                "Betty capacity source escapes the repository"
            ) from exc
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != binding.get("bytes")
            or _sha256_file(source) != binding.get("sha256")
        ):
            raise RuntimePhasePlanError(
                f"Betty capacity source binding drifted: {relative}; "
                "historical admission requires the original source checkout (see docs/code-layout-migration.md)"
            )
    return {
        "path": BETTY_CAPACITY_EVIDENCE_PATH,
        "sha256": payload_sha,
        "decision_id": str(value["decision_id"]),
        "observed_at_betty": str(value.get("observed_at_betty", "")),
    }


def betty_runtime_v3_resource_contract() -> dict[str, Any]:
    """Return the exact resource/capacity identity used by runtime-plan v3."""

    peak_cpu = (
        BETTY_MIG90_LANES * BETTY_MIG90_CPUS_PER_JOB
        + BETTY_CPU_LANES * BETTY_CPU_CPUS_PER_JOB
    )
    if (
        peak_cpu > BETTY_ACCOUNT_CPU_LIMIT
        or BETTY_MIG90_LANES > BETTY_ACCOUNT_GPU_LIMIT
        or BETTY_ACCOUNT_GPU_LIMIT > BETTY_QOS_GPU_GROUP_LIMIT
    ):
        raise RuntimePhasePlanError("Betty v3 resource arithmetic drifted")
    return {
        "schema_version": RUNTIME_V3_RESOURCE_SCHEMA_VERSION,
        "walltime_seconds": BETTY_WALLTIME_SECONDS,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "account_cpu_limit": BETTY_ACCOUNT_CPU_LIMIT,
        "account_gpu_limit": BETTY_ACCOUNT_GPU_LIMIT,
        "qos_gpu_group_limit": BETTY_QOS_GPU_GROUP_LIMIT,
        "peak_scheduled_cpus": peak_cpu,
        "spare_account_cpus": BETTY_ACCOUNT_CPU_LIMIT - peak_cpu,
        "capacity_evidence": {
            "path": BETTY_CAPACITY_EVIDENCE_PATH,
            "sha256": BETTY_CAPACITY_EVIDENCE_SHA256,
        },
        "gpu": {
            "resource_class": "b200-mig90",
            "wrapper": "mig90_14cpu.sbatch",
            "lanes": BETTY_MIG90_LANES,
            "cpus_per_job": BETTY_MIG90_CPUS_PER_JOB,
            "gpus_per_job": 1,
        },
        "cpu": {
            "resource_class": "genoa-std-mem",
            "wrapper": "cpu.sbatch",
            "lanes": BETTY_CPU_LANES,
            "cpus_per_job": BETTY_CPU_CPUS_PER_JOB,
            "gpus_per_job": 0,
        },
    }


def _attempt9_probe_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the compact, immutable v2 provenance summary."""

    expected = {
        "schema_version": BETTY_DETERMINISM_PROBE_BINDING_SCHEMA_VERSION,
        "path": BETTY_DETERMINISM_PROBE_BINDING_PATH,
        "sha256": BETTY_DETERMINISM_PROBE_BINDING_SHA256,
        "binding_hash": BETTY_DETERMINISM_PROBE_BINDING_HASH,
        "raw_comparison_sha256": BETTY_DETERMINISM_PROBE_COMPARISON_SHA256,
        "probe_hash": BETTY_DETERMINISM_PROBE_HASH,
        "terminal_green": True,
        "bdb2021_paired_repeat_match": True,
        "predictive_scores_inspected": False,
        "gpu_job_ids": ["7784273", "7784274"],
        "comparison_job_id": "7784275",
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimePhasePlanError(
            "attempt9 determinism-probe provenance summary drifted"
        )
    return expected


def _read_attempt9_bound_artifact(
    root: Path, record: Mapping[str, Any], *, label: str
) -> tuple[bytes, dict[str, Any]]:
    """Read one preserved probe artifact and verify both immutable files."""

    relative = record.get("path")
    sidecar_relative = record.get("sidecar_path")
    if (
        not isinstance(relative, str)
        or not isinstance(sidecar_relative, str)
        or Path(relative).is_absolute()
        or Path(sidecar_relative).is_absolute()
        or ".." in Path(relative).parts
        or ".." in Path(sidecar_relative).parts
    ):
        raise RuntimePhasePlanError(f"attempt9 {label} path is unsafe")
    path = resolve_record_path(root, relative)
    sidecar = resolve_record_path(root, sidecar_relative)
    try:
        path.relative_to(root)
        sidecar.relative_to(root)
        payload = path.read_bytes()
        sidecar_payload = sidecar.read_bytes()
        fields = sidecar_payload.decode("ascii").split()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimePhasePlanError(
            f"attempt9 preserved {label} is unreadable"
        ) from exc
    if (
        path.is_symlink()
        or sidecar.is_symlink()
        or len(fields) != 2
        or fields[0] != record.get("sha256")
        or fields[1] != path.name
        or hashlib.sha256(payload).hexdigest() != record.get("sha256")
        or hashlib.sha256(sidecar_payload).hexdigest()
        != record.get("sidecar_sha256")
        or not isinstance(value, dict)
    ):
        raise RuntimePhasePlanError(
            f"attempt9 preserved {label} checksum replay failed"
        )
    return payload, value


def _read_attempt9_source_copy(
    root: Path, record: Mapping[str, Any], *, label: str
) -> tuple[bytes, dict[str, Any]]:
    """Read the source-controlled byte copy of one remote probe artifact."""

    source_record = {
        "path": record.get("source_copy_path"),
        "sidecar_path": record.get("source_copy_sidecar_path"),
        "sha256": record.get("sha256"),
        "sidecar_sha256": record.get("sidecar_sha256"),
    }
    return _read_attempt9_bound_artifact(
        root, source_record, label=f"{label} source copy"
    )


def _read_attempt9_replay_artifact(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    require_preserved: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Replay the durable source copy and, when present, its data-tree twin."""

    copied_payload, copied_value = _read_attempt9_source_copy(
        root, record, label=label
    )
    preserved_path = root / str(record.get("path", ""))
    preserved_sidecar = root / str(record.get("sidecar_path", ""))
    preserved_present = any(
        path.exists() or path.is_symlink()
        for path in (preserved_path, preserved_sidecar)
    )
    if require_preserved or preserved_present:
        preserved_payload, preserved_value = _read_attempt9_bound_artifact(
            root, record, label=label
        )
        if (
            preserved_payload != copied_payload
            or preserved_value != copied_value
        ):
            raise RuntimePhasePlanError(
                f"attempt9 preserved {label} differs from its source copy"
            )
    return copied_payload, copied_value


def validate_betty_determinism_probe(
    repo_root: str | Path,
    value: Mapping[str, Any],
    *,
    require_live_source_match: bool = True,
) -> dict[str, Any]:
    """Replay source-bound provenance and the raw score-blind comparator."""

    compact: dict[str, Any] | None = None
    try:
        compact = _attempt9_probe_binding(value)
    except RuntimePhasePlanError:
        pass
    root = Path(repo_root).resolve()
    path = (root / BETTY_DETERMINISM_PROBE_BINDING_PATH).resolve()
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimePhasePlanError(
            "attempt9 determinism probe escapes the repository"
        ) from exc
    if (
        relative != BETTY_DETERMINISM_PROBE_BINDING_PATH
        or path.is_symlink()
        or sidecar.is_symlink()
    ):
        raise RuntimePhasePlanError("attempt9 determinism probe path is unsafe")
    try:
        payload = path.read_bytes()
        fields = sidecar.read_text(encoding="ascii").split()
        binding = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimePhasePlanError(
            "attempt9 determinism probe/checksum is unreadable"
        ) from exc
    if (
        len(fields) != 2
        or fields[0] != BETTY_DETERMINISM_PROBE_BINDING_SHA256
        or fields[1] != path.name
        or hashlib.sha256(payload).hexdigest()
        != BETTY_DETERMINISM_PROBE_BINDING_SHA256
        or not isinstance(binding, Mapping)
        or (compact is None and dict(value) != dict(binding))
    ):
        raise RuntimePhasePlanError(
            "attempt9 determinism probe/checksum replay failed"
        )

    unsigned = dict(binding)
    binding_hash = unsigned.pop("binding_hash", None)
    if (
        binding.get("schema_version")
        != BETTY_DETERMINISM_PROBE_BINDING_SCHEMA_VERSION
        or binding_hash != BETTY_DETERMINISM_PROBE_BINDING_HASH
        or hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        != BETTY_DETERMINISM_PROBE_BINDING_HASH
    ):
        raise RuntimePhasePlanError("attempt9 determinism binding identity drifted")

    raw = binding.get("raw_comparison")
    assertions = binding.get("assertions")
    jobs = binding.get("sacct_jobs")
    sources = binding.get("source_bindings")
    if (
        not isinstance(raw, Mapping)
        or not isinstance(assertions, Mapping)
        or not isinstance(jobs, list)
        or not isinstance(sources, Mapping)
        or assertions
        != {
            "allocation_shape": "independent_allocations",
            "bdb2021_paired_repeat_match": True,
            "full_diagnostic_equal": True,
            "preflight_contract_equal": True,
            "predictive_scores_inspected": False,
            "terminal_green": True,
        }
        or [job.get("job_id") for job in jobs if isinstance(job, Mapping)]
        != ["7784273", "7784274", "7784275"]
        or any(
            job.get("state") != "COMPLETED" or job.get("exit_code") != "0:0"
            for job in jobs
            if isinstance(job, Mapping)
        )
        or jobs[0].get("partition") != "b200-mig45"
        or jobs[1].get("partition") != "b200-mig45"
        or jobs[0].get("qos") != "wharton-dgx-b200"
        or jobs[1].get("qos") != "wharton-dgx-b200"
        or jobs[2].get("qos") != "wharton-genoa"
        or any(job.get("account") != "ajw-wharton" for job in jobs)
        or jobs[0].get("node_list") != "dgx028"
        or jobs[1].get("node_list") != "dgx028"
        or "cpu=6" not in str(jobs[0].get("alloc_tres"))
        or "gres/gpu:45gb=1" not in str(jobs[0].get("alloc_tres"))
        or "cpu=6" not in str(jobs[1].get("alloc_tres"))
        or "gres/gpu:45gb=1" not in str(jobs[1].get("alloc_tres"))
    ):
        raise RuntimePhasePlanError("attempt9 terminal allocation evidence drifted")

    expected_source_registry = {
        "bdb2021_adapter",
        "bdb_adapter_common",
        "bdb_determinism",
        "bdb_metrics",
        "bdb_models",
        "bdb_representations",
        "bdb_requirements_lock",
        "bdb_runner",
        "common",
        "compare_wrapper",
        "gpu_wrapper",
        "neural_models",
        "probe",
        "project_requirements_lock",
    }
    if set(sources) != expected_source_registry:
        raise RuntimePhasePlanError("attempt9 probe source registry drifted")
    for source in sources.values():
        if not isinstance(source, Mapping):
            raise RuntimePhasePlanError("attempt9 probe source binding is malformed")
        relative_source = str(source.get("path", ""))
        if not require_live_source_match:
            continue
        try:
            source_path = resolve_record_path(root, relative_source)
        except ValueError as exc:
            raise RuntimePhasePlanError(
                "attempt9 probe source binding escapes the repository"
            ) from exc
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or _sha256_file(source_path) != source.get("sha256")
        ):
            raise RuntimePhasePlanError(
                "attempt9 probe source binding drifted; historical admission requires "
                "the original source checkout (see docs/code-layout-migration.md)"
            )

    scientific = binding.get("scientific_run_bindings")
    frozen = scientific.get("frozen_task") if isinstance(scientific, Mapping) else None
    run_manifests = (
        scientific.get("run_manifests") if isinstance(scientific, Mapping) else None
    )
    if (
        not isinstance(frozen, Mapping)
        or not isinstance(run_manifests, Mapping)
        or set(run_manifests) != {"run_a", "run_b"}
    ):
        raise RuntimePhasePlanError("attempt9 scientific run binding drifted")
    frozen_copy = resolve_record_path(root, str(frozen.get("source_copy_path", "")))
    try:
        frozen_copy.relative_to(root)
        frozen_copy_payload = frozen_copy.read_bytes()
        frozen_value = json.loads(frozen_copy_payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimePhasePlanError("attempt9 frozen TaskSpec is unreadable") from exc
    if (
        frozen.get("path")
        != "configs/bdb_suite/frozen/bdb2021_completion.json"
        or frozen_copy.is_symlink()
        or hashlib.sha256(frozen_copy_payload).hexdigest()
        != frozen.get("sha256")
        or frozen_value.get("task_spec_hash") != frozen.get("task_spec_hash")
        or frozen.get("task_spec_hash")
        != "f4e2cf180ad8732db1d1b1d43b8b5185224f0945d4d04e37e658ece0665a3cca"
    ):
        raise RuntimePhasePlanError("attempt9 frozen TaskSpec binding drifted")
    for run_name in ("run_a", "run_b"):
        record = run_manifests[run_name]
        if not isinstance(record, Mapping):
            raise RuntimePhasePlanError("attempt9 run manifest binding is malformed")
        copied_payload, copied_manifest = _read_attempt9_replay_artifact(
            root,
            record,
            label=f"{run_name} manifest",
            require_preserved=require_live_source_match,
        )
        manifest = copied_manifest
        if (
            manifest.get("manifest_hash") != record.get("manifest_hash")
            or manifest.get("manifest_hash")
            != "bc8fe1e808b7405b5246123ac1563ba7c70982e5b685e0570b0ea1b64653088a"
            or manifest.get("task_spec_hash") != frozen.get("task_spec_hash")
            or manifest.get("task_id") != "bdb2021_completion"
        ):
            raise RuntimePhasePlanError(
                f"attempt9 {run_name} manifest binding drifted"
            )
    probe_manifest = _read_attempt9_source_copy(
        root, run_manifests["run_a"], label="probe manifest"
    )[1]
    code_registry = probe_manifest.get("provenance", {}).get("code")
    operational_allowlist = set(ATTEMPT9_OPERATIONAL_SOURCE_BINDINGS)
    if (
        not isinstance(code_registry, list)
        or len(code_registry) != 48
        or hashlib.sha256(
            json.dumps(
                {"code": code_registry},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        != "ed8d1fd63943858cd25e502ca5de049e4fe22cffc8c69fae7b5e99b08a69c7d6"
    ):
        raise RuntimePhasePlanError("attempt9 probe code registry drifted")
    observed_operational_drift: set[str] = set()
    for record in code_registry:
        if not isinstance(record, Mapping):
            raise RuntimePhasePlanError("attempt9 probe code record is malformed")
        relative_code = record.get("path")
        if (
            not isinstance(relative_code, str)
            or not relative_code
            or Path(relative_code).is_absolute()
            or ".." in Path(relative_code).parts
        ):
            raise RuntimePhasePlanError("attempt9 probe code path is unsafe")
        if not require_live_source_match:
            continue
        code_path = (root / relative_code).resolve()
        try:
            observed_relative = code_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimePhasePlanError(
                "attempt9 probe code path escapes the repository"
            ) from exc
        matches_probe = (
            not code_path.is_symlink()
            and code_path.is_file()
            and observed_relative == relative_code
            and code_path.stat().st_size == record.get("size_bytes")
            and _sha256_file(code_path) == record.get("sha256")
        )
        if not matches_probe:
            if relative_code not in operational_allowlist:
                raise RuntimePhasePlanError(
                    f"attempt9 probe-tested scientific source drifted: {relative_code}"
                )
            observed_operational_drift.add(relative_code)
    if require_live_source_match:
        if observed_operational_drift != operational_allowlist:
            raise RuntimePhasePlanError(
                "attempt9 operational-only probe drift allowlist is not exact"
            )
        for relative_code, expected in ATTEMPT9_OPERATIONAL_SOURCE_BINDINGS.items():
            code_path = (root / relative_code).resolve()
            if (
                not code_path.is_file()
                or code_path.is_symlink()
                or code_path.stat().st_size != expected.get("size_bytes")
                or _sha256_file(code_path) != expected.get("sha256")
            ):
                raise RuntimePhasePlanError(
                    f"attempt9 post-probe operational source drifted: {relative_code}"
                )

    execution_receipts = binding.get("execution_receipts")
    if not isinstance(execution_receipts, Mapping) or set(execution_receipts) != {
        "run_a",
        "run_b",
    }:
        raise RuntimePhasePlanError("attempt9 execution receipt registry drifted")
    observed_mig_devices: list[str] = []
    for label, job_id, run_name in (
        ("run_a", "7784273", "run_a"),
        ("run_b", "7784274", "run_b"),
    ):
        record = execution_receipts[label]
        if not isinstance(record, Mapping):
            raise RuntimePhasePlanError("attempt9 execution receipt is malformed")
        copied_payload, copied_execution = _read_attempt9_replay_artifact(
            root,
            record,
            label=f"{label} execution receipt",
            require_preserved=require_live_source_match,
        )
        execution = copied_execution
        description = execution.get("description")
        if (
            execution.get("schema_version")
            != "bdb2021-attn-determinism-probe-v1"
            or execution.get("kind") != "execution"
            or execution.get("allocation_shape") != "one_replica"
            or execution.get("slurm", {}).get("job_id") != job_id
            or execution.get("slurm", {}).get("partition") != "b200-mig45"
            or execution.get("slurm", {}).get("node_list") != "dgx028"
            or not str(execution.get("slurm", {}).get("cuda_visible_devices", ""))
            .startswith("MIG-")
            or not isinstance(description, Mapping)
            or description.get("task_id") != "bdb2021_completion"
            or description.get("branch") != "fixed_main"
            or description.get("repeat") != 1
            or description.get("anchor") != 130
            or description.get("model_id") != "attn_relnet"
            or description.get("tf32_mode") != "default"
            or description.get("nvidia_tf32_override") is not None
            or not str(description.get("run_dir", "")).endswith(f"/{run_name}")
        ):
            raise RuntimePhasePlanError(
                f"attempt9 {label} execution provenance drifted"
            )
        observed_mig_devices.append(
            str(execution.get("slurm", {}).get("cuda_visible_devices"))
        )
    if len(set(observed_mig_devices)) != 2:
        raise RuntimePhasePlanError(
            "attempt9 replicas did not use distinct MIG device UUIDs"
        )

    raw_payload, receipt = _read_attempt9_replay_artifact(
        root,
        raw,
        label="raw comparison",
        require_preserved=require_live_source_match,
    )
    comparisons = receipt.get("comparison") if isinstance(receipt, Mapping) else None
    equality_flags = []
    if isinstance(comparisons, Mapping):
        equality_flags = [
            comparisons.get("arrays", {}).get("equal"),
            comparisons.get("history_without_timing", {}).get("equal"),
            comparisons.get("predictions", {}).get("equal"),
            comparisons.get("scientific_metrics", {}).get("equal"),
            comparisons.get("phase_history", {}).get("selector_history", {}).get("equal"),
            comparisons.get("phase_history", {}).get("refit_history", {}).get("equal"),
        ]
    if (
        raw.get("sha256") != BETTY_DETERMINISM_PROBE_COMPARISON_SHA256
        or raw.get("probe_hash") != BETTY_DETERMINISM_PROBE_HASH
        or receipt.get("schema_version") != "bdb2021-attn-determinism-probe-v1"
        or receipt.get("probe_hash") != BETTY_DETERMINISM_PROBE_HASH
        or receipt.get("allocation_shape") != "independent_allocations"
        or receipt.get("preflight_contract_equal") is not True
        or receipt.get("full_diagnostic_equal") is not True
        or equality_flags != [True] * 6
        or receipt.get("slurm", {}).get("job_id") != "7784275"
        or receipt.get("slurm", {}).get("partition") != "genoa-std-mem"
        or receipt.get("slurm", {}).get("node_list") != "epyc-2-10"
        or receipt.get("task_id") != "bdb2021_completion"
        or receipt.get("cell")
        != {
            "anchor": 130,
            "branch": "fixed_main",
            "model_id": "attn_relnet",
            "repeat": 1,
        }
        or receipt.get("tf32_mode_a") != "default"
        or receipt.get("tf32_mode_b") != "default"
    ):
        raise RuntimePhasePlanError("attempt9 raw comparator replay failed")

    raw_unsigned = dict(receipt)
    observed_probe_hash = raw_unsigned.pop("probe_hash", None)
    if (
        observed_probe_hash != BETTY_DETERMINISM_PROBE_HASH
        or hashlib.sha256(
            json.dumps(
                raw_unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=True,
            ).encode("utf-8")
        ).hexdigest()
        != BETTY_DETERMINISM_PROBE_HASH
    ):
        raise RuntimePhasePlanError("attempt9 raw comparator probe_hash drifted")
    observed = _attempt9_probe_binding(
        {
            "schema_version": BETTY_DETERMINISM_PROBE_BINDING_SCHEMA_VERSION,
            "path": BETTY_DETERMINISM_PROBE_BINDING_PATH,
            "sha256": BETTY_DETERMINISM_PROBE_BINDING_SHA256,
            "binding_hash": BETTY_DETERMINISM_PROBE_BINDING_HASH,
            "raw_comparison_sha256": BETTY_DETERMINISM_PROBE_COMPARISON_SHA256,
            "probe_hash": BETTY_DETERMINISM_PROBE_HASH,
            "terminal_green": True,
            "bdb2021_paired_repeat_match": True,
            "predictive_scores_inspected": False,
            "gpu_job_ids": ["7784273", "7784274"],
            "comparison_job_id": "7784275",
        }
    )
    if compact is not None and compact != observed:
        raise RuntimePhasePlanError("attempt9 compact provenance summary drifted")
    return observed


def betty_runtime_v4_resource_contract(
    determinism_probe: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return attempt 9's exact four-MIG45/eight-CPU capacity contract.

    Runtime-plan v3 remains immutable for the settled attempt7/8 evidence.
    This v4 contract is a new operational identity and cannot be constructed
    without the checksum-bound green probe that justifies rejecting MIG90.
    """

    # The historical attempt-9 probe remains immutable.  New scientific code
    # normally has to match its live source replay exactly.  The separately
    # named retrospective BDB2020 task is the one deliberate exception: its
    # exact protocol amendment is checksum-bound here, while its current model
    # code is exercised by a fresh paired smoke A/B gate before this runtime
    # plan can be built.  We therefore consume only the immutable compact
    # attempt-9 hardware/probe identity for that task instead of pretending
    # its newly added adapter existed in the historical probe tree.
    if repo_root is not None and task_id == BDB2020_HARMONIZED_TASK_ID:
        root = Path(repo_root).resolve()
        amendment_path = (root / BDB2020_HARMONIZED_AMENDMENT_PATH).resolve()
        try:
            amendment_path.relative_to(root)
        except ValueError as exc:
            raise RuntimePhasePlanError(
                "BDB2020 harmonized amendment escapes the repository"
            ) from exc
        if (
            amendment_path.is_symlink()
            or not amendment_path.is_file()
            or _sha256_file(amendment_path)
            != BDB2020_HARMONIZED_AMENDMENT_SHA256
        ):
            raise RuntimePhasePlanError(
                "BDB2020 harmonized runtime amendment drifted"
            )
        try:
            from .contracts import load_bdb2020_harmonized_amendment

            amendment = load_bdb2020_harmonized_amendment(amendment_path)
        except Exception as exc:
            raise RuntimePhasePlanError(
                f"BDB2020 harmonized runtime amendment failed: {exc}"
            ) from exc
        if (
            amendment.get("new_task", {}).get("task_id")
            != BDB2020_HARMONIZED_TASK_ID
            or amendment.get("execution", {}).get("profile") != "full100"
            or amendment.get("execution", {}).get("paired_smoke_replays") != 2
            or amendment.get("execution", {}).get("successor_automatic_start")
            is not False
        ):
            raise RuntimePhasePlanError(
                "BDB2020 harmonized runtime amendment scope drifted"
            )
        probe = validate_betty_determinism_probe(
            root,
            determinism_probe,
            require_live_source_match=False,
        )
    else:
        probe = (
            validate_betty_determinism_probe(repo_root, determinism_probe)
            if repo_root is not None
            else _attempt9_probe_binding(determinism_probe)
        )
    peak_cpu = (
        BETTY_MIG45_LANES * BETTY_MIG45_CPUS_PER_JOB
        + BETTY_ATTEMPT9_CPU_LANES * BETTY_CPU_CPUS_PER_JOB
    )
    if (
        peak_cpu != BETTY_ATTEMPT9_PEAK_CPUS
        or BETTY_ACCOUNT_CPU_LIMIT - peak_cpu != BETTY_ATTEMPT9_SPARE_CPUS
        or peak_cpu > BETTY_ACCOUNT_CPU_LIMIT
        or BETTY_MIG45_LANES != BETTY_ACCOUNT_GPU_LIMIT
        or BETTY_ACCOUNT_GPU_LIMIT > BETTY_QOS_GPU_GROUP_LIMIT
    ):
        raise RuntimePhasePlanError("Betty v4 resource arithmetic drifted")
    return {
        "schema_version": RUNTIME_V4_RESOURCE_SCHEMA_VERSION,
        "walltime_seconds": BETTY_WALLTIME_SECONDS,
        "job_guard_seconds": RUNTIME_JOB_GUARD_SECONDS,
        "job_overhead_seconds": RUNTIME_JOB_OVERHEAD_SECONDS,
        "account_cpu_limit": BETTY_ACCOUNT_CPU_LIMIT,
        "account_gpu_limit": BETTY_ACCOUNT_GPU_LIMIT,
        "qos_gpu_group_limit": BETTY_QOS_GPU_GROUP_LIMIT,
        "peak_scheduled_cpus": peak_cpu,
        "spare_account_cpus": BETTY_ACCOUNT_CPU_LIMIT - peak_cpu,
        "capacity_evidence": {
            "path": BETTY_CAPACITY_EVIDENCE_PATH,
            "sha256": BETTY_CAPACITY_EVIDENCE_SHA256,
        },
        "determinism_probe": probe,
        "selection": "mig45_terminal_green_determinism_fallback",
        "gpu": {
            "resource_class": "b200-mig45",
            "wrapper": "mig45_6cpu.sbatch",
            "lanes": BETTY_MIG45_LANES,
            "cpus_per_job": BETTY_MIG45_CPUS_PER_JOB,
            "gpus_per_job": 1,
        },
        "cpu": {
            "resource_class": "genoa-std-mem",
            "wrapper": "cpu.sbatch",
            "lanes": BETTY_ATTEMPT9_CPU_LANES,
            "cpus_per_job": BETTY_CPU_CPUS_PER_JOB,
            "gpus_per_job": 0,
        },
    }


def _positive_seconds(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimePhasePlanError(f"invalid phase runtime bound: {label}")
    return float(value)


def canonical_neural_phase_shards(
    *,
    experimental_phase: str,
    anchors: Sequence[int],
    models: Sequence[str],
    model_phase_seconds_bound: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Build the canonical maximum-parallelism neural exact cover.

    Each shard is one immutable anchor/model cell chain.  Repeats may be
    grouped only when *both* selector and refit allocations remain inside the
    usable guard.  Atomic rectangles deliberately maximize available
    parallelism; the task launcher later assigns concrete repeat chunks to
    the admitted lanes with deterministic longest-processing-time scheduling.
    """

    ordered_anchors = [int(value) for value in anchors]
    ordered_models = [str(value) for value in models]
    if not experimental_phase or not ordered_anchors or not ordered_models:
        raise RuntimePhasePlanError("neural phase registry is empty")
    if len(set(ordered_anchors)) != len(ordered_anchors):
        raise RuntimePhasePlanError("neural phase repeats an anchor")
    if ordered_anchors != sorted(ordered_anchors):
        raise RuntimePhasePlanError("neural phase anchors are not canonical")
    if len(set(ordered_models)) != len(ordered_models):
        raise RuntimePhasePlanError("neural phase repeats a model")

    usable = RUNTIME_JOB_GUARD_SECONDS - RUNTIME_JOB_OVERHEAD_SECONDS
    result: list[dict[str, Any]] = []
    for anchor in ordered_anchors:
        for model in ordered_models:
            raw = model_phase_seconds_bound.get(model)
            if not isinstance(raw, Mapping) or set(raw) != set(NEURAL_PHASES):
                raise RuntimePhasePlanError(
                    f"neural model phase timing registry differs: {model}"
                )
            selector = _positive_seconds(
                raw["selector"], label=f"{model}/selector"
            )
            refit = _positive_seconds(raw["refit"], label=f"{model}/refit")
            if selector > usable or refit > usable:
                bad_phase = "selector" if selector > usable else "refit"
                bad_value = selector if selector > usable else refit
                raise RuntimePhasePlanError(
                    "one atomic neural phase exceeds the Betty runtime guard: "
                    f"{experimental_phase}/{anchor}/{model}/{bad_phase}="
                    f"{bad_value:.6f}s>{usable}s"
                )
            repeats_per_chain = int(usable // max(selector, refit))
            if repeats_per_chain < 1:
                raise RuntimePhasePlanError("neural phase repeat chunk is empty")
            selector_predicted = (
                RUNTIME_JOB_OVERHEAD_SECONDS
                + repeats_per_chain * selector
            )
            refit_predicted = (
                RUNTIME_JOB_OVERHEAD_SECONDS + repeats_per_chain * refit
            )
            if (
                selector_predicted > RUNTIME_JOB_GUARD_SECONDS
                or refit_predicted > RUNTIME_JOB_GUARD_SECONDS
            ):
                raise RuntimePhasePlanError(
                    "neural phase predicted allocation exceeds its guard"
                )
            result.append(
                {
                    "shard_id": (
                        f"gpu_neural-{experimental_phase}-a{anchor:04d}-{model}"
                    ),
                    "anchors": [anchor],
                    "models": [model],
                    "cells_per_repeat": 1,
                    "repeats_per_chain": repeats_per_chain,
                    "selector_seconds_per_repeat_bound": selector,
                    "selector_predicted_job_seconds": selector_predicted,
                    "refit_seconds_per_repeat_bound": refit,
                    "refit_predicted_job_seconds": refit_predicted,
                    "chain_seconds_per_repeat_bound": selector + refit,
                }
            )
    return result


def validate_neural_phase_shards(
    observed: Any,
    *,
    experimental_phase: str,
    anchors: Sequence[int],
    models: Sequence[str],
    model_phase_seconds_bound: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Regenerate and require the exact canonical atomic phase cover."""

    expected = canonical_neural_phase_shards(
        experimental_phase=experimental_phase,
        anchors=anchors,
        models=models,
        model_phase_seconds_bound=model_phase_seconds_bound,
    )
    if observed != expected:
        raise RuntimePhasePlanError(
            "neural selector/refit shards are not the canonical exact cover"
        )
    return expected


def concrete_chain_units(
    shards: Sequence[Mapping[str, Any]], *, repeats: int
) -> list[dict[str, Any]]:
    """Expand canonical shards into immutable concrete repeat chunks."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise RuntimePhasePlanError("phase repeat count must be positive")
    units: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        chunk = shard.get("repeats_per_chain")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise RuntimePhasePlanError("neural chain shard chunk is invalid")
        selector = _positive_seconds(
            shard.get("selector_seconds_per_repeat_bound"),
            label=f"shard-{shard_index}/selector",
        )
        refit = _positive_seconds(
            shard.get("refit_seconds_per_repeat_bound"),
            label=f"shard-{shard_index}/refit",
        )
        for chunk_index, first in enumerate(range(1, repeats + 1, chunk)):
            selected = list(range(first, min(repeats, first + chunk - 1) + 1))
            units.append(
                {
                    "unit_id": f"{shard['shard_id']}-r{first:03d}",
                    "shard_index": shard_index,
                    "chunk_index": chunk_index,
                    "shard_id": str(shard["shard_id"]),
                    "anchors": list(shard["anchors"]),
                    "models": list(shard["models"]),
                    "repeats": selected,
                    "selector_predicted_job_seconds": (
                        RUNTIME_JOB_OVERHEAD_SECONDS + len(selected) * selector
                    ),
                    "refit_predicted_job_seconds": (
                        RUNTIME_JOB_OVERHEAD_SECONDS + len(selected) * refit
                    ),
                    "chain_predicted_seconds": (
                        2 * RUNTIME_JOB_OVERHEAD_SECONDS
                        + len(selected) * (selector + refit)
                    ),
                }
            )
    return units


def lpt_lane_assignment(
    units: Sequence[Mapping[str, Any]], *, lanes: int = BETTY_MIG90_LANES
) -> list[list[dict[str, Any]]]:
    """Deterministically balance concrete cell chains over MIG90 lanes."""

    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
        raise RuntimePhasePlanError("GPU lane count must be positive")
    normalized = [dict(unit) for unit in units]
    if len({str(unit.get("unit_id")) for unit in normalized}) != len(normalized):
        raise RuntimePhasePlanError("neural chain unit IDs are not unique")
    ordered = sorted(
        normalized,
        key=lambda unit: (
            -_positive_seconds(
                unit.get("chain_predicted_seconds"),
                label=str(unit.get("unit_id")),
            ),
            str(unit.get("unit_id")),
        ),
    )
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(lanes)]
    loads = [0.0] * lanes
    for unit in ordered:
        lane = min(range(lanes), key=lambda index: (loads[index], index))
        assignments[lane].append(unit)
        loads[lane] += float(unit["chain_predicted_seconds"])
    return assignments


def validate_lpt_lane_assignment(
    observed: Any,
    *,
    shards: Sequence[Mapping[str, Any]],
    repeats: int,
    lanes: int = BETTY_MIG90_LANES,
) -> list[list[dict[str, Any]]]:
    """Replay the exact concrete cover and deterministic lane balance."""

    expected = lpt_lane_assignment(
        concrete_chain_units(shards, repeats=repeats), lanes=lanes
    )
    if observed != expected:
        raise RuntimePhasePlanError(
            "neural chain lane assignment is not the canonical LPT schedule"
        )
    return expected
