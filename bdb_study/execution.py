"""Manifest-bound CPU/GPU queue execution and resumable cell dispatch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import TaskSpec
from .prepared import load_prepared_task
from .prepared import load_prepared_receipt
from .contracts import (
    load_task_spec,
    prepared_contract_for_variant,
    sha256_file,
    validate_task_spec,
)
from .design import validate_task_design
from .determinism import deterministic_environment
from .manifest import (
    ManifestError,
    build_shared_frozen_registry_binding,
    parse_tensorflow_probe_output,
    tensorflow_probe_script,
    validate_prepared_binding,
    validate_primary_reference_binding,
)
from .runner import (
    TaskRuntime,
    run_cell,
    run_neural_selector,
    validate_neural_selector_for_cell,
)
from .storage import (
    CellKey,
    CellStatus,
    GenericRunStorage,
    claim_phase_lease,
    estimate_queue_eta_seconds,
    manifest_sha256,
    pending_cells_by_queue,
    release_phase_lease,
    status_by_queue,
    verify_storage_backend_receipt,
)


class ExecutionError(RuntimeError):
    pass


ENVIRONMENT_POLICY_QUEUE_NEUTRAL = "queue_neutral"
ENVIRONMENT_POLICY_GPU_EXACT = "gpu_exact"
ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY = "host_artifact_only"
_ENVIRONMENT_POLICIES = frozenset(
    {
        ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
        ENVIRONMENT_POLICY_GPU_EXACT,
        ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY,
    }
)
_SCIENTIFIC_ENVIRONMENT_FIELDS = ("python", "implementation")
_HARDWARE_ENVIRONMENT_FIELDS = ("os", "machine", "processor", "cpu_count")


def _resolve_environment_policy(
    environment_policy: str | None,
    verify_tensorflow_runtime: bool | None,
) -> str:
    """Resolve the explicit policy and the legacy TensorFlow verification flag.

    ``verify_tensorflow_runtime=False`` historically marked CPU-only/status
    callers.  It remains a supported alias for the queue-neutral policy so
    existing aggregate and preflight consumers stay portable.  New execution
    callers use an explicit policy because TensorFlow visibility and host
    hardware are one runtime-admission decision, not independent guarantees.

    ``host_artifact_only`` is narrower still and is reserved for a host-side
    Slurm dispatcher that cannot share the Pyxis container's interpreter.  It
    replays every immutable scientific artifact below, but it cannot authorize
    model execution or aggregation and therefore never compares the live host
    package, hardware, or TensorFlow inventory.
    """

    if verify_tensorflow_runtime is not None and not isinstance(
        verify_tensorflow_runtime, bool
    ):
        raise ExecutionError("verify_tensorflow_runtime must be boolean or null")
    legacy_policy = (
        None
        if verify_tensorflow_runtime is None
        else (
            ENVIRONMENT_POLICY_GPU_EXACT
            if verify_tensorflow_runtime
            else ENVIRONMENT_POLICY_QUEUE_NEUTRAL
        )
    )
    if environment_policy is None:
        return legacy_policy or ENVIRONMENT_POLICY_GPU_EXACT
    policy = str(environment_policy)
    if policy not in _ENVIRONMENT_POLICIES:
        raise ExecutionError(f"unsupported runtime environment policy: {policy}")
    if legacy_policy is not None and legacy_policy != policy:
        raise ExecutionError(
            "runtime environment policy conflicts with verify_tensorflow_runtime"
        )
    return policy


def _verify_installed_versions(
    planned: Mapping[str, Any] | None,
    *,
    category: str,
) -> None:
    """Verify one frozen package registry against the active interpreter."""

    if planned is None:
        planned = {}
    if not isinstance(planned, Mapping):
        raise ExecutionError(f"runtime {category} registry is invalid")
    for name, expected in planned.items():
        try:
            observed = importlib.metadata.version(str(name))
        except importlib.metadata.PackageNotFoundError:
            observed = None
        if observed != expected:
            raise ExecutionError(
                f"runtime {category} drift for {name}: "
                f"planned={expected}, observed={observed}"
            )


def _validate_frozen_environment(
    environment: Mapping[str, Any] | Any,
) -> None:
    """Validate frozen environment provenance without consulting this host."""

    if not isinstance(environment, Mapping):
        raise ExecutionError("runtime environment provenance is invalid")
    for field in _SCIENTIFIC_ENVIRONMENT_FIELDS:
        value = environment.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ExecutionError(
                f"runtime environment provenance has invalid {field}"
            )
    packages = environment.get("packages")
    if not isinstance(packages, Mapping) or any(
        not isinstance(name, str)
        or not name
        or (version is not None and not isinstance(version, str))
        for name, version in packages.items()
    ):
        raise ExecutionError("runtime environment package registry is invalid")
    deterministic = environment.get("deterministic_environment")
    if not isinstance(deterministic, Mapping) or not deterministic or any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        for name, value in deterministic.items()
    ):
        raise ExecutionError(
            "runtime environment deterministic settings are invalid"
        )
    tensorflow_runtime = environment.get("tensorflow_runtime")
    if (
        not isinstance(tensorflow_runtime, Mapping)
        or not isinstance(tensorflow_runtime.get("available"), bool)
    ):
        raise ExecutionError("runtime TensorFlow receipt is invalid")


def _verify_environment(
    environment: Mapping[str, Any],
    *,
    policy: str,
    require_deterministic_environment: bool,
) -> None:
    """Verify scientific provenance, with optional exact GPU-host admission.

    Queue-neutral verification deliberately excludes physical host inventory
    (OS/machine/processor/CPU count) and the TensorFlow device/build probe.  It
    still enforces the interpreter, frozen package registry, and deterministic
    worker settings.  GPU-exact verification adds exact host and TensorFlow
    build/device/thread replay.
    """

    _validate_frozen_environment(environment)
    if policy not in _ENVIRONMENT_POLICIES:
        raise ExecutionError(f"unsupported runtime environment policy: {policy}")

    observed_environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    fields = _SCIENTIFIC_ENVIRONMENT_FIELDS
    if policy == ENVIRONMENT_POLICY_GPU_EXACT:
        observed_environment.update(
            {
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            }
        )
        fields += _HARDWARE_ENVIRONMENT_FIELDS
    for field in fields:
        observed = observed_environment[field]
        if environment.get(field) != observed:
            raise ExecutionError(
                f"runtime environment drift for {field}: "
                f"planned={environment.get(field)!r}, observed={observed!r}"
            )

    _verify_installed_versions(
        environment.get("packages"), category="package"
    )
    if require_deterministic_environment:
        deterministic = environment.get("deterministic_environment")
        if not isinstance(deterministic, Mapping) or not deterministic:
            raise ExecutionError(
                "runtime environment lacks frozen deterministic worker settings"
            )
        required = deterministic_environment()
        missing_or_wrong = [
            name
            for name, expected in required.items()
            if deterministic.get(name) != expected
        ]
        if missing_or_wrong:
            raise ExecutionError(
                "runtime environment lacks canonical deterministic worker "
                f"settings: {', '.join(missing_or_wrong)}"
            )
        for name, expected in deterministic.items():
            if os.environ.get(str(name)) != expected:
                raise ExecutionError(
                    f"deterministic worker environment differs for {name}: "
                    f"expected={expected!r}, observed={os.environ.get(str(name))!r}"
                )

    if policy != ENVIRONMENT_POLICY_GPU_EXACT:
        return
    probe = tensorflow_probe_script()
    environment_for_probe = dict(os.environ)
    environment_for_probe.update(deterministic_environment())
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            env=environment_for_probe,
        )
        observed_tf = parse_tensorflow_probe_output(completed.stdout)
    except (OSError, subprocess.CalledProcessError, ManifestError) as exc:
        observed_tf = {"available": False, "probe_error_type": type(exc).__name__}
    if observed_tf != environment.get("tensorflow_runtime"):
        raise ExecutionError("TensorFlow build/device/thread runtime differs from planning")


def _verify_pilot_execution_contract(
    manifest: Mapping[str, Any], design: Mapping[str, Any]
) -> None:
    """Fail closed on every scientific and evidence boundary of pilot10."""

    execution = manifest.get("execution", {})
    if (
        not isinstance(execution, Mapping)
        or execution.get("mode") != "pilot"
        or execution.get("profile") != "pilot10"
        or design.get("profile") != "pilot10"
    ):
        raise ExecutionError(
            "pilot10 requires an exact pilot mode/profile/design identity"
        )
    if design.get("cell_counts") != {
        "primary": 300,
        "structural_ablation": 0,
        "frozen_sensitivity": 0,
        "required": 300,
    }:
        raise ExecutionError("pilot10 manifest does not contain exactly 300 primary cells")
    expected_lanes = _runtime_plan_external_gpu_lanes(manifest)
    if execution.get("external_gpu_lanes") != expected_lanes:
        raise ExecutionError(
            f"pilot10 execution must use exactly {expected_lanes} external GPU lanes"
        )
    if execution.get("evidence_status") != "exploratory_provisional":
        raise ExecutionError("pilot10 evidence status is not exploratory/provisional")
    if execution.get("inference_scope") != (
        "descriptive_only_no_confirmatory_inference"
    ):
        raise ExecutionError("pilot10 inference scope is not descriptive-only")
    if manifest.get("primary_reference") is not None:
        raise ExecutionError("pilot10 execution cannot consume a primary reference")


def _runtime_plan_external_gpu_lanes(manifest: Mapping[str, Any]) -> int:
    """Bind GPU concurrency exclusively to the exact phased resource plan."""

    runtime_plan = manifest.get("runtime_plan")
    if not isinstance(runtime_plan, Mapping):
        return 4
    schema = runtime_plan.get("schema_version", "bdb-runtime-plan-v2")
    if schema == "bdb-runtime-plan-v2":
        if "resource_contract" in runtime_plan:
            raise ExecutionError("runtime-plan v2 cannot carry a v3 resource contract")
        return 4
    if schema not in {"bdb-runtime-plan-v3", "bdb-runtime-plan-v4"}:
        raise ExecutionError("runtime plan has an unsupported execution schema")
    from .runtime_phases import (
        betty_runtime_v3_resource_contract,
        betty_runtime_v4_resource_contract,
    )

    if schema == "bdb-runtime-plan-v4":
        embedded = runtime_plan.get("resource_contract")
        if not isinstance(embedded, Mapping):
            raise ExecutionError("runtime-plan v4 resource contract is missing")
        try:
            expected_resource = betty_runtime_v4_resource_contract(
                embedded.get("determinism_probe"),
                task_id=str(runtime_plan.get("task_id", "")),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ExecutionError(
                "runtime-plan v4 determinism evidence is missing or drifted"
            ) from exc
    else:
        expected_resource = betty_runtime_v3_resource_contract()
    if runtime_plan.get("resource_contract") != expected_resource:
        raise ExecutionError(f"{schema} resource contract is missing or drifted")
    if runtime_plan.get("gpu_lanes") != int(expected_resource["gpu"]["lanes"]):
        raise ExecutionError(f"{schema} GPU lane binding is inconsistent")
    return int(expected_resource["gpu"]["lanes"])


def verify_run_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
    require_deterministic_environment: bool = False,
    environment_policy: str | None = None,
    verify_tensorflow_runtime: bool | None = None,
) -> None:
    policy = _resolve_environment_policy(
        environment_policy, verify_tensorflow_runtime
    )
    if (
        policy == ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY
        and require_deterministic_environment
    ):
        raise ExecutionError(
            "host artifact-only admission cannot authorize a deterministic worker"
        )
    if manifest.get("schema_version") != "bdb-task-run-v1":
        raise ExecutionError("unsupported task run manifest")
    try:
        actual = manifest_sha256(manifest)
    except Exception as exc:
        raise ExecutionError(f"run manifest identity is invalid: {exc}") from exc
    if manifest.get("manifest_hash") != actual:
        raise ExecutionError("manifest_hash does not match the run manifest")
    if manifest.get("run_hash") != actual[:24]:
        raise ExecutionError("run_hash does not match the manifest hash prefix")
    verify_storage_backend_receipt(manifest.get("storage_backend"), repo_root=repo_root)
    design = manifest.get("task_design")
    if not isinstance(design, Mapping) or design.get("design_hash") is None:
        raise ExecutionError("run manifest has no task design")
    if manifest.get("required_cells") != design.get("required_cells"):
        raise ExecutionError("run required-cell registry differs from its task design")
    if manifest.get("primary_required_cells") != design.get("primary_cells"):
        raise ExecutionError("run primary-cell registry differs from its task design")
    if manifest.get("ablation_required_cells") != design.get("ablation_cells"):
        raise ExecutionError("run ablation-cell registry differs from its task design")
    if manifest.get("sensitivity_required_cells") != design.get("sensitivity_cells"):
        raise ExecutionError("run sensitivity-cell registry differs from its task design")
    if manifest.get("task_id") != design.get("task_id"):
        raise ExecutionError("run task ID differs from its task design")
    root = Path(repo_root).resolve()
    provenance = manifest.get("provenance", {})
    for record in provenance.get("code", []):
        path = root / str(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ExecutionError(f"code/config provenance drift: {path}")
    task_receipt = manifest.get("task_spec_receipt", {})
    task_path = root / str(task_receipt.get("path", ""))
    if not task_path.is_file() or sha256_file(task_path) != task_receipt.get("sha256"):
        raise ExecutionError("frozen TaskSpec receipt changed after planning")
    parsed_spec = validate_task_spec(
        manifest["task_spec"], repo_root=root, require_source=False
    )
    if parsed_spec.spec_hash != manifest.get("task_spec_hash"):
        raise ExecutionError("run task-spec hash differs from its embedded TaskSpec")
    frozen_spec = load_task_spec(task_path, repo_root=root, require_source=False)
    if frozen_spec.as_dict() != parsed_spec.as_dict():
        raise ExecutionError("embedded TaskSpec differs from the frozen task receipt")
    try:
        from .manifest import ManifestError, validate_task_scientific_receipts

        validate_task_scientific_receipts(parsed_spec, repo_root=root)
    except ManifestError as exc:
        raise ExecutionError(f"frozen scientific receipt failed: {exc}") from exc
    try:
        shared_registry_peer = build_shared_frozen_registry_binding(
            parsed_spec, repo_root=root
        )
    except ManifestError as exc:
        raise ExecutionError(f"shared frozen game registry failed: {exc}") from exc
    if manifest.get("shared_registry_peer") != shared_registry_peer:
        raise ExecutionError(
            "shared frozen game-registry receipt differs from run planning"
        )
    for model_id, model in parsed_spec.models.items():
        development = model.get("development_receipt")
        if not isinstance(development, Mapping):
            raise ExecutionError(f"model {model_id} has no frozen development receipt")
        development_path = (root / str(development.get("path", ""))).resolve()
        try:
            development_path.relative_to(root)
            value = json.loads(development_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExecutionError(
                f"could not verify development receipt for {model_id}: {exc}"
            ) from exc
        unsigned_development = {
            key: item for key, item in value.items() if key != "receipt_hash"
        }
        development_hash = hashlib.sha256(
            json.dumps(
                unsigned_development, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if (
            sha256_file(development_path) != development.get("file_sha256")
            or
            value.get("receipt_hash") != development_hash
            or development_hash != development.get("receipt_hash")
            or value.get("selected", {}).get("config") != model.get("selected_config")
        ):
            raise ExecutionError(f"development selection drift for model {model_id}")
        source_path_value = development.get("source_path")
        if source_path_value is not None:
            source_path = (root / str(source_path_value)).resolve()
            try:
                source_path.relative_to(root)
                source_value = json.loads(source_path.read_text(encoding="utf-8"))
            except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ExecutionError(
                    f"could not verify raw development receipt for {model_id}: {exc}"
                ) from exc
            source_unsigned = {
                key: item
                for key, item in source_value.items()
                if key != "receipt_hash"
            }
            source_hash = hashlib.sha256(
                json.dumps(
                    source_unsigned, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if (
                sha256_file(source_path) != development.get("source_file_sha256")
                or source_value.get("receipt_hash") != source_hash
                or source_hash != development.get("source_receipt_hash")
            ):
                raise ExecutionError(
                    f"raw development receipt drift for model {model_id}"
                )
    profile = str(design.get("profile", "custom"))
    validate_task_design(
        design,
        parsed_spec,
        profile=None if profile == "custom" else profile,
    )
    execution_mode = manifest.get("execution", {}).get("mode")
    if execution_mode == "pilot":
        _verify_pilot_execution_contract(manifest, design)
    elif execution_mode == "definitive":
        if profile == "pilot10":
            _verify_pilot_execution_contract(manifest, design)
        if manifest.get("execution", {}).get("profile") != profile:
            raise ExecutionError("execution profile differs from its task design")
        if profile == "full100":
            counts = design.get("cell_counts")
            if counts != {
                "primary": 3000,
                "structural_ablation": 80,
                "frozen_sensitivity": 0,
                "required": 3080,
            }:
                raise ExecutionError("full100 manifest does not contain 3000+80 cells")
            expected_lanes = _runtime_plan_external_gpu_lanes(manifest)
            if manifest.get("execution", {}).get("external_gpu_lanes") != expected_lanes:
                raise ExecutionError(
                    f"full100 execution must use exactly {expected_lanes} external GPU lanes"
                )
            if manifest.get("primary_reference") is not None:
                raise ExecutionError("full100 execution cannot consume a primary reference")
        elif profile == "sensitivity20":
            if design.get("cell_counts") != {
                "primary": 0,
                "structural_ablation": 0,
                "frozen_sensitivity": 200,
                "required": 200,
            }:
                raise ExecutionError("sensitivity20 manifest does not contain 200 cells")
            expected_lanes = _runtime_plan_external_gpu_lanes(manifest)
            if manifest.get("execution", {}).get("external_gpu_lanes") != expected_lanes:
                raise ExecutionError(
                    "sensitivity20 execution must use exactly "
                    f"{expected_lanes} external GPU lanes"
                )
            try:
                validate_primary_reference_binding(
                    manifest.get("primary_reference"),
                    parsed_spec,
                    design,
                    repo_root=root,
                )
            except ManifestError as exc:
                raise ExecutionError(
                    f"sensitivity primary reference failed: {exc}"
                ) from exc
    prepared = manifest.get("prepared", {})
    prepared_root = root / str(prepared.get("path", ""))
    prepared_receipt_path = prepared_root / "receipt.json"
    if (
        not prepared_receipt_path.is_file()
        or sha256_file(prepared_receipt_path) != prepared.get("receipt_sha256")
    ):
        raise ExecutionError("prepared receipt changed after planning")
    prepared_receipt = load_prepared_receipt(prepared_root, verify_files=True)
    if prepared_receipt.get("prepared_hash") != prepared.get("prepared_hash"):
        raise ExecutionError("prepared scientific hash differs from run manifest")
    expected_variant = "primary"
    if profile == "sensitivity20":
        frozen_sensitivities = [
            item
            for item in parsed_spec.sensitivities
            if item["selection"] == "frozen_prespecified"
        ]
        if len(frozen_sensitivities) != 1:
            raise ExecutionError(
                "sensitivity20 lacks exactly one frozen sensitivity contract"
            )
        expected_variant = str(
            frozen_sensitivities[0]["execution"]["prepared_variant"]
        )
    prepared_task = load_prepared_task(
        prepared_root, mmap_mode="r", verify_files=False
    )
    try:
        prepared_semantics = validate_prepared_binding(
            parsed_spec,
            prepared_receipt,
            prepared_task,
            require_frozen_identity=profile != "sensitivity20",
            prepared_contract_override=prepared_contract_for_variant(
                parsed_spec, expected_variant
            ),
        )
    except ManifestError as exc:
        raise ExecutionError(f"prepared semantic binding failed: {exc}") from exc
    if prepared_semantics.get("semantic_hash") != prepared.get("semantic_hash"):
        raise ExecutionError("prepared semantic hash differs from run manifest")
    preflight = manifest.get("preflight")
    if preflight is not None:
        preflight_path = root / str(preflight.get("path", ""))
        if (
            not preflight_path.is_file()
            or sha256_file(preflight_path) != preflight.get("sha256")
        ):
            raise ExecutionError("preflight receipt changed after definitive planning")
    runtime_plan = manifest.get("runtime_plan")
    if runtime_plan is not None:
        runtime_path = root / str(runtime_plan.get("path", ""))
        if (
            not runtime_path.is_file()
            or sha256_file(runtime_path) != runtime_plan.get("sha256")
        ):
            raise ExecutionError("runtime plan changed after definitive planning")
    lock = provenance.get("dependency_lock", {})
    if not isinstance(lock, Mapping):
        raise ExecutionError("runtime dependency lock provenance is invalid")
    lock_relative = Path(str(lock.get("path", "")))
    if lock_relative.is_absolute() or ".." in lock_relative.parts:
        raise ExecutionError("runtime dependency lock path is unsafe")
    lock_path = (root / lock_relative).resolve()
    try:
        lock_path.relative_to(root)
    except ValueError as exc:
        raise ExecutionError("runtime dependency lock path escapes repo_root") from exc
    expected_dependencies = lock.get("expected")
    planned_dependencies = lock.get("installed")
    if (
        not lock_path.is_file()
        or sha256_file(lock_path) != lock.get("sha256")
        or not isinstance(expected_dependencies, Mapping)
        or not isinstance(planned_dependencies, Mapping)
        or dict(expected_dependencies) != dict(planned_dependencies)
        or lock.get("mismatches") != {}
    ):
        raise ExecutionError("runtime dependency lock provenance is invalid")
    environment = provenance.get("environment", {})
    _validate_frozen_environment(environment)
    if policy != ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY:
        _verify_installed_versions(
            planned_dependencies, category="dependency"
        )
        _verify_environment(
            environment,
            policy=policy,
            require_deterministic_environment=require_deterministic_environment,
        )


def open_run(
    run_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    require_deterministic_environment: bool = False,
    environment_policy: str | None = None,
    verify_tensorflow_runtime: bool | None = None,
) -> GenericRunStorage:
    storage = GenericRunStorage.open(run_dir)
    verify_run_manifest(
        storage.manifest,
        repo_root=repo_root,
        require_deterministic_environment=require_deterministic_environment,
        environment_policy=environment_policy,
        verify_tensorflow_runtime=verify_tensorflow_runtime,
    )
    return storage


def _execution_design(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the full design or its explicit smoke/benchmark projection."""

    design = copy.deepcopy(manifest["task_design"])
    smoke = manifest.get("execution", {}).get("smoke", {})
    benchmark = manifest.get("execution", {}).get("benchmark", {})
    projection = None
    expected_count = None
    model_count = len(
        {
            str(record.get("model"))
            for record in design.get("required_cells", [])
            if record.get("model") is not None
        }
    )
    if model_count < 1:
        raise ExecutionError("execution projection design has no model panel")
    if smoke.get("enabled") is True:
        projection = smoke
        expected_count = model_count
    elif benchmark.get("enabled") is True:
        projection = benchmark
        expected_count = 2 * model_count
    if projection is None:
        return design
    identities = {
        (
            str(record.get("source_branch", record["branch"])),
            record.get("source_ablation_id", record.get("ablation_id")),
            int(record["repeat"]),
            int(record["n_train"]), str(record["model"]),
        )
        for record in projection.get("cells", [])
    }
    design["required_cells"] = [
        record
        for record in design["required_cells"]
        if (
            str(record["branch"]), record.get("ablation_id"), int(record["repeat"]),
            int(record["n_train"]), str(record["model"]),
        ) in identities
    ]
    if len(design["required_cells"]) != expected_count:
        raise ExecutionError("execution projection is incomplete or ambiguous")
    intervention = projection.get("intervention")
    if intervention is not None:
        if (
            not isinstance(intervention, Mapping)
            or set(intervention)
            != {"branch", "sensitivity_id", "prepared_variant"}
            or intervention.get("branch") != "frozen_sensitivity"
            or not intervention.get("sensitivity_id")
            or not intervention.get("prepared_variant")
        ):
            raise ExecutionError("benchmark intervention projection is invalid")
        for record in design["required_cells"]:
            record["branch"] = "frozen_sensitivity"
            record["ablation_id"] = None
            record["sensitivity_id"] = str(intervention["sensitivity_id"])
    return design


def filter_execution_design(
    design: Mapping[str, Any],
    *,
    queues: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Independently project a design without changing its scientific identity."""

    output = copy.deepcopy(design)
    records = output.get("required_cells")
    if not isinstance(records, list) or not records:
        raise ExecutionError("execution design has no required cells")
    available = {
        "queues": {str(record["queue"]) for record in records},
        "repeats": {int(record["repeat"]) for record in records},
        "anchors": {int(record["n_train"]) for record in records},
        "models": {str(record["model"]) for record in records},
        "ablations": {
            "none" if record.get("ablation_id") is None else str(record["ablation_id"])
            for record in records
        },
        "sensitivities": {
            "none" if record.get("sensitivity_id") is None else str(record["sensitivity_id"])
            for record in records
        },
    }
    requested = {
        "queues": None if queues is None else {str(value) for value in queues},
        "repeats": None if repeats is None else {int(value) for value in repeats},
        "anchors": None if anchors is None else {int(value) for value in anchors},
        "models": None if models is None else {str(value) for value in models},
        "ablations": None if ablations is None else {str(value) for value in ablations},
        "sensitivities": (
            None if sensitivities is None else {str(value) for value in sensitivities}
        ),
    }
    for name, values in requested.items():
        if values is not None:
            if not values:
                raise ExecutionError(f"{name} filter cannot be empty")
            unknown = values - available[name]
            if unknown:
                raise ExecutionError(f"unknown {name} filter values: {sorted(unknown)}")

    def keep(record: Mapping[str, Any]) -> bool:
        ablation = "none" if record.get("ablation_id") is None else str(record["ablation_id"])
        sensitivity = (
            "none"
            if record.get("sensitivity_id") is None
            else str(record["sensitivity_id"])
        )
        checks = {
            "queues": str(record["queue"]),
            "repeats": int(record["repeat"]),
            "anchors": int(record["n_train"]),
            "models": str(record["model"]),
            "ablations": ablation,
            "sensitivities": sensitivity,
        }
        return all(
            requested[name] is None or value in requested[name]
            for name, value in checks.items()
        )

    output["required_cells"] = [record for record in records if keep(record)]
    if not output["required_cells"]:
        raise ExecutionError("execution filters select no cells")
    output["execution_filter"] = {
        name: None if values is None else sorted(values)
        for name, values in requested.items()
    }
    return output


def _effective_task_spec(manifest: Mapping[str, Any]) -> TaskSpec:
    mapping = copy.deepcopy(manifest["task_spec"])
    smoke = manifest.get("execution", {}).get("smoke", {})
    if smoke.get("enabled") is True:
        for entry in mapping["models"].values():
            if entry.get("family") in {
                "relnet", "attn_relnet", "set_transformer"
            }:
                config = dict(entry.get("selected_config", {}))
                config["max_epochs"] = int(smoke["neural_max_epochs"])
                config["patience"] = int(smoke["neural_patience"])
                entry["selected_config"] = config
    return TaskSpec.from_mapping(mapping)


def _cell_record(design: Mapping[str, Any], key: CellKey) -> Mapping[str, Any]:
    matches = [
        record
        for record in design["required_cells"]
        if record["branch"] == key.branch
        and int(record["repeat"]) == key.repeat
        and int(record["n_train"]) == key.n_train
        and record["model"] == key.model
    ]
    if len(matches) != 1:
        raise ExecutionError(f"cell registry lookup is not unique: {key}")
    return matches[0]


NEURAL_PHASES = frozenset({"monolithic", "selector", "refit"})
NEURAL_SELECTOR_RECEIPT_SCHEMA_VERSION = "bdb-neural-selector-receipt-v1"


def neural_selector_receipt_path(run_dir: str | Path, key: CellKey) -> Path:
    """Return the phase receipt path without changing the final cell namespace."""

    root = Path(run_dir).resolve()
    target = (
        root
        / "neural_selection"
        / key.branch
        / f"repeat_{key.repeat:03d}"
        / f"n_{key.n_train:04d}"
        / key.model
        / "selector.json"
    )
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise ExecutionError("selector receipt path escapes the run directory") from exc
    return target


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _immutable_write_bytes(path: Path, payload: bytes) -> None:
    """Hard-link commit immutable bytes, accepting only an exact prior value."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ExecutionError(f"immutable selector artifact cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ExecutionError(
                    f"refusing to replace a different immutable selector artifact: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _selector_run_binding(storage: GenericRunStorage) -> dict[str, Any]:
    manifest = storage.manifest
    return {
        "manifest_hash": str(storage.manifest_hash),
        "task_id": str(manifest["task_id"]),
        "task_spec_hash": str(manifest["task_spec_hash"]),
        "prepared_hash": str(manifest["prepared"]["prepared_hash"]),
        "profile": str(manifest["execution"]["profile"]),
    }


def _selector_cell_identity(
    key: CellKey, record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "branch": key.branch,
        "repeat": int(key.repeat),
        "n_train": int(key.n_train),
        "model": key.model,
        "ablation_id": record.get("ablation_id"),
        "sensitivity_id": record.get("sensitivity_id"),
    }


def _build_selector_receipt(
    storage: GenericRunStorage,
    key: CellKey,
    record: Mapping[str, Any],
    core: Mapping[str, Any],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    selection = core.get("selection") if isinstance(core, Mapping) else None
    history = selection.get("selector_history") if isinstance(selection, Mapping) else None
    validation_losses = history.get("val_loss") if isinstance(history, Mapping) else None
    if not isinstance(validation_losses, list) or not validation_losses:
        raise ExecutionError("selector core has no observed validation epochs")
    best_epoch = selection.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 1:
        raise ExecutionError("selector core has an invalid best epoch")
    body = {
        "schema_version": NEURAL_SELECTOR_RECEIPT_SCHEMA_VERSION,
        "run_binding": _selector_run_binding(storage),
        "cell": _selector_cell_identity(key, record),
        "selector_core": dict(core),
        "selector_elapsed_seconds": float(elapsed_seconds),
        "selector_observed_epochs": int(len(validation_losses)),
        "best_epoch": int(best_epoch),
        "refit_epochs": int(best_epoch),
    }
    body["receipt_hash"] = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    return body


def _write_selector_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = _pretty_json_bytes(dict(receipt))
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar_payload = f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n".encode(
        "ascii"
    )
    if sidecar.exists() and not path.exists():
        raise ExecutionError(f"orphan selector checksum sidecar: {sidecar}")
    _immutable_write_bytes(path, payload)
    _immutable_write_bytes(sidecar, sidecar_payload)


def _read_selector_receipt(
    path: Path,
    *,
    expected_run_binding: Mapping[str, Any],
    expected_cell: Mapping[str, Any],
    repair_missing_sidecar: bool = False,
) -> dict[str, Any]:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not path.exists():
        if sidecar.exists():
            raise ExecutionError(f"orphan selector checksum sidecar: {sidecar}")
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file() or sidecar.is_symlink():
        raise ExecutionError(f"selector receipt path is unsafe: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    missing_sidecar = not sidecar.exists()
    if missing_sidecar:
        if not repair_missing_sidecar:
            raise ExecutionError(f"selector receipt lacks checksum sidecar: {path}")
    elif not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        raise ExecutionError(f"selector receipt checksum mismatch: {path}")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"selector receipt is not canonical JSON: {path}") from exc
    expected_fields = {
        "schema_version",
        "run_binding",
        "cell",
        "selector_core",
        "selector_elapsed_seconds",
        "selector_observed_epochs",
        "best_epoch",
        "refit_epochs",
        "receipt_hash",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ExecutionError("selector receipt fields are invalid")
    if payload != _pretty_json_bytes(value):
        raise ExecutionError("selector receipt JSON bytes are not canonical")
    claimed_hash = value.pop("receipt_hash")
    actual_hash = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    value["receipt_hash"] = claimed_hash
    if claimed_hash != actual_hash:
        raise ExecutionError("selector receipt embedded hash mismatch")
    if value.get("schema_version") != NEURAL_SELECTOR_RECEIPT_SCHEMA_VERSION:
        raise ExecutionError("selector receipt schema is invalid")
    if value.get("run_binding") != dict(expected_run_binding):
        raise ExecutionError("selector receipt belongs to a different run")
    if value.get("cell") != dict(expected_cell):
        raise ExecutionError("selector receipt belongs to a different cell")
    elapsed = value.get("selector_elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ExecutionError("selector elapsed time is invalid")
    core = value.get("selector_core")
    selection = core.get("selection") if isinstance(core, Mapping) else None
    history = selection.get("selector_history") if isinstance(selection, Mapping) else None
    observed = history.get("val_loss") if isinstance(history, Mapping) else None
    best_epoch = selection.get("best_epoch") if isinstance(selection, Mapping) else None
    if (
        not isinstance(observed, list)
        or not observed
        or value.get("selector_observed_epochs") != len(observed)
        or value.get("best_epoch") != best_epoch
        or value.get("refit_epochs") != best_epoch
    ):
        raise ExecutionError("selector phase metadata does not match its scientific core")
    if missing_sidecar:
        _immutable_write_bytes(sidecar, expected_sidecar)
    return value


def load_neural_selector_receipt(
    run_dir: str | Path,
    storage: GenericRunStorage,
    key: CellKey,
    record: Mapping[str, Any],
    *,
    repair_missing_sidecar: bool = False,
) -> dict[str, Any]:
    """Load one checksum- and provenance-bound selector receipt."""

    return _read_selector_receipt(
        neural_selector_receipt_path(run_dir, key),
        expected_run_binding=_selector_run_binding(storage),
        expected_cell=_selector_cell_identity(key, record),
        repair_missing_sidecar=repair_missing_sidecar,
    )


def run_repeat_queue(
    run_dir: str | Path,
    *,
    repeat: int,
    queue: str,
    repo_root: str | Path = ".",
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
    neural_phase: str = "monolithic",
) -> dict[str, Any]:
    """Load one queue representation once and execute every anchor/model cell."""

    if neural_phase not in NEURAL_PHASES:
        raise ExecutionError(f"unsupported neural phase: {neural_phase}")
    if neural_phase != "monolithic" and queue != "gpu_neural":
        raise ExecutionError("selector/refit phases require the exact gpu_neural queue")
    root = Path(repo_root).resolve()
    storage = open_run(
        run_dir,
        repo_root=root,
        require_deterministic_environment=True,
        environment_policy=(
            ENVIRONMENT_POLICY_QUEUE_NEUTRAL
            if queue == "cpu_tabular"
            else ENVIRONMENT_POLICY_GPU_EXACT
        ),
    )
    manifest = storage.manifest
    design = filter_execution_design(
        _execution_design(manifest),
        queues=[queue],
        repeats=[repeat],
        anchors=anchors,
        models=models,
        ablations=ablations,
        sensitivities=sensitivities,
    )
    spec = _effective_task_spec(manifest)
    prepared_path = root / manifest["prepared"]["path"]
    task = load_prepared_task(prepared_path, mmap_mode="r", verify_files=True)
    runtime = TaskRuntime(task, queue=queue)
    keys = []
    for record in design["required_cells"]:
        if int(record["repeat"]) == int(repeat) and record["queue"] == queue:
            keys.append(
                CellKey(
                    str(record["branch"]), int(record["repeat"]),
                    int(record["n_train"]), str(record["model"]),
                )
            )
    keys.sort(key=lambda key: (key.n_train, key.model))
    if not keys:
        raise ExecutionError(f"repeat {repeat} has no cells in queue {queue}")
    completed = skipped = 0
    elapsed: list[float] = []
    phase_records: list[dict[str, Any]] = []
    for key in keys:
        record = _cell_record(design, key)
        validation = storage.validate_cell(key)
        if validation.status is CellStatus.COMPLETE:
            skipped += 1
            continue
        if (
            validation.status is CellStatus.RUNNING
            and neural_phase != "refit"
        ):
            skipped += 1
            continue
        selector_receipt = None
        selector_path = neural_selector_receipt_path(run_dir, key)
        expected_run_binding = _selector_run_binding(storage)
        expected_cell = _selector_cell_identity(key, record)
        if neural_phase == "selector":
            try:
                selector_receipt = _read_selector_receipt(
                    selector_path,
                    expected_run_binding=expected_run_binding,
                    expected_cell=expected_cell,
                    repair_missing_sidecar=True,
                )
            except FileNotFoundError:
                selector_receipt = None
            if selector_receipt is not None:
                validate_neural_selector_for_cell(
                    selector_receipt["selector_core"],
                    runtime,
                    spec,
                    design,
                    repeat=key.repeat,
                    n_train=key.n_train,
                    model_id=key.model,
                    branch=key.branch,
                    ablation_id=record.get("ablation_id"),
                    sensitivity_id=record.get("sensitivity_id"),
                )
                skipped += 1
                phase_records.append(
                    {
                        "cell": expected_cell,
                        "selector_observed_epochs": int(
                            selector_receipt["selector_observed_epochs"]
                        ),
                        "best_epoch": int(selector_receipt["best_epoch"]),
                        "refit_epochs": int(selector_receipt["refit_epochs"]),
                        "receipt_hash": str(selector_receipt["receipt_hash"]),
                    }
                )
                continue
            selector_claim = claim_phase_lease(
                selector_path.parent / "_RUNNING",
                manifest_hash=str(storage.manifest_hash),
                key=key,
                phase="selector",
                completion_path=selector_path,
                completion_validator=lambda: bool(
                    _read_selector_receipt(
                        selector_path,
                        expected_run_binding=expected_run_binding,
                        expected_cell=expected_cell,
                        repair_missing_sidecar=True,
                    )
                ),
            )
            if selector_claim.status == "running":
                skipped += 1
                continue
            if selector_claim.status == "complete":
                selector_receipt = _read_selector_receipt(
                    selector_path,
                    expected_run_binding=expected_run_binding,
                    expected_cell=expected_cell,
                    repair_missing_sidecar=True,
                )
                validate_neural_selector_for_cell(
                    selector_receipt["selector_core"],
                    runtime,
                    spec,
                    design,
                    repeat=key.repeat,
                    n_train=key.n_train,
                    model_id=key.model,
                    branch=key.branch,
                    ablation_id=record.get("ablation_id"),
                    sensitivity_id=record.get("sensitivity_id"),
                )
                skipped += 1
                phase_records.append(
                    {
                        "cell": expected_cell,
                        "selector_observed_epochs": int(
                            selector_receipt["selector_observed_epochs"]
                        ),
                        "best_epoch": int(selector_receipt["best_epoch"]),
                        "refit_epochs": int(selector_receipt["refit_epochs"]),
                        "receipt_hash": str(selector_receipt["receipt_hash"]),
                    }
                )
                continue
            started = time.monotonic()
            try:
                core = run_neural_selector(
                    runtime,
                    spec,
                    design,
                    repeat=key.repeat,
                    n_train=key.n_train,
                    model_id=key.model,
                    branch=key.branch,
                    ablation_id=record.get("ablation_id"),
                    sensitivity_id=record.get("sensitivity_id"),
                )
                duration = float(time.monotonic() - started)
                validate_neural_selector_for_cell(
                    core,
                    runtime,
                    spec,
                    design,
                    repeat=key.repeat,
                    n_train=key.n_train,
                    model_id=key.model,
                    branch=key.branch,
                    ablation_id=record.get("ablation_id"),
                    sensitivity_id=record.get("sensitivity_id"),
                )
                receipt = _build_selector_receipt(
                    storage, key, record, core, elapsed_seconds=duration
                )
                _write_selector_receipt(selector_path, receipt)
                selector_receipt = _read_selector_receipt(
                    selector_path,
                    expected_run_binding=expected_run_binding,
                    expected_cell=expected_cell,
                )
                completed += 1
                elapsed.append(duration)
                phase_records.append(
                    {
                        "cell": expected_cell,
                        "selector_observed_epochs": int(
                            selector_receipt["selector_observed_epochs"]
                        ),
                        "best_epoch": int(selector_receipt["best_epoch"]),
                        "refit_epochs": int(selector_receipt["refit_epochs"]),
                        "receipt_hash": str(selector_receipt["receipt_hash"]),
                    }
                )
            finally:
                release_phase_lease(selector_claim)
            continue
        if neural_phase == "refit":
            try:
                selector_receipt = _read_selector_receipt(
                    selector_path,
                    expected_run_binding=expected_run_binding,
                    expected_cell=expected_cell,
                    repair_missing_sidecar=True,
                )
            except FileNotFoundError as exc:
                raise ExecutionError(
                    f"refit requires a valid selector receipt: {selector_path}"
                ) from exc
            validate_neural_selector_for_cell(
                selector_receipt["selector_core"],
                runtime,
                spec,
                design,
                repeat=key.repeat,
                n_train=key.n_train,
                model_id=key.model,
                branch=key.branch,
                ablation_id=record.get("ablation_id"),
                sensitivity_id=record.get("sensitivity_id"),
            )
        phase_claim = None
        if neural_phase == "refit":
            phase_claim = claim_phase_lease(
                Path(run_dir) / key.relative_dir / "_RUNNING",
                manifest_hash=str(storage.manifest_hash),
                key=key,
                phase="refit",
                completion_path=Path(run_dir) / key.relative_dir / "_SUCCESS",
                completion_validator=lambda: (
                    storage.validate_cell(key).status is CellStatus.COMPLETE
                ),
            )
            if phase_claim.status == "running":
                skipped += 1
                continue
            if phase_claim.status == "complete":
                if storage.validate_cell(key).status is not CellStatus.COMPLETE:
                    raise ExecutionError(
                        f"completed cell marker is invalid during refit claim: {key}"
                    )
                skipped += 1
                continue
        else:
            overwrite = validation.status is CellStatus.CORRUPT
            try:
                storage.mark_cell_running(key, overwrite=overwrite)
            except Exception:
                if storage.validate_cell(key).status in {
                    CellStatus.COMPLETE,
                    CellStatus.RUNNING,
                }:
                    skipped += 1
                    continue
                raise
        started = time.monotonic()
        try:
            result = run_cell(
                runtime,
                spec,
                design,
                repeat=key.repeat,
                n_train=key.n_train,
                model_id=key.model,
                branch=key.branch,
                ablation_id=record.get("ablation_id"),
                sensitivity_id=record.get("sensitivity_id"),
                neural_selector_core=(
                    None
                    if selector_receipt is None
                    else selector_receipt["selector_core"]
                ),
            )
            duration = float(time.monotonic() - started)
            result.metrics["elapsed_seconds"] = duration
            from .preflight import process_peak_rss_bytes

            result.metrics["process_peak_rss_bytes"] = process_peak_rss_bytes()
            result.history["elapsed_seconds"] = duration
            if manifest.get("execution", {}).get("smoke", {}).get("enabled") is True:
                result.history["smoke_override"] = dict(manifest["execution"]["smoke"])
            wrote = storage.write_cell_artifacts(
                key,
                metrics=result.metrics,
                predictions=result.predictions,
                history=result.history,
                arrays=result.arrays or None,
                preferred_predictions="parquet",
            )
            if not wrote and storage.validate_cell(key).status is not CellStatus.COMPLETE:
                raise ExecutionError(f"storage declined incomplete cell {key}")
            final_validation = storage.validate_cell(key)
            if final_validation.status is not CellStatus.COMPLETE:
                raise ExecutionError(f"cell failed post-write validation: {final_validation}")
            completed += 1
            elapsed.append(duration)
            if selector_receipt is not None:
                phase_records.append(
                    {
                        "cell": expected_cell,
                        "selector_observed_epochs": int(
                            selector_receipt["selector_observed_epochs"]
                        ),
                        "best_epoch": int(selector_receipt["best_epoch"]),
                        "refit_epochs": int(selector_receipt["refit_epochs"]),
                        "refit_elapsed_seconds": duration,
                        "receipt_hash": str(selector_receipt["receipt_hash"]),
                    }
                )
        finally:
            if phase_claim is None:
                storage.clear_cell_running(key)
            else:
                release_phase_lease(phase_claim)
    result = {
        "repeat": int(repeat),
        "queue": queue,
        "filters": design.get("execution_filter"),
        "completed": completed,
        "skipped": skipped,
        "elapsed_seconds": float(sum(elapsed)),
    }
    if neural_phase != "monolithic":
        result["neural_phase"] = neural_phase
        result["phase_records"] = phase_records
    return result


def _worker_environment(root: Path, queue: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            **deterministic_environment(),
            "MPLCONFIGDIR": "/tmp/bdb-suite-mpl",
            "XDG_CACHE_HOME": "/tmp/bdb-suite-cache",
        }
    )
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) if not prior else f"{root}{os.pathsep}{prior}"
    if queue == "cpu_tabular":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    return environment


def _run_worker_subprocess(
    run_dir: Path,
    repeat: int,
    queue: str,
    root: Path,
    *,
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
    neural_phase: str = "monolithic",
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "bdb_study",
        "--repo-root",
        str(root),
        "_worker",
        "--run-dir",
        str(run_dir),
        "--repeat",
        str(repeat),
        "--queue",
        queue,
    ]
    if neural_phase != "monolithic":
        command.extend(["--neural-phase", neural_phase])
    for anchor in anchors or ():
        command.extend(["--anchor", str(int(anchor))])
    for model in models or ():
        command.extend(["--model", str(model)])
    for ablation in ablations or ():
        command.extend(["--ablation", str(ablation)])
    for sensitivity in sensitivities or ():
        command.extend(["--sensitivity", str(sensitivity)])
    result = subprocess.run(
        command,
        cwd=root,
        env=_worker_environment(root, queue),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExecutionError(
            f"{queue} repeat {repeat} failed ({result.returncode})\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"worker returned invalid JSON: {result.stdout[-2000:]}") from exc


def _run_queue(
    run_dir: Path,
    queue: str,
    repeats: list[int],
    workers: int,
    root: Path,
    *,
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
    neural_phase: str = "monolithic",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=queue) as pool:
        futures = {
            pool.submit(
                _run_worker_subprocess,
                run_dir,
                repeat,
                queue,
                root,
                anchors=anchors,
                models=models,
                ablations=ablations,
                sensitivities=sensitivities,
                neural_phase=neural_phase,
            ): repeat
            for repeat in repeats
        }
        for future in as_completed(futures):
            try:
                output.append(future.result())
            except Exception as exc:
                failures.append(str(exc))
    if failures:
        raise ExecutionError("\n\n".join(failures))
    return sorted(output, key=lambda row: int(row["repeat"]))


def run_missing_cells(
    run_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    queues: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
    neural_phase: str = "monolithic",
) -> dict[str, Any]:
    """Run CPU and GPU queues concurrently with manifest-frozen concurrency."""

    if neural_phase not in NEURAL_PHASES:
        raise ExecutionError(f"unsupported neural phase: {neural_phase}")
    if neural_phase != "monolithic" and (
        queues is None
        or len(list(queues)) != 1
        or str(list(queues)[0]) != "gpu_neural"
    ):
        raise ExecutionError(
            "selector/refit execution requires exactly --queue gpu_neural"
        )
    root = Path(repo_root).resolve()
    directory = Path(run_dir).resolve()
    requested_queues = None if queues is None else {str(value) for value in queues}
    storage = open_run(
        directory,
        repo_root=root,
        environment_policy=(
            ENVIRONMENT_POLICY_QUEUE_NEUTRAL
            if requested_queues == {"cpu_tabular"}
            else ENVIRONMENT_POLICY_GPU_EXACT
        ),
    )
    manifest = storage.manifest
    queue_specs = manifest["execution"]["queues"]
    execution_design = filter_execution_design(
        _execution_design(manifest),
        queues=queues,
        repeats=repeats,
        anchors=anchors,
        models=models,
        ablations=ablations,
        sensitivities=sensitivities,
    )
    pending = pending_cells_by_queue(storage, execution_design)
    repeats_by_queue = {
        queue: sorted({key.repeat for key in keys}) for queue, keys in pending.items()
    }
    started = time.monotonic()
    results: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bdb-queue") as pool:
        futures = {}
        selected_queues = (
            [str(value) for value in queues]
            if queues is not None
            else ["cpu_tabular", "gpu_neural"]
        )
        for queue in selected_queues:
            repeats = repeats_by_queue[queue]
            if not repeats:
                results[queue] = []
                continue
            workers = int(queue_specs[queue]["workers"])
            futures[
                pool.submit(
                    _run_queue,
                    directory,
                    queue,
                    repeats,
                    workers,
                    root,
                    anchors=anchors,
                    models=models,
                    ablations=ablations,
                    sensitivities=sensitivities,
                    neural_phase=neural_phase,
                )
            ] = queue
        failures = []
        for future, queue in futures.items():
            try:
                results[queue] = future.result()
            except Exception as exc:
                failures.append(f"{queue}: {exc}")
        if failures:
            raise ExecutionError("\n\n".join(failures))
    output = {
        "elapsed_seconds": float(time.monotonic() - started),
        "queues": results,
        "filters": execution_design.get("execution_filter"),
        "status": queue_status(directory, repo_root=root),
    }
    if neural_phase != "monolithic":
        output["neural_phase"] = neural_phase
    return output


def queue_status(
    run_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    queues: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
    anchors: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
    ablations: Sequence[str] | None = None,
    sensitivities: Sequence[str] | None = None,
) -> dict[str, Any]:
    # Status is a checksum/metadata operation and is intentionally portable to
    # login and CPU-only nodes after GPU suitability was frozen by preflight.
    storage = open_run(
        run_dir,
        repo_root=repo_root,
        environment_policy=ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
    )
    design = filter_execution_design(
        _execution_design(storage.manifest),
        queues=queues,
        repeats=repeats,
        anchors=anchors,
        models=models,
        ablations=ablations,
        sensitivities=sensitivities,
    )
    status = status_by_queue(storage, design)
    durations: dict[str, list[float]] = {}
    for record in design["required_cells"]:
        key = CellKey(
            str(record["branch"]), int(record["repeat"]),
            int(record["n_train"]), str(record["model"]),
        )
        if not storage.validate_cell(key).is_complete:
            continue
        metrics = storage.load_metrics(key)
        elapsed = metrics.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            durations.setdefault(key.model, []).append(float(elapsed))
    configured = storage.manifest.get("execution", {}).get("seconds_per_model", {})
    estimates = {
        model: float(np.mean(durations[model])) if model in durations else float(configured[model])
        for model in {str(record["model"]) for record in design["required_cells"]}
        if model in durations or model in configured
    }
    queue_estimate = None
    all_models = {str(record["model"]) for record in design["required_cells"]}
    if set(estimates) == all_models:
        queue_config = storage.manifest["execution"]["queues"]
        queue_estimate = estimate_queue_eta_seconds(
            storage,
            design,
            estimates,
            cpu_workers=int(queue_config["cpu_tabular"]["workers"]),
            gpu_workers=int(queue_config["gpu_neural"]["workers"]),
        )
    return {
        "manifest_hash": storage.manifest_hash,
        "filters": design.get("execution_filter"),
        "queues": {
            queue: {
                **counts.as_dict(),
                "eta_seconds": (
                    queue_estimate["queue_seconds"][queue]
                    if queue_estimate is not None else None
                ),
            }
            for queue, counts in status.items()
        },
        "concurrent_eta_seconds": (
            queue_estimate["eta_seconds"] if queue_estimate is not None else None
        ),
    }
