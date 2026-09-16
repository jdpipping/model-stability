"""Command-line workflow for the additive harmonized BDB suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _default_raw(root: Path, task_id: str) -> Path:
    year = task_id[3:7]
    return root / "data" / f"bdb{year}" / "raw"


def _default_prepared(root: Path, task_id: str) -> Path:
    return root / "data" / "processed" / "bdb_suite" / task_id


def _default_prepared_variant(root: Path, task_id: str, variant: str) -> Path:
    if variant == "primary":
        return _default_prepared(root, task_id)
    return (
        root
        / "data"
        / "processed"
        / "bdb_suite"
        / "variants"
        / task_id
        / variant
    )


def _default_draft(root: Path, task_id: str) -> Path:
    return root / "configs" / "bdb_suite" / "tasks" / f"{task_id}.json"


def _inside_repo(root: Path, path: str | Path, label: str) -> Path:
    """Resolve a scientific artifact path and fail if it escapes the repo."""

    target = Path(path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be beneath repo_root: {target}") from exc
    return target


def command_inventory(arguments: argparse.Namespace) -> None:
    from .adapters import ADAPTER_MODULES, get_adapter
    from .data_import import inventory

    root = Path(arguments.repo_root).resolve()
    result = inventory(root)
    result["tasks"] = {}
    for task_id in ADAPTER_MODULES:
        raw = _default_raw(root, task_id)
        entry: dict[str, Any] = {"raw_dir": str(raw), "raw_exists": raw.is_dir()}
        if arguments.audit and raw.is_dir():
            entry["cohort_audit"] = get_adapter(task_id).audit_cohort(raw).as_dict()
        result["tasks"][task_id] = entry
    _print(result)


def command_import_data(arguments: argparse.Namespace) -> None:
    from .data_import import import_release

    releases = arguments.release or [2021, 2022, 2024]
    output = {
        str(release): import_release(release, repo_root=arguments.repo_root)
        for release in releases
    }
    _print({key: {field: value[field] for field in ("receipt_sha256", "tree_sha256", "file_count", "byte_count")} for key, value in output.items()})


def command_prepare(arguments: argparse.Namespace) -> None:
    from .adapters import get_adapter
    from .prepared import validate_prepared_task_contract, write_prepared_task
    from .contracts import (
        load_task_spec,
        prepared_contract_for_variant,
        validate_source_tree_receipts,
    )

    root = Path(arguments.repo_root).resolve()
    variant = str(getattr(arguments, "prepared_variant", "primary"))
    if variant != "primary" and not (
        arguments.task == "bdb2024_tackle"
        and variant == "common_closest_approach_minus_ten"
    ):
        raise ValueError(
            f"unsupported prepared variant {variant!r} for {arguments.task}"
        )
    raw = Path(arguments.raw_dir).resolve() if arguments.raw_dir else _default_raw(root, arguments.task)
    canonical_output = _default_prepared_variant(root, arguments.task, variant).resolve()
    output = Path(arguments.output_dir).resolve() if arguments.output_dir else canonical_output
    adapter = get_adapter(arguments.task)
    if arguments.audit_only:
        _print(adapter.audit_cohort(raw).as_dict())
        return
    output = _inside_repo(root, output, "prepared output")
    if variant != "primary" and output != canonical_output:
        raise ValueError(
            "a definitive sensitivity prepared bundle must use its canonical "
            f"variant path: {canonical_output}"
        )
    if arguments.max_examples is not None and (
        arguments.output_dir is None
        or output == _default_prepared(root, arguments.task).resolve()
    ):
        raise ValueError(
            "--max-examples is diagnostic only and requires an explicit, "
            "noncanonical --output-dir beneath repo_root"
        )
    spec = load_task_spec(_default_draft(root, arguments.task), repo_root=root)
    destinations = {
        (root / receipt.destination).resolve()
        for receipt in spec.source_tree_receipts
    }
    if len(destinations) != 1:
        raise ValueError("definitive prepare requires exactly one canonical raw source tree")
    canonical_raw = next(iter(destinations))
    if raw != canonical_raw:
        raise ValueError(
            f"definitive prepare must read the TaskSpec source tree: {canonical_raw}"
        )
    validate_source_tree_receipts(
        spec.source_tree_receipts,
        repo_root=root,
        verify_tree=True,
    )
    keyword: dict[str, Any] = {"max_examples": arguments.max_examples}
    if arguments.task == "bdb2024_tackle":
        keyword["cutoff_policy"] = (
            variant if variant != "primary" else "primary_candidate_event"
        )
        if arguments.max_examples is None and variant == "primary":
            keyword["require_fidelity_counts"] = True
    task = adapter.prepare(raw, **keyword)
    if arguments.max_examples is None:
        validate_prepared_task_contract(
            task,
            (
                spec.prepared_contract
                if variant == "primary"
                else prepared_contract_for_variant(spec, variant)
            ),
        )
    source_hashes = {
        receipt.source_id: receipt.tree_sha256 for receipt in spec.source_tree_receipts
    }
    receipt = write_prepared_task(task, output, source_receipt_hashes=source_hashes)
    _print({"output_dir": str(output), "prepared_variant": variant, "prepared_hash": receipt["prepared_hash"], "examples": receipt["examples"], "games": receipt["games"]})


def command_dev_cv(arguments: argparse.Namespace) -> None:
    from .contracts import load_task_spec
    from .design import build_task_design
    from .devcv import (
        run_development_cv,
        run_development_cv_shard,
        write_development_receipt,
    )
    from .manifest import build_development_provenance, game_records_from_prepared
    from .prepared import load_prepared_task
    from .runner import TaskRuntime, development_evaluator

    fold_filters = getattr(arguments, "fold", None)
    candidate_filters = getattr(arguments, "candidate", None)
    sharded = fold_filters is not None or candidate_filters is not None
    if sharded and arguments.family is None:
        raise ValueError(
            "development fold/candidate filters require exactly one --family"
        )
    root = Path(arguments.repo_root).resolve()
    prepared_dir = Path(arguments.prepared_dir).resolve() if arguments.prepared_dir else _default_prepared(root, arguments.task)
    prepared_dir = _inside_repo(root, prepared_dir, "development prepared input")
    draft_path = _default_draft(root, arguments.task)
    provenance = build_development_provenance(
        draft_path,
        prepared_dir,
        repo_root=root,
    )
    task = load_prepared_task(prepared_dir, mmap_mode="r", verify_files=False)
    spec = load_task_spec(draft_path, repo_root=root)
    design = build_task_design(spec, game_records_from_prepared(task))
    families = (
        [arguments.family]
        if arguments.family
        else [
            "glm",
            "lightgbm",
            "relnet",
            "attn_relnet",
            "set_transformer",
        ]
    )
    model_by_family = {entry["family"]: model_id for model_id, entry in spec.models.items()}
    output_dir = Path(arguments.output_dir).resolve() if arguments.output_dir else root / "data" / "bdb_suite_runs" / "development" / arguments.task
    output_dir = _inside_repo(root, output_dir, "development receipt output")
    def progress(record: Any) -> None:
        print(
            "development-cv "
            f"family={record['family']} fold={record['fold']} "
            f"candidate={record['candidate_index']} status={record['status']}",
            file=sys.stderr,
            flush=True,
        )

    written = {}
    for family in families:
        queue = "cpu_tabular" if family in {"glm", "lightgbm"} else "gpu_neural"
        runtime = TaskRuntime(task, queue=queue)
        if sharded:
            shard = run_development_cv_shard(
                task,
                design,
                family,
                development_evaluator(runtime),
                model_id=model_by_family[family],
                provenance=provenance,
                checkpoint_dir=output_dir / ".checkpoints" / family,
                folds=fold_filters,
                candidate_indices=candidate_filters,
                resume=arguments.resume,
                progress=progress,
            )
            written[family] = {"checkpoint_only": shard}
        else:
            receipt = run_development_cv(
                task,
                design,
                family,
                development_evaluator(runtime),
                model_id=model_by_family[family],
                provenance=provenance,
                checkpoint_dir=output_dir / ".checkpoints" / family,
                resume=arguments.resume,
                progress=progress,
            )
            path = write_development_receipt(
                receipt, output_dir / f"{family}.json"
            )
            written[family] = {
                "path": str(path),
                "selected": receipt["selected"],
            }
    _print(written)


def command_dev_cv_reduce(arguments: argparse.Namespace) -> None:
    """Reduce an exact complete checkpoint grid without fitting any model."""

    from .contracts import load_task_spec
    from .design import build_task_design
    from .devcv import (
        reduce_development_checkpoints,
        write_development_receipt,
    )
    from .manifest import build_development_provenance, game_records_from_prepared
    from .prepared import load_prepared_task

    root = Path(arguments.repo_root).resolve()
    prepared_dir = (
        Path(arguments.prepared_dir).resolve()
        if arguments.prepared_dir
        else _default_prepared(root, arguments.task)
    )
    prepared_dir = _inside_repo(
        root, prepared_dir, "development prepared input"
    )
    draft_path = _default_draft(root, arguments.task)
    provenance = build_development_provenance(
        draft_path,
        prepared_dir,
        repo_root=root,
    )
    task = load_prepared_task(prepared_dir, mmap_mode="r", verify_files=False)
    spec = load_task_spec(draft_path, repo_root=root)
    design = build_task_design(spec, game_records_from_prepared(task))
    model_by_family = {
        entry["family"]: model_id for model_id, entry in spec.models.items()
    }
    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else root
        / "data"
        / "bdb_suite_runs"
        / "development"
        / arguments.task
    )
    output_dir = _inside_repo(root, output_dir, "development receipt output")
    family = str(arguments.family)
    receipt = reduce_development_checkpoints(
        task,
        design,
        family,
        model_id=model_by_family[family],
        provenance=provenance,
        checkpoint_dir=output_dir / ".checkpoints" / family,
    )
    path = write_development_receipt(receipt, output_dir / f"{family}.json")
    _print(
        {
            "family": family,
            "path": str(path),
            "receipt_hash": receipt["receipt_hash"],
            "selected": receipt["selected"],
            "checkpoint_cells": len(receipt["fold_scores"]),
        }
    )


def command_freeze(arguments: argparse.Namespace) -> None:
    from .manifest import freeze_task_from_development

    root = Path(arguments.repo_root).resolve()
    prepared = Path(arguments.prepared_dir).resolve() if arguments.prepared_dir else _default_prepared(root, arguments.task)
    development = Path(arguments.development_dir).resolve() if arguments.development_dir else root / "data" / "bdb_suite_runs" / "development" / arguments.task
    output = Path(arguments.output).resolve() if arguments.output else root / "configs" / "bdb_suite" / "frozen" / f"{arguments.task}.json"
    prepared = _inside_repo(root, prepared, "freeze prepared input")
    development = _inside_repo(root, development, "freeze development input")
    output = _inside_repo(root, output, "frozen TaskSpec output")
    receipt = freeze_task_from_development(
        _default_draft(root, arguments.task),
        prepared,
        development,
        output,
        repo_root=root,
        verify_full_source_trees=arguments.verify_full_source_trees,
    )
    _print({"output": str(output), "task_spec_hash": receipt["task_spec_hash"]})


def command_plan(arguments: argparse.Namespace) -> None:
    from .contracts import load_task_spec
    from .manifest import build_run_manifest
    from .storage import initialize_run_dir

    root = Path(arguments.repo_root).resolve()
    frozen = Path(arguments.frozen_task).resolve() if arguments.frozen_task else root / "configs" / "bdb_suite" / "frozen" / f"{arguments.task}.json"
    frozen = _inside_repo(root, frozen, "plan frozen TaskSpec input")
    if arguments.prepared_dir:
        prepared = Path(arguments.prepared_dir).resolve()
    elif arguments.profile == "sensitivity20":
        sensitivity_spec = load_task_spec(
            frozen, repo_root=root, require_source=False
        )
        frozen_sensitivities = [
            item
            for item in sensitivity_spec.sensitivities
            if item["selection"] == "frozen_prespecified"
        ]
        if len(frozen_sensitivities) != 1:
            raise ValueError(
                "sensitivity20 requires exactly one frozen sensitivity contract"
            )
        prepared = _default_prepared_variant(
            root,
            arguments.task,
            str(frozen_sensitivities[0]["execution"]["prepared_variant"]),
        )
    else:
        prepared = _default_prepared(root, arguments.task)
    prepared = _inside_repo(root, prepared, "plan prepared input")
    preflight_receipt = arguments.preflight_receipt
    if preflight_receipt is not None:
        preflight_path = Path(preflight_receipt)
        if not preflight_path.is_absolute():
            preflight_path = root / preflight_path
        preflight_receipt = str(
            _inside_repo(root, preflight_path, "plan preflight input")
        )
    runtime_plan = arguments.runtime_plan
    if runtime_plan is not None:
        runtime_path = Path(runtime_plan)
        if not runtime_path.is_absolute():
            runtime_path = root / runtime_path
        runtime_plan = str(_inside_repo(root, runtime_path, "plan runtime input"))
    primary_reference_run = getattr(arguments, "primary_reference_run", None)
    if primary_reference_run is not None:
        reference_path = Path(primary_reference_run)
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        primary_reference_run = str(
            _inside_repo(root, reference_path, "plan primary-reference input")
        )
    manifest = build_run_manifest(
        frozen,
        prepared,
        repo_root=root,
        repeats=(1 if arguments.smoke or arguments.benchmark else arguments.repeats),
        profile=arguments.profile,
        cpu_workers=arguments.cpu_workers,
        smoke=arguments.smoke,
        benchmark=arguments.benchmark,
        preflight_receipt=preflight_receipt,
        runtime_plan=runtime_plan,
        primary_reference_run=primary_reference_run,
    )
    storage = initialize_run_dir(arguments.run_dir, manifest, repo_root=root)
    _print({"run_dir": str(storage.run_dir), "manifest_hash": storage.manifest_hash, "run_hash": manifest["run_hash"], "cells": len(manifest["required_cells"])})


def command_runtime_plan(arguments: argparse.Namespace) -> None:
    from .preflight import build_runtime_plan

    root = Path(arguments.repo_root).resolve()
    output = _inside_repo(root, arguments.output, "runtime plan output")
    receipt = build_runtime_plan(
        arguments.benchmark_run,
        output,
        repo_root=root,
        safety_factor=arguments.safety_factor,
    )
    _print(
        {
            "output": str(output),
            "runtime_plan_hash": receipt["runtime_plan_hash"],
            "gpu_lanes": receipt["gpu_lanes"],
            "job_guard_seconds": receipt["job_guard_seconds"],
            "queue_plans": receipt["queue_plans"],
        }
    )


def command_runtime_plan_phased(arguments: argparse.Namespace) -> None:
    from .preflight import build_runtime_plan_v3

    root = Path(arguments.repo_root).resolve()
    output = _inside_repo(root, arguments.output, "phased runtime plan output")
    receipt = build_runtime_plan_v3(
        arguments.benchmark_run,
        output,
        repo_root=root,
        safety_factor=arguments.safety_factor,
    )
    _print(
        {
            "output": str(output),
            "runtime_plan_hash": receipt["runtime_plan_hash"],
            "gpu_lanes": receipt["gpu_lanes"],
            "cpu_lanes": receipt["cpu_lanes"],
            "job_guard_seconds": receipt["job_guard_seconds"],
            "queue_plans": receipt["queue_plans"],
        }
    )


def command_runtime_plan_phased_v4(arguments: argparse.Namespace) -> None:
    from .preflight import build_runtime_plan_v4

    root = Path(arguments.repo_root).resolve()
    output = _inside_repo(root, arguments.output, "v4 runtime plan output")
    binding_path = _inside_repo(
        root, arguments.determinism_probe_binding, "determinism probe binding"
    )
    try:
        value = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("determinism probe binding is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("determinism probe binding must be an object")
    probe = value.get("probe") if isinstance(value.get("probe"), dict) else value
    receipt = build_runtime_plan_v4(
        arguments.benchmark_run,
        output,
        determinism_probe=probe,
        repo_root=root,
        safety_factor=arguments.safety_factor,
    )
    _print(
        {
            "output": str(output),
            "runtime_plan_hash": receipt["runtime_plan_hash"],
            "gpu_lanes": receipt["gpu_lanes"],
            "cpu_lanes": receipt["cpu_lanes"],
            "resource_contract": receipt["resource_contract"],
            "job_guard_seconds": receipt["job_guard_seconds"],
            "queue_plans": receipt["queue_plans"],
        }
    )


def command_verify_determinism(arguments: argparse.Namespace) -> None:
    from .preflight import build_preflight_receipt

    root = Path(arguments.repo_root).resolve()
    output = _inside_repo(root, arguments.output, "preflight receipt output")
    receipt = build_preflight_receipt(
        arguments.run_a,
        arguments.run_b,
        output,
        repo_root=root,
    )
    _print(
        {
            "output": str(output),
            "receipt_hash": receipt["receipt_hash"],
            "determinism": receipt["determinism"],
            "recommended_cpu_workers": receipt["memory"]["recommended_cpu_workers"],
        }
    )


def command_synthetic_smoke(arguments: argparse.Namespace) -> None:
    from .synthetic import run_synthetic_study

    receipt = run_synthetic_study(
        arguments.output_dir, repo_root=arguments.repo_root
    )
    _print(
        {
            "output_dir": str(Path(arguments.output_dir).resolve()),
            "cells": receipt["cells"],
            "summary_groups": receipt["summary_groups"],
            "status": "complete",
        }
    )


def command_run(arguments: argparse.Namespace) -> None:
    from .execution import run_missing_cells

    _print(
        run_missing_cells(
            arguments.run_dir,
            repo_root=arguments.repo_root,
            queues=arguments.queue,
            repeats=arguments.repeat,
            anchors=arguments.anchor,
            models=arguments.model,
            ablations=arguments.ablation,
            sensitivities=arguments.sensitivity,
            neural_phase=arguments.neural_phase,
        )
    )


def command_status(arguments: argparse.Namespace) -> None:
    from .execution import queue_status

    _print(
        queue_status(
            arguments.run_dir,
            repo_root=arguments.repo_root,
            queues=arguments.queue,
            repeats=arguments.repeat,
            anchors=arguments.anchor,
            models=arguments.model,
            ablations=arguments.ablation,
            sensitivities=arguments.sensitivity,
        )
    )


def command_worker(arguments: argparse.Namespace) -> None:
    from .execution import run_repeat_queue

    print(
        json.dumps(
            run_repeat_queue(
                arguments.run_dir,
                repeat=arguments.repeat,
                queue=arguments.queue,
                repo_root=arguments.repo_root,
                anchors=arguments.anchor,
                models=arguments.model,
                ablations=arguments.ablation,
                sensitivities=arguments.sensitivity,
                neural_phase=arguments.neural_phase,
            ),
            sort_keys=True,
        )
    )


def command_aggregate_task(arguments: argparse.Namespace) -> None:
    from .aggregate import aggregate_task_run

    _print(aggregate_task_run(arguments.run_dir, repo_root=arguments.repo_root, require_complete=arguments.require_complete))


def _validated_v2_suite_component(
    run_dir: str | Path,
    task_id: str,
    profile: str,
    *,
    repo_root: str | Path,
) -> tuple[Any, dict[str, Any], Path]:
    """Validate and bind one finalized protocol-v2 task or sensitivity run."""

    from .contracts import sha256_file
    from .execution import open_run
    from .storage import cell_keys_from_design, validate_checksum

    storage = open_run(
        Path(run_dir).resolve(),
        repo_root=repo_root,
        verify_tensorflow_runtime=False,
    )
    manifest = storage.manifest
    design = manifest.get("task_design", {})
    expected_mode = "pilot" if profile == "pilot10" else "definitive"
    if (
        manifest.get("task_id") != task_id
        or manifest.get("execution", {}).get("mode") != expected_mode
        or manifest.get("execution", {}).get("profile") != profile
        or design.get("profile") != profile
    ):
        raise ValueError(
            f"suite component {task_id} is not an admitted {profile} run"
        )
    expected_counts_by_profile = {
        "pilot10": {
            "primary": 300,
            "structural_ablation": 0,
            "frozen_sensitivity": 0,
            "required": 300,
        },
        "full100": {
            "primary": 3000,
            "structural_ablation": 80,
            "frozen_sensitivity": 0,
            "required": 3080,
        },
        "sensitivity20": {
            "primary": 0,
            "structural_ablation": 0,
            "frozen_sensitivity": 200,
            "required": 200,
        },
    }
    try:
        expected_counts = expected_counts_by_profile[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported suite component profile: {profile}") from exc
    if design.get("cell_counts") != expected_counts:
        raise ValueError(f"suite component {task_id} has the wrong {profile} grid")
    required_cells = cell_keys_from_design(design)
    if len(required_cells) != expected_counts["required"] or not storage.validate_final(
        required_cells
    ):
        raise ValueError(f"suite component {task_id} is not completely finalized")
    receipt_path = storage.run_dir / "final" / "receipt.json"
    metrics_name = (
        "primary_metrics.csv"
        if profile in {"pilot10", "full100"}
        else "frozen_sensitivity_metrics.csv"
    )
    metrics_path = storage.run_dir / "final" / metrics_name
    if not validate_checksum(receipt_path) or not validate_checksum(metrics_path):
        raise ValueError(f"suite component checksums failed: {task_id}/{profile}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"suite component receipt is unreadable: {task_id}") from exc
    expected_receipt_counts_by_profile = {
        "pilot10": {
            "cells": 300,
            "primary_cells": 300,
            "ablation_cells": 0,
            "sensitivity_cells": 0,
            "primary_summary_groups": 30,
        },
        "full100": {
            "cells": 3080,
            "primary_cells": 3000,
            "ablation_cells": 80,
            "sensitivity_cells": 0,
            "primary_summary_groups": 30,
        },
        "sensitivity20": {
            "cells": 200,
            "primary_cells": 0,
            "ablation_cells": 0,
            "sensitivity_cells": 200,
            "primary_summary_groups": 0,
        },
    }
    expected_receipt_counts = expected_receipt_counts_by_profile[profile]
    if (
        receipt.get("task_id") != task_id
        or receipt.get("manifest_hash") != storage.manifest_hash
        or any(receipt.get(key) != value for key, value in expected_receipt_counts.items())
        or receipt.get("scientific_revalidation")
        != "ordered_identities_splits_seeds_config_history_predictions_all_metrics"
    ):
        raise ValueError(f"suite component aggregate receipt is incomplete: {task_id}")
    if profile == "pilot10":
        forbidden = {
            "paired_contrasts",
            "supported_best",
            "bca_sd_intervals",
            "bca_log_sd_ratios",
            "interval_width_contrasts",
            "coverage_lower_bounds",
            "controlled_sharpness",
        }
        if (
            manifest.get("execution", {}).get("evidence_status")
            != "exploratory_provisional"
            or receipt.get("evidence_status") != "exploratory_provisional"
            or receipt.get("analysis_scope")
            != "pilot10_descriptive_only_no_confirmatory_inference"
            or receipt.get("bootstrap_draws") != 0
            or receipt.get("max_t_family") != "none"
            or forbidden & set(receipt.get("artifacts", {}))
        ):
            raise ValueError(
                f"suite pilot receipt is not descriptive/provisional: {task_id}"
            )
    if profile == "sensitivity20":
        required_secondary = {
            "frozen_sensitivity_summary",
            "frozen_sensitivity_effects",
            "frozen_sensitivity_pairwise",
        }
        artifacts = receipt.get("artifacts", {})
        if (
            receipt.get("primary_reference") != manifest.get("primary_reference")
            or not required_secondary <= set(artifacts)
            or any(int(receipt.get(field, 0)) <= 0 for field in (
                "sensitivity_summary_groups",
                "sensitivity_effect_groups",
                "sensitivity_pairwise_groups",
            ))
        ):
            raise ValueError(
                f"suite sensitivity receipt lacks its paired comparison: {task_id}"
            )
        for name in required_secondary:
            artifact_path = storage.run_dir / "final" / str(artifacts[name]["file"])
            if not validate_checksum(artifact_path, artifacts[name]["sha256"]):
                raise ValueError(
                    f"suite sensitivity comparison checksum failed: {task_id}/{name}"
                )
    binding = {
        "task_id": task_id,
        "run_dir": str(storage.run_dir.resolve()),
        "manifest_hash": storage.manifest_hash,
        "manifest_file_sha256": sha256_file(storage.run_dir / "manifest.json"),
        "final_marker_sha256": sha256_file(storage.run_dir / "_SUCCESS"),
        "final_receipt_sha256": sha256_file(receipt_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    if profile == "sensitivity20":
        binding["primary_reference"] = dict(manifest["primary_reference"])
    return storage, binding, metrics_path


def _validated_bdb2020_reference(
    run_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> tuple[dict[str, Any], Any]:
    """Validate the completed BDB2020 panel without scheduling a new fit."""

    from .contracts import sha256_file, sha256_json
    from rushing_study.execution import expected_cell_keys
    from rushing_study.data import load_study_data
    from rushing_study.metrics import crps_contributions
    from rushing_study.storage import CellKey, RunStorage
    import numpy as np
    import pandas as pd

    storage = RunStorage.open(Path(run_dir).resolve())
    manifest = storage.manifest
    config = manifest.get("config", {})
    if (
        manifest.get("study_id") != "rushing_confirmatory_v1"
        or int(config.get("execution", {}).get("confirmatory_repeats", -1)) != 100
        or manifest.get("manifest_hash") != storage.manifest_hash
    ):
        raise ValueError("BDB2020 reference is not the completed full100 campaign")
    main_keys = expected_cell_keys(manifest, "main")
    if len(main_keys) != 2400:
        raise ValueError("BDB2020 reference does not contain 2,400 primary cells")
    success_path = storage.run_dir / "_SUCCESS"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("BDB2020 reference has no readable final marker") from exc
    recorded: dict[CellKey, str] = {}
    try:
        for entry in success["cells"]:
            key = CellKey(**entry["key"])
            recorded[key] = str(entry["marker_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BDB2020 final marker cell index is malformed") from exc
    if not set(main_keys) <= set(recorded):
        raise ValueError("BDB2020 final marker omits completed primary cells")
    for key in main_keys:
        marker = storage.cell_dir(key) / "_SUCCESS"
        if not marker.is_file() or sha256_file(marker) != recorded[key]:
            raise ValueError(f"BDB2020 committed cell drifted: {key.relative_dir}")
    metrics_path = storage.run_dir / "final" / "metrics.csv"
    metrics_sha256 = sha256_file(metrics_path)
    indexed = {
        str(item.get("file")): str(item.get("sha256"))
        for item in success.get("final_artifacts", [])
        if isinstance(item, dict)
    }
    if indexed.get("final/metrics.csv") != metrics_sha256:
        raise ValueError("BDB2020 primary metrics are not indexed by finalization")
    frame = pd.read_csv(metrics_path)
    required_columns = {"branch", "repeat", "n_train", "model", "crps"}
    if required_columns - set(frame) or len(frame) != 2400:
        raise ValueError("BDB2020 primary metrics have the wrong schema or row count")
    expected_models = {
        "ridge_sgd_l2", "lightgbm_multiclass", "zoo_cnn", "set_transformer"
    }
    expected_anchors = {20, 40, 80, 160, 240, 360}
    identities = set(
        zip(
            frame["repeat"].astype(int),
            frame["n_train"].astype(int),
            frame["model"].astype(str),
        )
    )
    expected_identities = {
        (repeat, anchor, model)
        for repeat in range(1, 101)
        for anchor in expected_anchors
        for model in expected_models
    }
    if (
        set(frame["branch"].astype(str)) != {"main"}
        or identities != expected_identities
    ):
        raise ValueError("BDB2020 primary metrics are not the exact full100 grid")
    reference = frame.loc[frame["n_train"].astype(int).isin({20, 40})].copy()
    reference["task_id"] = "bdb2020_rushing"
    root = Path(repo_root).resolve()
    data_provenance = manifest.get("provenance", {}).get("data", {}).get("files", {})
    for name, record in data_provenance.items():
        path = (root / str(record.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"BDB2020 data pin escapes repo_root: {name}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"BDB2020 data provenance drifted: {name}")
    data_config = json.loads(json.dumps(config))
    configured_data = data_config["data"]
    for key in ("processed_dir", "raw_train_csv", "train_x", "train_x_set", "train_y"):
        if key in configured_data:
            configured_data[key] = str((root / configured_data[key]).resolve())
    data = load_study_data(data_config, mmap_mode="r")
    split_lookup = {
        int(split["repeat_id"]): split for split in manifest.get("split_manifests", [])
    }
    if set(split_lookup) != set(range(1, 101)):
        raise ValueError("BDB2020 reference lacks its 100 frozen split manifests")
    null_lookup: dict[tuple[int, int], tuple[float, float]] = {}
    game_ids = data.metadata["game_id"].astype(str)
    for repeat in range(1, 101):
        split = split_lookup[repeat]
        for anchor in (20, 40):
            train_games = {
                str(value)
                for value in split["anchors"][str(anchor)]["train_games"]
            }
            test_games = {str(value) for value in split["test_games"]}
            train_index = np.flatnonzero(game_ids.isin(train_games).to_numpy())
            test_index = np.flatnonzero(game_ids.isin(test_games).to_numpy())
            if (
                set(game_ids.iloc[train_index]) != train_games
                or set(game_ids.iloc[test_index]) != test_games
            ):
                raise ValueError("BDB2020 pinned split references unavailable games")
            train_y = np.asarray(data.y[train_index], dtype=np.int64)
            test_y = np.asarray(data.y[test_index], dtype=np.int64)
            counts = np.bincount(train_y, minlength=80).astype(np.float64)
            probability = counts / counts.sum()
            contributions = crps_contributions(
                test_y, np.repeat(probability[None, :], len(test_y), axis=0)
            )
            by_game = pd.DataFrame(
                {
                    "game_id": game_ids.iloc[test_index].to_numpy(),
                    "contribution": contributions,
                }
            ).groupby("game_id", sort=False)["contribution"].mean()
            null_lookup[(repeat, anchor)] = (
                float(np.mean(contributions)), float(by_game.mean())
            )
    reference["null_loss"] = [
        null_lookup[(int(repeat), int(anchor))][0]
        for repeat, anchor in zip(reference["repeat"], reference["n_train"])
    ]
    reference["null_game_equal_loss"] = [
        null_lookup[(int(repeat), int(anchor))][1]
        for repeat, anchor in zip(reference["repeat"], reference["n_train"])
    ]
    reference["primary_loss"] = reference["crps"].astype(float)
    reference["game_equal_loss"] = reference["crps_game_equal"].astype(float)
    reference["skill"] = 1.0 - reference["primary_loss"] / reference["null_loss"]
    # Preserve the historical architecture identity.  The synthesis role is a
    # descriptive alignment used only to put the discovery task beside the
    # prospective roles. The Zoo CNN remains a separate legacy row; BDB2020
    # has no RelNet or AttnRelNet result and neither missing role is imputed.
    synthesis_role = {
        "ridge_sgd_l2": "glm",
        "lightgbm_multiclass": "lightgbm",
        "zoo_cnn": "legacy_zoo_cnn",
        "set_transformer": "set_transformer",
    }
    reference["original_model"] = reference["model"].astype(str)
    reference["synthesis_role"] = reference["original_model"].map(synthesis_role)
    if reference["synthesis_role"].isna().any():
        raise ValueError("BDB2020 reference contains an unmapped legacy model role")
    reference["synthesis_role_interpretation"] = (
        "descriptive_role_alignment_only_not_architecture_identity"
    )
    # ``descriptive_suite_summary`` consumes its historical ``family`` field;
    # keep that mechanical alias alongside, never in place of, original_model.
    reference["family"] = reference["synthesis_role"]
    reference = reference.sort_values(
        ["repeat", "n_train", "model"], kind="stable"
    ).reset_index(drop=True)
    reference_rows_sha256 = sha256_json(
        reference[
            [
                "repeat", "n_train", "model", "original_model",
                "synthesis_role", "synthesis_role_interpretation", "family",
                "primary_loss", "game_equal_loss", "null_loss", "skill",
            ]
        ].to_dict(orient="records")
    )
    binding = {
        "schema_version": "bdb2020-readonly-reference-v1",
        "run_dir": str(storage.run_dir.resolve()),
        "manifest_hash": storage.manifest_hash,
        "manifest_file_sha256": sha256_file(storage.run_dir / "manifest.json"),
        "final_marker_sha256": sha256_file(success_path),
        "primary_metrics_sha256": metrics_sha256,
        "primary_cells": 2400,
        "reference_cells": 800,
        "reference_anchors": [20, 40],
        "repeats": 100,
        "models": sorted(expected_models),
        "synthesis_role_mapping": synthesis_role,
        "synthesis_role_caveat": (
            "descriptive alignment only; BDB2020 has no RelNet or AttnRelNet; "
            "the Zoo CNN remains an architecture-specific legacy row"
        ),
        "null_model": "train_only_empirical_pmf_on_each_frozen_nested_subset",
        "reference_rows_sha256": reference_rows_sha256,
    }
    return binding, reference


def _aggregate_pilot_suite(
    suite: dict[str, Any],
    manifest_path: Path,
    arguments: argparse.Namespace,
) -> None:
    """Validate and reduce six pilot10 tasks into a provisional mockup."""

    import hashlib
    from .analysis import descriptive_suite_summary, validate_task_grid
    from .storage import atomic_write_csv, atomic_write_json
    import pandas as pd

    expected_fields = {
        "schema_version",
        "profile",
        "evidence_status",
        "task_runs",
        "suite_hash",
    }
    if set(suite) != expected_fields:
        raise ValueError("pilot suite manifest fields differ from its frozen schema")
    if (
        suite.get("schema_version") != "bdb-pilot-suite-manifest-v1"
        or suite.get("profile") != "pilot10"
        or suite.get("evidence_status") != "exploratory_provisional"
    ):
        raise ValueError("unsupported or mislabeled pilot suite manifest")
    unsigned = {key: value for key, value in suite.items() if key != "suite_hash"}
    calculated_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if suite.get("suite_hash") != calculated_hash:
        raise ValueError("pilot suite manifest hash is invalid")
    required = {
        "bdb2021_completion",
        "bdb2022_punt_returns",
        "bdb2023_sack",
        "bdb2024_tackle",
        "bdb2025_man_zone",
        "bdb2026_trajectory",
    }
    task_runs = suite.get("task_runs", {})
    if set(task_runs) != required:
        raise ValueError(f"pilot suite must contain exactly {sorted(required)}")

    def resolve(value: str) -> Path:
        path = Path(value)
        return (
            path.resolve()
            if path.is_absolute()
            else (manifest_path.parent / path).resolve()
        )

    component_fields = {
        "task_id",
        "run_dir",
        "manifest_hash",
        "manifest_file_sha256",
        "final_marker_sha256",
        "final_receipt_sha256",
        "metrics_sha256",
    }
    frames: dict[str, pd.DataFrame] = {}
    run_manifests: dict[str, Any] = {}
    pinned_components: dict[str, Any] = {}
    for task_id, entry in sorted(task_runs.items()):
        if (
            not isinstance(entry, dict)
            or set(entry) != component_fields
            or entry.get("task_id") != task_id
        ):
            raise ValueError(f"pilot suite component pin is invalid: {task_id}")
        storage, observed, metrics_path = _validated_v2_suite_component(
            resolve(str(entry["run_dir"])),
            task_id,
            "pilot10",
            repo_root=arguments.repo_root,
        )
        observed["run_dir"] = entry["run_dir"]
        if observed != entry:
            raise ValueError(f"pilot suite component pin drifted: {task_id}")
        frame = validate_task_grid(
            pd.read_csv(metrics_path),
            task_id=task_id,
            anchors=storage.manifest["task_design"]["anchors"],
            repeats=10,
            models=storage.manifest["task_spec"]["models"].keys(),
        )
        if len(frame) != 300:
            raise ValueError(f"pilot task {task_id} lacks its exact 300-cell grid")
        frame["evidence_status"] = "exploratory_provisional"
        frame["profile"] = "pilot10"
        frames[task_id] = frame
        run_manifests[task_id] = storage.manifest
        pinned_components[task_id] = entry

    if (
        run_manifests["bdb2024_tackle"]["task_design"]["game_registry"]
        != run_manifests["bdb2025_man_zone"]["task_design"]["game_registry"]
    ):
        raise ValueError(
            "pilot BDB2024 and BDB2025 do not share the identical frozen registry"
        )
    summary = descriptive_suite_summary(
        frames,
        require_complete_task_ids=tuple(sorted(required)),
    )
    if len(summary) != 20 or set(summary["tasks"].astype(int)) != {6}:
        raise ValueError("pilot suite summary lacks its exact 5x4 task-equal grid")
    summary["evidence_status"] = "exploratory_provisional"
    summary["profile"] = "pilot10"
    long_metrics = pd.concat(
        [frames[task_id] for task_id in sorted(frames)], ignore_index=True
    ).sort_values(["task_id", "repeat", "n_train", "model"]).reset_index(drop=True)
    if len(long_metrics) != 1_800:
        raise ValueError("pilot combined metrics must contain exactly 1,800 cells")

    output = (
        Path(arguments.output).resolve()
        if arguments.output
        else manifest_path.parent / "pilot10_suite_summary.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_record = atomic_write_csv(output, summary)
    metrics_output = output.with_name(f"{output.stem}_task_metrics.csv")
    metrics_record = atomic_write_csv(metrics_output, long_metrics)
    receipt = {
        "schema_version": "bdb-pilot-suite-aggregate-v1",
        "profile": "pilot10",
        "evidence_status": "exploratory_provisional",
        "suite_manifest": str(manifest_path),
        "suite_hash": calculated_hash,
        "tasks": sorted(required),
        "task_components": pinned_components,
        "task_count": 6,
        "primary_cells_per_task": 300,
        "combined_primary_cells": 1_800,
        "repeat_ids": list(range(1, 11)),
        "common_anchors": [10, 20, 40, 60],
        "analysis_scope": "provisional_task_equal_descriptive_mockup_only",
        "inference": "none_no_cross_task_ci_no_confirmatory_claims",
        "model_winner_claim": "prohibited_for_pilot_suite",
        "overlap_disclosure": (
            "NFL seasons overlap across releases; BDB2024 and BDB2025 use the same 2022 games"
        ),
        "summary": summary_record.as_dict(),
        "task_metrics": metrics_record.as_dict(),
    }
    receipt_record = atomic_write_json(output.with_suffix(".receipt.json"), receipt)
    _print(
        {
            "output": str(output),
            "task_metrics": str(metrics_output),
            "receipt": str(output.parent / receipt_record.file),
            "rows": len(summary),
            "evidence_status": "exploratory_provisional",
        }
    )


def command_aggregate_suite(arguments: argparse.Namespace) -> None:
    import hashlib
    from .analysis import descriptive_suite_summary, validate_task_grid
    from .storage import atomic_write_csv, atomic_write_json
    import pandas as pd

    manifest_path = Path(arguments.suite_manifest).resolve()
    suite = json.loads(manifest_path.read_text(encoding="utf-8"))
    if suite.get("schema_version") == "bdb-pilot-suite-manifest-v1":
        _aggregate_pilot_suite(suite, manifest_path, arguments)
        return
    required_fields = {
        "schema_version", "task_runs", "sensitivity_runs",
        "bdb2020_reference", "suite_hash",
    }
    if set(suite) != required_fields:
        raise ValueError("suite manifest fields differ from bdb-suite-manifest-v3")
    if suite.get("schema_version") != "bdb-suite-manifest-v3":
        raise ValueError("unsupported suite manifest version")
    unsigned_suite = {key: value for key, value in suite.items() if key != "suite_hash"}
    calculated_suite_hash = hashlib.sha256(
        json.dumps(unsigned_suite, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if suite.get("suite_hash") != calculated_suite_hash:
        raise ValueError("suite manifest hash is invalid")
    required = {
        "bdb2021_completion", "bdb2022_punt_returns", "bdb2023_sack",
        "bdb2024_tackle", "bdb2025_man_zone", "bdb2026_trajectory",
    }
    required_sensitivities = {"bdb2024_tackle", "bdb2025_man_zone"}
    task_runs = suite.get("task_runs", {})
    if set(task_runs) != required:
        raise ValueError(f"suite manifest must contain exactly {sorted(required)}")
    sensitivity_runs = suite.get("sensitivity_runs", {})
    if set(sensitivity_runs) != required_sensitivities:
        raise ValueError(
            "suite manifest must pin the BDB2024 and BDB2025 sensitivities"
        )

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()

    frames: dict[str, Any] = {}
    run_manifests: dict[str, Any] = {}
    component_fields = {
        "task_id", "run_dir", "manifest_hash", "manifest_file_sha256",
        "final_marker_sha256", "final_receipt_sha256", "metrics_sha256",
    }
    for task_id, entry in sorted(task_runs.items()):
        if not isinstance(entry, dict) or set(entry) != component_fields:
            raise ValueError(f"suite component pin is invalid: {task_id}")
        if entry.get("task_id") != task_id:
            raise ValueError(f"suite task key/identity mismatch: {task_id}")
        storage, observed, path = _validated_v2_suite_component(
            resolve(entry["run_dir"]), task_id, "full100",
            repo_root=arguments.repo_root,
        )
        observed["run_dir"] = entry["run_dir"]
        if observed != entry:
            raise ValueError(f"suite component pin drifted: {task_id}")
        frame = validate_task_grid(
            pd.read_csv(path),
            task_id=task_id,
            anchors=storage.manifest["task_design"]["anchors"],
            repeats=100,
            models=storage.manifest["task_spec"]["models"].keys(),
        )
        if len(frame) != 3000:
            raise ValueError(f"task {task_id} does not contain the 3,000-cell primary grid")
        frames[task_id] = frame
        run_manifests[task_id] = storage.manifest

    sensitivity_fields = component_fields | {"primary_reference"}
    for task_id, entry in sorted(sensitivity_runs.items()):
        if (
            not isinstance(entry, dict)
            or set(entry) != sensitivity_fields
            or entry.get("task_id") != task_id
        ):
            raise ValueError(f"suite sensitivity pin is invalid: {task_id}")
        _, observed, _ = _validated_v2_suite_component(
            resolve(entry["run_dir"]), task_id, "sensitivity20",
            repo_root=arguments.repo_root,
        )
        observed["run_dir"] = entry["run_dir"]
        if observed != entry:
            raise ValueError(f"suite sensitivity pin drifted: {task_id}")
        if entry["primary_reference"].get("manifest_hash") != task_runs[task_id].get(
            "manifest_hash"
        ):
            raise ValueError(
                f"suite sensitivity is paired to a different primary run: {task_id}"
            )

    tackle_registry = run_manifests["bdb2024_tackle"]["task_design"]["game_registry"]
    coverage_registry = run_manifests["bdb2025_man_zone"]["task_design"]["game_registry"]
    if tackle_registry != coverage_registry:
        raise ValueError(
            "BDB2024 and BDB2025 do not use the identical frozen 2022 game registry"
        )
    bdb2020_entry = suite.get("bdb2020_reference")
    if not isinstance(bdb2020_entry, dict):
        raise ValueError("suite manifest lacks its read-only BDB2020 reference")
    observed_bdb2020, bdb2020_frame = _validated_bdb2020_reference(
        resolve(str(bdb2020_entry.get("run_dir", ""))),
        repo_root=arguments.repo_root,
    )
    observed_bdb2020["run_dir"] = bdb2020_entry.get("run_dir")
    if observed_bdb2020 != bdb2020_entry:
        raise ValueError("BDB2020 read-only reference pin drifted")
    frames["bdb2020_rushing"] = bdb2020_frame

    result = descriptive_suite_summary(
        frames,
        require_complete_task_ids=tuple(sorted({*required, "bdb2020_rushing"})),
        partial_task_anchors={"bdb2020_rushing": (20, 40)},
        partial_task_families={
            "bdb2020_rushing": (
                "glm", "lightgbm", "set_transformer", "legacy_zoo_cnn"
            )
        },
    )
    output = Path(arguments.output).resolve() if arguments.output else manifest_path.parent / "suite_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_record = atomic_write_csv(output, result)
    bdb2020_output = output.with_name(f"{output.stem}_bdb2020_reference.csv")
    bdb2020_record = atomic_write_csv(bdb2020_output, bdb2020_frame)
    receipt = {
        "schema_version": "bdb-suite-aggregate-v2",
        "suite_manifest": str(manifest_path),
        "suite_hash": calculated_suite_hash,
        "tasks": sorted(frames),
        "sensitivity_tasks": sorted(sensitivity_runs),
        "repeat_ids": list(range(1, 101)),
        "common_anchors": [10, 20, 40, 60],
        "bdb2020_reference_anchors": [20, 40],
        "inference": "descriptive_task_equal_fixed_dependent_tasks_no_cross_task_ci",
        "overlap_disclosure": (
            "NFL seasons overlap across releases; BDB2024 and BDB2025 use the same 2022 games"
        ),
        "summary": summary_record.as_dict(),
        "bdb2020_reference": bdb2020_record.as_dict(),
        "bdb2020_pooling": (
            "included_at_exact_20_40_anchors_only; six_later_tasks_at_10_20_40_60"
        ),
        "bdb2020_synthesis_role_caveat": bdb2020_entry.get(
            "synthesis_role_caveat"
        ),
    }
    receipt_record = atomic_write_json(output.with_suffix(".receipt.json"), receipt)
    _print(
        {
            "output": str(output),
            "receipt": str(output.parent / receipt_record.file),
            "rows": len(result),
            "bdb2020_reference": str(bdb2020_output),
            "inference": receipt["inference"],
        }
    )


def command_suite_plan(arguments: argparse.Namespace) -> None:
    import hashlib
    import os
    from .storage import immutable_write_json

    required = {
        "bdb2021_completion", "bdb2022_punt_returns", "bdb2023_sack",
        "bdb2024_tackle", "bdb2025_man_zone", "bdb2026_trajectory",
    }
    profile = str(getattr(arguments, "profile", "full100"))
    if profile not in {"pilot10", "full100"}:
        raise ValueError("suite plan profile must be pilot10 or full100")
    task_run_arguments: dict[str, str] = {}
    for item in arguments.task_run:
        if "=" not in item:
            raise ValueError("--task-run must use TASK_ID=RUN_DIR")
        task_id, run_dir = item.split("=", 1)
        if task_id in task_run_arguments or task_id not in required or not run_dir:
            raise ValueError(f"invalid or duplicate --task-run entry: {item}")
        task_run_arguments[task_id] = run_dir
    if set(task_run_arguments) != required:
        raise ValueError(f"suite plan requires exactly {sorted(required)}")
    required_sensitivities = {"bdb2024_tackle", "bdb2025_man_zone"}
    sensitivity_arguments: dict[str, str] = {}
    for item in (getattr(arguments, "sensitivity_run", None) or []):
        if "=" not in item:
            raise ValueError("--sensitivity-run must use TASK_ID=RUN_DIR")
        task_id, run_dir = item.split("=", 1)
        if (
            task_id in sensitivity_arguments
            or task_id not in required_sensitivities
            or not run_dir
        ):
            raise ValueError(f"invalid or duplicate --sensitivity-run entry: {item}")
        sensitivity_arguments[task_id] = run_dir
    if profile == "pilot10" and sensitivity_arguments:
        raise ValueError("pilot10 suite plan does not accept sensitivity runs")
    if profile == "full100" and set(sensitivity_arguments) != required_sensitivities:
        raise ValueError(
            "suite plan requires finalized BDB2024 and BDB2025 sensitivity20 runs"
        )
    if profile == "full100" and not getattr(arguments, "bdb2020_run_dir", None):
        raise ValueError("full100 suite plan requires --bdb2020-run-dir")

    root = Path(arguments.repo_root).resolve()
    target = Path(arguments.output).resolve()

    def resolve_input(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    def portable(path: Path) -> str:
        return Path(os.path.relpath(path, start=target.parent)).as_posix()

    task_runs: dict[str, dict[str, Any]] = {}
    for task_id in sorted(task_run_arguments):
        _, binding, _ = _validated_v2_suite_component(
            resolve_input(task_run_arguments[task_id]),
            task_id,
            profile,
            repo_root=root,
        )
        binding["run_dir"] = portable(Path(binding["run_dir"]))
        task_runs[task_id] = binding

    if profile == "pilot10":
        if getattr(arguments, "bdb2020_run_dir", None):
            raise ValueError("pilot10 suite plan does not accept a BDB2020 reference")
        payload = {
            "schema_version": "bdb-pilot-suite-manifest-v1",
            "profile": "pilot10",
            "evidence_status": "exploratory_provisional",
            "task_runs": task_runs,
        }
        payload["suite_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        target = immutable_write_json(target, payload)
        _print(
            {
                "output": str(target.resolve()),
                "suite_hash": payload["suite_hash"],
                "profile": "pilot10",
                "evidence_status": "exploratory_provisional",
            }
        )
        return

    sensitivity_runs: dict[str, dict[str, Any]] = {}
    for task_id in sorted(sensitivity_arguments):
        _, binding, _ = _validated_v2_suite_component(
            resolve_input(sensitivity_arguments[task_id]),
            task_id,
            "sensitivity20",
            repo_root=root,
        )
        if binding["primary_reference"].get("manifest_hash") != task_runs[task_id].get(
            "manifest_hash"
        ):
            raise ValueError(
                f"sensitivity20 run is paired to a different primary run: {task_id}"
            )
        binding["run_dir"] = portable(Path(binding["run_dir"]))
        sensitivity_runs[task_id] = binding

    bdb2020_reference, _ = _validated_bdb2020_reference(
        resolve_input(arguments.bdb2020_run_dir), repo_root=root
    )
    bdb2020_reference["run_dir"] = portable(Path(bdb2020_reference["run_dir"]))
    payload = {
        "schema_version": "bdb-suite-manifest-v3",
        "task_runs": task_runs,
        "sensitivity_runs": sensitivity_runs,
        "bdb2020_reference": bdb2020_reference,
    }
    payload["suite_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = immutable_write_json(target, payload)
    _print({"output": str(target.resolve()), "suite_hash": payload["suite_hash"]})


def command_pilot_report(arguments: argparse.Namespace) -> None:
    """Build the checksum-gated six-task provisional pilot mock-up."""

    from .pilot_report import generate_pilot_report

    _print(
        generate_pilot_report(
            arguments.aggregate_receipt,
            output_dir=arguments.output_dir,
            output_stem=arguments.output_stem,
        )
    )


def command_fidelity_bdb2024(arguments: argparse.Namespace) -> None:
    import hashlib
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score
    from .fidelity.bdb2024 import (
        fidelity_receipt,
        fit_executed_xgboost,
        persist_reference_artifacts,
        predict_fidelity_score,
        run_framewise_fidelity_from_raw,
        validate_fidelity_counts,
    )
    from .prepared import load_prepared_receipt, load_prepared_task
    from .storage import immutable_write_json

    task = load_prepared_task(arguments.prepared_dir, mmap_mode="r", verify_files=True)
    prepared_receipt = load_prepared_receipt(arguments.prepared_dir, verify_files=False)
    if task.task_id != "bdb2024_tackle":
        raise ValueError("fidelity branch requires the BDB2024 prepared task")
    development = task.examples["week"].to_numpy(dtype=int) <= 8
    test = task.examples["week"].to_numpy(dtype=int) == 9
    validate_fidelity_counts(
        int(np.sum(development & (np.asarray(task.y) == 1))),
        int(np.sum(development & (np.asarray(task.y) == 0))),
        int(np.sum(test & (np.asarray(task.y) == 1))),
        int(np.sum(test & (np.asarray(task.y) == 0))),
    )
    fitted = fit_executed_xgboost(task.tabular.loc[development], task.y[development], verbose=arguments.verbose)
    score = predict_fidelity_score(fitted, task.tabular.loc[test])
    target = task.y[test]
    result: dict[str, Any] = {
        "schema_version": "bdb2024-fidelity-result-v2",
        "branch": "bdb2024_executed_notebook_xgboost_reference",
        "prepared_hash": prepared_receipt["prepared_hash"],
        "development_examples": int(development.sum()),
        "week9_examples": int(test.sum()),
        "week9_auc": float(roc_auc_score(target, score)),
        "week9_log_loss": float(log_loss(target, score)),
        "parameters": dict(fitted.parameters),
        "fidelity_receipt": fidelity_receipt(),
        "score_interpretation": "case_control_score_not_unconditional_probability",
    }
    # Direct callers constructed before the end-to-end artifact branch existed
    # intentionally retain the old summary-only behavior. Every invocation
    # parsed by ``build_parser`` has ``artifact_dir`` and executes the complete
    # persisted reference + all-frame path.
    if hasattr(arguments, "artifact_dir"):
        root = Path(arguments.repo_root).resolve()
        output_path = Path(arguments.output).resolve()
        artifact_dir = (
            Path(arguments.artifact_dir).resolve()
            if arguments.artifact_dir
            else output_path.parent / f"{output_path.stem}.artifacts"
        )
        reference = persist_reference_artifacts(
            fitted,
            task.examples.loc[test].reset_index(drop=True),
            target,
            score,
            artifact_dir / "reference",
            prepared_hash=prepared_receipt["prepared_hash"],
        )
        raw_dir = (
            Path(arguments.raw_dir).resolve()
            if arguments.raw_dir
            else _default_raw(root, "bdb2024_tackle")
        )
        framewise = run_framewise_fidelity_from_raw(
            fitted.model,
            raw_dir,
            artifact_dir / "framewise",
            model_sha256=reference["artifacts"]["model"]["sha256"],
            artifact_format=arguments.artifact_format,
            chunk_rows=arguments.chunk_rows,
        )
        result["artifacts"] = {
            "root": str(artifact_dir),
            "reference_receipt_hash": reference["receipt_hash"],
            "reference_receipt": "reference/reference_receipt.json",
            "framewise_receipt_hash": framewise["receipt_hash"],
            "framewise_receipt": "framewise/framewise_receipt.json",
        }
        result["framewise"] = {
            "complete_weeks": framewise["complete_weeks"],
            "scored_defender_frames": int(
                sum(
                    item["scored_defender_frames"]
                    for item in framewise["week_receipts"]
                )
            ),
            "play_defender_summaries": int(
                sum(
                    item["play_defender_summaries"]
                    for item in framewise["week_receipts"]
                )
            ),
        }
    result["result_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = immutable_write_json(arguments.output, result)
    _print({"output": str(output), **result})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bdb_study")
    parser.add_argument("--repo-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    item = subparsers.add_parser("inventory")
    item.add_argument("--audit", action="store_true")
    item.set_defaults(func=command_inventory)

    item = subparsers.add_parser("import-data")
    item.add_argument("--release", type=int, action="append", choices=[2021, 2022, 2024])
    item.set_defaults(func=command_import_data)

    for name in ("prepare", "dev-cv", "freeze", "plan"):
        item = subparsers.add_parser(name)
        item.add_argument("--task", required=True)
        if name == "prepare":
            item.add_argument("--raw-dir")
            item.add_argument("--output-dir")
            item.add_argument(
                "--prepared-variant",
                default="primary",
                choices=["primary", "common_closest_approach_minus_ten"],
            )
            item.add_argument("--max-examples", type=int)
            item.add_argument("--audit-only", action="store_true")
            item.set_defaults(func=command_prepare)
        elif name == "dev-cv":
            item.add_argument("--prepared-dir")
            item.add_argument("--output-dir")
            item.add_argument(
                "--family",
                choices=[
                    "glm",
                    "lightgbm",
                    "relnet",
                    "attn_relnet",
                    "set_transformer",
                ],
            )
            item.add_argument(
                "--resume",
                action="store_true",
                help="reuse valid immutable fold/candidate checkpoints",
            )
            item.add_argument(
                "--fold",
                type=int,
                action="append",
                help=(
                    "checkpoint-only development fold filter; requires one "
                    "--family and emits no final receipt"
                ),
            )
            item.add_argument(
                "--candidate",
                type=int,
                action="append",
                help=(
                    "checkpoint-only candidate-index filter; requires one "
                    "--family and emits no final receipt"
                ),
            )
            item.set_defaults(func=command_dev_cv)
        elif name == "freeze":
            item.add_argument("--prepared-dir")
            item.add_argument("--development-dir")
            item.add_argument("--output")
            item.add_argument(
                "--verify-full-source-trees",
                action="store_true",
                help="compatibility flag; definitive freeze always verifies full source trees",
            )
            item.set_defaults(func=command_freeze)
        else:
            item.add_argument("--frozen-task")
            item.add_argument("--prepared-dir")
            item.add_argument("--run-dir", required=True)
            item.add_argument(
                "--profile",
                choices=["full50", "pilot10", "full100", "sensitivity20"],
                default="full100",
            )
            item.add_argument("--repeats", type=int)
            item.add_argument("--cpu-workers", type=int)
            mode = item.add_mutually_exclusive_group()
            mode.add_argument("--smoke", action="store_true")
            mode.add_argument("--benchmark", action="store_true")
            item.add_argument("--preflight-receipt")
            item.add_argument("--runtime-plan")
            item.add_argument("--primary-reference-run")
            item.set_defaults(func=command_plan)

    item = subparsers.add_parser("dev-cv-reduce")
    item.add_argument("--task", required=True)
    item.add_argument("--prepared-dir")
    item.add_argument("--output-dir")
    item.add_argument(
        "--family",
        required=True,
        choices=[
            "glm",
            "lightgbm",
            "relnet",
            "attn_relnet",
            "set_transformer",
        ],
    )
    item.set_defaults(func=command_dev_cv_reduce)

    def add_execution_filters(item: argparse.ArgumentParser) -> None:
        item.add_argument("--queue", choices=["cpu_tabular", "gpu_neural"], action="append")
        item.add_argument("--repeat", type=int, action="append")
        item.add_argument("--anchor", type=int, action="append")
        item.add_argument("--model", action="append")
        item.add_argument("--ablation", action="append")
        item.add_argument("--sensitivity", action="append")

    item = subparsers.add_parser("run")
    item.add_argument("--run-dir", required=True)
    item.add_argument("--resume", action="store_true", help="accepted explicitly; valid cells always skip")
    item.add_argument(
        "--neural-phase",
        choices=["monolithic", "selector", "refit"],
        default="monolithic",
    )
    add_execution_filters(item)
    item.set_defaults(func=command_run)
    item = subparsers.add_parser("status")
    item.add_argument("--run-dir", required=True)
    add_execution_filters(item)
    item.set_defaults(func=command_status)
    item = subparsers.add_parser("aggregate-task")
    item.add_argument("--run-dir", required=True)
    item.add_argument("--require-complete", action="store_true")
    item.set_defaults(func=command_aggregate_task)
    item = subparsers.add_parser("aggregate-suite")
    item.add_argument("--suite-manifest", required=True)
    item.add_argument("--output")
    item.set_defaults(func=command_aggregate_suite)
    item = subparsers.add_parser("suite-plan")
    item.add_argument("--profile", choices=["pilot10", "full100"], default="full100")
    item.add_argument("--task-run", action="append", required=True)
    item.add_argument("--sensitivity-run", action="append")
    item.add_argument("--bdb2020-run-dir")
    item.add_argument("--output", required=True)
    item.set_defaults(func=command_suite_plan)
    item = subparsers.add_parser("pilot-report")
    item.add_argument("--aggregate-receipt", required=True)
    item.add_argument("--output-dir")
    item.add_argument(
        "--output-stem",
        default="bdb2021_2026_pilot10_provisional",
    )
    item.set_defaults(func=command_pilot_report)
    item = subparsers.add_parser("fidelity-bdb2024")
    item.add_argument("--prepared-dir", required=True)
    item.add_argument("--raw-dir")
    item.add_argument("--output", required=True)
    item.add_argument(
        "--artifact-dir",
        help="immutable model, candidate-score, frame-score, and event artifact root",
    )
    item.add_argument(
        "--artifact-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Parquet is the definitive format; CSV is available for dependency-light smoke tests",
    )
    item.add_argument("--chunk-rows", type=int, default=50000)
    item.add_argument("--verbose", action="store_true")
    item.set_defaults(func=command_fidelity_bdb2024)
    item = subparsers.add_parser("verify-determinism")
    item.add_argument("--run-a", required=True)
    item.add_argument("--run-b", required=True)
    item.add_argument("--output", required=True)
    item.set_defaults(func=command_verify_determinism)
    item = subparsers.add_parser("runtime-plan")
    item.add_argument("--benchmark-run", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--safety-factor", type=float, default=1.25)
    item.set_defaults(func=command_runtime_plan)
    item = subparsers.add_parser("runtime-plan-phased")
    item.add_argument("--benchmark-run", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--safety-factor", type=float, default=1.25)
    item.set_defaults(func=command_runtime_plan_phased)
    item = subparsers.add_parser("runtime-plan-phased-v4")
    item.add_argument("--benchmark-run", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--determinism-probe-binding", required=True)
    item.add_argument("--safety-factor", type=float, default=1.25)
    item.set_defaults(func=command_runtime_plan_phased_v4)
    item = subparsers.add_parser("synthetic-smoke")
    item.add_argument("--output-dir", required=True)
    item.set_defaults(func=command_synthetic_smoke)
    item = subparsers.add_parser("_worker")
    item.add_argument("--run-dir", required=True)
    item.add_argument("--repeat", type=int, required=True)
    item.add_argument("--queue", choices=["cpu_tabular", "gpu_neural"], required=True)
    item.add_argument("--anchor", type=int, action="append")
    item.add_argument("--model", action="append")
    item.add_argument("--ablation", action="append")
    item.add_argument("--sensitivity", action="append")
    item.add_argument(
        "--neural-phase",
        choices=["monolithic", "selector", "refit"],
        default="monolithic",
    )
    item.set_defaults(func=command_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        arguments.func(arguments)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
