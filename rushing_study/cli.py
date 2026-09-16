"""Command-line interface for planning, running, and aggregating the study."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .analysis import (
    flatten_sensitivity_candidate_scores,
    run_aggregation,
    run_sensitivity_aggregation,
)
from .data import load_study_data
from .design import (
    FULL_CONFIRMATORY_REPEATS,
    HYBRID_EXECUTION_PROFILE,
    HYBRID_EXECUTION_QUEUES,
    build_study_manifest,
    declared_execution_settings,
    execution_profile,
    load_config,
    sha256_json,
    verify_runtime_provenance,
)
from .execution import (
    SENSITIVITY_STAGE1_RECEIPT,
    classification_counts,
    classify_scientific_cells,
    configure_worker_device,
    configured_anchors,
    configured_execution_queues,
    configured_models,
    execute_repeat,
    execution_queue_for_model,
    expected_cell_keys,
    load_metrics_frame,
    resolve_execution_queue,
    schedule_subprocesses,
    validate_sensitivity_stage1_receipt,
)
from .runner import run_cell
from .storage import (
    CellStatus,
    RunStorage,
    atomic_write_csv,
    atomic_write_json,
    finalize_run,
    initialize_run_dir,
    register_existing_artifact,
    validate_checksum,
)


DEFAULT_CONFIG = Path("configs/rushing_confirmatory_v1.json")
def _comma_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _comma_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _determinism_receipts(
    storage: RunStorage, *, exact_neural_preflight: bool = False
) -> list[tuple[Path, dict[str, Any]]]:
    """Return checksummed determinism receipts that satisfy their declared contract."""

    receipts: list[tuple[Path, dict[str, Any]]] = []
    verification_dir = storage.run_dir / "verification"
    if not verification_dir.is_dir():
        return receipts
    settings = declared_execution_settings(storage.manifest["config"])
    required_environment = settings["environment_variables"]
    repo_root = storage.manifest.get("provenance", {}).get("code", {}).get("repo_root")
    for path in sorted(verification_dir.glob("*.json")):
        if not validate_checksum(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        comparisons = payload.get("comparisons") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("passed") is not True
            or payload.get("manifest_hash") != storage.manifest_hash
            or payload.get("branch") not in {"main", "sensitivity"}
            or payload.get("model") not in configured_models(storage.manifest)
            or payload.get("execution_queue")
            != execution_queue_for_model(storage.manifest, payload.get("model"))
            or not isinstance(payload.get("repeat"), int)
            or not isinstance(payload.get("n_train"), int)
            or payload.get("environment") != required_environment
            or not isinstance(comparisons, dict)
            or set(comparisons) != {"calibration_proba", "test_proba"}
        ):
            continue
        valid_comparisons = True
        for comparison in comparisons.values():
            if not isinstance(comparison, dict) or comparison.get("within_tolerance") is not True:
                valid_comparisons = False
                break
            difference = comparison.get("max_abs_difference")
            if not isinstance(difference, (int, float)) or not np.isfinite(difference) or difference < 0:
                valid_comparisons = False
                break
        if not valid_comparisons:
            continue
        if exact_neural_preflight and not (
            payload.get("branch") == "main"
            and payload.get("repeat") == 1
            and payload.get("n_train") == 20
            and payload.get("model") in {"zoo_cnn", "set_transformer"}
            and float(payload.get("rtol", float("nan"))) == 0.0
            and float(payload.get("atol", float("nan"))) == 0.0
            and all(
                comparison.get("exact") is True
                and float(comparison.get("max_abs_difference", float("nan"))) == 0.0
                for comparison in comparisons.values()
            )
        ):
            continue
        receipts.append((path, payload))
    return receipts


def _attach_preflight_attestation(
    manifest: dict[str, Any], preflight_run_dir: Path
) -> dict[str, Any]:
    """Validate the smoke protocol and bind its evidence into a final manifest."""

    smoke = RunStorage.open(preflight_run_dir)
    if smoke.manifest.get("config", {}).get("execution", {}).get("plan") != "smoke":
        raise RuntimeError("--preflight-run-dir must contain a smoke-only manifest.")
    smoke_repeats = smoke.manifest.get("config", {}).get("execution", {}).get(
        "confirmatory_repeats"
    )
    final_repeats = manifest.get("config", {}).get("execution", {}).get(
        "confirmatory_repeats"
    )
    if smoke_repeats != final_repeats:
        raise RuntimeError(
            "Smoke preflight and definitive plan must select the same 50- or "
            "100-repeat study profile."
        )
    smoke_execution_profile = execution_profile(smoke.manifest["config"])
    final_execution_profile = execution_profile(manifest["config"])
    if smoke_execution_profile != final_execution_profile:
        raise RuntimeError(
            "Smoke preflight and definitive plan must select the same sequential "
            "or hybrid execution profile."
        )
    smoke_neural = smoke.manifest.get("config", {}).get("neural_training", {})
    if (
        smoke_neural.get("max_epochs") != 1
        or smoke_neural.get("early_stopping_patience") != 0
    ):
        raise RuntimeError(
            "Smoke preflight must use exactly one neural epoch with patience zero."
        )
    final_provenance = manifest.get("provenance", {})
    smoke_provenance = smoke.manifest.get("provenance", {})
    final_code = final_provenance.get("code", {})
    smoke_code = smoke_provenance.get("code", {})
    if (
        final_code.get("files") != smoke_code.get("files")
        or final_provenance.get("data") != smoke_provenance.get("data")
        or final_provenance.get("environment") != smoke_provenance.get("environment")
    ):
        raise RuntimeError(
            "Smoke preflight and definitive plan must use identical declared code, "
            "data, environment, and hardware."
        )

    repo_root = smoke_code.get("repo_root")
    prior_cwd = Path.cwd()
    try:
        if repo_root:
            os.chdir(repo_root)
        data = load_study_data(smoke.manifest["config"])
        cell_keys = expected_cell_keys(
            smoke.manifest,
            "main",
            repeat_ids=[1],
            anchors=[20],
        )
        classified = classify_scientific_cells(smoke, cell_keys, study_data=data)
    finally:
        os.chdir(prior_cwd)
    invalid = {
        str(key.relative_dir): result.reason
        for key, result in classified.items()
        if not result.is_complete
    }
    if invalid:
        raise RuntimeError(
            "Smoke preflight requires all four repeat-1/n=20 main cells: "
            + json.dumps(invalid, sort_keys=True)
        )
    receipts = _determinism_receipts(smoke, exact_neural_preflight=True)
    receipts_by_model = {
        payload["model"]: (path, payload)
        for path, payload in receipts
        if payload.get("model") in {"zoo_cnn", "set_transformer"}
    }
    if set(receipts_by_model) != {"zoo_cnn", "set_transformer"}:
        raise RuntimeError(
            "Smoke preflight requires exact (rtol=atol=0) determinism receipts for both neural models."
        )
    cells = []
    for key in sorted(cell_keys):
        marker = smoke.cell_dir(key) / "_SUCCESS"
        cells.append(
            {
                "path": str(key.relative_dir),
                "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            }
        )
    attestation = {
        "schema_version": 1,
        "protocol": "one-repeat-one-epoch-smoke-preflight-v1",
        "source_manifest_hash": smoke.manifest_hash,
        "source_run_hash": smoke.manifest.get("run_hash"),
        "cells": cells,
        "determinism_receipts": {
            model: {
                "file": str(receipt_path.relative_to(smoke.run_dir)),
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "payload": receipt_payload,
            }
            for model, (receipt_path, receipt_payload) in sorted(
                receipts_by_model.items()
            )
        },
        "code_files_sha256": sha256_json(smoke_code.get("files", {})),
        "data_provenance_sha256": sha256_json(smoke_provenance.get("data", {})),
        "environment_provenance_sha256": sha256_json(
            smoke_provenance.get("environment", {})
        ),
    }
    updated = deepcopy(manifest)
    updated["preflight_attestation"] = attestation
    updated.pop("manifest_hash", None)
    updated.pop("run_hash", None)
    manifest_hash = sha256_json(updated)
    updated["manifest_hash"] = manifest_hash
    updated["run_hash"] = manifest_hash[:24]
    return updated


def _write_manifest_views(storage: RunStorage) -> None:
    manifest = storage.manifest
    views = [
        ("config.json", manifest["config"]),
        ("split_manifests.json", manifest["split_manifests"]),
        ("seed_registry.json", manifest["seed_registry"]),
        ("provenance.json", manifest["provenance"]),
    ]
    if "preflight_attestation" in manifest:
        views.append(("preflight_attestation.json", manifest["preflight_attestation"]))
    for filename, value in views:
        path = storage.run_dir / filename
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != value:
                raise RuntimeError(f"Immutable manifest view differs: {path}")
            continue
        atomic_write_json(path, value)


def _verify_runtime_if_frozen(
    storage: RunStorage, *, require_environment: bool
) -> None:
    """Enforce a definitive manifest while allowing synthetic test manifests."""

    provenance = storage.manifest.get("provenance")
    if not isinstance(provenance, dict):
        return
    code = provenance.get("code")
    repo_root = code.get("repo_root", ".") if isinstance(code, dict) else "."
    verify_runtime_provenance(
        storage.manifest,
        repo_root,
        require_environment=require_environment,
    )


def command_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.full:
        config["execution"]["confirmatory_repeats"] = FULL_CONFIRMATORY_REPEATS
    if args.hybrid:
        config["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        config["execution"]["queues"] = deepcopy(HYBRID_EXECUTION_QUEUES)
    if args.smoke:
        config = deepcopy(config)
        config["execution"]["plan"] = "smoke"
        config["neural_training"]["max_epochs"] = int(args.smoke_neural_epochs)
        config["neural_training"]["early_stopping_patience"] = 0
        config["description"] = f"SMOKE ONLY: {config['description']}"
    raw_path = Path(args.repo_root) / config["data"]["raw_train_csv"]
    game_table = pd.read_csv(raw_path, usecols=["GameId", "Season"]).drop_duplicates()
    manifest = build_study_manifest(config, game_table, args.repo_root)
    if config["execution"]["plan"] == "final":
        if args.preflight_run_dir is None:
            raise RuntimeError(
                "Definitive planning requires --preflight-run-dir from the completed smoke protocol."
            )
        manifest = _attach_preflight_attestation(manifest, args.preflight_run_dir)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.repo_root) / "results" / "rushing" / manifest["run_hash"]
    storage = initialize_run_dir(run_dir, manifest, manifest_hash=manifest["manifest_hash"])
    _write_manifest_views(storage)
    print(
        json.dumps(
            {
                "run_dir": str(storage.run_dir),
                "manifest_hash": storage.manifest_hash,
                "run_hash": manifest["run_hash"],
                "plan": config["execution"]["plan"],
                "profile": (
                    "full_100"
                    if config["execution"]["confirmatory_repeats"]
                    == FULL_CONFIRMATORY_REPEATS
                    else "default_50"
                ),
                "execution_profile": execution_profile(config),
                "main_cells": len(expected_cell_keys(manifest, "main")),
                "sensitivity_stage1_cells": len(expected_cell_keys(manifest, "sensitivity")),
            },
            indent=2,
        )
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    schedule_subprocesses(
        args.run_dir,
        args.branch,
        sensitivity_extended=args.extend_sensitivity,
        resume=args.resume,
        repeat_ids=_comma_ints(args.repeats),
        anchors=_comma_ints(args.anchors),
        models=_comma_strings(args.models),
        workers=args.workers,
        queue=args.queue,
    )
    return 0


def command_run_repeat(args: argparse.Namespace) -> int:
    storage = RunStorage.open(Path(args.run_dir).resolve())
    repo_root = storage.manifest.get("provenance", {}).get("code", {}).get("repo_root")
    if repo_root:
        os.chdir(repo_root)
    queue_name, queue_models = resolve_execution_queue(
        storage.manifest,
        args.queue,
        _comma_strings(args.models),
    )
    expected_queue = os.environ.get("RUSHING_STUDY_QUEUE")
    if expected_queue is not None and expected_queue != queue_name:
        raise RuntimeError(
            f"Worker environment queue {expected_queue!r} does not match "
            f"requested queue {queue_name!r}."
        )
    result = execute_repeat(
        storage,
        args.branch,
        args.repeat_id,
        sensitivity_extended=args.extend_sensitivity,
        anchors=_comma_ints(args.anchors),
        models=queue_models,
        queue=queue_name,
        resume=args.resume,
    )
    result["execution_queue"] = queue_name
    result["device"] = configured_execution_queues(storage.manifest)[queue_name][
        "device"
    ]
    print(json.dumps(result, indent=2))
    return 0


def _status_for(
    storage: RunStorage,
    branch: str,
    extended: bool,
    *,
    models: Iterable[str] | None = None,
    worker_count: int = 1,
) -> dict[str, Any]:
    keys = expected_cell_keys(
        storage.manifest,
        branch,
        sensitivity_extended=extended,
        models=models,
    )
    classified = classify_scientific_cells(
        storage, keys, recompute_conformal=False
    )
    counts = classification_counts(classified).as_dict()
    corrupt = [str(key.relative_dir) for key, result in classified.items() if result.status is CellStatus.CORRUPT]
    counts["corrupt_examples"] = corrupt[:10]
    elapsed_by_shape: dict[tuple[int, str], list[float]] = {}
    all_elapsed: list[float] = []
    for key, result in classified.items():
        if not result.is_complete:
            continue
        elapsed = float(storage.load_metrics(key).get("elapsed_seconds", 0.0))
        if elapsed > 0:
            elapsed_by_shape.setdefault((key.n_train, key.model), []).append(elapsed)
            all_elapsed.append(elapsed)
    fallback = float(np.mean(all_elapsed)) if all_elapsed else None
    eta = 0.0
    estimable = True
    repeat_work: dict[int, float] = {}
    for key, result in classified.items():
        if result.is_complete:
            continue
        observations = elapsed_by_shape.get((key.n_train, key.model), [])
        estimate = float(np.mean(observations)) if observations else fallback
        if estimate is None:
            estimable = False
            break
        eta += estimate
        repeat_work[key.repeat] = repeat_work.get(key.repeat, 0.0) + estimate
    counts["eta_seconds"] = eta if estimable else None
    if estimable and repeat_work:
        slots = [0.0] * min(max(int(worker_count), 1), len(repeat_work))
        for work in sorted(repeat_work.values(), reverse=True):
            slot = min(range(len(slots)), key=slots.__getitem__)
            slots[slot] += work
        counts["approximate_wall_eta_seconds"] = max(slots)
    else:
        counts["approximate_wall_eta_seconds"] = 0.0 if estimable else None
    return counts


def command_status(args: argparse.Namespace) -> int:
    storage = RunStorage.open(args.run_dir)
    queues: dict[str, Any] = {}
    for queue_name, spec in configured_execution_queues(storage.manifest).items():
        main_status = _status_for(
            storage,
            "main",
            False,
            models=spec["models"],
            worker_count=int(spec["workers"]),
        )
        sensitivity_status = _status_for(
            storage,
            "sensitivity",
            False,
            models=spec["models"],
            worker_count=int(spec["workers"]),
        )
        worker_count = int(spec["workers"])
        queue_status: dict[str, Any] = {
            "device": spec["device"],
            "workers": worker_count,
            "models": list(spec["models"]),
            "main": main_status,
            "sensitivity_stage1": sensitivity_status,
        }
        if args.extend_sensitivity:
            extended_status = _status_for(
                storage,
                "sensitivity",
                True,
                models=spec["models"],
                worker_count=worker_count,
            )
            queue_status["sensitivity_extended"] = extended_status
        queues[queue_name] = queue_status
    result = {
        "run_dir": str(storage.run_dir),
        "manifest_hash": storage.manifest_hash,
        "profile": (
            "full_100"
            if storage.manifest["config"]["execution"]["confirmatory_repeats"]
            == FULL_CONFIRMATORY_REPEATS
            else "default_50"
        ),
        "execution_profile": execution_profile(storage.manifest["config"]),
        "main": _status_for(storage, "main", False),
        "sensitivity_stage1": _status_for(storage, "sensitivity", False),
        "queues": queues,
        "finalized": (storage.run_dir / "_SUCCESS").exists(),
    }
    if args.extend_sensitivity:
        result["sensitivity_extended"] = _status_for(storage, "sensitivity", True)
    print(json.dumps(result, indent=2))
    return 0


def _register_analysis_outputs(result: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for path in result["output_paths"].values():
        target = Path(path)
        register_existing_artifact(target)
        paths.append(target)
    return paths


def _write_stage1_decision_receipt(
    storage: RunStorage,
    sensitivity_result: dict[str, Any],
    analysis_paths: Iterable[Path],
) -> Path:
    """Commit the prespecified 20-repeat decision without allowing later rewrites."""

    keys = expected_cell_keys(storage.manifest, "sensitivity")
    cell_markers = []
    for key in sorted(keys):
        marker = storage.cell_dir(key) / "_SUCCESS"
        if not marker.is_file():
            raise RuntimeError(f"Cannot receipt an incomplete stage-one cell: {key}")
        cell_markers.append(
            {
                "path": str(key.relative_dir),
                "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            }
        )
    artifact_index = []
    for path in sorted((Path(value) for value in analysis_paths), key=str):
        if not validate_checksum(path):
            raise RuntimeError(f"Cannot receipt an invalid stage-one artifact: {path}")
        artifact_index.append(
            {
                "file": str(path.resolve().relative_to(storage.run_dir.resolve())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    trigger_manifest = sensitivity_result.get("trigger_manifest", {})
    receipt = {
        "schema_version": 1,
        "manifest_hash": storage.manifest_hash,
        "branch": "sensitivity",
        "analysis_stage": "stage1",
        "repeat_ids": list(range(1, 21)),
        "anchors": configured_anchors(storage.manifest, "sensitivity"),
        "models": configured_models(storage.manifest),
        "extension_trigger": bool(sensitivity_result["extension_trigger"]),
        "trigger_reasons": list(sensitivity_result.get("trigger_reasons", [])),
        "trigger_manifest": trigger_manifest,
        "cell_markers": cell_markers,
        "analysis_artifacts": artifact_index,
    }
    path = storage.run_dir / SENSITIVITY_STAGE1_RECEIPT
    if path.exists():
        if not validate_checksum(path):
            raise RuntimeError("Existing sensitivity stage-one receipt is corrupt.")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Existing sensitivity stage-one receipt is unreadable.") from exc
        if existing != receipt:
            raise RuntimeError("Refusing to overwrite a different stage-one sensitivity decision.")
        return path
    atomic_write_json(path, receipt)
    return path


def _validated_preflight_view(storage: RunStorage) -> Path | None:
    """Return the indexed preflight view required by definitive manifests."""

    if storage.manifest.get("config", {}).get("execution", {}).get("plan") != "final":
        return None
    attestation = storage.manifest.get("preflight_attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("Definitive manifest lacks its required smoke preflight attestation.")
    determinism = attestation.get("determinism_receipts")
    expected_neural_models = {"zoo_cnn", "set_transformer"}

    def valid_receipt(model: str) -> bool:
        receipt = determinism.get(model) if isinstance(determinism, dict) else None
        payload = receipt.get("payload") if isinstance(receipt, dict) else None
        comparisons = payload.get("comparisons") if isinstance(payload, dict) else None
        return bool(
            isinstance(payload, dict)
            and payload.get("passed") is True
            and payload.get("branch") == "main"
            and payload.get("repeat") == 1
            and payload.get("n_train") == 20
            and payload.get("model") == model
            and float(payload.get("rtol", float("nan"))) == 0.0
            and float(payload.get("atol", float("nan"))) == 0.0
            and isinstance(comparisons, dict)
            and set(comparisons) == {"calibration_proba", "test_proba"}
            and all(
                isinstance(value, dict)
                and value.get("exact") is True
                and value.get("within_tolerance") is True
                and float(value.get("max_abs_difference", float("nan"))) == 0.0
                for value in comparisons.values()
            )
        )

    if (
        attestation.get("protocol") != "one-repeat-one-epoch-smoke-preflight-v1"
        or not isinstance(attestation.get("cells"), list)
        or len(attestation["cells"]) != 4
        or not isinstance(determinism, dict)
        or set(determinism) != expected_neural_models
        or not all(valid_receipt(model) for model in expected_neural_models)
    ):
        raise RuntimeError("Definitive manifest has an invalid smoke determinism attestation.")
    path = storage.run_dir / "preflight_attestation.json"
    if not validate_checksum(path):
        raise RuntimeError("The definitive run's preflight attestation view is missing or corrupt.")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("The definitive run's preflight attestation view is unreadable.") from exc
    if saved != attestation:
        raise RuntimeError("Preflight attestation view differs from the immutable manifest.")
    return path


def _write_sensitivity_outputs(
    storage: RunStorage,
    main_metrics: pd.DataFrame,
    tuned_metrics: pd.DataFrame,
    *,
    extended: bool,
    destination: Path,
) -> tuple[dict[str, Any], list[Path]]:
    result = run_sensitivity_aggregation(
        main_metrics,
        tuned_metrics,
        storage.manifest,
        destination,
        bootstrap_draws=int(storage.manifest["config"]["analysis"]["bootstrap_draws"]),
        extended=extended,
    )
    paths = _register_analysis_outputs(result)
    candidate_cells: list[dict[str, Any]] = []
    for key in expected_cell_keys(
        storage.manifest,
        "sensitivity",
        sensitivity_extended=extended,
    ):
        history = storage.load_history(key)
        metrics = storage.load_metrics(key)
        if not isinstance(history, dict) or "candidate_scores" not in history:
            raise RuntimeError(f"Sensitivity cell {key} has no saved candidate scores.")
        candidate_cells.append(
            {
                "branch": key.branch,
                "repeat": key.repeat,
                "n_train": key.n_train,
                "model": key.model,
                "candidate_scores": history["candidate_scores"],
                "selected_candidate_index": history.get("selected_candidate_index"),
                "selected_config": metrics.get("selected_config"),
            }
        )
    candidates = flatten_sensitivity_candidate_scores(candidate_cells, extended=extended)
    candidate_path = destination / "candidate_scores.csv"
    atomic_write_csv(candidate_path, candidates)
    paths.append(candidate_path)
    result["candidate_scores"] = candidates
    result["output_paths"]["candidate_scores"] = candidate_path
    return result, paths


def command_aggregate(args: argparse.Namespace) -> int:
    storage = RunStorage.open(args.run_dir)
    _verify_runtime_if_frozen(storage, require_environment=False)
    repo_root = storage.manifest.get("provenance", {}).get("code", {}).get("repo_root")
    if repo_root:
        os.chdir(repo_root)
    main_keys = expected_cell_keys(storage.manifest, "main")
    stage1_keys = expected_cell_keys(storage.manifest, "sensitivity")
    extended_keys = expected_cell_keys(
        storage.manifest, "sensitivity", sensitivity_extended=True
    )
    preflight_path = _validated_preflight_view(storage)
    if (storage.run_dir / "_SUCCESS").exists():
        if storage.validate_final(main_keys + extended_keys):
            final_keys = main_keys + extended_keys
            validate_sensitivity_stage1_receipt(storage, require_trigger=True)
        elif storage.validate_final(main_keys + stage1_keys):
            final_keys = main_keys + stage1_keys
            _, decision = validate_sensitivity_stage1_receipt(
                storage, require_trigger=False
            )
            if decision["extension_trigger"]:
                raise RuntimeError(
                    "A finalized stage-one run cannot omit a sensitivity extension that triggered."
                )
        else:
            raise RuntimeError("The finalized run marker or one of its indexed artifacts is corrupt.")
        data = load_study_data(storage.manifest["config"])
        scientific = classify_scientific_cells(
            storage, final_keys, study_data=data
        )
        invalid = {
            str(key.relative_dir): result.reason
            for key, result in scientific.items()
            if not result.is_complete
        }
        if invalid:
            raise RuntimeError(
                "The finalized run fails scientific/data-bound validation: "
                + json.dumps(dict(list(invalid.items())[:10]), sort_keys=True)
            )
        print(json.dumps({"run_dir": str(storage.run_dir), "finalized": True, "changed": False}, indent=2))
        return 0

    data = load_study_data(storage.manifest["config"])
    main_metrics = load_metrics_frame(storage, main_keys, study_data=data)
    final_dir = storage.run_dir / "final" / "main"
    main_result = run_aggregation(
        main_metrics,
        storage.manifest,
        final_dir,
        bootstrap_draws=int(storage.manifest["config"]["analysis"]["bootstrap_draws"]),
    )
    final_paths = _register_analysis_outputs(main_result)
    metrics_path = storage.run_dir / "final" / "metrics.csv"
    atomic_write_csv(metrics_path, main_metrics)
    final_paths.append(metrics_path)

    stage1_classified = classify_scientific_cells(
        storage, stage1_keys, study_data=data
    )
    stage1_complete = all(result.is_complete for result in stage1_classified.values())
    if args.require_complete and not stage1_complete:
        raise RuntimeError("The all-four sensitivity stage is incomplete.")
    sensitivity_result: dict[str, Any] | None = None
    sensitivity_keys = stage1_keys
    extended_complete = False
    stage1_trigger: bool | None = None
    if stage1_complete:
        stage1_metrics = load_metrics_frame(
            storage, stage1_keys, study_data=data
        )
        stage1_result, stage1_paths = _write_sensitivity_outputs(
            storage,
            main_metrics,
            stage1_metrics,
            extended=False,
            destination=storage.run_dir / "final" / "sensitivity" / "stage1",
        )
        stage1_receipt = _write_stage1_decision_receipt(
            storage, stage1_result, stage1_paths
        )
        final_paths.extend([*stage1_paths, stage1_receipt])
        stage1_trigger = bool(stage1_result["extension_trigger"])
        sensitivity_result = stage1_result

        if stage1_trigger:
            validate_sensitivity_stage1_receipt(storage, require_trigger=True)
            extended_classified = classify_scientific_cells(
                storage, extended_keys, study_data=data
            )
            extended_complete = all(
                result.is_complete for result in extended_classified.values()
            )
            if extended_complete:
                extended_metrics = load_metrics_frame(
                    storage, extended_keys, study_data=data
                )
                sensitivity_result, extended_paths = _write_sensitivity_outputs(
                    storage,
                    main_metrics,
                    extended_metrics,
                    extended=True,
                    destination=storage.run_dir
                    / "final"
                    / "sensitivity"
                    / "all_50",
                )
                final_paths.extend(extended_paths)
                sensitivity_keys = extended_keys

    if args.require_complete:
        if sensitivity_result is None:
            raise RuntimeError("Sensitivity analysis is required for finalization.")
        if stage1_trigger and not extended_complete:
            raise RuntimeError(
                "The prespecified sensitivity extension was triggered. Run sensitivity repeats 21-50 "
                "with --extend-sensitivity, then aggregate again."
            )
        if preflight_path is not None:
            final_paths.append(preflight_path)
        required = main_keys + sensitivity_keys
        finalize_run(
            storage,
            required,
            final_artifacts=[path.relative_to(storage.run_dir) for path in final_paths],
        )
    print(
        json.dumps(
            {
                "main_cells": len(main_metrics),
                "sensitivity_cells": 0 if sensitivity_result is None else len(sensitivity_keys),
                "sensitivity_extended": extended_complete,
                "extension_trigger": stage1_trigger,
                "finalized": storage.validate_final(main_keys + sensitivity_keys)
                if (storage.run_dir / "_SUCCESS").exists()
                else False,
            },
            indent=2,
        )
    )
    return 0


def command_verify_determinism(args: argparse.Namespace) -> int:
    storage = RunStorage.open(Path(args.run_dir).resolve())
    queue_name = execution_queue_for_model(storage.manifest, args.model)
    _verify_runtime_if_frozen(storage, require_environment=bool(args.runtime_ready))
    device_record = (
        configure_worker_device(storage.manifest, queue_name)
        if args.runtime_ready
        else None
    )
    if (storage.run_dir / "_SUCCESS").exists():
        raise RuntimeError("Determinism verification must be recorded before the run is finalized.")
    settings = declared_execution_settings(storage.manifest["config"])
    required_environment = settings["environment_variables"]
    repo_root = (
        storage.manifest.get("provenance", {})
        .get("code", {})
        .get("repo_root")
    )
    mismatched = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in required_environment.items()
        if os.environ.get(name) != value
    }
    if mismatched and not args.runtime_ready:
        command = [
            sys.executable,
            "-m",
            "rushing_study",
            "verify-determinism",
            "--run-dir",
            str(storage.run_dir),
            "--branch",
            args.branch,
            "--repeat-id",
            str(args.repeat_id),
            "--n-train",
            str(args.n_train),
            "--model",
            args.model,
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--runtime-ready",
        ]
        environment = os.environ.copy()
        environment.update(required_environment)
        if repo_root:
            frozen_root = str(Path(repo_root).resolve())
            prior_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                frozen_root
                if not prior_pythonpath
                else frozen_root + os.pathsep + prior_pythonpath
            )
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            cwd=None if not repo_root else str(Path(repo_root).resolve()),
        )
        return int(completed.returncode)
    if mismatched:
        raise RuntimeError(f"Deterministic subprocess environment mismatch: {mismatched}")
    if repo_root:
        os.chdir(repo_root)
    data = load_study_data(storage.manifest["config"])
    first = run_cell(data, storage.manifest, args.branch, args.repeat_id, args.n_train, args.model)
    second = run_cell(data, storage.manifest, args.branch, args.repeat_id, args.n_train, args.model)
    comparisons = {}
    for name in ("calibration_proba", "test_proba"):
        left, right = first.arrays[name], second.arrays[name]
        comparisons[name] = {
            "exact": bool(np.array_equal(left, right)),
            "max_abs_difference": float(np.max(np.abs(left.astype(float) - right.astype(float)))),
            "within_tolerance": bool(np.allclose(left, right, rtol=args.rtol, atol=args.atol)),
        }
    passed = all(item["within_tolerance"] for item in comparisons.values())
    result = {
        "passed": passed,
        "manifest_hash": storage.manifest_hash,
        "branch": args.branch,
        "repeat": int(args.repeat_id),
        "n_train": int(args.n_train),
        "model": args.model,
        "execution_queue": queue_name,
        "device": None if device_record is None else device_record["device"],
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {name: os.environ.get(name) for name in required_environment},
        "hardware": storage.manifest.get("provenance", {}).get("environment", {}),
        "comparisons": comparisons,
    }
    verification_path = (
        storage.run_dir
        / "verification"
        / f"{args.branch}_repeat_{args.repeat_id:03d}_n_{args.n_train:04d}_{args.model}.json"
    )
    artifact = atomic_write_json(verification_path, result)
    print(
        json.dumps(
            {
                **result,
                "artifact": str(verification_path),
                "artifact_sha256": artifact.sha256,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


def command_cluster_interval_sensitivity(args: argparse.Namespace) -> int:
    """Build the separate, immutable BDB2020 no-refit interval companion."""

    from .cluster_intervals import generate_cluster_interval_sensitivity

    result = generate_cluster_interval_sensitivity(args.run_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


def command_verify_cluster_interval_sensitivity(args: argparse.Namespace) -> int:
    """Verify the companion receipt, checksums, and optional source binding."""

    from .cluster_intervals import validate_cluster_interval_artifact

    receipt = validate_cluster_interval_artifact(
        args.output_dir,
        source_run_dir=args.run_dir,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "output_dir": str(args.output_dir),
                "protocol": receipt["protocol"],
                "source_manifest_hash": receipt["source_manifest_hash"],
                "cell_count": receipt["cell_count"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prospective four-model rushing stability study.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Validate and freeze an immutable study manifest.")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--repo-root", type=Path, default=Path("."))
    plan.add_argument("--run-dir", type=Path)
    plan.add_argument(
        "--full",
        action="store_true",
        help="Freeze the 100-repeat primary profile instead of the 50-repeat default.",
    )
    plan.add_argument(
        "--hybrid",
        action="store_true",
        help="Use the locked 12-CPU tabular plus one-GPU neural scheduler.",
    )
    plan.add_argument(
        "--preflight-run-dir",
        type=Path,
        help="Completed one-epoch smoke run whose exact determinism receipt is bound into a final manifest.",
    )
    plan.add_argument("--smoke", action="store_true", help="Create a one-epoch preflight manifest.")
    plan.add_argument("--smoke-neural-epochs", type=int, default=1)
    plan.set_defaults(func=command_plan)

    run = subparsers.add_parser("run", help="Dispatch missing repeat workers.")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--branch", choices=["main", "sensitivity"], default="main")
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--workers",
        type=int,
        help="Queue worker count; omit to use the manifest's complete topology.",
    )
    run.add_argument(
        "--queue",
        choices=["sequential", "cpu_tabular", "gpu_neural"],
        help="Run only one manifest queue (normally omitted).",
    )
    run.add_argument("--extend-sensitivity", action="store_true")
    run.add_argument("--repeats", help="Optional comma-separated repeat IDs (useful for smoke runs).")
    run.add_argument("--anchors")
    run.add_argument("--models")
    run.set_defaults(func=command_run)

    worker = subparsers.add_parser(
        "run-repeat", help="Execute one repeat worker (normally dispatched by run)."
    )
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--branch", choices=["main", "sensitivity"], required=True)
    worker.add_argument("--repeat-id", type=int, required=True)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--extend-sensitivity", action="store_true")
    worker.add_argument("--anchors")
    worker.add_argument("--models")
    worker.add_argument(
        "--queue",
        choices=["sequential", "cpu_tabular", "gpu_neural"],
    )
    worker.set_defaults(func=command_run_repeat)

    status = subparsers.add_parser("status", help="Report complete, corrupt, and missing cells.")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--extend-sensitivity", action="store_true")
    status.set_defaults(func=command_status)

    aggregate = subparsers.add_parser("aggregate", help="Validate and publish paired inference.")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--require-complete", action="store_true")
    aggregate.set_defaults(func=command_aggregate)

    deterministic = subparsers.add_parser(
        "verify-determinism", help="Refit one cell twice and compare its probability arrays."
    )
    deterministic.add_argument("--run-dir", type=Path, required=True)
    deterministic.add_argument("--branch", choices=["main", "sensitivity"], default="main")
    deterministic.add_argument("--repeat-id", type=int, default=1)
    deterministic.add_argument("--n-train", type=int, default=20)
    deterministic.add_argument("--model", default="zoo_cnn")
    deterministic.add_argument("--rtol", type=float, default=1e-7)
    deterministic.add_argument("--atol", type=float, default=1e-7)
    deterministic.add_argument("--runtime-ready", action="store_true", help=argparse.SUPPRESS)
    deterministic.set_defaults(func=command_verify_determinism)

    cluster_interval = subparsers.add_parser(
        "cluster-interval-sensitivity",
        help=(
            "Build the immutable no-refit, game-clustered 90%% interval sensitivity "
            "from a finalized rushing full100 run."
        ),
    )
    cluster_interval.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Finalized full100 rushing source run (opened read-only).",
    )
    cluster_interval.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, disjoint destination; an existing path is never overwritten.",
    )
    cluster_interval.set_defaults(func=command_cluster_interval_sensitivity)

    verify_cluster_interval = subparsers.add_parser(
        "verify-cluster-interval-sensitivity",
        help="Verify a cluster-interval receipt/checksum chain and its full100 source binding.",
    )
    verify_cluster_interval.add_argument("--output-dir", type=Path, required=True)
    verify_cluster_interval.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Finalized full100 source run bound by the sensitivity receipt.",
    )
    verify_cluster_interval.set_defaults(func=command_verify_cluster_interval_sensitivity)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
