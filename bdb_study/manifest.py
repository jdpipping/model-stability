"""Freeze task receipts and construct provenance-bound run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from packaging.requirements import InvalidRequirement, Requirement

from .contracts import (
    TaskSpec,
    TaskSpecError,
    build_source_tree_receipt,
    canonical_json,
    freeze_task_spec,
    load_bdb2020_harmonized_amendment,
    load_bdb2026_horizon_scale_amendment,
    load_global_set_transformer_amendment,
    load_task_spec,
    prepared_contract_for_variant,
    sha256_file,
    trajectory_output_head_contract,
    validate_horizon_scale_config,
)
from .design import (
    EXECUTION_PROFILES,
    analysis_seed_parts,
    build_game_registry,
    build_task_design,
    validate_task_design,
)
from .determinism import deterministic_environment
from .prepared import (
    PreparedArtifactError,
    load_prepared_receipt,
    load_prepared_task,
    validate_prepared_task_contract,
)
from .runtime_phases import (
    BETTY_MIG90_LANES as BETTY_PHASED_GPU_LANES,
    RUNTIME_PLAN_V3_SCHEMA_VERSION,
    RUNTIME_PLAN_V4_SCHEMA_VERSION,
)
from .storage import (
    GenericRunStorage,
    bind_storage_backend,
    cell_keys_from_design,
    manifest_sha256,
    queue_manifest,
    validate_checksum,
)


RUN_SCHEMA_VERSION = "bdb-task-run-v1"
DEVELOPMENT_PROVENANCE_SCHEMA_VERSION = "bdb-development-provenance-v2"
TENSORFLOW_PROBE_PREFIX = "BDB_TF_PROBE_JSON="
EXPECTED_COHORTS = {
    "bdb2020_rushing_harmonized": (31_007, 688),
    "bdb2021_completion": (17_846, 253),
    "bdb2022_punt_returns": (2_273, 712),
    "bdb2023_sack": (8_533, 122),
    "bdb2024_tackle": (10_599, 136),
    "bdb2025_man_zone": (9_229, 136),
    "bdb2026_trajectory": (46_045, 272),
}
SOURCE_RECEIPT_BY_RELEASE = {
    2020: "configs/bdb_suite/sources/bdb2020_local.json",
    2021: "configs/bdb_suite/sources/bdb2021_import.json",
    2022: "configs/bdb_suite/sources/bdb2022_import.json",
    2023: "configs/bdb_suite/sources/bdb2023_local.json",
    2024: "configs/bdb_suite/sources/bdb2024_import.json",
    2025: "configs/bdb_suite/sources/bdb2025_local.json",
    2026: "configs/bdb_suite/sources/bdb2026_local.json",
}
SHARED_REGISTRY_PEERS = {
    "bdb2024_tackle": "bdb2025_man_zone",
    "bdb2025_man_zone": "bdb2024_tackle",
}
PROTOCOL_AMENDMENT_PATH = (
    "configs/bdb_suite/protocol_amendments/"
    "20260821_global_set_transformer_primary.json"
)
BDB2026_HORIZON_SCALE_AMENDMENT_PATH = (
    "configs/bdb_suite/protocol_amendments/"
    "20260821_bdb2026_horizon_scale_tail.json"
)
BDB2020_HARMONIZED_TASK_ID = "bdb2020_rushing_harmonized"
BDB2020_HARMONIZED_AMENDMENT_PATH = (
    "configs/bdb_suite/protocol_amendments/"
    "20260906_bdb2020_harmonized_five_role.json"
)


class ManifestError(RuntimeError):
    pass


PRIMARY_REFERENCE_SCHEMA_VERSION = "bdb-primary-reference-v1"


def tensorflow_probe_script() -> str:
    """Return a sentinel-framed TensorFlow runtime probe.

    TensorFlow plugins may write informational text to stdout after Python has
    printed its result.  A distinct prefix lets callers recover the one JSON
    record without assuming that it is the final output line.
    """

    return (
        "import json\n"
        "try:\n"
        " import tensorflow as tf\n"
        " value={'available':True,'version':tf.__version__,"
        "'devices':[{'name':d.name,'device_type':d.device_type} for d in tf.config.list_physical_devices()],"
        "'build_info':tf.sysconfig.get_build_info(),"
        "'intra_op_threads':tf.config.threading.get_intra_op_parallelism_threads(),"
        "'inter_op_threads':tf.config.threading.get_inter_op_parallelism_threads()}\n"
        "except Exception as exc:\n"
        " value={'available':False,'error_type':type(exc).__name__}\n"
        f"print({TENSORFLOW_PROBE_PREFIX!r}+json.dumps(value,sort_keys=True,default=str))\n"
    )


def parse_tensorflow_probe_output(stdout: str) -> dict[str, Any]:
    """Extract exactly one sentinel-framed TensorFlow probe record."""

    records = [
        line[len(TENSORFLOW_PROBE_PREFIX) :]
        for line in str(stdout).splitlines()
        if line.startswith(TENSORFLOW_PROBE_PREFIX)
    ]
    if len(records) != 1:
        raise ManifestError(
            f"TensorFlow probe emitted {len(records)} framed records; expected exactly one"
        )
    try:
        value = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ManifestError("TensorFlow probe emitted malformed JSON") from exc
    if not isinstance(value, dict):
        raise ManifestError("TensorFlow probe record must be a JSON object")
    return value


def validate_prepared_binding(
    spec: TaskSpec,
    prepared_receipt: Mapping[str, Any],
    prepared_task: Any,
    *,
    require_frozen_identity: bool,
    prepared_contract_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a checksum-valid prepared payload to its scientific TaskSpec."""

    prepared_contract = (
        spec.prepared_contract
        if prepared_contract_override is None
        else dict(prepared_contract_override)
    )
    identity = {
        "task_id": spec.task_id,
        "outcome_type": spec.outcome_type,
        "primary_metric": spec.primary_metric,
        "examples": int(prepared_contract["examples"]),
        "games": int(prepared_contract["games"]),
    }
    for field, expected in identity.items():
        observed = prepared_receipt.get(field)
        if field in {"examples", "games"}:
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                pass
        if observed != expected:
            raise ManifestError(
                f"prepared {field} differs from the TaskSpec prepared contract"
            )
    try:
        semantic = validate_prepared_task_contract(
            prepared_task,
            prepared_contract,
            embedded_receipt=prepared_receipt.get("semantic_receipt"),
        )
    except PreparedArtifactError as exc:
        raise ManifestError(f"prepared semantic binding failed: {exc}") from exc

    if require_frozen_identity:
        cohort = spec.cohort
        if cohort.get("prepared_hash") != prepared_receipt.get("prepared_hash"):
            raise ManifestError(
                "prepared hash differs from the exact artifact frozen in the TaskSpec"
            )
        if int(cohort.get("prepared_examples", -1)) != identity["examples"] or int(
            cohort.get("prepared_games", -1)
        ) != identity["games"]:
            raise ManifestError("frozen prepared cohort counts differ from the TaskSpec")
        if canonical_json(cohort.get("prepared_audit")) != canonical_json(
            prepared_receipt.get("audit")
        ):
            raise ManifestError(
                "prepared cohort audit differs from the audit frozen in the TaskSpec"
            )
    return semantic


def model_implementation_receipt(
    outcome_type: str,
    family: str,
    outcome: Mapping[str, Any],
    *,
    selected_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact task-neutral architecture and output-head contract."""

    modeled = "binary" if outcome_type == "frame_event" else str(outcome_type)
    if not isinstance(selected_config, Mapping) or not selected_config:
        raise ManifestError("an implementation receipt requires a frozen selected_config")
    # Round-trip through canonical JSON so the duplicated, human-readable
    # optimization receipt cannot retain aliases or mutable caller objects.
    frozen_config = json.loads(
        json.dumps(dict(selected_config), sort_keys=True, separators=(",", ":"))
    )
    trajectory_target: str | None = None
    if modeled == "trajectory":
        trajectory_target = str(frozen_config.get("trajectory_target", "residual"))
        if trajectory_target not in {"residual", "absolute"}:
            raise ManifestError(
                "trajectory implementation receipt requires trajectory_target "
                "to be residual or absolute"
            )
    if modeled == "binary":
        output_head = {
            "type": "positive_class_probability",
            "shape": ["examples"],
            "prediction": "raw_positive_class_probability",
        }
    elif modeled == "distribution":
        support = outcome.get("support")
        if not isinstance(support, (list, tuple)) or len(support) != 2:
            raise ManifestError("distribution output head requires a two-endpoint support")
        classes = int(support[1]) - int(support[0]) + 1
        output_head = {
            "type": "punt_cdf_residual_v1",
            "shape": ["examples", classes],
            "support": [int(support[0]), int(support[1])],
            "baseline": "train_only_empirical_cdf",
            "projection": "bounded_pava_then_pmf_difference",
        }
    elif modeled == "trajectory":
        if trajectory_target == "absolute":
            output_head = {
                "type": "horizon_conditioned_masked_absolute_xy_tensor",
                "shape": ["examples", "prepared_max_horizon", 2],
                "coordinates_per_horizon": 2,
                "horizon": "prepared_target_values_axis_1",
                "training_target": "official_absolute_xy",
                "reconstruction": (
                    "direct_absolute_prediction_then_constant_velocity_residualization_"
                    "for_common_evaluation"
                ),
            }
        else:
            output_head = {
                "type": "horizon_conditioned_masked_xy_residual_tensor",
                "shape": ["examples", "prepared_max_horizon", 2],
                "coordinates_per_horizon": 2,
                "horizon": "prepared_target_values_axis_1",
                "training_target": "official_absolute_xy_minus_constant_velocity_baseline",
                "reconstruction": "constant_velocity_baseline_plus_residual",
            }
    else:
        raise ManifestError(f"unsupported modeled outcome {modeled!r}")

    if family == "glm":
        architecture = (
            {
                "implementation": "sklearn.linear_model.Ridge",
                "fit": (
                    "one_shared_horizon_conditioned_absolute_coordinate_regressor"
                    if trajectory_target == "absolute"
                    else "one_shared_horizon_conditioned_residual_regressor"
                ),
                "penalty": "l2",
            }
            if modeled == "trajectory"
            else {
                "implementation": "sklearn.linear_model.Ridge",
                "fit": "one_multioutput_cdf_residual_regressor",
                "objective": "cdf_residual_squared_error",
                "penalty": "l2",
            }
            if modeled == "distribution"
            else {
                "implementation": "sklearn.linear_model.SGDClassifier",
                "objective": "log_loss",
                "multiclass": "one_vs_rest",
                "penalty": "l2",
                "epochs": 50,
                "batch_size": 64,
            }
        )
    elif family == "lightgbm":
        architecture = {
            "implementation": (
                "lightgbm.LGBMRegressor_shared_horizon_conditioned"
                if modeled == "trajectory"
                else "lightgbm.LGBMRegressor_threshold_query"
                if modeled == "distribution"
                else "lightgbm.LGBMClassifier"
            ),
            "objective": {
                "binary": "binary",
                "distribution": "regression_l2_cdf_residual",
                "trajectory": "regression_l2",
            }[modeled],
            **(
                {
                    "fit": "one_threshold_query_regressor",
                    "threshold_feature": "support_value",
                    "total_trees": 200,
                }
                if modeled == "distribution"
                else {}
            ),
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
        }
    elif family == "relnet":
        architecture = {
            "implementation": "tensorflow.keras",
            "architecture_id": "bdb_relnet_v2",
            "inputs": "task_tokens_typed_edges_masks_context",
            "edge_messages": "shared_typed_edge_mlp",
            "aggregation": "fixed_masked_sum_normalized_by_observed_degree",
            "temporal_encoder": "shared_masked_temporal_encoder",
            "optimizer": "adam",
            "epoch_selection": "game_and_stratum_grouped_80_20_then_reinitialize_refit_all_n",
        }
    elif family == "attn_relnet":
        architecture = {
            "implementation": "tensorflow.keras",
            "architecture_id": "bdb_attn_relnet_v2",
            "inputs": "task_tokens_typed_edges_masks_context",
            "edge_messages": "shared_typed_edge_mlp",
            "aggregation": "masked_attention_weighted_edge_aggregation",
            "temporal_encoder": "shared_masked_temporal_encoder",
            "optimizer": "adam",
            "epoch_selection": "game_and_stratum_grouped_80_20_then_reinitialize_refit_all_n",
        }
    elif family == "set_transformer":
        architecture = {
            "implementation": "tensorflow.keras",
            "architecture_id": "bdb_global_set_transformer_v1",
            "inputs": "task_tokens_player_frame_masks_time_context",
            "attention_contract": "global_set_time_attention_v1",
            "relation_scope": "global_masked_all_player_self_attention",
            "typed_graph_edges_consumed": False,
            "temporal_encoder": "factorized_masked_temporal_attention",
            "protocol_note": (
                "protocol_v2_factorized_temporal_not_exact_bdb2020_snapshot"
            ),
            "optimizer": "adam",
            "epoch_selection": "game_and_stratum_grouped_80_20_then_reinitialize_refit_all_n",
        }
    else:
        raise ManifestError(f"unsupported model family {family!r}")
    if modeled == "trajectory":
        architecture["training_target"] = trajectory_target
    optimization_loss = {
        "binary": {
            "glm": "log_loss",
            "lightgbm": "binary_logloss_objective",
            "relnet": "log_loss",
            "attn_relnet": "log_loss",
            "set_transformer": "log_loss",
        },
        "distribution": {
            "glm": "cdf_residual_squared_error_crps_equivalent",
            "lightgbm": "cdf_residual_squared_error_crps_equivalent",
            "relnet": "crps",
            "attn_relnet": "crps",
            "set_transformer": "crps",
        },
        "trajectory": {
            "glm": (
                "ridge_absolute_coordinate_squared_error"
                if trajectory_target == "absolute"
                else "ridge_residual_coordinate_squared_error"
            ),
            "lightgbm": (
                "absolute_coordinate_regression_l2"
                if trajectory_target == "absolute"
                else "residual_coordinate_regression_l2"
            ),
            "relnet": (
                "masked_absolute_coordinate_mse"
                if trajectory_target == "absolute"
                else "masked_residual_coordinate_mse"
            ),
            "attn_relnet": (
                "masked_absolute_coordinate_mse"
                if trajectory_target == "absolute"
                else "masked_residual_coordinate_mse"
            ),
            "set_transformer": (
                "masked_absolute_coordinate_mse"
                if trajectory_target == "absolute"
                else "masked_residual_coordinate_mse"
            ),
        },
    }[modeled][family]
    output_head["optimization_loss"] = optimization_loss
    return {
        "schema_version": "bdb-model-implementation-v2",
        "family": family,
        "modeled_outcome": modeled,
        "architecture": architecture,
        "output_head": output_head,
        "frozen_optimization_config": frozen_config,
    }


def validate_model_implementation_receipts(spec: TaskSpec) -> None:
    for model_id, entry in spec.models.items():
        expected = model_implementation_receipt(
            spec.outcome_type,
            str(entry.get("family")),
            spec.outcome,
            selected_config=entry.get("selected_config"),
        )
        if entry.get("implementation_receipt") != expected:
            raise ManifestError(
                f"model {model_id} lacks the exact architecture/output-head receipt"
            )


def _validate_joint_neural_target_binding(
    spec: TaskSpec, *, repo_root: str | Path | None = None
) -> None:
    """Validate the frozen BDB2026 three-neural-model decoder decision."""

    if spec.task_id != "bdb2026_trajectory":
        return
    contract = spec.joint_neural_target_selection
    if not isinstance(contract, Mapping):
        raise ManifestError("BDB2026 lacks its joint neural target contract")
    neural_families = ("relnet", "attn_relnet", "set_transformer")
    neural = {family: spec.models[family] for family in neural_families}
    development_bound = any(
        isinstance(entry.get("development_receipt"), Mapping)
        for entry in neural.values()
    )
    binding = contract.get("receipt")
    if not development_bound:
        # A prospective draft has no scored joint receipt yet. The static rule
        # is already validated by TaskSpec; the definitive freeze fills it.
        if binding is not None:
            raise ManifestError("unscored BDB2026 contract cannot bind a joint receipt")
        return
    if not isinstance(binding, Mapping):
        raise ManifestError("frozen BDB2026 lacks its joint neural target receipt")
    selected_target = str(binding.get("selected_target", ""))
    if (
        spec.output_head != trajectory_output_head_contract(selected_target)
        or spec.outcome.get("neural_training_target") != selected_target
    ):
        raise ManifestError(
            "BDB2026 frozen output head/outcome differs from the joint target"
        )
    for family, entry in neural.items():
        selected_config = entry.get("selected_config")
        development = entry.get("development_receipt")
        if (
            not isinstance(selected_config, Mapping)
            or selected_config.get("trajectory_target") != selected_target
            or not isinstance(development, Mapping)
        ):
            raise ManifestError(
                "BDB2026 neural models must share the jointly selected target"
            )
        local = development.get("joint_neural_target_selection")
        development_path = PurePosixPath(str(development.get("path", "")))
        local_joint_path = (
            development_path.parent
            / str(local.get("joint_receipt_path", ""))
            if isinstance(local, Mapping)
            else PurePosixPath("")
        )
        if (
            not isinstance(local, Mapping)
            or local.get("schema_version") != binding.get("schema_version")
            or local.get("protocol") != binding.get("protocol")
            or local.get("joint_receipt_hash") != binding.get("receipt_hash")
            or local.get("selected_target") != selected_target
            or local.get("source_development_receipt_hash")
            != binding.get("source_development_receipt_hashes", {}).get(family)
            or development.get("source_receipt_hash")
            != local.get("source_development_receipt_hash")
            or local_joint_path.as_posix() != binding.get("path")
        ):
            raise ManifestError(
                f"BDB2026 {family} development receipt is not bound to the joint decision"
            )
    if repo_root is None:
        return
    root = Path(repo_root).resolve()
    joint_path = (root / str(binding.get("path", ""))).resolve()
    try:
        joint_path.relative_to(root)
        joint = _load_json(joint_path)
    except (ValueError, ManifestError) as exc:
        raise ManifestError("joint neural target receipt path is invalid") from exc
    if (
        sha256_file(joint_path) != binding.get("file_sha256")
        or _json_hash({key: value for key, value in joint.items() if key != "receipt_hash"})
        != joint.get("receipt_hash")
        or joint.get("receipt_hash") != binding.get("receipt_hash")
        or joint.get("schema_version") != binding.get("schema_version")
        or joint.get("protocol") != binding.get("protocol")
        or joint.get("selected_target") != selected_target
        or {
            family: joint.get("source_development_receipts", {})
            .get(family, {})
            .get("development_receipt_hash")
            for family in neural_families
        }
        != binding.get("source_development_receipt_hashes")
    ):
        raise ManifestError("joint neural target receipt checksum or binding drifted")


def validate_task_scientific_receipts(
    spec: TaskSpec, *, repo_root: str | Path | None = None
) -> None:
    """Validate model implementations and task-specific reference branches."""

    validate_model_implementation_receipts(spec)
    _validate_joint_neural_target_binding(spec, repo_root=repo_root)
    if spec.task_id == "bdb2024_tackle":
        # Lazy import keeps the general manifest path independent of the
        # optional XGBoost runtime while still binding the executed notebook
        # parameters, exact feature order, cohort counts, and source commit.
        from .fidelity.bdb2024 import fidelity_receipt

        if spec.cohort.get("fidelity_reference") != fidelity_receipt():
            raise ManifestError(
                "BDB2024 frozen TaskSpec lacks the exact winner-fidelity receipt"
            )


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_relative(path: str | Path, root: Path, label: str) -> tuple[Path, str]:
    """Resolve an immutable scientific input/output beneath ``repo_root``."""

    target = Path(path).resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestError(f"{label} must be beneath repo_root") from exc
    return target, relative


def _shared_peer_paths(root: Path, task_id: str) -> tuple[str, Path, Path]:
    try:
        peer_id = SHARED_REGISTRY_PEERS[task_id]
    except KeyError as exc:
        raise ManifestError(f"{task_id!r} has no shared-registry peer") from exc
    return (
        peer_id,
        root / "configs" / "bdb_suite" / "tasks" / f"{peer_id}.json",
        root / "data" / "processed" / "bdb_suite" / peer_id,
    )


def build_shared_prepared_registry_binding(
    spec: TaskSpec,
    current_registry: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any] | None:
    """Validate and bind the paired 2024/2025 prepared game registry.

    Both tasks describe the same 136 games.  Tuning either task is therefore
    admitted only after the peer's checksum-valid, semantically valid prepared
    artifact produces the byte-identical sequestering registry.
    """

    if spec.task_id not in SHARED_REGISTRY_PEERS:
        return None
    root = Path(repo_root).resolve()
    peer_id, peer_draft, peer_prepared = _shared_peer_paths(root, spec.task_id)
    try:
        peer_spec = load_task_spec(peer_draft, repo_root=root, require_source=False)
        peer_receipt = load_prepared_receipt(peer_prepared, verify_files=True)
        peer_task = load_prepared_task(
            peer_prepared, mmap_mode="r", verify_files=False
        )
    except (TaskSpecError, PreparedArtifactError, OSError) as exc:
        raise ManifestError(
            f"shared-registry peer prepared artifact is required: {peer_id}: {exc}"
        ) from exc
    if peer_spec.task_id != peer_id:
        raise ManifestError("shared-registry peer TaskSpec identity differs from its path")
    peer_semantics = validate_prepared_binding(
        peer_spec,
        peer_receipt,
        peer_task,
        require_frozen_identity=False,
    )
    peer_registry = build_game_registry(peer_spec, game_records_from_prepared(peer_task))
    if canonical_json(peer_registry) != canonical_json(current_registry):
        raise ManifestError(
            f"shared 2024/2025 game registries differ: {spec.task_id} vs {peer_id}"
        )
    return {
        "schema_version": "bdb-shared-prepared-registry-v1",
        "task_id": peer_id,
        "task_spec": {
            "path": peer_draft.relative_to(root).as_posix(),
            "task_spec_hash": peer_spec.spec_hash,
            "file_sha256": sha256_file(peer_draft),
        },
        "prepared": {
            "path": peer_prepared.relative_to(root).as_posix(),
            "prepared_hash": peer_receipt["prepared_hash"],
            "receipt_sha256": sha256_file(peer_prepared / "receipt.json"),
            "semantic_hash": peer_semantics["semantic_hash"],
        },
        "game_registry_hash": str(peer_registry["registry_hash"]),
    }


def build_shared_frozen_registry_binding(
    spec: TaskSpec,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any] | None:
    """Validate and bind the paired frozen 2024/2025 registry receipt."""

    if spec.task_id not in SHARED_REGISTRY_PEERS:
        return None
    root = Path(repo_root).resolve()
    peer_id = SHARED_REGISTRY_PEERS[spec.task_id]
    peer_path = root / "configs" / "bdb_suite" / "frozen" / f"{peer_id}.json"
    try:
        peer_spec = load_task_spec(peer_path, repo_root=root, require_source=False)
    except (TaskSpecError, OSError) as exc:
        raise ManifestError(
            f"shared-registry frozen peer is required: {peer_id}: {exc}"
        ) from exc
    if peer_spec.task_id != peer_id:
        raise ManifestError("shared-registry frozen peer identity differs from its path")
    validate_task_scientific_receipts(peer_spec, repo_root=root)
    current_registry = spec.cohort.get("game_registry")
    peer_registry = peer_spec.cohort.get("game_registry")
    if not isinstance(current_registry, Mapping) or not isinstance(peer_registry, Mapping):
        raise ManifestError("both shared-registry frozen tasks require game registries")
    if canonical_json(peer_registry) != canonical_json(current_registry):
        raise ManifestError(
            f"shared frozen 2024/2025 game registries differ: {spec.task_id} vs {peer_id}"
        )
    return {
        "schema_version": "bdb-shared-frozen-registry-v1",
        "task_id": peer_id,
        "path": peer_path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(peer_path),
        "task_spec_hash": peer_spec.spec_hash,
        "game_registry_hash": str(peer_registry.get("registry_hash")),
    }


def bind_run_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a run payload using the storage backend's canonical hash rules."""

    if "manifest_hash" in value or "run_hash" in value:
        raise ManifestError("run identity fields must not be supplied by callers")
    manifest = dict(value)
    manifest_hash = manifest_sha256(manifest)
    manifest["manifest_hash"] = manifest_hash
    manifest["run_hash"] = manifest_hash[:24]
    # This also guards future backend changes from silently disagreeing with
    # the identity emitted here.
    if manifest_sha256(manifest) != manifest_hash:
        raise ManifestError("storage backend rejected the bound run identity")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def _family_model_ids(task: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for model_id, entry in task.get("models", {}).items():
        family = str(entry.get("family", ""))
        if family in result:
            raise ManifestError(f"two models use family {family!r}")
        result[family] = str(model_id)
    if set(result) != {
        "glm", "lightgbm", "relnet", "attn_relnet", "set_transformer"
    }:
        raise ManifestError("task must contain exactly the five canonical families")
    return result


def freeze_task_from_development(
    draft_path: str | Path,
    prepared_dir: str | Path,
    development_dir: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    verify_full_source_trees: bool = False,
) -> dict[str, Any]:
    """Bind exact cohort audit, source tree, and five selected configurations."""

    root = Path(repo_root).resolve()
    draft_target, _ = _repo_relative(draft_path, root, "draft TaskSpec")
    prepared_root, _ = _repo_relative(prepared_dir, root, "prepared artifact")
    development_root, _ = _repo_relative(
        development_dir, root, "development receipt directory"
    )
    output_target, _ = _repo_relative(output_path, root, "frozen TaskSpec output")
    draft = _load_json(draft_target)
    # A definitive freeze always re-reads every byte named by the aggregate
    # source receipt.  The legacy flag remains accepted by callers but no
    # longer weakens this admission gate when omitted.
    draft_spec = load_task_spec(draft_target, repo_root=root, require_source=True)
    prepared = load_prepared_receipt(prepared_root, verify_files=True)
    if draft.get("task_id") != prepared.get("task_id"):
        raise ManifestError("draft TaskSpec and prepared artifact differ")
    if int(draft.get("total_games", -1)) != int(prepared.get("games", -2)):
        raise ManifestError("prepared game count differs from the locked task design")
    expected_examples, expected_games = EXPECTED_COHORTS[str(draft["task_id"])]
    if (int(prepared.get("examples", -1)), int(prepared.get("games", -1))) != (
        expected_examples,
        expected_games,
    ):
        raise ManifestError(
            f"prepared cohort differs from locked count {expected_examples}/{expected_games}"
        )
    expected_sources = {
        str(receipt["source_id"]): str(receipt["tree_sha256"])
        for receipt in draft.get("source_tree_receipts", [])
    }
    if dict(prepared.get("source_receipt_hashes", {})) != expected_sources:
        raise ManifestError("prepared source-tree hashes differ from the draft TaskSpec")
    prepared_task = load_prepared_task(
        prepared_root, mmap_mode="r", verify_files=False
    )
    validate_prepared_binding(
        draft_spec,
        prepared,
        prepared_task,
        require_frozen_identity=False,
    )
    development_design = build_task_design(
        draft_spec, game_records_from_prepared(prepared_task)
    )
    expected_development_provenance = build_development_provenance(
        draft_target,
        prepared_root,
        repo_root=root,
    )
    cohort = dict(draft.get("cohort", {}))
    cohort["prepared_audit"] = dict(prepared.get("audit", {}))
    cohort["prepared_examples"] = int(prepared["examples"])
    cohort["prepared_games"] = int(prepared["games"])
    cohort["prepared_hash"] = str(prepared["prepared_hash"])
    # Freeze the prospective sequestering decision itself, not merely the
    # algorithm that can regenerate it.  This receipt therefore contains the
    # exact development/confirmatory game IDs, all five development folds,
    # the normalized source game records, and the registry content hash.
    # Main-run planning later refuses any registry that differs byte-for-byte
    # in canonical JSON from this pre-score decision.
    cohort["game_registry"] = dict(development_design["game_registry"])
    if draft["task_id"] == "bdb2024_tackle":
        from .fidelity.bdb2024 import fidelity_receipt

        cohort["fidelity_reference"] = fidelity_receipt()
    draft["cohort"] = cohort

    family_ids = _family_model_ids(draft)
    models = {key: dict(value) for key, value in draft["models"].items()}
    raw_development: dict[str, dict[str, Any]] = {}
    raw_paths: dict[str, Path] = {}
    from .devcv import validate_development_receipt

    for family, model_id in family_ids.items():
        receipt_path = development_root / f"{family}.json"
        receipt = _load_json(receipt_path)
        try:
            receipt = validate_development_receipt(
                receipt,
                prepared_task,
                development_design,
                family,
                model_id=model_id,
                expected_provenance=expected_development_provenance,
            )
        except ValueError as exc:
            raise ManifestError(
                f"development receipt failed scientific validation: {receipt_path}: {exc}"
            ) from exc
        raw_development[family] = receipt
        raw_paths[family] = receipt_path

    effective_development = dict(raw_development)
    effective_paths = dict(raw_paths)
    if draft["task_id"] == "bdb2026_trajectory":
        from .devcv import (
            JOINT_NEURAL_TARGET_DEFAULT_PATH,
            finalize_joint_neural_development_receipts,
            write_development_receipt,
            write_joint_neural_target_receipt,
        )

        try:
            finalized = finalize_joint_neural_development_receipts(
                {
                    family: raw_development[family]
                    for family in ("relnet", "attn_relnet", "set_transformer")
                },
                prepared_task,
                development_design,
                expected_provenance=expected_development_provenance,
                joint_receipt_path=JOINT_NEURAL_TARGET_DEFAULT_PATH,
            )
        except ValueError as exc:
            raise ManifestError(
                "joint BDB2026 neural decoder-target selection failed replay"
            ) from exc
        joint = finalized["joint_receipt"]
        joint_path = development_root / JOINT_NEURAL_TARGET_DEFAULT_PATH
        write_joint_neural_target_receipt(joint, joint_path)
        joint_target, joint_relative = _repo_relative(
            joint_path, root, "joint neural target receipt"
        )
        for family in ("relnet", "attn_relnet", "set_transformer"):
            receipt = finalized["finalized_development_receipts"][family]
            receipt_path = development_root / f"{family}.joint_finalized.json"
            write_development_receipt(receipt, receipt_path)
            effective_development[family] = receipt
            effective_paths[family] = receipt_path
        joint_contract = dict(draft["joint_neural_target_selection"])
        joint_contract["receipt"] = {
            "schema_version": joint["schema_version"],
            "protocol": joint["protocol"],
            "path": joint_relative,
            "file_sha256": sha256_file(joint_target),
            "receipt_hash": joint["receipt_hash"],
            "selected_target": joint["selected_target"],
            "source_development_receipt_hashes": {
                family: joint["source_development_receipts"][family][
                    "development_receipt_hash"
                ]
                for family in ("relnet", "attn_relnet", "set_transformer")
            },
        }
        draft["joint_neural_target_selection"] = joint_contract
        selected_target = str(joint["selected_target"])
        draft["output_head"] = trajectory_output_head_contract(selected_target)
        outcome = dict(draft["outcome"])
        outcome["neural_training_target"] = selected_target
        draft["outcome"] = outcome

    for family, model_id in family_ids.items():
        receipt = effective_development[family]
        receipt_path = effective_paths[family]
        claimed = receipt["receipt_hash"]
        selected = receipt.get("selected", {})
        if not isinstance(selected.get("config"), dict):
            raise ManifestError(f"development receipt has no selected config: {receipt_path}")
        selected_config = dict(selected["config"])
        models[model_id]["selected_config"] = selected_config
        models[model_id]["development_receipt"] = {
            "path": receipt_path.resolve().relative_to(root).as_posix(),
            "receipt_hash": claimed,
            "file_sha256": sha256_file(receipt_path),
            "mean_validation_loss": selected["mean_validation_loss"],
        }
        if family in {"relnet", "attn_relnet", "set_transformer"} and draft["task_id"] == "bdb2026_trajectory":
            source_path = raw_paths[family]
            models[model_id]["development_receipt"].update(
                {
                    "source_path": source_path.resolve().relative_to(root).as_posix(),
                    "source_receipt_hash": raw_development[family]["receipt_hash"],
                    "source_file_sha256": sha256_file(source_path),
                    "joint_neural_target_selection": dict(
                        receipt["joint_neural_target_selection"]
                    ),
                }
            )
        models[model_id]["implementation_receipt"] = model_implementation_receipt(
            draft_spec.outcome_type,
            family,
            draft_spec.outcome,
            selected_config=selected_config,
        )
    draft["models"] = models
    release = int(draft["release_year"])
    aggregate_path = root / SOURCE_RECEIPT_BY_RELEASE[release]
    tree_reference = build_source_tree_receipt(aggregate_path, repo_root=root)
    draft["source_receipts"] = []
    draft["source_tree_receipts"] = [tree_reference.as_dict()]
    return freeze_task_spec(
        draft,
        output_target,
        repo_root=root,
        require_source=True,
    )


def _code_receipts(root: Path) -> list[dict[str, Any]]:
    # Fail closed before any development or run identity can be created.  The
    # same file is also included below in the checksummed code tree.
    _protocol_amendment_binding(root)
    paths = sorted((root / "bdb_study").rglob("*.py"))
    paths.extend(
        path
        for path in sorted((root / "configs" / "bdb_suite").rglob("*.json"))
        # Frozen TaskSpecs and preflight attestations are generated receipts,
        # not executable/scientific inputs.  Each is bound separately where
        # consumed, so including them here would make one task's receipt alter
        # every other task's code identity.
        if not {
            "frozen",
            "preflight",
        }
        & set(path.relative_to(root / "configs" / "bdb_suite").parts)
    )
    lock = root / "configs" / "bdb_suite" / "requirements-lock.txt"
    if lock.exists():
        paths.append(lock)
    records = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def _protocol_amendment_binding(root: Path) -> dict[str, Any]:
    path = Path(root).resolve() / PROTOCOL_AMENDMENT_PATH
    amendment = load_global_set_transformer_amendment(path)
    horizon_path = (
        Path(root).resolve() / BDB2026_HORIZON_SCALE_AMENDMENT_PATH
    )
    horizon_amendment = load_bdb2026_horizon_scale_amendment(horizon_path)
    return {
        "path": PROTOCOL_AMENDMENT_PATH,
        "sha256": sha256_file(path),
        "amendment_id": amendment["amendment_id"],
        "evidence_status": amendment["evidence_status"],
        "withdrawn_campaign": amendment["campaign_disposition"][
            "withdrawn_campaign"
        ],
        "replacement_campaign": amendment["campaign_disposition"][
            "replacement_campaign"
        ],
        "bdb2026_horizon_scale": {
            "path": BDB2026_HORIZON_SCALE_AMENDMENT_PATH,
            "sha256": sha256_file(horizon_path),
            "amendment_id": horizon_amendment["amendment_id"],
            "evidence_status": horizon_amendment["evidence_status"],
            "superseded_campaign": horizon_amendment["superseded_campaign"][
                "namespace"
            ],
            "replacement_campaign": horizon_amendment["replacement_campaign"],
            "support_requirement": horizon_amendment["amended_scale_rule"][
                "support_requirement"
            ],
            "unobserved_tail_policy": horizon_amendment[
                "amended_scale_rule"
            ]["unobserved_tail_policy"],
        },
    }


def _bdb2020_harmonized_amendment_binding(
    root: Path,
    task_id: str,
) -> dict[str, Any] | None:
    """Bind the retrospective amendment only to its separate new task."""

    if task_id != BDB2020_HARMONIZED_TASK_ID:
        return None
    path = Path(root).resolve() / BDB2020_HARMONIZED_AMENDMENT_PATH
    amendment = load_bdb2020_harmonized_amendment(path)
    return {
        "path": BDB2020_HARMONIZED_AMENDMENT_PATH,
        "sha256": sha256_file(path),
        "amendment_id": amendment["amendment_id"],
        "evidence_status": amendment["evidence_status"],
    }


def _environment_provenance() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "tensorflow",
        "tensorflow-metal", "keras", "venn-abers", "xgboost", "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tensorflow_runtime: dict[str, Any]
    probe = tensorflow_probe_script()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe], check=True, capture_output=True, text=True,
            env={
                **os.environ,
                **deterministic_environment(),
            },
        )
        tensorflow_runtime = parse_tensorflow_probe_output(completed.stdout)
    except (OSError, subprocess.CalledProcessError, ManifestError) as exc:
        tensorflow_runtime = {"available": False, "probe_error_type": type(exc).__name__}
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "tensorflow_runtime": tensorflow_runtime,
        "deterministic_environment": deterministic_environment(),
    }


def _development_environment_provenance(
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Strip queue hardware from the scientific development identity.

    CPU tabular tuning, GPU neural tuning, and CPU reduction/freeze must bind
    the same code/data/dependency environment. Exact physical-GPU identity is
    admitted later by the deterministic smoke preflight, not by per-family
    development receipts.
    """

    deterministic = environment.get("deterministic_environment")
    if not isinstance(deterministic, Mapping):
        raise ManifestError("development environment lacks deterministic settings")
    required = deterministic_environment()
    if any(deterministic.get(name) != expected for name, expected in required.items()):
        raise ManifestError(
            "development environment lacks the canonical deterministic settings"
        )
    packages = environment.get("packages")
    if packages is not None and not isinstance(packages, Mapping):
        raise ManifestError("development environment package registry is invalid")
    return {
        "schema_version": "bdb-development-environment-v1",
        "python": environment.get("python"),
        "implementation": environment.get("implementation"),
        "packages": {} if packages is None else dict(packages),
        "deterministic_environment": dict(deterministic),
        "hardware_scope": (
            "queue_neutral_cpu_gpu_development_identity; physical_gpu_frozen_by_preflight"
        ),
    }


def _require_tensorflow_gpu(environment: Mapping[str, Any], purpose: str) -> None:
    runtime = environment.get("tensorflow_runtime", {})
    devices = runtime.get("devices", []) if isinstance(runtime, Mapping) else []
    has_gpu = runtime.get("available") is True and any(
        isinstance(device, Mapping)
        and str(device.get("device_type", "")).upper() == "GPU"
        for device in devices
    )
    if not has_gpu:
        raise ManifestError(f"{purpose} requires a TensorFlow-visible physical GPU")


def verify_dependency_lock(root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Compare every pinned suite dependency with the executing environment."""

    path = Path(root).resolve() / "configs" / "bdb_suite" / "requirements-lock.txt"
    if not path.is_file():
        raise ManifestError(f"suite dependency lock is missing: {path}")
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ManifestError(f"invalid dependency lock line {line_number}") from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or not specifiers[0].version
            or "*" in specifiers[0].version
        ):
            raise ManifestError(f"invalid dependency lock line {line_number}")
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = requirement.name.lower()
        version = specifiers[0].version
        if name in expected:
            raise ManifestError(f"duplicate dependency lock entry for {name}")
        expected[name.lower()] = version
    installed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[name] = actual
        if actual != version:
            mismatches[name] = {"expected": version, "observed": actual}
    if require_complete and mismatches:
        raise ManifestError(f"suite dependency lock is not satisfied: {mismatches}")
    return {
        "path": path.relative_to(Path(root).resolve()).as_posix(),
        "sha256": sha256_file(path),
        "expected": expected,
        "installed": installed,
        "mismatches": mismatches,
    }


def build_development_provenance(
    draft_path: str | Path,
    prepared_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind development tuning to exact code, data, and environment bytes.

    This receipt deliberately does not consult version-control state.  It is
    recomputed during ``freeze`` so a tuning result cannot be carried across
    a changed TaskSpec, prepared artifact, implementation, dependency set, or
    execution environment.
    """

    root = Path(repo_root).resolve()
    draft_target = Path(draft_path).resolve()
    prepared_root = Path(prepared_dir).resolve()
    prepared_receipt_path = prepared_root / "receipt.json"
    _, draft_relative = _repo_relative(
        draft_target, root, "development TaskSpec"
    )
    _, prepared_relative = _repo_relative(
        prepared_root, root, "development prepared artifact"
    )

    spec = load_task_spec(draft_target, repo_root=root, require_source=False)
    prepared = load_prepared_receipt(prepared_root, verify_files=True)
    if prepared.get("task_id") != spec.task_id:
        raise ManifestError("development TaskSpec and prepared artifact differ")
    prepared_task = load_prepared_task(
        prepared_root, mmap_mode="r", verify_files=False
    )
    semantic = validate_prepared_binding(
        spec,
        prepared,
        prepared_task,
        require_frozen_identity=False,
    )
    current_registry = build_game_registry(
        spec, game_records_from_prepared(prepared_task)
    )
    shared_registry_peer = build_shared_prepared_registry_binding(
        spec,
        current_registry,
        repo_root=root,
    )
    amendment = _protocol_amendment_binding(root)
    bdb2020_harmonized_amendment = _bdb2020_harmonized_amendment_binding(
        root, spec.task_id
    )
    code = _code_receipts(root)
    dependency_lock = verify_dependency_lock(root, require_complete=True)
    environment = _development_environment_provenance(_environment_provenance())
    for name, expected in environment["deterministic_environment"].items():
        if os.environ.get(name) != expected:
            raise ManifestError(
                "development tuning requires the declared deterministic environment: "
                f"{name}={expected!r}, observed={os.environ.get(name)!r}"
            )
    provenance = {
        "schema_version": DEVELOPMENT_PROVENANCE_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "task_spec": {
            "path": draft_relative,
            "task_spec_hash": spec.spec_hash,
            "file_sha256": sha256_file(draft_target),
        },
        "prepared": {
            "path": prepared_relative,
            "prepared_hash": prepared["prepared_hash"],
            "receipt_sha256": sha256_file(prepared_receipt_path),
            "semantic_hash": semantic["semantic_hash"],
            "game_registry_hash": current_registry["registry_hash"],
        },
        "code": code,
        "code_identity_sha256": _json_hash({"code": code}),
        "protocol_amendment": amendment,
        "dependency_lock": dependency_lock,
        "environment": environment,
    }
    if bdb2020_harmonized_amendment is not None:
        provenance[
            "bdb2020_harmonized_amendment"
        ] = bdb2020_harmonized_amendment
    if shared_registry_peer is not None:
        provenance["shared_registry_peer"] = shared_registry_peer
    return provenance


def build_primary_reference_binding(
    primary_run_dir: str | Path,
    task_spec: TaskSpec | Mapping[str, Any],
    sensitivity_design: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind a sensitivity20 design to its finalized full100 primary panel."""

    root = Path(repo_root).resolve()
    run_root, run_relative = _repo_relative(
        primary_run_dir, root, "sensitivity primary-reference run"
    )
    spec = (
        task_spec
        if isinstance(task_spec, TaskSpec)
        else TaskSpec.from_mapping(task_spec, require_source_receipts=False)
    )
    try:
        storage = GenericRunStorage.open(run_root)
    except Exception as exc:
        raise ManifestError(f"could not open sensitivity primary reference: {exc}") from exc
    primary = storage.manifest
    if (
        primary.get("schema_version") != RUN_SCHEMA_VERSION
        or primary.get("manifest_hash") != manifest_sha256(primary)
        or primary.get("task_id") != spec.task_id
        or primary.get("task_spec_hash") != spec.spec_hash
        or primary.get("execution", {}).get("mode") != "definitive"
        or primary.get("execution", {}).get("profile") != "full100"
    ):
        raise ManifestError(
            "sensitivity primary reference must be the matching definitive full100 run"
        )
    primary_design = primary.get("task_design")
    if not isinstance(primary_design, Mapping) or primary_design.get("cell_counts") != {
        "primary": 3000,
        "structural_ablation": 80,
        "frozen_sensitivity": 0,
        "required": 3080,
    }:
        raise ManifestError("sensitivity primary reference has the wrong full100 grid")
    required_keys = cell_keys_from_design(primary_design)
    if len(required_keys) != 3080 or not storage.validate_final(required_keys):
        raise ManifestError("sensitivity primary reference is not finalized over 3080 cells")
    receipt_path = run_root / "final" / "receipt.json"
    primary_metrics_path = run_root / "final" / "primary_metrics.csv"
    success_path = run_root / "_SUCCESS"
    if not validate_checksum(receipt_path) or not validate_checksum(primary_metrics_path):
        raise ManifestError("sensitivity primary aggregate artifacts failed checksums")
    receipt = _load_json(receipt_path)
    if (
        receipt.get("task_id") != spec.task_id
        or receipt.get("manifest_hash") != storage.manifest_hash
        or receipt.get("cells") != 3080
        or receipt.get("primary_cells") != 3000
        or receipt.get("ablation_cells") != 80
        or receipt.get("sensitivity_cells") != 0
        or receipt.get("primary_summary_groups") != 30
    ):
        raise ManifestError("sensitivity primary aggregate receipt is incomplete")
    if canonical_json(primary_design.get("game_registry")) != canonical_json(
        sensitivity_design.get("game_registry")
    ):
        raise ManifestError("sensitivity and primary game registries differ")
    sensitivity_splits = sensitivity_design.get("split_manifests")
    primary_splits = primary_design.get("split_manifests")
    if (
        not isinstance(sensitivity_splits, list)
        or not isinstance(primary_splits, list)
        or primary_splits[: len(sensitivity_splits)] != sensitivity_splits
    ):
        raise ManifestError("sensitivity and primary repeat splits differ")
    primary_configs = {
        model_id: entry.get("selected_config")
        for model_id, entry in primary.get("task_spec", {}).get("models", {}).items()
    }
    sensitivity_configs = {
        model_id: entry.get("selected_config")
        for model_id, entry in spec.models.items()
    }
    if canonical_json(primary_configs) != canonical_json(sensitivity_configs):
        raise ManifestError("sensitivity and primary frozen model configurations differ")

    primary_cells = {
        (int(cell["repeat"]), int(cell["n_train"]), str(cell["model"])): cell
        for cell in primary_design.get("primary_cells", [])
    }
    pairs: list[dict[str, Any]] = []
    for cell in sensitivity_design.get("sensitivity_cells", []):
        identity = (int(cell["repeat"]), int(cell["n_train"]), str(cell["model"]))
        reference = primary_cells.get(identity)
        if reference is None:
            raise ManifestError(f"sensitivity cell lacks a primary pair: {identity}")
        for field in (
            "role", "outer_split_hash", "nested_split_hash", "seeds"
        ):
            if canonical_json(cell.get(field)) != canonical_json(reference.get(field)):
                raise ManifestError(
                    f"sensitivity pair differs outside its intervention: {identity}.{field}"
                )
        pairs.append(
            {
                "repeat": identity[0],
                "n_train": identity[1],
                "model": identity[2],
                "outer_split_hash": str(cell["outer_split_hash"]),
                "nested_split_hash": str(cell["nested_split_hash"]),
                "seeds_sha256": sha256_json(cell["seeds"]),
            }
        )
    if len(pairs) != 200:
        raise ManifestError("sensitivity primary pairing must contain exactly 200 cells")
    return {
        "schema_version": PRIMARY_REFERENCE_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "run_dir": run_relative,
        "manifest_hash": storage.manifest_hash,
        "manifest_file_sha256": sha256_file(run_root / "manifest.json"),
        "final_marker_sha256": sha256_file(success_path),
        "aggregate_receipt_sha256": sha256_file(receipt_path),
        "primary_metrics_sha256": sha256_file(primary_metrics_path),
        "task_spec_hash": spec.spec_hash,
        "prepared_hash": str(primary.get("prepared", {}).get("prepared_hash")),
        "game_registry_hash": str(
            primary_design.get("game_registry", {}).get("registry_hash")
        ),
        "base_seed": int(primary_design.get("base_seed")),
        "selected_configs_sha256": sha256_json(primary_configs),
        "paired_cells": len(pairs),
        "paired_cells_sha256": sha256_json(pairs),
    }


def validate_primary_reference_binding(
    binding: Mapping[str, Any],
    task_spec: TaskSpec | Mapping[str, Any],
    sensitivity_design: Mapping[str, Any],
    *,
    repo_root: str | Path = ".",
) -> None:
    """Recompute a sensitivity's external primary reference and reject drift."""

    if not isinstance(binding, Mapping) or binding.get("schema_version") != (
        PRIMARY_REFERENCE_SCHEMA_VERSION
    ):
        raise ManifestError("sensitivity primary-reference binding is missing or invalid")
    root = Path(repo_root).resolve()
    run_dir = root / str(binding.get("run_dir", ""))
    observed = build_primary_reference_binding(
        run_dir, task_spec, sensitivity_design, repo_root=root
    )
    if canonical_json(binding) != canonical_json(observed):
        raise ManifestError("sensitivity primary-reference binding drifted")


def build_run_manifest(
    frozen_task_path: str | Path,
    prepared_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    repeats: int | None = None,
    profile: str = "full50",
    cpu_workers: int | None = None,
    smoke: bool = False,
    benchmark: bool = False,
    preflight_receipt: str | Path | None = None,
    runtime_plan: str | Path | None = None,
    primary_reference_run: str | Path | None = None,
) -> dict[str, Any]:
    """Create a full task manifest; callers persist it via RunStorage."""

    root = Path(repo_root).resolve()
    frozen_target, _ = _repo_relative(
        frozen_task_path, root, "frozen TaskSpec"
    )
    prepared_root, _ = _repo_relative(
        prepared_dir, root, "prepared artifact"
    )
    if smoke and benchmark:
        raise ManifestError("smoke and benchmark planning are mutually exclusive")
    try:
        execution_profile = EXECUTION_PROFILES[profile]
    except KeyError as exc:
        raise ManifestError(f"unknown execution profile {profile!r}") from exc
    if preflight_receipt is not None:
        preflight_target = Path(preflight_receipt)
        if not preflight_target.is_absolute():
            preflight_target = root / preflight_target
        preflight_target, _ = _repo_relative(
            preflight_target, root, "preflight receipt"
        )
    else:
        preflight_target = None
    if runtime_plan is not None:
        runtime_target = Path(runtime_plan)
        if not runtime_target.is_absolute():
            runtime_target = root / runtime_target
        runtime_target, _ = _repo_relative(runtime_target, root, "runtime plan")
    else:
        runtime_target = None
    external_gpu_lanes = 4
    if runtime_target is not None:
        try:
            runtime_schema = _load_json(runtime_target).get("schema_version")
        except (ManifestError, AttributeError):
            runtime_schema = None
        if runtime_schema in {
            RUNTIME_PLAN_V3_SCHEMA_VERSION,
            RUNTIME_PLAN_V4_SCHEMA_VERSION,
        }:
            external_gpu_lanes = BETTY_PHASED_GPU_LANES
    if primary_reference_run is not None:
        primary_reference_target = Path(primary_reference_run)
        if not primary_reference_target.is_absolute():
            primary_reference_target = root / primary_reference_target
        primary_reference_target, _ = _repo_relative(
            primary_reference_target, root, "sensitivity primary-reference run"
        )
    else:
        primary_reference_target = None
    if smoke:
        repeats = 1 if repeats is None else repeats
        if repeats != 1:
            raise ManifestError("smoke manifests must contain exactly one repeat")
        if preflight_receipt is not None:
            raise ManifestError("a smoke manifest cannot consume its own preflight receipt")
        if runtime_plan is not None:
            raise ManifestError("a smoke manifest cannot consume a runtime plan")
        if primary_reference_target is not None:
            raise ManifestError("a smoke manifest cannot consume a primary reference")
        cpu_workers = 1 if cpu_workers is None else int(cpu_workers)
        design_profile = None
        mode = "smoke"
    elif benchmark:
        repeats = 1 if repeats is None else repeats
        if repeats != 1:
            raise ManifestError("benchmark manifests must contain exactly one repeat")
        if preflight_receipt is None:
            raise ManifestError("benchmark planning requires a preflight receipt")
        if runtime_plan is not None:
            raise ManifestError("a benchmark manifest cannot consume its own runtime plan")
        if primary_reference_target is not None:
            raise ManifestError("a benchmark manifest cannot consume a primary reference")
        if cpu_workers is None:
            try:
                preflight_value = _load_json(preflight_target)
                cpu_workers = int(preflight_value["memory"]["recommended_cpu_workers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestError("preflight receipt has no recommended CPU concurrency") from exc
        design_profile = None
        mode = "benchmark"
    else:
        repeats = execution_profile.repeats if repeats is None else repeats
        if repeats != execution_profile.repeats:
            raise ManifestError(
                f"profile {profile!r} requires exactly {execution_profile.repeats} repeats"
            )
        if preflight_receipt is None:
            raise ManifestError(
                "definitive planning requires a deterministic smoke/memory preflight receipt"
            )
        if cpu_workers is None:
            try:
                preflight_value = _load_json(preflight_target)
                cpu_workers = int(preflight_value["memory"]["recommended_cpu_workers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ManifestError("preflight receipt has no recommended CPU concurrency") from exc
        if profile in {"pilot10", "full100", "sensitivity20"} and runtime_plan is None:
            raise ManifestError(
                f"{profile} planning requires a checksummed runtime plan"
            )
        if profile not in {"pilot10", "full100", "sensitivity20"} and runtime_plan is not None:
            raise ManifestError(f"profile {profile!r} does not consume a runtime plan")
        if profile == "sensitivity20" and primary_reference_target is None:
            raise ManifestError(
                "sensitivity20 planning requires --primary-reference-run"
            )
        if profile != "sensitivity20" and primary_reference_target is not None:
            raise ManifestError(
                "only sensitivity20 planning accepts a primary-reference run"
            )
        design_profile = profile
        mode = "pilot" if profile == "pilot10" else "definitive"
    spec = load_task_spec(frozen_target, repo_root=root, require_source=False)
    validate_task_scientific_receipts(spec, repo_root=root)
    if profile in {"pilot10", "full100"} and not smoke and not benchmark and spec.task_id == "bdb2026_trajectory":
        for model_id, entry in spec.models.items():
            try:
                validate_horizon_scale_config(entry["selected_config"])
            except (KeyError, TaskSpecError) as exc:
                raise ManifestError(
                    f"BDB2026 {profile} model {model_id} lacks its frozen 94-horizon scale"
                ) from exc
    if not isinstance(spec.cohort.get("game_registry"), Mapping):
        raise ManifestError(
            "frozen TaskSpec has no exact cohort.game_registry; freeze it from "
            "the sequestered development design before planning"
        )
    shared_registry_peer = build_shared_frozen_registry_binding(spec, repo_root=root)
    prepared_receipt = load_prepared_receipt(prepared_root, verify_files=True)
    if prepared_receipt["task_id"] != spec.task_id:
        raise ManifestError("prepared task and frozen TaskSpec differ")
    expected_variant = "primary"
    sensitivity_contract_mode = profile == "sensitivity20"
    sensitivity_mode = sensitivity_contract_mode and not smoke and not benchmark
    if sensitivity_contract_mode:
        frozen_sensitivities = [
            item for item in spec.sensitivities
            if item["selection"] == "frozen_prespecified"
        ]
        if len(frozen_sensitivities) != 1:
            raise ManifestError(
                "sensitivity20 requires exactly one frozen sensitivity contract"
            )
        expected_variant = frozen_sensitivities[0]["execution"]["prepared_variant"]
        semantic_receipt = prepared_receipt.get("semantic_receipt", {})
        adapter_metadata = (
            semantic_receipt.get("adapter_metadata", {})
            if isinstance(semantic_receipt, Mapping)
            else {}
        )
        observed_variant = (
            adapter_metadata.get("cutoff_policy")
            if spec.task_id == "bdb2024_tackle"
            else adapter_metadata.get("prepared_variant", "primary")
        )
        if observed_variant != expected_variant:
            raise ManifestError(
                "sensitivity20 prepared bundle variant differs from its frozen contract: "
                f"expected={expected_variant!r}, observed={observed_variant!r}"
            )
    task = load_prepared_task(prepared_root, mmap_mode="r", verify_files=False)
    prepared_semantics = validate_prepared_binding(
        spec,
        prepared_receipt,
        task,
        require_frozen_identity=not sensitivity_contract_mode,
        prepared_contract_override=(
            prepared_contract_for_variant(spec, expected_variant)
            if sensitivity_contract_mode
            else None
        ),
    )
    games = game_records_from_prepared(task)
    design = build_task_design(
        spec,
        games,
        repeats=repeats,
        profile=design_profile,
        seed_profile=(
            "pilot10"
            if profile == "pilot10" and design_profile is None
            else None
        ),
    )
    validate_task_design(design, spec, profile=design_profile)
    primary_reference = (
        build_primary_reference_binding(
            primary_reference_target, spec, design, repo_root=root
        )
        if sensitivity_mode
        else None
    )
    task_receipt_path = frozen_target
    prepared_receipt_path = prepared_root / "receipt.json"
    amendment = _protocol_amendment_binding(root)
    bdb2020_harmonized_amendment = _bdb2020_harmonized_amendment_binding(
        root, spec.task_id
    )
    code_receipts = _code_receipts(root)
    environment = _environment_provenance()
    _require_tensorflow_gpu(environment, "real-data smoke and definitive planning")
    dependency_lock = verify_dependency_lock(root, require_complete=True)
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "task_spec": spec.as_dict(),
        "task_spec_hash": spec.spec_hash,
        "task_spec_receipt": {
            "path": task_receipt_path.relative_to(root).as_posix(),
            "sha256": sha256_file(task_receipt_path),
        },
        "prepared": {
            "path": prepared_root.relative_to(root).as_posix(),
            "prepared_hash": prepared_receipt["prepared_hash"],
            "semantic_hash": prepared_semantics["semantic_hash"],
            "receipt_sha256": sha256_file(prepared_receipt_path),
        },
        "task_design": design,
        "required_cells": design["required_cells"],
        "primary_required_cells": design["primary_cells"],
        "ablation_required_cells": design["ablation_cells"],
        "sensitivity_required_cells": design["sensitivity_cells"],
        "execution": queue_manifest(
            cpu_workers=int(cpu_workers),
            gpu_workers=1,
            include_set_transformer=profile != "full50",
        ),
        "analysis": {
            "bootstrap_draws": 0 if profile == "pilot10" else 10_000,
            "bootstrap_seed": design["seed_registry"][
                json.dumps(
                    list(
                        analysis_seed_parts(
                            spec.task_id,
                            design.get("seed_profile", design_profile),
                        )
                    ),
                    separators=(",", ":"),
                )
            ],
            "max_t_family": (
                "none_pilot_descriptive_only"
                if profile == "pilot10"
                else "six_model_pairs_by_all_six_anchors_within_task"
                if profile == "full50"
                else "ten_model_pairs_by_all_six_anchors_within_task"
            ),
        },
        "provenance": {
            "code": code_receipts,
            "protocol_amendment": amendment,
            "environment": environment,
            "dependency_lock": dependency_lock,
        },
    }
    if bdb2020_harmonized_amendment is not None:
        manifest["provenance"][
            "bdb2020_harmonized_amendment"
        ] = bdb2020_harmonized_amendment
    manifest["execution"].update(
        {
            "mode": mode,
            "profile": profile,
            "external_gpu_lanes": (
                external_gpu_lanes
                if profile in {"pilot10", "full100", "sensitivity20"}
                else 1
            ),
        }
    )
    if profile == "pilot10":
        manifest["execution"].update(
            {
                "evidence_status": "exploratory_provisional",
                "inference_scope": "descriptive_only_no_confirmatory_inference",
            }
        )
    if shared_registry_peer is not None:
        manifest["shared_registry_peer"] = shared_registry_peer
    if primary_reference is not None:
        manifest["primary_reference"] = primary_reference
    if smoke:
        # The same five cells establish byte-exact determinism and the worker
        # peak used by the RAM concurrency rule.  Measure that peak at the
        # largest scientific workload, not at the cheapest anchor.
        resource_anchor = int(max(spec.anchors))
        frozen_sensitivity = (
            next(
                item for item in spec.sensitivities
                if item["selection"] == "frozen_prespecified"
            )
            if profile == "sensitivity20"
            else None
        )
        intervention = (
            {
                "branch": "frozen_sensitivity",
                "sensitivity_id": str(frozen_sensitivity["id"]),
                "prepared_variant": str(
                    frozen_sensitivity["execution"]["prepared_variant"]
                ),
            }
            if frozen_sensitivity is not None
            else None
        )
        smoke_cells = [
            {
                "branch": (
                    intervention["branch"] if intervention is not None
                    else record["branch"]
                ),
                "source_branch": record["branch"],
                "ablation_id": record.get("ablation_id"),
                "sensitivity_id": (
                    intervention["sensitivity_id"]
                    if intervention is not None
                    else record.get("sensitivity_id")
                ),
                "repeat": int(record["repeat"]),
                "n_train": int(record["n_train"]),
                "model": record["model"],
                "queue": record["queue"],
            }
            for record in design["required_cells"]
            if int(record["repeat"]) == 1
            and int(record["n_train"]) == resource_anchor
            and not (
                profile == "full50" and record["model"] == "set_transformer"
            )
        ]
        expected_smoke_cells = 4 if profile == "full50" else 5
        if len(smoke_cells) != expected_smoke_cells:
            raise ManifestError(
                "smoke projection did not resolve its profile-bound model panel"
            )
        manifest["execution"]["smoke"] = {
            "enabled": True,
            "cells": smoke_cells,
            "neural_max_epochs": 1,
            "neural_patience": 0,
            "resource_anchor": resource_anchor,
            "intervention": intervention,
            "purpose": "real_data_one_repeat_one_epoch_determinism_and_memory_preflight",
        }
        manifest["execution"]["benchmark"] = {"enabled": False}
    elif benchmark:
        benchmark_anchors = (
            {
                int(value)
                for item in spec.sensitivities
                if item["selection"] == "frozen_prespecified"
                for value in item["execution"]["anchors"]
            }
            if profile == "sensitivity20"
            else {int(min(spec.anchors)), int(max(spec.anchors))}
        )
        frozen_sensitivity = (
            next(
                item for item in spec.sensitivities
                if item["selection"] == "frozen_prespecified"
            )
            if profile == "sensitivity20"
            else None
        )
        intervention = (
            {
                "branch": "frozen_sensitivity",
                "sensitivity_id": str(frozen_sensitivity["id"]),
                "prepared_variant": str(
                    frozen_sensitivity["execution"]["prepared_variant"]
                ),
            }
            if frozen_sensitivity is not None
            else None
        )
        benchmark_cells = [
            {
                "branch": (
                    intervention["branch"] if intervention is not None
                    else record["branch"]
                ),
                "source_branch": record["branch"],
                "ablation_id": record.get("ablation_id"),
                "sensitivity_id": (
                    intervention["sensitivity_id"]
                    if intervention is not None
                    else record.get("sensitivity_id")
                ),
                "repeat": int(record["repeat"]),
                "n_train": int(record["n_train"]),
                "model": record["model"],
                "queue": record["queue"],
            }
            for record in design["primary_cells"]
            if int(record["repeat"]) == 1
            and int(record["n_train"]) in benchmark_anchors
            and not (
                profile == "full50" and record["model"] == "set_transformer"
            )
        ]
        expected_benchmark_cells = 8 if profile == "full50" else 10
        if len(benchmark_cells) != expected_benchmark_cells:
            raise ManifestError(
                "benchmark projection did not resolve its profile-bound "
                "two-anchor model panel"
            )
        manifest["execution"]["smoke"] = {"enabled": False}
        manifest["execution"]["benchmark"] = {
            "enabled": True,
            "cells": benchmark_cells,
            "anchors": sorted(benchmark_anchors),
            "intervention": intervention,
            "purpose": "full_epoch_endpoint_runtime_bound",
        }
    else:
        manifest["execution"]["smoke"] = {"enabled": False}
        manifest["execution"]["benchmark"] = {"enabled": False}
    if not smoke:
        from .preflight import validate_preflight_receipt

        manifest["preflight"] = validate_preflight_receipt(
            preflight_target, manifest, repo_root=root
        )
    if profile in {"pilot10", "full100", "sensitivity20"} and not smoke and not benchmark:
        from .preflight import validate_runtime_plan

        manifest["runtime_plan"] = validate_runtime_plan(
            runtime_target, manifest, repo_root=root
        )
    manifest = bind_storage_backend(manifest, repo_root=root)
    return bind_run_identity(manifest)


def game_records_from_prepared(task: Any) -> Any:
    """Return one deterministic season/week record per prepared game."""

    game_columns = ["game_id", "stratum"]
    games = task.examples[game_columns].drop_duplicates("game_id").copy()
    if "season" in task.examples:
        season = task.examples[["game_id", "season"]].drop_duplicates("game_id")
        games = games.merge(season, on="game_id", how="left")
        games["season"] = pd_to_numeric_stratum(games["season"])
    else:
        extracted = games["stratum"].astype(str).str.extract(r"(?P<season>20\d{2})")["season"]
        games["season"] = pd_to_numeric_stratum(extracted)
    # A week is optional in prepared examples; deterministic stratification is
    # by season except the shared 2024/25 registry, which uses the same records.
    if "week" in task.examples:
        week = task.examples[["game_id", "week"]].drop_duplicates("game_id")
        games = games.merge(week, on="game_id", how="left")
    else:
        extracted_week = games["stratum"].astype(str).str.extract(r"[wW](?P<week>\d{1,2})")["week"]
        games["week"] = extracted_week.fillna(1).astype(int)
    return games[["game_id", "season", "week"]]


def pd_to_numeric_stratum(values: Any) -> Any:
    """Normalize season-like strata without importing pandas at module import."""

    import pandas as pd

    converted = pd.to_numeric(values, errors="raise")
    if not np.all(np.isfinite(converted)):
        raise ManifestError("game strata must be finite season values")
    return converted.astype(int)
