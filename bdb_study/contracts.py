"""Versioned scientific contracts for the cross-release BDB study.

This module deliberately depends only on the Python standard library.  A task
adapter can therefore validate and freeze its scientific receipt without
importing a modeling framework (in particular, without importing TensorFlow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


TASK_SPEC_SCHEMA_VERSION = "bdb-task-spec-v2"
TASK_SPEC_RECEIPT_VERSION = "bdb-task-spec-receipt-v1"
SOURCE_RECEIPT_SCHEMA_VERSION = "bdb-source-receipt-v1"
SOURCE_TREE_RECEIPT_SCHEMA_VERSION = "bdb-source-tree-receipt-v1"
PREPARED_SEMANTIC_SCHEMA_VERSION = "bdb-prepared-semantics-v2"
PROTOCOL_AMENDMENT_SCHEMA_VERSION = "bdb-protocol-amendment-v1"
GLOBAL_SET_TRANSFORMER_AMENDMENT_ID = "20260821_global_set_transformer_primary"
BDB2026_HORIZON_SCALE_AMENDMENT_ID = "20260821_bdb2026_horizon_scale_tail"
BDB2026_HORIZON_SCALE_AMENDMENT_CANONICAL_SHA256 = (
    "9268e98dbfcbe80b31a75e57393ab04837bcd5a18e46d91e198ef09174a52698"
)
BDB2020_HARMONIZED_AMENDMENT_SCHEMA_VERSION = (
    "bdb2020-harmonized-protocol-amendment-v1"
)
BDB2020_HARMONIZED_AMENDMENT_ID = "20260906_bdb2020_harmonized_five_role"
BDB2020_HARMONIZED_AMENDMENT_CANONICAL_SHA256 = (
    "329b8d7984ce210b78898e8964e1a39a7916b0906783b5eaef49fc2208cdb8d4"
)
CANONICAL_MODEL_ROLES = (
    "linear_structure",
    "boosted_structure",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
CANONICAL_MODEL_FAMILIES = (
    "glm",
    "lightgbm",
    "relnet",
    "attn_relnet",
    "set_transformer",
)
ROLE_TO_FAMILY = dict(zip(CANONICAL_MODEL_ROLES, CANONICAL_MODEL_FAMILIES))
OUTCOME_TYPES = ("binary", "distribution", "frame_event", "trajectory")
GLM_ALPHA_GRID = (10.0, 10.0 / 3.0, 1.0, 1.0 / 3.0, 0.1)
TRAJECTORY_GLM_ALPHA_GRID = (0.1, 0.01, 0.001, 0.0001, 0.00001)
LIGHTGBM_GRID_ID = "bdb_suite_lgbm_four_v1"
NEURAL_LEARNING_RATE_GRID = (0.001, 0.0003)
NEURAL_DROPOUT_GRID = (0.3, 0.1)
TRANSFORMER_CAPACITY = {
    "d_model": 64,
    "layers": 3,
    "heads": 2,
    "ff_dim": 256,
}
HORIZON_SCALE_SUPPORT_REQUIREMENT = (
    "pooled_positive_counts_form_nonempty_prefix_v1"
)
HORIZON_SCALE_TAIL_POLICY = "carry_forward_last_fitted_pava_level_v1"

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Base class for an invalid or conflicting scientific contract."""


class TaskSpecError(ContractError):
    """A task specification is missing or violates a scientific invariant."""


class SourceReceiptError(ContractError):
    """A data-source receipt is malformed or does not match its file."""


class FrozenReceiptCollisionError(ContractError):
    """A frozen receipt path already contains different bytes."""


class ProtocolAmendmentError(ContractError):
    """An outcome-blind protocol amendment is malformed or has drifted."""


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("Canonical JSON does not permit NaN or infinity.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("Canonical JSON object keys must be strings.")
            output[key] = _canonicalize(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _canonicalize(converted)
    raise ContractError(f"Unsupported canonical JSON value: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for hashes and byte comparisons."""

    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
    except OSError as exc:
        raise SourceReceiptError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskSpecError(f"{path} must be a JSON object.")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TaskSpecError(f"{path} must be an integer >= {minimum}.")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSpecError(f"{path} must be a nonempty string.")
    return value.strip()


def _require_string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskSpecError(f"{path} must be an array of strings.")
    output = [_require_nonempty_string(item, f"{path}[]") for item in value]
    if not allow_empty and not output:
        raise TaskSpecError(f"{path} cannot be empty.")
    if len(output) != len(set(output)):
        raise TaskSpecError(f"{path} cannot contain duplicates.")
    return output


def _require_exact_numeric_grid(
    value: Any,
    expected: Sequence[float],
    path: str,
) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskSpecError(f"{path} must be the ordered array {list(expected)}")
    observed = list(value)
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in observed
    ) or observed != list(expected):
        raise TaskSpecError(f"{path} must be the ordered array {list(expected)}")


def _validate_model_grid_declaration(
    outcome_type: str,
    model_id: str,
    config: Mapping[str, Any],
) -> None:
    """Bind TaskSpec declarations to the exact development-CV tie order."""

    family = config["family"]
    path = f"models.{model_id}"
    if family == "glm":
        expected = (
            TRAJECTORY_GLM_ALPHA_GRID
            if outcome_type == "trajectory"
            else GLM_ALPHA_GRID
        )
        _require_exact_numeric_grid(config.get("alpha_grid"), expected, f"{path}.alpha_grid")
        return
    if family == "lightgbm":
        if config.get("grid_id") != LIGHTGBM_GRID_ID:
            raise TaskSpecError(f"{path}.grid_id must be {LIGHTGBM_GRID_ID!r}")
        return
    if family in {"relnet", "attn_relnet", "set_transformer"}:
        _require_exact_numeric_grid(
            config.get("learning_rates"),
            NEURAL_LEARNING_RATE_GRID,
            f"{path}.learning_rates",
        )
        _require_exact_numeric_grid(
            config.get("dropouts"),
            NEURAL_DROPOUT_GRID,
            f"{path}.dropouts",
        )
        return
    raise TaskSpecError(f"{path}.family is not a canonical model family")


def _validate_history_contract(value: Any) -> dict[str, Any]:
    history = dict(_require_mapping(value, "history"))
    required = {
        "scope", "endpoint", "max_frames", "padding", "stable_player_slots",
        "slot_policy",
    }
    if set(history) != required:
        raise TaskSpecError(f"history requires exactly {sorted(required)}")
    if history["scope"] != "causal_tracking_history":
        raise TaskSpecError("history.scope must be 'causal_tracking_history'")
    _require_nonempty_string(history["endpoint"], "history.endpoint")
    _require_int(history["max_frames"], "history.max_frames", minimum=1)
    if history["padding"] not in {"none", "left_masked"}:
        raise TaskSpecError("history.padding must be 'none' or 'left_masked'")
    if history["stable_player_slots"] is not True:
        raise TaskSpecError("history.stable_player_slots must be true")
    if history["slot_policy"] != "play_global_focal_independent_v1":
        raise TaskSpecError(
            "history.slot_policy must be 'play_global_focal_independent_v1'"
        )
    return _canonicalize(history)


def _validate_graph_contract(value: Any) -> dict[str, Any]:
    graph = dict(_require_mapping(value, "graph"))
    required = {"node_types", "edge_types", "construction", "masked_aggregation"}
    if set(graph) != required:
        raise TaskSpecError(f"graph requires exactly {sorted(required)}")
    _require_string_list(graph["node_types"], "graph.node_types")
    _require_string_list(graph["edge_types"], "graph.edge_types")
    _require_nonempty_string(graph["construction"], "graph.construction")
    if graph["masked_aggregation"] != "observed_nodes_and_edges_only":
        raise TaskSpecError(
            "graph.masked_aggregation must be 'observed_nodes_and_edges_only'"
        )
    return _canonicalize(graph)


def _validate_identity_policy(value: Any) -> dict[str, Any]:
    policy = dict(_require_mapping(value, "identity_policy"))
    required = {
        "player_ids",
        "player_id_feature",
        "team_identity_primary",
        "team_identity_sensitivity",
    }
    if set(policy) != required:
        raise TaskSpecError(f"identity_policy requires exactly {sorted(required)}")
    if policy["player_ids"] != "stable_slot_assignment_only":
        raise TaskSpecError(
            "identity_policy.player_ids must be 'stable_slot_assignment_only'"
        )
    if policy["player_id_feature"] is not False:
        raise TaskSpecError("player IDs cannot enter a primary model as features")
    if policy["team_identity_primary"] is not False:
        raise TaskSpecError("team identity must be excluded from primary models")
    if not isinstance(policy["team_identity_sensitivity"], bool):
        raise TaskSpecError("team_identity_sensitivity must be boolean")
    return _canonicalize(policy)


def _validate_model_role_contract(value: Any) -> dict[str, Any]:
    contract = dict(_require_mapping(value, "model_roles"))
    required = {
        "primary",
        "shared_tabular_summary",
        "shared_neural_inputs",
        "relational_pair_shared_inputs",
        "set_transformer_structure",
        "parameter_match_tolerance_fraction",
        "flop_match_tolerance_fraction",
        "neural_parameter_cap",
    }
    if set(contract) != required:
        raise TaskSpecError(f"model_roles requires exactly {sorted(required)}")
    primary = _require_string_list(contract["primary"], "model_roles.primary")
    if tuple(primary) != CANONICAL_MODEL_ROLES:
        raise TaskSpecError(
            f"model_roles.primary must be ordered as {list(CANONICAL_MODEL_ROLES)}"
        )
    if contract["shared_tabular_summary"] is not True:
        raise TaskSpecError("the two tabular roles must share one summary table")
    shared = _require_string_list(
        contract["shared_neural_inputs"], "model_roles.shared_neural_inputs"
    )
    if shared != [
        "tokenization",
        "causal_history",
        "player_and_frame_masks",
        "time_to_event",
        "context",
        "output_head",
        "loss",
        "tuning_grid",
        "seeds",
        "training_protocol",
    ]:
        raise TaskSpecError("model_roles.shared_neural_inputs is not the frozen list")
    relational_shared = _require_string_list(
        contract["relational_pair_shared_inputs"],
        "model_roles.relational_pair_shared_inputs",
    )
    if relational_shared != ["graph_edges", "relational_temporal_encoder"]:
        raise TaskSpecError(
            "model_roles.relational_pair_shared_inputs is not the frozen list"
        )
    set_structure = dict(
        _require_mapping(
            contract["set_transformer_structure"],
            "model_roles.set_transformer_structure",
        )
    )
    expected_set_structure = {
        "relation_scope": "global_masked_all_player_self_attention",
        "typed_graph_edges_consumed": False,
        "temporal_encoder": "factorized_masked_temporal_attention",
    }
    if set_structure != expected_set_structure:
        raise TaskSpecError("model_roles.set_transformer_structure drifted")
    for field, expected in (
        ("parameter_match_tolerance_fraction", 0.05),
        ("flop_match_tolerance_fraction", 0.15),
    ):
        observed = contract[field]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise TaskSpecError(f"model_roles.{field} must be numeric")
        if float(observed) != expected:
            raise TaskSpecError(f"model_roles.{field} must be {expected}")
    if _require_int(
        contract["neural_parameter_cap"],
        "model_roles.neural_parameter_cap",
        minimum=1,
    ) != 350_000:
        raise TaskSpecError("model_roles.neural_parameter_cap must be 350000")
    return _canonicalize(contract)


def trajectory_output_head_contract(
    selected_target: str | None = None,
) -> dict[str, Any]:
    """Return the prospective or exactly frozen shared-neural trajectory head."""

    if selected_target is None:
        return {
            "id": "horizon_development_selected_xy_v1",
            "scope": "matched_neural_models",
            "prediction": (
                "masked x/y in residual or absolute coordinates as selected on "
                "development games"
            ),
            "training_loss": "development_selected_masked_coordinate_mse",
            "decoder": "shared_horizon_conditioned",
            "selected_training_target": None,
            "reconstruction": (
                "absolute_identity_or_constant_velocity_plus_residual_as_selected"
            ),
        }
    if selected_target == "residual":
        return {
            "id": "horizon_residual_xy_v1",
            "scope": "matched_neural_models",
            "prediction": (
                "masked x/y residual from constant velocity for each requested horizon"
            ),
            "training_loss": "masked_residual_coordinate_mse",
            "decoder": "shared_horizon_conditioned",
            "selected_training_target": "residual",
            "reconstruction": "constant_velocity_baseline_plus_residual",
        }
    if selected_target == "absolute":
        return {
            "id": "horizon_absolute_xy_v1",
            "scope": "matched_neural_models",
            "prediction": "masked official absolute x/y for each requested horizon",
            "training_loss": "masked_absolute_coordinate_mse",
            "decoder": "shared_horizon_conditioned",
            "selected_training_target": "absolute",
            "reconstruction": (
                "direct_absolute_prediction_then_constant_velocity_residualization_"
                "for_common_evaluation"
            ),
        }
    raise TaskSpecError("trajectory selected target must be residual, absolute, or null")


def _validate_output_head(value: Any, outcome_type: str) -> dict[str, Any]:
    head = dict(_require_mapping(value, "output_head"))
    required = {"id", "prediction", "training_loss"}
    if not required <= set(head):
        raise TaskSpecError(f"output_head must include {sorted(required)}")
    _require_nonempty_string(head["id"], "output_head.id")
    _require_nonempty_string(head["prediction"], "output_head.prediction")
    _require_nonempty_string(head["training_loss"], "output_head.training_loss")
    expected_id = {
        "binary": "binary_probability_v1",
        "frame_event": "binary_probability_v1",
        "distribution": "punt_cdf_residual_v1",
        "trajectory": None,
    }[outcome_type]
    if outcome_type == "trajectory":
        allowed = [
            trajectory_output_head_contract(None),
            trajectory_output_head_contract("residual"),
            trajectory_output_head_contract("absolute"),
        ]
        if head not in allowed:
            raise TaskSpecError(
                "trajectory output_head differs from the prospective/frozen decoder contract"
            )
    elif head["id"] != expected_id:
        raise TaskSpecError(f"output_head.id must be {expected_id!r}")
    return _canonicalize(head)


def _validate_uncertainty_contract(value: Any, outcome_type: str) -> dict[str, Any]:
    contract = dict(_require_mapping(value, "uncertainty"))
    required = {
        "primary_method",
        "secondary_diagnostic",
        "secondary_score_source",
        "field_intersection",
        "horizon_scale_learner",
        "calibration_unit",
        "nominal_coverage",
        "report",
    }
    if set(contract) != required:
        raise TaskSpecError(f"uncertainty requires exactly {sorted(required)}")
    expected_primary = {
        "binary": "out_of_game_venn_abers_calibration_v1",
        "frame_event": "out_of_game_venn_abers_calibration_v1",
        "distribution": "game_clustered_central_interval_padding_v1",
        "trajectory": "whole_path_scaled_radial_conformal_tube_v1",
    }[outcome_type]
    expected_secondary = (
        "raw_score_class_conditional_conformal_sets_v1"
        if outcome_type in {"binary", "frame_event"}
        else None
    )
    expected_secondary_source = (
        "raw_model_score" if outcome_type in {"binary", "frame_event"} else None
    )
    expected_field_intersection = (
        "disk_intersect_legal_field_rectangle"
        if outcome_type == "trajectory"
        else None
    )
    if contract["primary_method"] != expected_primary:
        raise TaskSpecError(
            f"uncertainty.primary_method must be {expected_primary!r}"
        )
    if contract["secondary_diagnostic"] != expected_secondary:
        raise TaskSpecError(
            f"uncertainty.secondary_diagnostic must be {expected_secondary!r}"
        )
    if contract["secondary_score_source"] != expected_secondary_source:
        raise TaskSpecError(
            "uncertainty.secondary_score_source must preserve raw-score "
            "split-conformal separation"
        )
    if contract["field_intersection"] != expected_field_intersection:
        raise TaskSpecError(
            "uncertainty.field_intersection differs from the frozen output geometry"
        )
    expected_horizon_scale_learner = (
        {
            "source": "five_grouped_oof_development_game_folds",
            "fold_statistic": "horizon_radial_error_median_and_observed_path_count",
            "pooling": "count_weighted_median_across_folds",
            "isotonic_projection": "observed_count_weighted_nondecreasing_pava",
            "support_requirement": HORIZON_SCALE_SUPPORT_REQUIREMENT,
            "unobserved_tail_policy": HORIZON_SCALE_TAIL_POLICY,
            "floor_yards": 0.25,
            "horizon_count": 94,
            "artifact_fields": [
                "dev_horizon_scale",
                "dev_horizon_scale_sha256",
                "dev_horizon_scale_observed_through",
                "dev_horizon_scale_support_requirement",
                "dev_horizon_scale_tail_policy",
            ],
        }
        if outcome_type == "trajectory"
        else None
    )
    if contract["horizon_scale_learner"] != expected_horizon_scale_learner:
        raise TaskSpecError(
            "uncertainty.horizon_scale_learner differs from the frozen grouped-OOF rule"
        )
    if contract["calibration_unit"] != "game":
        raise TaskSpecError("uncertainty.calibration_unit must be 'game'")
    coverage = contract["nominal_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or float(coverage) != 0.9:
        raise TaskSpecError("uncertainty.nominal_coverage must be 0.9")
    _require_string_list(contract["report"], "uncertainty.report")
    return _canonicalize(contract)


def _validate_joint_neural_target_contract(
    value: Any, outcome_type: str
) -> dict[str, Any] | None:
    """Validate the joint BDB2026 decoder-target selection contract."""

    if outcome_type != "trajectory":
        if value is not None:
            raise TaskSpecError(
                "joint_neural_target_selection must be null outside trajectory tasks"
            )
        return None
    contract = dict(
        _require_mapping(value, "joint_neural_target_selection")
    )
    required = {
        "schema_version",
        "families",
        "candidate_targets",
        "criterion",
        "tie_rule",
        "receipt",
    }
    if set(contract) != required:
        raise TaskSpecError(
            "joint_neural_target_selection requires exactly "
            f"{sorted(required)}"
        )
    if contract["schema_version"] != "bdb-joint-neural-target-contract-v1":
        raise TaskSpecError("joint neural target contract schema differs")
    expected_families = ["relnet", "attn_relnet", "set_transformer"]
    if contract["families"] != expected_families:
        raise TaskSpecError(
            "joint neural target contract must cover RelNet, AttnRelNet, and "
            "Set Transformer"
        )
    if contract["candidate_targets"] != ["residual", "absolute"]:
        raise TaskSpecError("joint neural target candidate order differs")
    if (
        contract["criterion"]
        != "pooled_equal_model_weight_grouped_oof_mean_rmse_v1"
        or contract["tie_rule"] != "first_target_in_frozen_order"
    ):
        raise TaskSpecError("joint neural target selection rule differs")
    binding = contract["receipt"]
    if binding is None:
        return _canonicalize(contract)
    receipt = dict(
        _require_mapping(binding, "joint_neural_target_selection.receipt")
    )
    receipt_fields = {
        "schema_version",
        "protocol",
        "path",
        "file_sha256",
        "receipt_hash",
        "selected_target",
        "source_development_receipt_hashes",
    }
    if set(receipt) != receipt_fields:
        raise TaskSpecError(
            "joint neural target receipt binding requires exactly "
            f"{sorted(receipt_fields)}"
        )
    if (
        receipt["schema_version"]
        != "bdb-joint-neural-trajectory-target-v1"
        or receipt["protocol"] != contract["criterion"]
        or receipt["selected_target"] not in contract["candidate_targets"]
    ):
        raise TaskSpecError("joint neural target receipt protocol differs")
    path = PurePosixPath(
        _require_nonempty_string(
            receipt["path"], "joint_neural_target_selection.receipt.path"
        )
    )
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise TaskSpecError("joint neural target receipt path must be repository-relative")
    for field in ("file_sha256", "receipt_hash"):
        observed = receipt[field]
        if not isinstance(observed, str) or not _SHA256.fullmatch(observed):
            raise TaskSpecError(
                f"joint_neural_target_selection.receipt.{field} must be SHA-256"
            )
    source_hashes = _require_mapping(
        receipt["source_development_receipt_hashes"],
        "joint_neural_target_selection.receipt.source_development_receipt_hashes",
    )
    if set(source_hashes) != {"relnet", "attn_relnet", "set_transformer"} or any(
        not isinstance(item, str) or not _SHA256.fullmatch(item)
        for item in source_hashes.values()
    ):
        raise TaskSpecError(
            "joint neural target receipt must bind all three raw neural development hashes"
        )
    return _canonicalize(contract)


def _validate_structural_ablation(
    value: Any,
    *,
    anchors: Sequence[int],
) -> dict[str, Any]:
    ablation = dict(_require_mapping(value, "structural_ablation"))
    required = {"id", "description", "models", "anchors", "repeats"}
    if set(ablation) != required:
        raise TaskSpecError(f"structural_ablation requires exactly {sorted(required)}")
    ablation_id = _require_nonempty_string(ablation["id"], "structural_ablation.id")
    if not _SLUG.fullmatch(ablation_id):
        raise TaskSpecError("structural_ablation.id must be a lowercase path-safe slug")
    _require_nonempty_string(ablation["description"], "structural_ablation.description")
    models = _require_string_list(ablation["models"], "structural_ablation.models")
    if models != ["relnet", "attn_relnet"]:
        raise TaskSpecError("structural ablations must pair RelNet and AttnRelNet")
    raw_anchors = ablation["anchors"]
    if isinstance(raw_anchors, (str, bytes)) or not isinstance(raw_anchors, Sequence):
        raise TaskSpecError("structural_ablation.anchors must be an array")
    observed_anchors = [
        _require_int(item, "structural_ablation.anchors[]", minimum=1)
        for item in raw_anchors
    ]
    expected_anchors = [20, int(max(anchors))]
    if observed_anchors != expected_anchors:
        raise TaskSpecError(
            f"structural_ablation.anchors must be {expected_anchors}"
        )
    if _require_int(ablation["repeats"], "structural_ablation.repeats", minimum=1) != 20:
        raise TaskSpecError("structural_ablation.repeats must be 20")
    return _canonicalize(ablation)


def _validate_sensitivities(
    value: Any,
    *,
    anchors: Sequence[int],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskSpecError("sensitivities must be an array")
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        record = dict(_require_mapping(item, f"sensitivities[{index}]"))
        if set(record) != {"id", "description", "selection", "execution"}:
            raise TaskSpecError(
                "each sensitivity requires exactly id/description/selection/execution"
            )
        sensitivity_id = _require_nonempty_string(record["id"], f"sensitivities[{index}].id")
        if not _SLUG.fullmatch(sensitivity_id) or sensitivity_id in seen:
            raise TaskSpecError("sensitivity IDs must be unique lowercase path-safe slugs")
        seen.add(sensitivity_id)
        _require_nonempty_string(record["description"], f"sensitivities[{index}].description")
        selection = record["selection"]
        if selection not in {"frozen_prespecified", "development_games_only"}:
            raise TaskSpecError("sensitivity selection policy is invalid")
        execution = record["execution"]
        if selection == "development_games_only":
            if execution is not None:
                raise TaskSpecError(
                    "development-selected sensitivities cannot enter confirmatory grids"
                )
        else:
            execution = dict(
                _require_mapping(execution, f"sensitivities[{index}].execution")
            )
            if set(execution) != {
                "profile", "repeats", "anchors", "models", "prepared_variant"
            }:
                raise TaskSpecError(
                    "frozen sensitivity execution requires profile/repeats/anchors/models/prepared_variant"
                )
            if execution["profile"] != "sensitivity20":
                raise TaskSpecError("frozen sensitivities must use profile sensitivity20")
            if _require_int(
                execution["repeats"],
                f"sensitivities[{index}].execution.repeats",
                minimum=1,
            ) != 20:
                raise TaskSpecError("sensitivity20 requires exactly 20 repeats")
            if execution["anchors"] != [20, int(max(anchors))]:
                raise TaskSpecError("sensitivity20 must use n=20 and the largest anchor")
            if execution["models"] != list(CANONICAL_MODEL_ROLES):
                raise TaskSpecError("sensitivity20 must cover all five model roles")
            _require_nonempty_string(
                execution["prepared_variant"],
                f"sensitivities[{index}].execution.prepared_variant",
            )
            record["execution"] = execution
        output.append(_canonicalize(record))
    return tuple(output)


def validate_horizon_scale_config(config: Mapping[str, Any]) -> None:
    """Validate BDB2026's frozen grouped-OOF horizon scale artifact."""

    values = config.get("dev_horizon_scale")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 94:
        raise TaskSpecError("dev_horizon_scale must contain exactly 94 values")
    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TaskSpecError(f"dev_horizon_scale[{index}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise TaskSpecError(f"dev_horizon_scale[{index}] must be finite and positive")
        normalized.append(number)
    if any(right < left for left, right in zip(normalized, normalized[1:])):
        raise TaskSpecError("dev_horizon_scale must be nondecreasing")
    if config.get("dev_horizon_scale_sha256") != sha256_json(normalized):
        raise TaskSpecError("dev_horizon_scale_sha256 does not match its values")
    observed_through = config.get("dev_horizon_scale_observed_through")
    if (
        isinstance(observed_through, bool)
        or not isinstance(observed_through, int)
        or not 1 <= observed_through <= 94
    ):
        raise TaskSpecError(
            "dev_horizon_scale_observed_through must be an integer in [1,94]"
        )
    if config.get("dev_horizon_scale_tail_policy") != HORIZON_SCALE_TAIL_POLICY:
        raise TaskSpecError("dev_horizon_scale_tail_policy differs from the locked rule")
    if (
        config.get("dev_horizon_scale_support_requirement")
        != HORIZON_SCALE_SUPPORT_REQUIREMENT
    ):
        raise TaskSpecError(
            "dev_horizon_scale_support_requirement differs from the locked rule"
        )
    tail = normalized[observed_through:]
    if any(value != normalized[observed_through - 1] for value in tail):
        raise TaskSpecError(
            "unobserved dev_horizon_scale tail must equal its last fitted level"
        )


def _safe_relative_path(value: Any, path: str) -> str:
    text = _require_nonempty_string(value, path)
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() in {".", ""}:
        raise SourceReceiptError(f"{path} must be a safe repository-relative path.")
    return candidate.as_posix()


_PREPARED_ARRAY_AXES = {
    "player_tokens": ("example", "time", "tracked_object", "channel"),
    "player_mask": ("example", "time", "tracked_object"),
    "frame_mask": ("example", "time"),
    "y": ("example",),
    "target_values": ("example", "horizon", "coordinate"),
    "target_mask": ("example", "horizon"),
    "target_baseline": ("example", "horizon", "coordinate"),
}


def _prepared_array_record(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    mapping = dict(_require_mapping(value, f"prepared_contract.arrays.{name}"))
    if set(mapping) != {"axes", "shape", "dtype"}:
        raise TaskSpecError(
            f"prepared_contract.arrays.{name} requires exactly axes/shape/dtype"
        )
    axes = _require_string_list(mapping["axes"], f"prepared_contract.arrays.{name}.axes")
    if tuple(axes) != _PREPARED_ARRAY_AXES[name]:
        raise TaskSpecError(
            f"prepared_contract.arrays.{name}.axes must be "
            f"{list(_PREPARED_ARRAY_AXES[name])}"
        )
    raw_shape = mapping["shape"]
    if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Sequence):
        raise TaskSpecError(f"prepared_contract.arrays.{name}.shape must be an array")
    shape = [
        _require_int(item, f"prepared_contract.arrays.{name}.shape[]", minimum=1)
        for item in raw_shape
    ]
    if len(shape) != len(axes):
        raise TaskSpecError(
            f"prepared_contract.arrays.{name}.shape does not match its axes"
        )
    dtype = _require_nonempty_string(
        mapping["dtype"], f"prepared_contract.arrays.{name}.dtype"
    )
    return {"axes": axes, "shape": shape, "dtype": dtype}


def _validate_prepared_contract(
    value: Any,
    *,
    task_id: str,
    outcome_type: str,
    primary_metric: str,
    total_games: int,
    channels: Sequence[str],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(_require_mapping(value, "prepared_contract"))
    required = {
        "schema_version",
        "task_id",
        "outcome_type",
        "primary_metric",
        "examples",
        "games",
        "tabular_feature_order",
        "channel_names",
        "arrays",
        "support",
        "mask_contract",
        "target_contract",
        "adapter_metadata",
        "semantic_hash",
    }
    if set(contract) != required:
        raise TaskSpecError(
            "prepared_contract fields differ from its v2 contract: "
            f"missing={sorted(required - set(contract))}, "
            f"unexpected={sorted(set(contract) - required)}"
        )
    if contract["schema_version"] != PREPARED_SEMANTIC_SCHEMA_VERSION:
        raise TaskSpecError(
            f"prepared_contract.schema_version must be {PREPARED_SEMANTIC_SCHEMA_VERSION!r}"
        )
    if contract["task_id"] != task_id:
        raise TaskSpecError("prepared_contract.task_id differs from task_id")
    if contract["outcome_type"] != outcome_type:
        raise TaskSpecError("prepared_contract.outcome_type differs from outcome_type")
    if contract["primary_metric"] != primary_metric:
        raise TaskSpecError(
            "prepared_contract.primary_metric differs from primary_metric"
        )
    examples = _require_int(contract["examples"], "prepared_contract.examples", minimum=1)
    games = _require_int(contract["games"], "prepared_contract.games", minimum=1)
    if games != total_games:
        raise TaskSpecError("prepared_contract.games differs from total_games")
    tabular = _require_string_list(
        contract["tabular_feature_order"], "prepared_contract.tabular_feature_order"
    )
    contract_channels = _require_string_list(
        contract["channel_names"], "prepared_contract.channel_names"
    )
    if list(contract_channels) != list(channels):
        raise TaskSpecError(
            "prepared_contract.channel_names must exactly match features.channels order"
        )

    raw_arrays = dict(_require_mapping(contract["arrays"], "prepared_contract.arrays"))
    if set(raw_arrays) != set(_PREPARED_ARRAY_AXES):
        raise TaskSpecError(
            "prepared_contract.arrays must declare every canonical prepared array"
        )
    arrays = {
        name: _prepared_array_record(raw_arrays[name], name)
        for name in _PREPARED_ARRAY_AXES
    }
    tokens = arrays["player_tokens"]
    player_mask = arrays["player_mask"]
    frame_mask = arrays["frame_mask"]
    if tokens is None or player_mask is None or frame_mask is None:
        raise TaskSpecError("prepared_contract requires token and player/frame mask arrays")
    if tokens["shape"][0] != examples:
        raise TaskSpecError("player_tokens first dimension differs from examples")
    if tokens["shape"][:3] != player_mask["shape"]:
        raise TaskSpecError("player_mask shape differs from player_tokens")
    if tokens["shape"][:2] != frame_mask["shape"]:
        raise TaskSpecError("frame_mask shape differs from player_tokens")
    if tokens["shape"][3] != len(contract_channels):
        raise TaskSpecError("player_tokens channel dimension differs from channel_names")
    if player_mask["dtype"] != "bool" or frame_mask["dtype"] != "bool":
        raise TaskSpecError("player_mask and frame_mask contract dtypes must be bool")

    support_value = contract["support"]
    support: dict[str, Any] | None
    if support_value is None:
        support = None
    else:
        support = dict(_require_mapping(support_value, "prepared_contract.support"))
        if set(support) != {"count", "minimum", "maximum", "values_sha256"}:
            raise TaskSpecError(
                "prepared_contract.support requires count/minimum/maximum/values_sha256"
            )
        support["count"] = _require_int(
            support["count"], "prepared_contract.support.count", minimum=1
        )
        for endpoint in ("minimum", "maximum"):
            candidate = support[endpoint]
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                raise TaskSpecError(
                    f"prepared_contract.support.{endpoint} must be numeric"
                )
            if not math.isfinite(float(candidate)):
                raise TaskSpecError(
                    f"prepared_contract.support.{endpoint} must be finite"
                )
            support[endpoint] = float(candidate)
        digest = str(support["values_sha256"])
        if not _SHA256.fullmatch(digest):
            raise TaskSpecError("prepared_contract.support.values_sha256 is invalid")

    modeled = "binary" if outcome_type == "frame_event" else outcome_type
    y = arrays["y"]
    trajectory_names = ("target_values", "target_mask", "target_baseline")
    if modeled in {"binary", "distribution"}:
        if y is None or y["shape"] != [examples]:
            raise TaskSpecError("class/distribution y must have exact shape [examples]")
        if any(arrays[name] is not None for name in trajectory_names):
            raise TaskSpecError("non-trajectory contracts must omit trajectory arrays")
    elif modeled == "trajectory":
        if y is not None or any(arrays[name] is None for name in trajectory_names):
            raise TaskSpecError("trajectory contracts require values/mask/baseline and no y")
        values = arrays["target_values"]
        target_mask = arrays["target_mask"]
        baseline = arrays["target_baseline"]
        assert values is not None and target_mask is not None and baseline is not None
        if values["shape"][0] != examples or values["shape"][-1] != 2:
            raise TaskSpecError("trajectory target_values must have shape [examples,H,2]")
        if baseline["shape"] != values["shape"] or target_mask["shape"] != values["shape"][:2]:
            raise TaskSpecError("trajectory baseline/target_mask shapes do not align")
        if target_mask["dtype"] != "bool":
            raise TaskSpecError("trajectory target_mask dtype must be bool")

    if modeled == "distribution":
        endpoints = outcome.get("support")
        assert isinstance(endpoints, Sequence)
        expected_values = [float(value) for value in range(int(endpoints[0]), int(endpoints[1]) + 1)]
        expected_support = {
            "count": len(expected_values),
            "minimum": expected_values[0],
            "maximum": expected_values[-1],
            "values_sha256": sha256_json(expected_values),
        }
        if support != expected_support:
            raise TaskSpecError(
                "prepared_contract.support differs from the exact outcome support"
            )
    elif support is not None:
        raise TaskSpecError("only distribution tasks may declare prepared support")

    mask_contract = dict(
        _require_mapping(contract["mask_contract"], "prepared_contract.mask_contract")
    )
    if set(mask_contract) != {"player_mask", "frame_mask", "target_mask"}:
        raise TaskSpecError(
            "prepared_contract.mask_contract requires player_mask/frame_mask/target_mask"
        )
    player_contract = dict(
        _require_mapping(mask_contract["player_mask"], "prepared_contract.mask_contract.player_mask")
    )
    expected_player_contract = {
        "meaning": "true_iff_tracked_object_slot_is_observed",
        "observed_frame_slot_layout": (
            "stable_per_play_slots_with_sparse_observation"
        ),
        "slot_identity": "alignment_only_never_a_model_feature",
        "masked_token_value": 0.0,
        "requires_observed_frame": True,
    }
    if player_contract != expected_player_contract:
        raise TaskSpecError("prepared_contract player_mask semantics are not canonical")
    frame_contract = dict(
        _require_mapping(mask_contract["frame_mask"], "prepared_contract.mask_contract.frame_mask")
    )
    if set(frame_contract) != {"meaning", "layout"} or frame_contract["meaning"] != (
        "true_iff_time_slot_is_observed_not_padding"
    ) or frame_contract["layout"] not in {"all_observed", "left_padded", "right_padded"}:
        raise TaskSpecError("prepared_contract frame_mask semantics are invalid")
    if modeled == "trajectory":
        target_mask_contract = dict(
            _require_mapping(
                mask_contract["target_mask"],
                "prepared_contract.mask_contract.target_mask",
            )
        )
        if (
            set(target_mask_contract) != {"meaning", "layout", "masked_target_value"}
            or target_mask_contract["meaning"]
            != "true_iff_official_future_coordinate_is_observed_and_scored"
            or target_mask_contract["layout"] not in {"all_observed", "right_padded"}
            or target_mask_contract["masked_target_value"] != "nan"
        ):
            raise TaskSpecError("prepared_contract target_mask semantics are invalid")
    elif mask_contract["target_mask"] is not None:
        raise TaskSpecError("non-trajectory target_mask contract must be null")

    expected_target_contract = {
        "binary": {
            "encoding": "zero_one",
            "examples_target": "equals_y",
            "model_output": "positive_class_probability",
        },
        "distribution": {
            "encoding": "zero_based_support_index",
            "examples_target": "equals_support_at_y",
            "model_output": "ordered_support_probability_vector",
        },
        "trajectory": {
            "encoding": "absolute_xy",
            "examples_target": "nan_placeholder",
            "training_target": (
                "development_selected_residual_or_absolute_from_stored_"
                "absolute_xy_and_baseline"
            ),
            "model_output": (
                "masked_xy_in_development_selected_residual_or_absolute_coordinates"
            ),
            "coordinate_order": ["x", "y"],
            "reconstruction": (
                "absolute_identity_or_target_baseline_plus_residual_as_frozen"
            ),
        },
    }[modeled]
    if contract["target_contract"] != expected_target_contract:
        raise TaskSpecError("prepared_contract target encoding/output semantics are invalid")
    _require_mapping(contract["adapter_metadata"], "prepared_contract.adapter_metadata")
    digest = str(contract["semantic_hash"])
    if not _SHA256.fullmatch(digest):
        raise TaskSpecError("prepared_contract.semantic_hash is invalid")
    unsigned = {key: item for key, item in contract.items() if key != "semantic_hash"}
    if digest != sha256_json(unsigned):
        raise TaskSpecError("prepared_contract semantic_hash does not match its payload")
    return _canonicalize(contract)


@dataclass(frozen=True)
class SourceReceipt:
    """Content-addressed receipt for one copied competition source file."""

    schema_version: str
    source_id: str
    source_path: str
    destination_path: str
    size_bytes: int
    sha256: str
    row_count: int | None = None
    columns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceReceipt":
        if not isinstance(value, Mapping):
            raise SourceReceiptError("source receipt must be a JSON object")
        allowed = {
            "schema_version",
            "source_id",
            "source_path",
            "destination_path",
            "size_bytes",
            "sha256",
            "row_count",
            "columns",
        }
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise SourceReceiptError(f"unexpected source-receipt fields: {unexpected}")
        if value.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
            raise SourceReceiptError(
                f"schema_version must be {SOURCE_RECEIPT_SCHEMA_VERSION!r}"
            )
        source_id = value.get("source_id")
        if not isinstance(source_id, str) or not _SLUG.fullmatch(source_id):
            raise SourceReceiptError("source_id must be a lowercase path-safe slug")
        source_path = value.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise SourceReceiptError("source_path must be a nonempty string")
        destination = _safe_relative_path(value.get("destination_path"), "destination_path")
        size = value.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SourceReceiptError("size_bytes must be a nonnegative integer")
        digest = str(value.get("sha256", "")).lower()
        if not _SHA256.fullmatch(digest):
            raise SourceReceiptError("sha256 must be a 64-character lowercase hex digest")
        row_count = value.get("row_count")
        if row_count is not None and (
            isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0
        ):
            raise SourceReceiptError("row_count must be null or a nonnegative integer")
        raw_columns = value.get("columns", [])
        if isinstance(raw_columns, (str, bytes)) or not isinstance(raw_columns, Sequence):
            raise SourceReceiptError("columns must be an array of strings")
        columns = tuple(raw_columns)
        if any(not isinstance(column, str) or not column for column in columns):
            raise SourceReceiptError("columns must contain only nonempty strings")
        if len(columns) != len(set(columns)):
            raise SourceReceiptError("columns cannot contain duplicates")
        return cls(
            schema_version=SOURCE_RECEIPT_SCHEMA_VERSION,
            source_id=source_id,
            source_path=source_path,
            destination_path=destination,
            size_bytes=size,
            sha256=digest,
            row_count=row_count,
            columns=columns,
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["columns"] = list(self.columns)
        return value


def build_source_receipt(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    source_id: str,
    repo_root: str | Path,
    row_count: int | None = None,
    columns: Sequence[str] = (),
) -> SourceReceipt:
    """Verify a copy byte-for-byte and return its immutable source receipt."""

    source = Path(source_path).resolve()
    root = Path(repo_root).resolve()
    destination = Path(destination_path)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    try:
        relative = destination.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceReceiptError("destination_path must be inside repo_root") from exc
    if not source.is_file() or not destination.is_file():
        raise SourceReceiptError("source and destination must both be existing files")
    source_size = source.stat().st_size
    destination_size = destination.stat().st_size
    if source_size != destination_size:
        raise SourceReceiptError("source and destination sizes differ")
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise SourceReceiptError("source and destination hashes differ")
    return SourceReceipt.from_mapping(
        {
            "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
            "source_id": source_id,
            "source_path": str(source),
            "destination_path": relative,
            "size_bytes": source_size,
            "sha256": source_hash,
            "row_count": row_count,
            "columns": list(columns),
        }
    )


def validate_source_receipt(
    receipt: SourceReceipt | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    require_source: bool = False,
) -> SourceReceipt:
    """Validate structure and, when ``repo_root`` is supplied, file contents."""

    parsed = receipt if isinstance(receipt, SourceReceipt) else SourceReceipt.from_mapping(receipt)
    if repo_root is None:
        if require_source:
            raise SourceReceiptError("require_source=True requires repo_root")
        return parsed
    root = Path(repo_root).resolve()
    destination = (root / parsed.destination_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise SourceReceiptError("destination escapes repo_root") from exc
    if not destination.is_file():
        raise SourceReceiptError(f"receipt destination is missing: {parsed.destination_path}")
    if destination.stat().st_size != parsed.size_bytes:
        raise SourceReceiptError(f"receipt size mismatch: {parsed.destination_path}")
    if sha256_file(destination) != parsed.sha256:
        raise SourceReceiptError(f"receipt hash mismatch: {parsed.destination_path}")
    if require_source:
        source = Path(parsed.source_path)
        if not source.is_file():
            raise SourceReceiptError(f"receipt source is missing: {source}")
        if source.stat().st_size != parsed.size_bytes or sha256_file(source) != parsed.sha256:
            raise SourceReceiptError(f"receipt source no longer matches: {source}")
    return parsed


def validate_source_receipts(
    receipts: Sequence[SourceReceipt | Mapping[str, Any]],
    *,
    repo_root: str | Path | None = None,
    require_source: bool = False,
) -> tuple[SourceReceipt, ...]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise SourceReceiptError("source_receipts must be an array")
    parsed = tuple(
        validate_source_receipt(item, repo_root=repo_root, require_source=require_source)
        for item in receipts
    )
    destinations = [receipt.destination_path for receipt in parsed]
    if len(destinations) != len(set(destinations)):
        raise SourceReceiptError("source receipts contain duplicate destination paths")
    return parsed


@dataclass(frozen=True)
class SourceTreeReceipt:
    """Reference to one aggregate, content-addressed import receipt."""

    schema_version: str
    source_id: str
    receipt_path: str
    receipt_sha256: str
    tree_sha256: str
    destination: str
    file_count: int
    byte_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceTreeReceipt":
        if not isinstance(value, Mapping):
            raise SourceReceiptError("source tree receipt must be a JSON object")
        required = {
            "schema_version",
            "source_id",
            "receipt_path",
            "receipt_sha256",
            "tree_sha256",
            "destination",
            "file_count",
            "byte_count",
        }
        if set(value) != required:
            raise SourceReceiptError(
                "source tree receipt fields differ from its v1 contract: "
                f"missing={sorted(required - set(value))}, "
                f"unexpected={sorted(set(value) - required)}"
            )
        if value["schema_version"] != SOURCE_TREE_RECEIPT_SCHEMA_VERSION:
            raise SourceReceiptError(
                f"schema_version must be {SOURCE_TREE_RECEIPT_SCHEMA_VERSION!r}"
            )
        source_id = value["source_id"]
        if not isinstance(source_id, str) or not _SLUG.fullmatch(source_id):
            raise SourceReceiptError("source_id must be a lowercase path-safe slug")
        receipt_path = _safe_relative_path(value["receipt_path"], "receipt_path")
        destination = _safe_relative_path(value["destination"], "destination")
        receipt_hash = str(value["receipt_sha256"]).lower()
        tree_hash = str(value["tree_sha256"]).lower()
        if not _SHA256.fullmatch(receipt_hash) or not _SHA256.fullmatch(tree_hash):
            raise SourceReceiptError("receipt_sha256 and tree_sha256 must be SHA-256 hex digests")
        file_count = value["file_count"]
        byte_count = value["byte_count"]
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count <= 0
        ):
            raise SourceReceiptError("file_count must be a positive integer")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise SourceReceiptError("byte_count must be a nonnegative integer")
        return cls(
            schema_version=SOURCE_TREE_RECEIPT_SCHEMA_VERSION,
            source_id=source_id,
            receipt_path=receipt_path,
            receipt_sha256=receipt_hash,
            tree_sha256=tree_hash,
            destination=destination,
            file_count=file_count,
            byte_count=byte_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_aggregate_receipt_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "source_id",
        "destination",
        "tree_sha256",
        "file_count",
        "byte_count",
        "catalog_receipts",
        "files",
        "receipt_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SourceReceiptError(f"aggregate import receipt is missing fields: {missing}")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise SourceReceiptError("aggregate import receipt files must be a nonempty array")
    if payload["file_count"] != len(files):
        raise SourceReceiptError("aggregate import receipt file_count is wrong")
    paths: list[str] = []
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise SourceReceiptError("aggregate file entries require path/bytes/sha256")
        path = _safe_relative_path(entry["path"], "files[].path")
        size = entry["bytes"]
        digest = str(entry["sha256"]).lower()
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SourceReceiptError("aggregate file bytes must be nonnegative integers")
        if not _SHA256.fullmatch(digest):
            raise SourceReceiptError("aggregate file sha256 is invalid")
        paths.append(path)
        total_bytes += size
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise SourceReceiptError("aggregate file paths must be unique and sorted")
    if payload["byte_count"] != total_bytes:
        raise SourceReceiptError("aggregate import receipt byte_count is wrong")
    if payload["tree_sha256"] != sha256_json(files):
        raise SourceReceiptError("aggregate import receipt tree_sha256 is wrong")
    scientific_identity = {
        key: value
        for key, value in payload.items()
        if key not in {"receipt_sha256", "source_path_observed", "verified_at_utc"}
    }
    if payload["receipt_sha256"] != sha256_json(scientific_identity):
        raise SourceReceiptError("aggregate receipt_sha256 is wrong")


def build_source_tree_receipt(
    aggregate_receipt_path: str | Path,
    *,
    repo_root: str | Path,
) -> SourceTreeReceipt:
    """Build a compact TaskSpec reference from a verified import receipt."""

    root = Path(repo_root).resolve()
    receipt_path = Path(aggregate_receipt_path)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = receipt_path.resolve()
    try:
        relative = receipt_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceReceiptError("aggregate receipt must be inside repo_root") from exc
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceReceiptError(f"could not read aggregate receipt {receipt_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SourceReceiptError("aggregate receipt root must be an object")
    _validate_aggregate_receipt_payload(payload)
    return SourceTreeReceipt.from_mapping(
        {
            "schema_version": SOURCE_TREE_RECEIPT_SCHEMA_VERSION,
            "source_id": payload["source_id"],
            "receipt_path": relative,
            # This is the aggregate receipt's scientific identity hash, not a
            # hash of wall-clock/host audit metadata in the JSON file.
            "receipt_sha256": payload["receipt_sha256"],
            "tree_sha256": payload["tree_sha256"],
            "destination": payload["destination"],
            "file_count": payload["file_count"],
            "byte_count": payload["byte_count"],
        }
    )


def validate_source_tree_receipt(
    receipt: SourceTreeReceipt | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_tree: bool = False,
) -> SourceTreeReceipt:
    parsed = (
        receipt if isinstance(receipt, SourceTreeReceipt) else SourceTreeReceipt.from_mapping(receipt)
    )
    if repo_root is None:
        if verify_tree:
            raise SourceReceiptError("verify_tree=True requires repo_root")
        return parsed
    root = Path(repo_root).resolve()
    receipt_path = (root / parsed.receipt_path).resolve()
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise SourceReceiptError("aggregate receipt path escapes repo_root") from exc
    rebuilt = build_source_tree_receipt(receipt_path, repo_root=root)
    if rebuilt != parsed:
        raise SourceReceiptError(f"aggregate receipt reference mismatch: {parsed.receipt_path}")
    if verify_tree:
        destination = (root / parsed.destination).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise SourceReceiptError("aggregate destination escapes repo_root") from exc
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        for entry in payload["files"]:
            path = (destination / entry["path"]).resolve()
            try:
                path.relative_to(destination)
            except ValueError as exc:
                raise SourceReceiptError("aggregate file path escapes destination") from exc
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise SourceReceiptError(f"aggregate source file is missing or wrong-sized: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise SourceReceiptError(f"aggregate source file hash mismatch: {path}")
    return parsed


def validate_source_tree_receipts(
    receipts: Sequence[SourceTreeReceipt | Mapping[str, Any]],
    *,
    repo_root: str | Path | None = None,
    verify_tree: bool = False,
) -> tuple[SourceTreeReceipt, ...]:
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise SourceReceiptError("source_tree_receipts must be an array")
    parsed = tuple(
        validate_source_tree_receipt(item, repo_root=repo_root, verify_tree=verify_tree)
        for item in receipts
    )
    paths = [item.receipt_path for item in parsed]
    if len(paths) != len(set(paths)):
        raise SourceReceiptError("source tree receipts contain duplicate receipt paths")
    return parsed


@dataclass(frozen=True)
class TaskSpec:
    """Immutable, versioned adapter-to-runner scientific interface."""

    schema_version: str
    task_id: str
    release_year: int
    outcome_type: str
    primary_metric: str
    total_games: int
    development_games: int
    confirmatory_games: int
    outer_counts: Mapping[str, int]
    anchors: tuple[int, ...]
    event_cutoff: Mapping[str, Any]
    cohort: Mapping[str, Any]
    features: Mapping[str, Any]
    outcome: Mapping[str, Any]
    prepared_contract: Mapping[str, Any]
    history: Mapping[str, Any]
    graph: Mapping[str, Any]
    identity_policy: Mapping[str, Any]
    model_roles: Mapping[str, Any]
    output_head: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    joint_neural_target_selection: Mapping[str, Any] | None
    structural_ablation: Mapping[str, Any]
    sensitivities: tuple[Mapping[str, Any], ...]
    models: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    source_receipts: tuple[SourceReceipt, ...]
    source_tree_receipts: tuple[SourceTreeReceipt, ...]
    shared_game_registry_with: str | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        require_source_receipts: bool = True,
    ) -> "TaskSpec":
        if not isinstance(value, Mapping):
            raise TaskSpecError("task spec must be a JSON object")
        required = {
            "schema_version",
            "task_id",
            "release_year",
            "outcome_type",
            "primary_metric",
            "total_games",
            "development_games",
            "confirmatory_games",
            "outer_counts",
            "anchors",
            "event_cutoff",
            "cohort",
            "features",
            "outcome",
            "prepared_contract",
            "history",
            "graph",
            "identity_policy",
            "model_roles",
            "output_head",
            "uncertainty",
            "structural_ablation",
            "sensitivities",
            "models",
            "artifacts",
        }
        optional = {
            "joint_neural_target_selection",
            "shared_game_registry_with",
            "source_receipts",
            "source_tree_receipts",
        }
        missing = sorted(required - set(value))
        unexpected = sorted(set(value) - required - optional)
        if missing:
            raise TaskSpecError(f"task spec is missing fields: {missing}")
        if unexpected:
            raise TaskSpecError(f"task spec has unexpected fields: {unexpected}")
        if value["schema_version"] != TASK_SPEC_SCHEMA_VERSION:
            raise TaskSpecError(f"schema_version must be {TASK_SPEC_SCHEMA_VERSION!r}")
        task_id = _require_nonempty_string(value["task_id"], "task_id")
        if not _SLUG.fullmatch(task_id):
            raise TaskSpecError("task_id must be a lowercase path-safe slug")
        release_year = _require_int(value["release_year"], "release_year", minimum=2020)
        if release_year > 2026:
            raise TaskSpecError("release_year must be between 2020 and 2026")
        outcome_type = _require_nonempty_string(value["outcome_type"], "outcome_type")
        if outcome_type not in OUTCOME_TYPES:
            raise TaskSpecError(f"outcome_type must be one of {OUTCOME_TYPES}")
        primary_metric = _require_nonempty_string(value["primary_metric"], "primary_metric").lower()
        expected_metric = {
            "binary": "brier",
            "frame_event": "brier",
            "distribution": "crps",
            "trajectory": "rmse",
        }[outcome_type]
        if primary_metric != expected_metric:
            raise TaskSpecError(
                f"{outcome_type} tasks require primary_metric={expected_metric!r}"
            )
        total = _require_int(value["total_games"], "total_games", minimum=1)
        development = _require_int(value["development_games"], "development_games", minimum=1)
        confirmatory = _require_int(value["confirmatory_games"], "confirmatory_games", minimum=1)
        if development + confirmatory != total:
            raise TaskSpecError("development_games + confirmatory_games must equal total_games")
        outer_raw = _require_mapping(value["outer_counts"], "outer_counts")
        if set(outer_raw) != {"train", "calibration", "test"}:
            raise TaskSpecError("outer_counts must contain exactly train/calibration/test")
        outer = {
            name: _require_int(outer_raw[name], f"outer_counts.{name}", minimum=1)
            for name in ("train", "calibration", "test")
        }
        if sum(outer.values()) != confirmatory:
            raise TaskSpecError("outer_counts must sum to confirmatory_games")
        anchors_raw = value["anchors"]
        if isinstance(anchors_raw, (str, bytes)) or not isinstance(anchors_raw, Sequence):
            raise TaskSpecError("anchors must be an array")
        anchors = tuple(
            _require_int(item, "anchors[]", minimum=1) for item in anchors_raw
        )
        if len(anchors) != 6 or tuple(sorted(set(anchors))) != anchors:
            raise TaskSpecError("anchors must contain exactly six strictly increasing sizes")
        if anchors[-1] > outer["train"]:
            raise TaskSpecError("largest anchor cannot exceed outer training games")

        event_cutoff = dict(_require_mapping(value["event_cutoff"], "event_cutoff"))
        _require_nonempty_string(event_cutoff.get("event"), "event_cutoff.event")
        fallback_rules = event_cutoff.get("fallback_rules")
        _require_string_list(fallback_rules, "event_cutoff.fallback_rules", allow_empty=True)

        cohort = dict(_require_mapping(value["cohort"], "cohort"))
        _require_nonempty_string(cohort.get("description"), "cohort.description")
        _require_string_list(cohort.get("eligibility"), "cohort.eligibility")
        _require_string_list(cohort.get("exclusions"), "cohort.exclusions", allow_empty=True)

        features = dict(_require_mapping(value["features"], "features"))
        allowlist = _require_string_list(features.get("allowlist"), "features.allowlist")
        denylist = _require_string_list(features.get("denylist"), "features.denylist", allow_empty=True)
        channels = _require_string_list(features.get("channels"), "features.channels")
        _require_nonempty_string(
            features.get("coordinate_normalization"), "features.coordinate_normalization"
        )
        overlap = sorted(set(allowlist) & set(denylist))
        if overlap:
            raise TaskSpecError(f"feature allowlist and denylist overlap: {overlap}")
        if len(channels) != len(set(channels)):
            raise TaskSpecError("features.channels cannot contain duplicates")

        outcome = dict(_require_mapping(value["outcome"], "outcome"))
        _require_nonempty_string(outcome.get("target"), "outcome.target")
        _require_nonempty_string(outcome.get("null_model"), "outcome.null_model")
        if outcome_type in {"binary", "frame_event"} and "positive_label" not in outcome:
            raise TaskSpecError("binary/frame_event outcome requires positive_label")
        if outcome_type == "distribution":
            support = outcome.get("support")
            if (
                not isinstance(support, Sequence)
                or isinstance(support, (str, bytes))
                or len(support) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in support)
                or support[0] >= support[1]
            ):
                raise TaskSpecError("distribution outcome.support must be [integer_min, integer_max]")

        prepared_contract = _validate_prepared_contract(
            value["prepared_contract"],
            task_id=task_id,
            outcome_type=outcome_type,
            primary_metric=primary_metric,
            total_games=total,
            channels=channels,
            outcome=outcome,
        )

        history = _validate_history_contract(value["history"])
        graph = _validate_graph_contract(value["graph"])
        identity_policy = _validate_identity_policy(value["identity_policy"])
        model_roles = _validate_model_role_contract(value["model_roles"])
        output_head = _validate_output_head(value["output_head"], outcome_type)
        uncertainty = _validate_uncertainty_contract(value["uncertainty"], outcome_type)
        joint_neural_target_selection = _validate_joint_neural_target_contract(
            value.get("joint_neural_target_selection"), outcome_type
        )
        structural_ablation = _validate_structural_ablation(
            value["structural_ablation"], anchors=anchors
        )
        sensitivities = _validate_sensitivities(
            value["sensitivities"], anchors=anchors
        )
        token_shape = prepared_contract["arrays"]["player_tokens"]["shape"]
        if int(token_shape[1]) != int(history["max_frames"]):
            raise TaskSpecError(
                "history.max_frames must match the prepared player-token time axis"
            )
        frame_layout = prepared_contract["mask_contract"]["frame_mask"]["layout"]
        # A frozen left-padding *policy* can legitimately yield an all-observed
        # realized tensor when every example has at least ``max_frames`` causal
        # frames.  In contrast, a ``none`` policy may never produce padding.
        compatible_layouts = (
            {"left_padded", "all_observed"}
            if history["padding"] == "left_masked"
            else {"all_observed"}
        )
        if frame_layout not in compatible_layouts:
            raise TaskSpecError(
                "history.padding is incompatible with the prepared frame-mask layout"
            )
        stable_slot_metadata = str(
            prepared_contract["adapter_metadata"].get("stable_player_slots", "")
        )
        if not stable_slot_metadata.startswith(f"{history['slot_policy']};"):
            raise TaskSpecError(
                "prepared adapter metadata does not bind history.slot_policy"
            )

        models = dict(_require_mapping(value["models"], "models"))
        if len(models) != len(CANONICAL_MODEL_ROLES):
            raise TaskSpecError("models must contain exactly five primary models")
        families: list[str] = []
        roles: list[str] = []
        selected_config_models: set[str] = set()
        for model_id, config in models.items():
            if not isinstance(model_id, str) or not _SLUG.fullmatch(model_id):
                raise TaskSpecError("model IDs must be lowercase path-safe slugs")
            config_mapping = _require_mapping(config, f"models.{model_id}")
            role = _require_nonempty_string(
                config_mapping.get("role"), f"models.{model_id}.role"
            )
            family = _require_nonempty_string(
                config_mapping.get("family"), f"models.{model_id}.family"
            )
            if model_id != role or ROLE_TO_FAMILY.get(role) != family:
                raise TaskSpecError(
                    f"model {model_id!r} must use its canonical role/family mapping"
                )
            roles.append(role)
            families.append(family)
            _validate_model_grid_declaration(outcome_type, model_id, config_mapping)
            if "selected_config" in config_mapping:
                selected_config_models.add(model_id)
                if task_id == "bdb2026_trajectory":
                    validate_horizon_scale_config(
                        _require_mapping(
                            config_mapping["selected_config"],
                            f"models.{model_id}.selected_config",
                        )
                    )
        if tuple(sorted(families)) != tuple(sorted(CANONICAL_MODEL_FAMILIES)):
            raise TaskSpecError(
                f"models must represent each family exactly once: {CANONICAL_MODEL_FAMILIES}"
            )
        if set(roles) != set(CANONICAL_MODEL_ROLES) or set(models) != set(
            CANONICAL_MODEL_ROLES
        ):
            raise TaskSpecError(
                f"models must use each canonical role ID exactly once: {CANONICAL_MODEL_ROLES}"
            )
        if task_id == "bdb2026_trajectory":
            joint_is_frozen = (
                isinstance(joint_neural_target_selection, Mapping)
                and joint_neural_target_selection.get("receipt") is not None
            )
            if selected_config_models and selected_config_models != set(models):
                raise TaskSpecError(
                    "BDB2026 selected horizon-scale configs must be frozen for all five models"
                )
            if joint_is_frozen and selected_config_models != set(models):
                raise TaskSpecError(
                    "frozen BDB2026 joint target requires all selected model configs"
                )

        artifacts = dict(_require_mapping(value["artifacts"], "artifacts"))
        for required_artifact in ("metrics", "predictions"):
            if required_artifact not in artifacts:
                raise TaskSpecError(f"artifacts must declare {required_artifact!r}")

        receipts_raw = value.get("source_receipts", [])
        receipts = validate_source_receipts(receipts_raw)
        tree_receipts = validate_source_tree_receipts(value.get("source_tree_receipts", []))
        if require_source_receipts and not receipts and not tree_receipts:
            raise TaskSpecError("a frozen task requires at least one file or tree source receipt")
        shared = value.get("shared_game_registry_with")
        if shared is not None and (
            not isinstance(shared, str) or not _SLUG.fullmatch(shared)
        ):
            raise TaskSpecError("shared_game_registry_with must be null or a lowercase slug")

        return cls(
            schema_version=TASK_SPEC_SCHEMA_VERSION,
            task_id=task_id,
            release_year=release_year,
            outcome_type=outcome_type,
            primary_metric=primary_metric,
            total_games=total,
            development_games=development,
            confirmatory_games=confirmatory,
            outer_counts=outer,
            anchors=anchors,
            event_cutoff=_canonicalize(event_cutoff),
            cohort=_canonicalize(cohort),
            features=_canonicalize(features),
            outcome=_canonicalize(outcome),
            prepared_contract=prepared_contract,
            history=history,
            graph=graph,
            identity_policy=identity_policy,
            model_roles=model_roles,
            output_head=output_head,
            uncertainty=uncertainty,
            joint_neural_target_selection=joint_neural_target_selection,
            structural_ablation=structural_ablation,
            sensitivities=sensitivities,
            models=_canonicalize(models),
            artifacts=_canonicalize(artifacts),
            source_receipts=receipts,
            source_tree_receipts=tree_receipts,
            shared_game_registry_with=shared,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "release_year": self.release_year,
            "outcome_type": self.outcome_type,
            "primary_metric": self.primary_metric,
            "total_games": self.total_games,
            "development_games": self.development_games,
            "confirmatory_games": self.confirmatory_games,
            "outer_counts": dict(self.outer_counts),
            "anchors": list(self.anchors),
            "event_cutoff": _canonicalize(self.event_cutoff),
            "cohort": _canonicalize(self.cohort),
            "features": _canonicalize(self.features),
            "outcome": _canonicalize(self.outcome),
            "prepared_contract": _canonicalize(self.prepared_contract),
            "history": _canonicalize(self.history),
            "graph": _canonicalize(self.graph),
            "identity_policy": _canonicalize(self.identity_policy),
            "model_roles": _canonicalize(self.model_roles),
            "output_head": _canonicalize(self.output_head),
            "uncertainty": _canonicalize(self.uncertainty),
            "joint_neural_target_selection": _canonicalize(
                self.joint_neural_target_selection
            ),
            "structural_ablation": _canonicalize(self.structural_ablation),
            "sensitivities": [_canonicalize(item) for item in self.sensitivities],
            "models": _canonicalize(self.models),
            "artifacts": _canonicalize(self.artifacts),
            "source_receipts": [receipt.as_dict() for receipt in self.source_receipts],
            "source_tree_receipts": [receipt.as_dict() for receipt in self.source_tree_receipts],
            "shared_game_registry_with": self.shared_game_registry_with,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.as_dict())


def prepared_contract_for_variant(
    task_spec: TaskSpec | Mapping[str, Any],
    prepared_variant: str = "primary",
) -> dict[str, Any]:
    """Return the exact prepared-data contract for a frozen intervention.

    Only BDB2024's common-cutoff sensitivity changes causal preparation.  The
    BDB2025 team-identity intervention consumes the primary prepared bundle
    and is applied by the model adapter, so its variant remains ``primary``.
    """

    spec = validate_task_spec(task_spec, require_source_receipts=False)
    contract = json.loads(canonical_json(spec.prepared_contract))
    if prepared_variant == "primary":
        return contract
    if not (
        spec.task_id == "bdb2024_tackle"
        and prepared_variant == "common_closest_approach_minus_ten"
    ):
        raise TaskSpecError(
            f"unsupported prepared variant {prepared_variant!r} for {spec.task_id}"
        )
    metadata = contract.get("adapter_metadata")
    if not isinstance(metadata, dict):
        raise TaskSpecError("prepared contract has no adapter metadata")
    metadata["cutoff_policy"] = prepared_variant
    unsigned = {key: value for key, value in contract.items() if key != "semantic_hash"}
    contract["semantic_hash"] = sha256_json(unsigned)
    return contract


def validate_task_spec(
    value: TaskSpec | Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    require_source: bool = False,
    require_source_receipts: bool = True,
) -> TaskSpec:
    parsed = (
        value
        if isinstance(value, TaskSpec)
        else TaskSpec.from_mapping(
            value, require_source_receipts=require_source_receipts
        )
    )
    if require_source_receipts and not parsed.source_receipts and not parsed.source_tree_receipts:
        raise TaskSpecError("a frozen task requires at least one file or tree source receipt")
    validate_source_receipts(
        parsed.source_receipts,
        repo_root=repo_root,
        require_source=require_source,
    )
    validate_source_tree_receipts(
        parsed.source_tree_receipts,
        repo_root=repo_root,
        verify_tree=require_source,
    )
    # Round-trip validation prevents callers from constructing an invalid
    # dataclass directly and relying on frozen=True as scientific validation.
    return TaskSpec.from_mapping(
        parsed.as_dict(), require_source_receipts=require_source_receipts
    )


def _atomic_create_or_match(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FrozenReceiptCollisionError(
                f"refusing to overwrite different frozen receipt: {path}"
            )
        return
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
            if not path.is_file() or path.read_bytes() != payload:
                raise FrozenReceiptCollisionError(
                    f"another process froze a different receipt: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def freeze_task_spec(
    value: TaskSpec | Mapping[str, Any],
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    require_source: bool = False,
) -> dict[str, Any]:
    """Validate source artifacts and immutably write a hash-bound receipt.

    By default the destination files and aggregate receipt files are checked
    beneath the current repository.  ``require_source=True`` additionally
    rechecks original per-file sources and every file in an aggregate tree.
    """

    spec = validate_task_spec(value, repo_root=repo_root, require_source=require_source)
    receipt = {
        "receipt_schema_version": TASK_SPEC_RECEIPT_VERSION,
        "task_spec_hash": spec.spec_hash,
        "task_spec": spec.as_dict(),
    }
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_create_or_match(Path(path), payload)
    return receipt


def load_task_spec(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
    require_source: bool = False,
    require_source_receipts: bool = True,
) -> TaskSpec:
    """Load either a bare task spec or a frozen hash-bound receipt."""

    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskSpecError(f"could not load task spec {target}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TaskSpecError("task spec file root must be a JSON object")
    if "task_spec" in value:
        if set(value) != {"receipt_schema_version", "task_spec_hash", "task_spec"}:
            raise TaskSpecError("frozen task receipt has unexpected fields")
        if value.get("receipt_schema_version") != TASK_SPEC_RECEIPT_VERSION:
            raise TaskSpecError("unsupported task-spec receipt version")
        if not isinstance(value["task_spec"], Mapping):
            raise TaskSpecError("frozen task_spec must be an object")
        actual_hash = sha256_json(value["task_spec"])
        if value.get("task_spec_hash") != actual_hash:
            raise TaskSpecError("frozen task-spec hash does not match its payload")
        value = value["task_spec"]
    return validate_task_spec(
        value,
        repo_root=repo_root,
        require_source=require_source,
        require_source_receipts=require_source_receipts,
    )


def validate_global_set_transformer_amendment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact outcome-blind attempt2-to-attempt3 amendment.

    This is deliberately an amendment-specific, fail-closed validator.  Its
    purpose is to keep the cancellation boundary and uninspected archive
    evidence machine-readable and inseparable from the fifth-role contract.
    """

    def same_json(observed: Any, expected: Any) -> bool:
        try:
            return canonical_json(observed) == canonical_json(expected)
        except ContractError:
            return False

    if not isinstance(value, Mapping):
        raise ProtocolAmendmentError("protocol amendment root must be an object")
    expected_top_level = {
        "schema_version",
        "amendment_id",
        "decision_time_utc",
        "evidence_status",
        "campaign_disposition",
        "reason",
        "primary_role_added",
        "unchanged_relational_test",
        "bdb2026_joint_target_models",
        "profile_counts_per_task",
        "within_task_primary_multiplicity",
        "cross_task_bdb2020_policy",
    }
    if set(value) != expected_top_level:
        raise ProtocolAmendmentError("protocol amendment fields differ")
    for field, expected in (
        ("schema_version", PROTOCOL_AMENDMENT_SCHEMA_VERSION),
        ("amendment_id", GLOBAL_SET_TRANSFORMER_AMENDMENT_ID),
        ("decision_time_utc", "2026-08-21T07:24:25Z"),
        ("evidence_status", "outcome_blind_before_pilot_result_exposure"),
    ):
        if value.get(field) != expected:
            raise ProtocolAmendmentError(f"protocol amendment {field} drifted")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ProtocolAmendmentError("protocol amendment reason is missing")

    expected_campaign = {
        "withdrawn_campaign": "pilot10_attempt2",
        "replacement_campaign": "pilot10_attempt3",
        "withdrawal_boundary": (
            "before_any_final_aggregate_or_scientific_result_was_inspected"
        ),
        "full100_launched": False,
        "attempt2_campaign_hash": (
            "96c39edb8dc84282293c70fc456a77f3e18f2956a486c84df6f1b79e20de7f03"
        ),
        "attempt2_terminal_job_id": 7725228,
        "attempt2_submitted_job_ids": 710,
        "attempt2_initially_active_cancellation_targets": 651,
        "attempt2_receipt_sidecar_sha256": (
            "a744ad5e825359fe961a5cda70b6d30d03c22035de43bd52b318cd6c3ac45ae7"
        ),
        "attempt2_receipt_content_sha256": (
            "a744ad5e825359fe961a5cda70b6d30d03c22035de43bd52b318cd6c3ac45ae7"
        ),
        "attempt2_journal_sidecar_sha256": (
            "84d381a98c292e6293cf05a308a333c650e4af6d8e864e829770b043fa4de81d"
        ),
        "attempt2_journal_content_sha256": (
            "84d381a98c292e6293cf05a308a333c650e4af6d8e864e829770b043fa4de81d"
        ),
        "attempt2_embedded_journal_hash": (
            "185f2ab1705f0c5e530941f576640989c59528558725c64fa040e176611051b7"
        ),
        "pilot_outputs_created": 0,
        "selected_development_receipts_created": 0,
        "archived_uninspected_intermediates": {
            "scores_opened_or_interpreted": False,
            "total_files": 65,
            "total_bytes": 1_914_039,
            "inventories": [
                {
                    "path": (
                        "data/bdb_suite_runs/archive/"
                        "pilot10_attempt2_20260821/development"
                    ),
                    "files": 57,
                    "bytes": 1_908_689,
                    "inventory_sha256": (
                        "97b937e2318872a5c112d4a95001ed9e382a489cd5d269088d7971c0492e6e43"
                    ),
                },
                {
                    "path": (
                        "data/bdb_suite_runs/archive/"
                        "pilot10_attempt2_20260821/development_runtime"
                    ),
                    "files": 8,
                    "bytes": 5_350,
                    "inventory_sha256": (
                        "c06f100c3bcbc6c1628cd40fcde58c4b4c7d7b66482eab1a1fc65ccd6aeb9bca"
                    ),
                },
            ],
        },
    }
    if not same_json(value.get("campaign_disposition"), expected_campaign):
        raise ProtocolAmendmentError("attempt2 disposition evidence drifted")

    expected_role = {
        "role": "set_transformer",
        "family": "set_transformer",
        "display_name": "Global Set Transformer",
        "representation": (
            "the_same_causal_player_tokens_history_masks_time_context_output_head_loss_"
            "and_seeds_as_the_relational_neural_models"
        ),
        "relation_scope": "global_masked_all_player_self_attention",
        "development_grid": {
            "learning_rates": [0.001, 0.0003],
            "dropouts": [0.3, 0.1],
        },
    }
    if not same_json(value.get("primary_role_added"), expected_role):
        raise ProtocolAmendmentError("fifth primary role contract drifted")
    expected_relational = {
        "models": ["relnet", "attn_relnet"],
        "structural_ablations_apply_only_to_these_models": True,
        "parameter_match_tolerance_fraction": 0.05,
        "flop_match_tolerance_fraction": 0.15,
        "neural_parameter_cap": 350_000,
    }
    if not same_json(
        value.get("unchanged_relational_test"), expected_relational
    ):
        raise ProtocolAmendmentError("relational comparison contract drifted")
    if value.get("bdb2026_joint_target_models") != [
        "relnet",
        "attn_relnet",
        "set_transformer",
    ]:
        raise ProtocolAmendmentError("BDB2026 joint target family list drifted")
    expected_counts = {
        "full50": {
            "primary": 1_200,
            "structural_ablation": 0,
            "frozen_sensitivity": 0,
            "required": 1_200,
        },
        "pilot10": {
            "primary": 300,
            "structural_ablation": 0,
            "frozen_sensitivity": 0,
            "required": 300,
        },
        "full100": {
            "primary": 3_000,
            "structural_ablation": 80,
            "frozen_sensitivity": 0,
            "required": 3_080,
        },
        "sensitivity20": {
            "primary": 0,
            "structural_ablation": 0,
            "frozen_sensitivity": 200,
            "required": 200,
        },
    }
    if not same_json(value.get("profile_counts_per_task"), expected_counts):
        raise ProtocolAmendmentError("profile counts drifted")
    if value.get("within_task_primary_multiplicity") != {
        "model_pairs": 10,
        "anchors": 6,
        "max_t_contrasts": 60,
    }:
        raise ProtocolAmendmentError("primary multiplicity family drifted")
    if value.get("cross_task_bdb2020_policy") != {
        "aligned_roles": ["glm", "lightgbm", "set_transformer"],
        "legacy_only_role": "legacy_zoo_cnn",
        "missing_roles_not_imputed": ["relnet", "attn_relnet"],
    }:
        raise ProtocolAmendmentError("BDB2020 descriptive mapping drifted")
    return json.loads(canonical_json(value))


def load_global_set_transformer_amendment(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolAmendmentError(
            f"could not load protocol amendment {target}: {exc}"
        ) from exc
    return validate_global_set_transformer_amendment(value)


def validate_bdb2026_horizon_scale_amendment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact outcome-blind attempt4 horizon-support amendment."""

    if not isinstance(value, Mapping):
        raise ProtocolAmendmentError("BDB2026 horizon amendment root must be an object")
    if set(value) != {
        "schema_version",
        "amendment_id",
        "recorded_at_utc",
        "evidence_status",
        "superseded_campaign",
        "support_audit",
        "amended_scale_rule",
        "replacement_campaign",
    }:
        raise ProtocolAmendmentError("BDB2026 horizon amendment fields differ")
    if (
        value.get("schema_version") != PROTOCOL_AMENDMENT_SCHEMA_VERSION
        or value.get("amendment_id") != BDB2026_HORIZON_SCALE_AMENDMENT_ID
        or value.get("recorded_at_utc") != "2026-08-21T11:49:59Z"
        or value.get("evidence_status")
        != "outcome_blind_before_freeze_smoke_or_pilot_prediction"
        or value.get("replacement_campaign") != "pilot10_attempt5"
        or sha256_json(value)
        != BDB2026_HORIZON_SCALE_AMENDMENT_CANONICAL_SHA256
    ):
        raise ProtocolAmendmentError("BDB2026 horizon amendment evidence drifted")
    return json.loads(canonical_json(value))


def load_bdb2026_horizon_scale_amendment(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolAmendmentError(
            f"could not load BDB2026 horizon amendment {target}: {exc}"
        ) from exc
    return validate_bdb2026_horizon_scale_amendment(value)


def validate_bdb2020_harmonized_amendment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact retrospective BDB2020 harmonization amendment."""

    if not isinstance(value, Mapping):
        raise ProtocolAmendmentError(
            "BDB2020 harmonized amendment root must be an object"
        )
    if set(value) != {
        "schema_version",
        "amendment_id",
        "recorded_at_utc",
        "evidence_status",
        "legacy_evidence_policy",
        "new_task",
        "model_roles",
        "development_policy",
        "representation",
        "ordered_target",
        "relational_graph",
        "execution",
    }:
        raise ProtocolAmendmentError(
            "BDB2020 harmonized amendment fields differ"
        )
    try:
        canonical_sha256 = sha256_json(value)
    except ContractError as exc:
        raise ProtocolAmendmentError(
            "BDB2020 harmonized amendment is not canonical JSON"
        ) from exc
    if (
        value.get("schema_version")
        != BDB2020_HARMONIZED_AMENDMENT_SCHEMA_VERSION
        or value.get("amendment_id") != BDB2020_HARMONIZED_AMENDMENT_ID
        or value.get("recorded_at_utc") != "2026-09-06T21:00:00Z"
        or value.get("evidence_status")
        != "retrospective_harmonization_separate_from_legacy_confirmatory_evidence"
        or canonical_sha256
        != BDB2020_HARMONIZED_AMENDMENT_CANONICAL_SHA256
    ):
        raise ProtocolAmendmentError(
            "BDB2020 harmonized amendment evidence drifted"
        )
    return json.loads(canonical_json(value))


def load_bdb2020_harmonized_amendment(
    path: str | Path,
) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolAmendmentError(
            f"could not load BDB2020 harmonized amendment {target}: {exc}"
        ) from exc
    return validate_bdb2020_harmonized_amendment(value)
