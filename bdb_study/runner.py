"""Task-neutral model-cell execution for prepared BDB tasks.

The module itself does not import TensorFlow.  GPU initialization occurs only
when a neural cell calls the lazy functions in :mod:`bdb_study.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .adapters.common import PreparedTask
from .contracts import TaskSpec, validate_task_spec
from .metrics import (
    bernoulli_entropy,
    binary_log_loss_contributions,
    brier_contributions,
    calibration_bias,
    crps_contributions,
    empirical_distribution_null,
    fit_venn_abers,
    game_equal_mean,
    hierarchical_distribution_intervals,
    label_conditional_prediction_sets,
    pathwise_conformal_tube,
    prevalence_null,
    rms_reliability_error,
    skill_score,
    trajectory_path_rmse,
    trajectory_rmse,
    trajectory_squared_error,
)
from .models import (
    CLASSICAL_PARAMETER_COUNT_DEFINITION,
    NEURAL_PARAMETER_COUNT_DEFINITION,
    classical_parameter_count,
    fit_glm,
    fit_lightgbm,
    fit_neural_explicit_preprocessing,
    NeuralSelection,
    refit_neural_from_selection_explicit_preprocessing,
    select_neural_epoch_explicit_preprocessing,
    grouped_validation_indices,
    predict_neural,
    implementation_family,
    empirical_cdf_baseline,
    neural_head_loss_signature,
    PUNT_CDF_RESIDUAL_CONTRACT,
    GLOBAL_SET_FAMILIES,
    RELATIONAL_FAMILIES,
)
from .representations import (
    MaskedTokenScaler,
    LazyNeuralInputs,
    NeuralContextEncoder,
    RasterValueScaler,
    TabularEncoder,
    classical_feature_frame,
    primary_context_frame,
    rasterize_tokens,
    representative_neural_input_signature,
    representative_shared_neural_input_signature,
)


CLASSICAL_PREPROCESSING_SCOPE = "selected_training_games_only"
NEURAL_SELECTOR_PREPROCESSING_SCOPE = "epoch_selector_training_games_only"
NEURAL_FINAL_PREPROCESSING_SCOPE = "all_selected_training_games"


@dataclass
class CellComputation:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    history: dict[str, Any]
    arrays: dict[str, np.ndarray]


NEURAL_SELECTOR_CORE_SCHEMA_VERSION = "bdb-neural-cell-selector-v1"


@dataclass(frozen=True)
class _CellSetup:
    split: Mapping[str, Any]
    cell: Mapping[str, Any]
    train_games: tuple[str, ...]
    calibration_games: tuple[str, ...]
    test_games: tuple[str, ...]
    train_index: np.ndarray
    calibration_index: np.ndarray
    test_index: np.ndarray
    family: str
    config: dict[str, Any]
    seeds: dict[str, Any]


def _game_index(task: PreparedTask, games: list[str] | tuple[str, ...]) -> np.ndarray:
    requested = {str(value) for value in games}
    observed = task.examples["game_id"].astype(str).to_numpy()
    index = np.flatnonzero(np.isin(observed, list(requested)))
    present = set(observed[index])
    missing = sorted(requested - present)
    if missing:
        raise ValueError(f"prepared task has no examples for games {missing[:10]}")
    return index


def _model_entry(spec: TaskSpec, model_id: str) -> tuple[str, dict[str, Any]]:
    if model_id not in spec.models:
        raise KeyError(f"model {model_id!r} is not in the task receipt")
    entry = dict(spec.models[model_id])
    family = str(entry.pop("family"))
    selected = entry.get("selected_config")
    if not isinstance(selected, Mapping):
        raise ValueError(
            f"model {model_id!r} has no frozen selected_config; run development CV and freeze first"
        )
    return family, dict(selected)


def _modeled_outcome(task: PreparedTask) -> str:
    return "binary" if task.outcome_type == "frame_event" else task.outcome_type


def _n_outputs(task: PreparedTask) -> int:
    if _modeled_outcome(task) == "distribution":
        if task.support is None:
            raise ValueError("distribution task has no frozen support")
        return len(task.support)
    return 1


def _encoder_audit(encoder: TabularEncoder) -> dict[str, Any]:
    if encoder.transformer is None:
        return {}
    try:
        output_names = [str(value) for value in encoder.transformer.get_feature_names_out()]
    except Exception:
        output_names = []
    return {
        "input_columns": list(encoder.input_columns),
        "output_columns": output_names,
        "output_dimension": len(output_names),
        "fit_scope": CLASSICAL_PREPROCESSING_SCOPE,
    }


class TaskRuntime:
    """One queue-local view of a memory-mapped prepared task."""

    def __init__(self, task: PreparedTask, *, queue: str) -> None:
        if queue not in {"cpu_tabular", "gpu_neural"}:
            raise ValueError("queue must be cpu_tabular or gpu_neural")
        self.task = task
        self.queue = queue
        self._classical_frames: dict[bool, pd.DataFrame] = {}

    @property
    def classical_frame(self) -> pd.DataFrame:
        if self.queue != "cpu_tabular":
            raise RuntimeError("GPU runtime must not materialize the classical representation")
        if False not in self._classical_frames:
            self._classical_frames[False] = classical_feature_frame(
                self.task.tabular,
                self.task.player_tokens,
                self.task.player_mask,
                self.task.frame_mask,
                self.task.channel_names,
                task_id=self.task.task_id,
            )
        return self._classical_frames[False]

    def _classical_frame(self, *, include_team_identity: bool) -> pd.DataFrame:
        if self.queue != "cpu_tabular":
            raise RuntimeError("GPU runtime must not materialize the classical representation")
        key = bool(include_team_identity)
        if key not in self._classical_frames:
            self._classical_frames[key] = classical_feature_frame(
                self.task.tabular,
                self.task.player_tokens,
                self.task.player_mask,
                self.task.frame_mask,
                self.task.channel_names,
                include_team_identity=key,
                task_id=self.task.task_id,
            )
        return self._classical_frames[key]

    def classical_inputs(
        self,
        train_index: np.ndarray,
        *other_indices: np.ndarray,
        include_team_identity: bool = False,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        frame = self._classical_frame(
            include_team_identity=include_team_identity
        )
        encoder = TabularEncoder().fit(frame.iloc[train_index])
        arrays = [encoder.transform(frame.iloc[index]) for index in (train_index, *other_indices)]
        audit = _encoder_audit(encoder)
        audit["team_identity"] = "included_sensitivity" if include_team_identity else "excluded_primary"
        audit["structural_summary_shared_by_tabular_roles"] = True
        return arrays, audit

    def neural_inputs(
        self,
        family: str,
        train_index: np.ndarray,
        *other_indices: np.ndarray,
        fit_scope: str = CLASSICAL_PREPROCESSING_SCOPE,
        ablation_id: str | None = None,
        include_team_identity: bool = False,
    ) -> tuple[list[Any], dict[str, Any]]:
        if self.queue != "gpu_neural":
            raise RuntimeError("CPU runtime must not construct neural representations")
        task = self.task
        all_indices = (train_index, *other_indices)
        context_frame = primary_context_frame(
            task.tabular, include_team_identity=include_team_identity
        )
        context_encoder = NeuralContextEncoder().fit(context_frame.iloc[train_index])
        contexts = [
            context_encoder.transform(context_frame.iloc[index]) for index in all_indices
        ]
        context_audit = context_encoder.audit()
        context_audit["fit_scope"] = str(fit_scope)
        family = implementation_family(family)
        if family == "cnn":
            scaler = RasterValueScaler().fit_indexed_tokens(
                task.player_tokens,
                task.player_mask,
                train_index,
                task.channel_names,
            )
            inputs = [
                LazyNeuralInputs(
                    family="cnn",
                    player_tokens=task.player_tokens,
                    player_mask=task.player_mask,
                    frame_mask=task.frame_mask,
                    global_context=context,
                    example_indices=np.asarray(index, dtype=int),
                    channel_names=tuple(task.channel_names),
                    raster_scaler=scaler,
                    task_id=task.task_id,
                    ablation_id=ablation_id,
                )
                for index, context in zip(all_indices, contexts)
            ]
            audit = {
                "representation": "two_yard_spatial_raster",
                "fit_scope": str(fit_scope),
                "kinematic_means": scaler.means.tolist(),
                "kinematic_scales": scaler.scales.tolist(),
                "global_context": context_audit,
            }
            return inputs, audit
        if family in {*GLOBAL_SET_FAMILIES, *RELATIONAL_FAMILIES}:
            scaler = MaskedTokenScaler(tuple(task.channel_names)).fit_indexed(
                task.player_tokens,
                task.player_mask,
                train_index,
                frame_mask=task.frame_mask,
                final_frame_only=ablation_id in {
                    "release_snapshot_only",
                    "final_snapshot_only",
                },
                task_id=task.task_id,
                ablation_id=ablation_id,
            )
            inputs = [
                LazyNeuralInputs(
                    family=family,
                    player_tokens=task.player_tokens,
                    player_mask=task.player_mask,
                    frame_mask=task.frame_mask,
                    global_context=context,
                    example_indices=np.asarray(index, dtype=int),
                    channel_names=tuple(task.channel_names),
                    token_scaler=scaler,
                    task_id=task.task_id,
                    ablation_id=ablation_id,
                )
                for index, context in zip(all_indices, contexts)
            ]
            audit = {
                "representation": (
                    "matched_task_relational_graph"
                    if family in RELATIONAL_FAMILIES
                    else "global_set_masked_player_time_tokens"
                ),
                "fit_scope": str(fit_scope),
                "continuous_indices": list(scaler.continuous_indices),
                "means": scaler.means.tolist(),
                "scales": scaler.scales.tolist(),
                "global_context": context_audit,
                "task_id": task.task_id,
                "ablation_id": ablation_id,
                "ablation_mask_applied_to_scaler": ablation_id is not None,
                "team_identity": "included_sensitivity" if include_team_identity else "excluded_primary",
                "player_identity": "alignment_only_never_encoded",
            }
            return inputs, audit
        raise ValueError(f"unknown neural family {family!r}")


def _training_target(
    task: PreparedTask,
    index: np.ndarray,
    *,
    trajectory_target: str = "residual",
) -> tuple[np.ndarray, np.ndarray | None]:
    outcome = _modeled_outcome(task)
    if outcome == "trajectory":
        values = np.asarray(task.target_values[index], dtype=np.float32)
        baseline = np.asarray(task.target_baseline[index], dtype=np.float32)
        mask = np.asarray(task.target_mask[index], dtype=bool)
        if trajectory_target == "residual":
            target = values - baseline
        elif trajectory_target == "absolute":
            target = values.copy()
        else:
            raise ValueError("trajectory target must be residual or absolute")
        target[~mask] = 0.0
        return target, mask
    if task.y is None:
        raise ValueError("non-trajectory task has no encoded target")
    return np.asarray(task.y[index]), None


def _resolve_cell_setup(
    runtime: TaskRuntime,
    task_spec: TaskSpec | Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str,
    ablation_id: str | None,
    sensitivity_id: str | None,
) -> _CellSetup:
    task = runtime.task
    spec = validate_task_spec(task_spec)
    if task.task_id != spec.task_id or design.get("task_id") != spec.task_id:
        raise ValueError("task, task receipt, and design are not aligned")
    split = design["split_manifests"][int(repeat) - 1]
    if int(split["repeat"]) != int(repeat) or str(n_train) not in split["nested_train_game_ids"]:
        raise ValueError("requested repeat/anchor is not in the design")
    train_games = tuple(split["nested_train_game_ids"][str(n_train)])
    calibration_games = tuple(split["calibration_game_ids"])
    test_games = tuple(split["test_game_ids"])
    train_index = _game_index(task, train_games)
    calibration_index = _game_index(task, calibration_games)
    test_index = _game_index(task, test_games)
    if len(set(task.examples.iloc[train_index]["game_id"].astype(str))) != n_train:
        raise ValueError("training anchor does not contain the literal requested game count")
    family, config = _model_entry(spec, model_id)
    family = implementation_family(family)
    expected_queue = "cpu_tabular" if family in {"glm", "lightgbm"} else "gpu_neural"
    if runtime.queue != expected_queue:
        raise ValueError(f"model {model_id} belongs to {expected_queue}, not {runtime.queue}")
    matching = [
        record
        for record in design["required_cells"]
        if record["repeat"] == repeat
        and record["n_train"] == n_train
        and record["model"] == model_id
        and str(record.get("branch", "fixed_main")) == str(branch)
        and record.get("ablation_id") == ablation_id
        and record.get("sensitivity_id") == sensitivity_id
    ]
    if len(matching) != 1:
        raise ValueError("requested main/ablation cell is absent or ambiguous")
    cell = matching[0]
    seeds = dict(cell["seeds"])
    if "uncertainty_subsample" not in seeds:
        raise ValueError("cell lacks the frozen uncertainty_subsample seed")
    return _CellSetup(
        split=split,
        cell=cell,
        train_games=train_games,
        calibration_games=calibration_games,
        test_games=test_games,
        train_index=train_index,
        calibration_index=calibration_index,
        test_index=test_index,
        family=family,
        config=config,
        seeds=seeds,
    )


def _neural_selector_binding(
    task: PreparedTask,
    setup: _CellSetup,
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str,
    ablation_id: str | None,
    sensitivity_id: str | None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "cell": {
            "branch": str(branch),
            "repeat": int(repeat),
            "n_train": int(n_train),
            "model": str(model_id),
            "ablation_id": ablation_id,
            "sensitivity_id": sensitivity_id,
        },
        "family": setup.family,
        "model_config": dict(setup.config),
        "effective_model_config": _effective_model_config(
            task,
            setup.family,
            setup.config,
            sensitivity_id=sensitivity_id,
        ),
        "seeds": dict(setup.seeds),
        "outer_split_hash": str(setup.split["outer_split_hash"]),
        "nested_split_hash": str(setup.split["nested_split_hash"]),
        "train_game_ids": list(setup.train_games),
        "calibration_game_ids": list(setup.calibration_games),
        "test_game_ids": list(setup.test_games),
    }


def run_neural_selector(
    runtime: TaskRuntime,
    task_spec: TaskSpec | Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str = "fixed_main",
    ablation_id: str | None = None,
    sensitivity_id: str | None = None,
) -> dict[str, Any]:
    """Run only epoch selection and return its immutable scientific core."""

    setup = _resolve_cell_setup(
        runtime,
        task_spec,
        design,
        repeat=repeat,
        n_train=n_train,
        model_id=model_id,
        branch=branch,
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
    )
    if setup.family in {"glm", "lightgbm"}:
        raise ValueError("neural selector phase cannot execute a tabular model")
    task = runtime.task
    effective_config = _effective_model_config(
        task,
        setup.family,
        setup.config,
        sensitivity_id=sensitivity_id,
    )
    include_team_identity = bool(effective_config.get("include_team_identity", False))
    trajectory_target = str(effective_config.get("trajectory_target", "residual"))
    local_fit, local_validation = grouped_validation_indices(
        task.game_ids[setup.train_index],
        int(setup.seeds["validation_split"]),
        strata=task.strata[setup.train_index],
    )
    selector_train_index = setup.train_index[local_fit]
    selector_validation_index = setup.train_index[local_validation]
    selector_inputs, selector_preprocessing = runtime.neural_inputs(
        setup.family,
        selector_train_index,
        selector_validation_index,
        fit_scope=NEURAL_SELECTOR_PREPROCESSING_SCOPE,
        ablation_id=ablation_id,
        include_team_identity=include_team_identity,
    )
    selector_train_y, selector_train_mask = _training_target(
        task, selector_train_index, trajectory_target=trajectory_target
    )
    selector_validation_y, selector_validation_mask = _training_target(
        task, selector_validation_index, trajectory_target=trajectory_target
    )
    selection = select_neural_epoch_explicit_preprocessing(
        setup.family,
        selector_inputs[0],
        selector_train_y,
        selector_inputs[1],
        selector_validation_y,
        outcome_type=_modeled_outcome(task),
        n_outputs=_n_outputs(task),
        selector_train_mask=selector_train_mask,
        selector_validation_mask=selector_validation_mask,
        config=effective_config,
        selection_seed=int(setup.seeds["epoch_selection"]),
        validation_game_ids=tuple(
            sorted(
                str(value)
                for value in np.unique(task.game_ids[selector_validation_index])
            )
        ),
        validation_split_seed=int(setup.seeds["validation_split"]),
    )
    return {
        "schema_version": NEURAL_SELECTOR_CORE_SCHEMA_VERSION,
        "binding": _neural_selector_binding(
            task,
            setup,
            repeat=repeat,
            n_train=n_train,
            model_id=model_id,
            branch=branch,
            ablation_id=ablation_id,
            sensitivity_id=sensitivity_id,
        ),
        "selector_preprocessing": selector_preprocessing,
        "selector_train_game_ids": sorted(
            str(value) for value in np.unique(task.game_ids[selector_train_index])
        ),
        "selection": selection.as_dict(),
    }


def _validate_neural_selector_core(
    value: Mapping[str, Any],
    task: PreparedTask,
    setup: _CellSetup,
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str,
    ablation_id: str | None,
    sensitivity_id: str | None,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "binding",
        "selector_preprocessing",
        "selector_train_game_ids",
        "selection",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("neural selector core fields are invalid")
    if value.get("schema_version") != NEURAL_SELECTOR_CORE_SCHEMA_VERSION:
        raise ValueError("neural selector core schema is invalid")
    expected_binding = _neural_selector_binding(
        task,
        setup,
        repeat=repeat,
        n_train=n_train,
        model_id=model_id,
        branch=branch,
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
    )
    if value.get("binding") != expected_binding:
        raise ValueError("neural selector core does not bind the requested cell")
    if not isinstance(value.get("selector_preprocessing"), Mapping):
        raise ValueError("neural selector preprocessing audit is invalid")
    local_fit, local_validation = grouped_validation_indices(
        task.game_ids[setup.train_index],
        int(setup.seeds["validation_split"]),
        strata=task.strata[setup.train_index],
    )
    expected_train_games = sorted(
        str(value)
        for value in np.unique(task.game_ids[setup.train_index[local_fit]])
    )
    if value.get("selector_train_game_ids") != expected_train_games:
        raise ValueError("neural selector training games do not match the frozen split")
    selected = NeuralSelection.from_mapping(value["selection"])
    expected_validation_games = tuple(
        sorted(
            str(item)
            for item in np.unique(task.game_ids[setup.train_index[local_validation]])
        )
    )
    if (
        selected.family != setup.family
        or selected.outcome_type != _modeled_outcome(task)
        or selected.n_outputs != _n_outputs(task)
        or selected.validation_game_ids != expected_validation_games
        or selected.validation_split_seed != int(setup.seeds["validation_split"])
        or selected.selection_seed != int(setup.seeds["epoch_selection"])
    ):
        raise ValueError("neural selection state does not match the frozen cell contract")
    return dict(value)


def validate_neural_selector_for_cell(
    value: Mapping[str, Any],
    runtime: TaskRuntime,
    task_spec: TaskSpec | Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str = "fixed_main",
    ablation_id: str | None = None,
    sensitivity_id: str | None = None,
) -> dict[str, Any]:
    """Semantically bind a decoded selector core to one frozen cell."""

    setup = _resolve_cell_setup(
        runtime,
        task_spec,
        design,
        repeat=repeat,
        n_train=n_train,
        model_id=model_id,
        branch=branch,
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
    )
    return _validate_neural_selector_core(
        value,
        runtime.task,
        setup,
        repeat=repeat,
        n_train=n_train,
        model_id=model_id,
        branch=branch,
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
    )


def _effective_model_config(
    task: PreparedTask,
    family: str,
    config: Mapping[str, Any],
    *,
    sensitivity_id: str | None,
) -> dict[str, Any]:
    """Apply the same deterministic task defaults in every neural phase."""

    outcome = _modeled_outcome(task)
    output = dict(config)
    if outcome == "distribution":
        output.setdefault("output_contract", PUNT_CDF_RESIDUAL_CONTRACT)
    if family in {*RELATIONAL_FAMILIES, *GLOBAL_SET_FAMILIES} and outcome == "binary":
        output.setdefault("binary_loss", "log_loss")
    if family in {*RELATIONAL_FAMILIES, *GLOBAL_SET_FAMILIES}:
        output.setdefault(
            "representative_time_steps", int(task.player_tokens.shape[1])
        )
    if outcome == "trajectory":
        output.setdefault("trajectory_target", "residual")
        output.setdefault("trajectory_decoder", "shared_horizon_conditioned_v1")
    if task.task_id == "bdb2025_man_zone" and sensitivity_id == "include_team_identity":
        output["include_team_identity"] = True
    return output


def _fit_and_predict(
    runtime: TaskRuntime,
    family: str,
    config: Mapping[str, Any],
    train_index: np.ndarray,
    prediction_indices: tuple[np.ndarray, ...],
    *,
    fit_seed: int,
    validation_split_seed: int,
    selection_seed: int,
    refit_seed: int,
    ablation_id: str | None = None,
    sensitivity_id: str | None = None,
    neural_selector_core: Mapping[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    task = runtime.task
    family = implementation_family(family)
    outcome = _modeled_outcome(task)
    n_outputs = _n_outputs(task)
    effective_config = _effective_model_config(
        task, family, config, sensitivity_id=sensitivity_id
    )
    include_team_identity = bool(effective_config.get("include_team_identity", False))
    trajectory_target = str(effective_config.get("trajectory_target", "residual"))
    train_y, train_mask = _training_target(
        task, train_index, trajectory_target=trajectory_target
    )
    if family in {"glm", "lightgbm"}:
        inputs, preprocessing = runtime.classical_inputs(
            train_index,
            *prediction_indices,
            include_team_identity=include_team_identity,
        )
        fitter = fit_glm if family == "glm" else fit_lightgbm
        fit = fitter(
            inputs[0],
            train_y,
            outcome_type=outcome,
            config=effective_config,
            seed=fit_seed,
            n_outputs=n_outputs,
            target_mask=train_mask,
        )
        predictions = [fit.predict(values) for values in inputs[1:]]
        if outcome == "trajectory" and trajectory_target == "absolute":
            predictions = [
                np.asarray(values) - np.asarray(task.target_baseline[index])
                for values, index in zip(predictions, prediction_indices)
            ]
        parameter_count = classical_parameter_count(fit)
        return predictions, {
            "family": family,
            "parameter_count": parameter_count,
            "parameter_count_definition": CLASSICAL_PARAMETER_COUNT_DEFINITION,
            "preprocessing": preprocessing,
            "effective_model_config": effective_config,
        }
    local_fit, local_validation = grouped_validation_indices(
        runtime.task.game_ids[train_index],
        validation_split_seed,
        strata=runtime.task.strata[train_index],
    )
    selector_train_index = train_index[local_fit]
    selector_validation_index = train_index[local_validation]
    selector_inputs = None
    if neural_selector_core is None:
        selector_inputs, selector_preprocessing = runtime.neural_inputs(
            family,
            selector_train_index,
            selector_validation_index,
            fit_scope=NEURAL_SELECTOR_PREPROCESSING_SCOPE,
            ablation_id=ablation_id,
            include_team_identity=include_team_identity,
        )
    else:
        selection = NeuralSelection.from_mapping(neural_selector_core["selection"])
        selector_preprocessing = dict(neural_selector_core["selector_preprocessing"])
    final_inputs, preprocessing = runtime.neural_inputs(
        family,
        train_index,
        *prediction_indices,
        fit_scope=NEURAL_FINAL_PREPROCESSING_SCOPE,
        ablation_id=ablation_id,
        include_team_identity=include_team_identity,
    )
    selector_train_y = selector_train_mask = None
    selector_validation_y = selector_validation_mask = None
    if neural_selector_core is None:
        selector_train_y, selector_train_mask = _training_target(
            task, selector_train_index, trajectory_target=trajectory_target
        )
        selector_validation_y, selector_validation_mask = _training_target(
            task, selector_validation_index, trajectory_target=trajectory_target
        )
    representative_input = None
    representative_shared_input = None
    head_loss_signature = None
    if family in {*RELATIONAL_FAMILIES, *GLOBAL_SET_FAMILIES}:
        representative_input = representative_neural_input_signature(final_inputs[0])
        representative_shared_input = representative_shared_neural_input_signature(
            final_inputs[0]
        )
        signature_config = dict(effective_config)
        if (
            outcome == "distribution"
            and signature_config.get("output_contract")
            == PUNT_CDF_RESIDUAL_CONTRACT
        ):
            signature_config["cdf_baseline"] = empirical_cdf_baseline(
                train_y, n_outputs
            ).tolist()
        head_loss_signature = neural_head_loss_signature(
            outcome,
            n_outputs=n_outputs,
            max_horizon=(int(np.asarray(train_y).shape[1]) if outcome == "trajectory" else 1),
            config=signature_config,
        )
    if neural_selector_core is None:
        assert selector_inputs is not None
        fit = fit_neural_explicit_preprocessing(
            family,
            selector_inputs[0],
            selector_train_y,
            selector_inputs[1],
            selector_validation_y,
            final_inputs[0],
            train_y,
            outcome_type=outcome,
            n_outputs=n_outputs,
            selector_train_mask=selector_train_mask,
            selector_validation_mask=selector_validation_mask,
            final_mask=train_mask,
            config=effective_config,
            selection_seed=selection_seed,
            refit_seed=refit_seed,
            validation_game_ids=tuple(
                sorted(
                    str(value)
                    for value in np.unique(task.game_ids[selector_validation_index])
                )
            ),
            validation_split_seed=validation_split_seed,
        )
    else:
        fit = refit_neural_from_selection_explicit_preprocessing(
            family,
            final_inputs[0],
            train_y,
            outcome_type=outcome,
            n_outputs=n_outputs,
            final_mask=train_mask,
            config=effective_config,
            refit_seed=refit_seed,
            selection=selection,
        )
    predictions = [
        predict_neural(
            fit.model,
            values,
            batch_size=int(config.get("batch_size", 64)),
        )
        for values in final_inputs[1:]
    ]
    if outcome == "binary":
        predictions = [values.reshape(-1) for values in predictions]
    if outcome == "trajectory" and trajectory_target == "absolute":
        predictions = [
            np.asarray(values) - np.asarray(task.target_baseline[index])
            for values, index in zip(predictions, prediction_indices)
        ]
    history = {
        "family": family,
        "parameter_count": fit.parameter_count,
        "parameter_count_definition": NEURAL_PARAMETER_COUNT_DEFINITION,
        "best_epoch": fit.best_epoch,
        "selector_history": fit.selector_history,
        "refit_history": fit.refit_history,
        "preprocessing": preprocessing,
        "selector_preprocessing": selector_preprocessing,
        "validation_game_ids": list(fit.validation_game_ids),
        "validation_split_seed": fit.validation_split_seed,
        "training_seeds": {
            "epoch_selection": int(selection_seed),
            "refit": int(refit_seed),
            "validation_split": int(validation_split_seed),
        },
        "selector_train_game_ids": sorted(
            str(value) for value in np.unique(task.game_ids[selector_train_index])
        ),
        "selector_fit_games": int(len(np.unique(task.game_ids[selector_train_index]))),
        "selector_fit_examples": int(len(selector_train_index)),
        "final_fit_games": int(len(np.unique(runtime.task.game_ids[train_index]))),
        "final_fit_examples": int(len(train_index)),
        "effective_model_config": effective_config,
        "ablation_id": ablation_id,
        "sensitivity_id": sensitivity_id,
    }
    if family in {*RELATIONAL_FAMILIES, *GLOBAL_SET_FAMILIES}:
        history["representative_neural_input"] = representative_input
        history["representative_shared_neural_input"] = representative_shared_input
        history["output_head_loss_signature"] = head_loss_signature
    if fit.fairness_metadata:
        history["matched_architecture"] = fit.fairness_metadata
        history["representative_forward_flops"] = int(
            fit.fairness_metadata["representative_forward_flops"]
        )
    if fit.complexity_metadata:
        history["global_set_architecture"] = fit.complexity_metadata
        history["representative_forward_flops"] = int(
            fit.complexity_metadata["representative_forward_flops"]
        )
    # Hundreds of cells run sequentially in one GPU worker. Release the final
    # graph after predictions so model/session state cannot accumulate.
    del fit.model
    import tensorflow as tf

    tf.keras.backend.clear_session()
    return predictions, history


def _identity_frame(task: PreparedTask, index: np.ndarray, partition: str) -> pd.DataFrame:
    columns = [column for column in ("example_id", "game_id", "stratum", "target") if column in task.examples]
    frame = task.examples.iloc[index][columns].reset_index(drop=True).copy()
    frame.insert(0, "partition", partition)
    return frame


def _game_equal_quantile(
    values: np.ndarray, game_ids: np.ndarray, quantile: float
) -> float:
    observed = np.asarray(values, dtype=float).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    if len(observed) != len(games) or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("game-equal quantile inputs are invalid")
    return float(
        np.mean(
            [
                np.quantile(observed[games == game], quantile)
                for game in np.unique(games)
            ]
        )
    )


def _game_equal_label_mean(
    values: np.ndarray,
    labels: np.ndarray,
    game_ids: np.ndarray,
    *,
    label: int,
) -> float:
    observed = np.asarray(values, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    if not (len(observed) == len(y) == len(games)):
        raise ValueError("label-wise game-equal inputs are misaligned")
    per_game = [
        float(np.mean(observed[(games == game) & (y == int(label))]))
        for game in np.unique(games)
        if np.any((games == game) & (y == int(label)))
    ]
    if not per_game:
        raise ValueError(f"no test game contains binary label {label}")
    return float(np.mean(per_game))


def _game_equal_reliability(
    probability: np.ndarray,
    labels: np.ndarray,
    game_ids: np.ndarray,
    boundaries: np.ndarray,
) -> float:
    """Average per-game RMS reliability using calibration-frozen bins."""

    p = np.asarray(probability, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    games = np.asarray(game_ids).reshape(-1)
    bins = np.searchsorted(np.asarray(boundaries, dtype=float), p, side="right")
    if not (len(p) == len(y) == len(games)):
        raise ValueError("game-equal reliability inputs are misaligned")
    per_game: list[float] = []
    for game in np.unique(games):
        selected_game = games == game
        squared = 0.0
        count = 0
        for value in np.unique(bins[selected_game]):
            selected = selected_game & (bins == value)
            n_bin = int(np.sum(selected))
            difference = float(np.mean(y[selected]) - np.mean(p[selected]))
            squared += n_bin * difference**2
            count += n_bin
        per_game.append(math.sqrt(squared / count))
    return float(np.mean(per_game))


def _binary_result(
    task: PreparedTask,
    train_index: np.ndarray,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
    calibration_probability: np.ndarray,
    test_probability: np.ndarray,
    *,
    uncertainty_seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, np.ndarray]]:
    train_y = np.asarray(task.y[train_index], dtype=int)
    calibration_y = np.asarray(task.y[calibration_index], dtype=int)
    test_y = np.asarray(task.y[test_index], dtype=int)
    calibration_probability = np.asarray(calibration_probability).reshape(-1)
    test_probability = np.asarray(test_probability).reshape(-1)
    # One fitted Venn--Abers calibrator supplies both the calibration-score
    # registry used to freeze diagnostic bins/sets and the untouched test
    # probabilities.  Concatenating prediction inputs avoids fitting the
    # calibrator twice while keeping test labels completely out of the fit.
    calibrated_joint = fit_venn_abers(
        calibration_probability,
        calibration_y,
        np.concatenate([calibration_probability, test_probability]),
    )
    calibration_count = len(calibration_probability)
    calibration_calibrated_probability = (
        calibrated_joint.calibrated_probability[:calibration_count]
    )
    raw_test_probability = calibrated_joint.raw_probability[calibration_count:]
    calibrated_test_probability = (
        calibrated_joint.calibrated_probability[calibration_count:]
    )
    test_p0 = calibrated_joint.p0[calibration_count:]
    test_p1 = calibrated_joint.p1[calibration_count:]
    test_imprecision = calibrated_joint.imprecision[calibration_count:]
    # Class-conditional split-conformal sets use the raw model score. Using
    # in-sample Venn--Abers values here would reuse calibration labels inside
    # the nonconformity score and invalidate the split-conformal construction.
    prediction_sets = label_conditional_prediction_sets(
        calibration_probability,
        calibration_y,
        task.game_ids[calibration_index],
        raw_test_probability,
        seed=int(uncertainty_seed),
    )
    raw_contribution = brier_contributions(test_y, raw_test_probability)
    calibrated_contribution = brier_contributions(
        test_y, calibrated_test_probability
    )
    calibrated_log_contribution = binary_log_loss_contributions(
        test_y, calibrated_test_probability
    )
    null_probability = prevalence_null(train_y)
    null_contribution = brier_contributions(test_y, np.full(len(test_y), null_probability))
    reliability, reliability_boundaries = rms_reliability_error(
        calibration_calibrated_probability,
        calibrated_test_probability,
        test_y,
    )
    set_included = np.asarray(prediction_sets["included"], dtype=bool)
    set_size = np.asarray(prediction_sets["set_size"], dtype=int)
    if set(np.unique(test_y)) != {0, 1}:
        raise ValueError("binary test partition must contain both classes")
    label0 = test_y == 0
    label1 = test_y == 1
    test_games = task.game_ids[test_index]
    raw_log_contribution = binary_log_loss_contributions(
        test_y, raw_test_probability
    )
    entropy = bernoulli_entropy(calibrated_test_probability)
    metrics = {
        "primary_loss": float(raw_contribution.mean()),
        "null_loss": float(null_contribution.mean()),
        "game_equal_loss": game_equal_mean(raw_contribution, task.game_ids[test_index]),
        "game_equal_brier": game_equal_mean(raw_contribution, task.game_ids[test_index]),
        "raw_brier": float(raw_contribution.mean()),
        "calibrated_brier": float(calibrated_contribution.mean()),
        "raw_log_loss": float(raw_log_contribution.mean()),
        "calibrated_log_loss": float(calibrated_log_contribution.mean()),
        "game_equal_calibrated_brier": game_equal_mean(
            calibrated_contribution, task.game_ids[test_index]
        ),
        "calibration_bias": calibration_bias(test_y, calibrated_test_probability),
        "rms_reliability": float(reliability),
        "mean_entropy": float(bernoulli_entropy(calibrated_test_probability).mean()),
        "null_probability": float(null_probability),
        "mean_va_imprecision": float(test_imprecision.mean()),
        "p90_va_imprecision": float(np.quantile(test_imprecision, 0.90)),
        "label0_set_coverage": float(set_included[label0, 0].mean()),
        "label1_set_coverage": float(set_included[label1, 1].mean()),
        "set_singleton_rate": float(np.mean(set_size == 1)),
        "set_doubleton_rate": float(np.mean(set_size == 2)),
        "set_empty_rate": float(np.mean(set_size == 0)),
        "game_equal_raw_log_loss": game_equal_mean(
            raw_log_contribution, test_games
        ),
        "game_equal_calibrated_log_loss": game_equal_mean(
            calibrated_log_contribution, test_games
        ),
        "game_equal_calibration_bias": game_equal_mean(
            test_y - calibrated_test_probability, test_games
        ),
        "game_equal_rms_reliability": _game_equal_reliability(
            calibrated_test_probability,
            test_y,
            test_games,
            reliability_boundaries,
        ),
        "game_equal_mean_entropy": game_equal_mean(entropy, test_games),
        "game_equal_mean_va_imprecision": game_equal_mean(
            test_imprecision, test_games
        ),
        "game_equal_p90_va_imprecision": _game_equal_quantile(
            test_imprecision, test_games, 0.90
        ),
        "game_equal_label0_set_coverage": _game_equal_label_mean(
            set_included[:, 0], test_y, test_games, label=0
        ),
        "game_equal_label1_set_coverage": _game_equal_label_mean(
            set_included[:, 1], test_y, test_games, label=1
        ),
        "game_equal_set_singleton_rate": game_equal_mean(
            (set_size == 1).astype(float), test_games
        ),
        "game_equal_set_doubleton_rate": game_equal_mean(
            (set_size == 2).astype(float), test_games
        ),
        "game_equal_set_empty_rate": game_equal_mean(
            (set_size == 0).astype(float), test_games
        ),
        "uncertainty_subsample_seed": int(uncertainty_seed),
    }
    calibration_frame = _identity_frame(task, calibration_index, "calibration")
    calibration_frame["raw_probability"] = np.asarray(calibration_probability).reshape(-1)
    for column in ("calibrated_probability", "venn_abers_p0", "venn_abers_p1", "venn_abers_imprecision", "loss_contribution"):
        calibration_frame[column] = np.nan
    test_frame = _identity_frame(task, test_index, "test")
    test_frame["raw_probability"] = raw_test_probability
    if task.task_id == "bdb2024_tackle":
        test_frame["raw_case_control_score"] = raw_test_probability
    test_frame["calibrated_probability"] = calibrated_test_probability
    test_frame["venn_abers_probability"] = calibrated_test_probability
    test_frame["venn_abers_p0"] = test_p0
    test_frame["venn_abers_p1"] = test_p1
    test_frame["venn_abers_imprecision"] = test_imprecision
    test_frame["set_includes_0"] = set_included[:, 0]
    test_frame["set_includes_1"] = set_included[:, 1]
    test_frame["conformal_set_size"] = set_size
    test_frame["loss_contribution"] = raw_contribution
    arrays = {
        "binary_calibration_calibrated_probability": np.asarray(
            calibration_calibrated_probability, dtype=np.float64
        ),
        "binary_reliability_boundaries": np.asarray(
            reliability_boundaries, dtype=np.float64
        ),
        "binary_set_quantiles": np.asarray(
            prediction_sets["quantiles"], dtype=np.float64
        ),
        "binary_set_ranks": np.asarray(prediction_sets["ranks"], dtype=np.int64),
        "binary_set_label0_calibration_indices": np.asarray(
            prediction_sets["selected_label0_indices"], dtype=np.int64
        ),
        "binary_set_label1_calibration_indices": np.asarray(
            prediction_sets["selected_label1_indices"], dtype=np.int64
        ),
        "uncertainty_subsample_seed": np.asarray(
            [int(uncertainty_seed)], dtype=np.int64
        ),
    }
    return metrics, pd.concat([calibration_frame, test_frame], ignore_index=True), arrays


def _distribution_result(
    task: PreparedTask,
    train_index: np.ndarray,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
    calibration_probability: np.ndarray,
    test_probability: np.ndarray,
    *,
    uncertainty_seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, np.ndarray]]:
    train_y = np.asarray(task.y[train_index], dtype=int)
    calibration_y = np.asarray(task.y[calibration_index], dtype=int)
    test_y = np.asarray(task.y[test_index], dtype=int)
    contribution = crps_contributions(test_y, test_probability)
    null = empirical_distribution_null(train_y, test_probability.shape[1])
    null_contribution = crps_contributions(test_y, np.repeat(null[None, :], len(test_y), axis=0))
    intervals = hierarchical_distribution_intervals(
        calibration_probability,
        calibration_y,
        task.game_ids[calibration_index],
        test_probability,
        alpha=0.10,
        seed=int(uncertainty_seed),
    )
    covered = (test_y >= intervals["lower"]) & (test_y <= intervals["upper"])
    width = intervals["width"].astype(float)
    coverage_game_equal = game_equal_mean(
        covered.astype(float), task.game_ids[test_index]
    )
    width_game_equal = game_equal_mean(width, task.game_ids[test_index])
    metrics = {
        "primary_loss": float(contribution.mean()),
        "null_loss": float(null_contribution.mean()),
        "game_equal_loss": game_equal_mean(contribution, task.game_ids[test_index]),
        "game_equal_crps": game_equal_mean(contribution, task.game_ids[test_index]),
        "crps": float(contribution.mean()),
        "raw_crps": float(contribution.mean()),
        "coverage": coverage_game_equal,
        "coverage_game_equal": coverage_game_equal,
        "coverage_example_weighted": float(covered.mean()),
        "interval_width": width_game_equal,
        "width_game_equal": width_game_equal,
        "interval_width_example_weighted": float(width.mean()),
        "inclusive_class_count": float(np.mean(width + 1.0)),
        "interval_width_sd": float(width.std(ddof=1)) if len(width) > 1 else 0.0,
        "uncertainty_subsample_seed": int(uncertainty_seed),
    }
    calibration_frame = _identity_frame(task, calibration_index, "calibration")
    calibration_frame["loss_contribution"] = np.nan
    test_frame = _identity_frame(task, test_index, "test")
    test_frame["loss_contribution"] = contribution
    support = np.asarray(task.support, dtype=float)
    test_frame["interval_lower_index"] = intervals["lower"]
    test_frame["interval_upper_index"] = intervals["upper"]
    test_frame["interval_lower"] = support[intervals["lower"]]
    test_frame["interval_upper"] = support[intervals["upper"]]
    test_frame["conformal_padding"] = intervals["padding"]
    test_frame["covered"] = covered
    arrays = {
        # Persist the precision used by the metric functions.  Casting a
        # classical model's float64 output to float32 here can change a
        # recomputed CRPS, conformal tie, or interval endpoint.
        "calibration_probabilities": np.asarray(calibration_probability, dtype=np.float64),
        "test_probabilities": np.asarray(test_probability, dtype=np.float64),
        "null_probability": np.asarray(null, dtype=np.float64),
        "conformal_selected_calibration_indices": np.asarray(
            intervals["selected_calibration_indices"], dtype=np.int64
        ),
        "conformal_quantile": np.asarray([intervals["quantile"]], dtype=np.float64),
        "conformal_rank": np.asarray([intervals["rank"]], dtype=np.int64),
        "uncertainty_subsample_seed": np.asarray(
            [int(uncertainty_seed)], dtype=np.int64
        ),
    }
    return metrics, pd.concat([calibration_frame, test_frame], ignore_index=True), arrays


def _game_equal_trajectory_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    game_ids: np.ndarray,
) -> float:
    values = []
    for game_id in np.unique(game_ids):
        selected = np.asarray(game_ids) == game_id
        values.append(trajectory_rmse(truth[selected], prediction[selected], mask[selected]))
    return float(np.mean(values))


def _trajectory_result(
    task: PreparedTask,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
    calibration_residual: np.ndarray,
    test_residual: np.ndarray,
    *,
    horizon_scale: np.ndarray | None,
    uncertainty_seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, np.ndarray]]:
    calibration_residual = np.asarray(calibration_residual).reshape(np.asarray(task.target_values[calibration_index]).shape)
    test_residual = np.asarray(test_residual).reshape(np.asarray(task.target_values[test_index]).shape)
    cal_absolute = np.asarray(task.target_baseline[calibration_index]) + calibration_residual
    test_absolute = np.asarray(task.target_baseline[test_index]) + test_residual
    calibration_truth = np.asarray(task.target_values[calibration_index])
    calibration_mask = np.asarray(task.target_mask[calibration_index], dtype=bool)
    test_truth = np.asarray(task.target_values[test_index])
    test_mask = np.asarray(task.target_mask[test_index], dtype=bool)
    model_rmse = trajectory_rmse(test_truth, test_absolute, test_mask)
    null_rmse = trajectory_rmse(test_truth, np.asarray(task.target_baseline[test_index]), test_mask)
    per_trajectory = trajectory_squared_error(test_truth, test_absolute, test_mask)
    per_path_rmse = trajectory_path_rmse(test_truth, test_absolute, test_mask)
    if horizon_scale is None:
        raise ValueError("trajectory evaluation requires a frozen development horizon scale")
    scale = np.asarray(horizon_scale, dtype=np.float64).reshape(-1)
    if task.task_id == "bdb2026_trajectory" and scale.shape != (94,):
        raise ValueError("BDB2026 trajectory evaluation requires exactly 94 horizon scales")
    tube = pathwise_conformal_tube(
        calibration_truth,
        cal_absolute,
        calibration_mask,
        task.game_ids[calibration_index],
        test_truth,
        test_absolute,
        test_mask,
        scale,
        seed=int(uncertainty_seed),
    )
    path_covered = np.asarray(tube["path_covered"], dtype=bool)
    mean_diameter = np.asarray(tube["per_path_mean_diameter"], dtype=float)
    metrics = {
        "primary_loss": model_rmse,
        "null_loss": null_rmse,
        "game_equal_loss": _game_equal_trajectory_rmse(
            test_truth, test_absolute, test_mask, task.game_ids[test_index]
        ),
        "rmse": model_rmse,
        "official_pooled_rmse": model_rmse,
        "game_equal_rmse": _game_equal_trajectory_rmse(
            test_truth, test_absolute, test_mask, task.game_ids[test_index]
        ),
        "path_equal_rmse": float(per_path_rmse.mean()),
        "path_coverage": float(path_covered.mean()),
        "game_equal_path_coverage": game_equal_mean(
            path_covered.astype(float), task.game_ids[test_index]
        ),
        "mean_tube_diameter": float(mean_diameter.mean()),
        "game_equal_tube_diameter": game_equal_mean(
            mean_diameter, task.game_ids[test_index]
        ),
        "uncertainty_subsample_seed": int(uncertainty_seed),
    }
    horizon_rmse = []
    for horizon in range(test_truth.shape[1]):
        valid = test_mask[:, horizon]
        if not np.any(valid):
            horizon_rmse.append(np.nan)
            continue
        error = test_absolute[valid, horizon] - test_truth[valid, horizon]
        horizon_rmse.append(float(np.sqrt(np.mean(np.square(error)))))
    calibration_frame = _identity_frame(task, calibration_index, "calibration")
    calibration_frame["loss_contribution"] = np.nan
    test_frame = _identity_frame(task, test_index, "test")
    test_frame["loss_contribution"] = per_trajectory
    test_frame["path_rmse"] = per_path_rmse
    test_frame["path_covered"] = path_covered
    test_frame["mean_tube_diameter"] = mean_diameter
    arrays = {
        # Keep scientific artifacts at the precision used to compute the
        # reported metrics so aggregate validation can reproduce them exactly.
        "calibration_prediction": np.asarray(cal_absolute, dtype=np.float64),
        "calibration_target": np.asarray(calibration_truth, dtype=np.float64),
        "calibration_mask": np.asarray(calibration_mask, dtype=bool),
        "test_prediction": np.asarray(test_absolute, dtype=np.float64),
        "test_predicted_residual": np.asarray(test_residual, dtype=np.float64),
        "test_target": np.asarray(test_truth, dtype=np.float64),
        "test_baseline": np.asarray(task.target_baseline[test_index], dtype=np.float64),
        "test_mask": test_mask,
        "horizon_rmse": np.asarray(horizon_rmse, dtype=np.float64),
        "path_tube_horizon_scale": np.asarray(tube["horizon_scale"], dtype=np.float64),
        "path_tube_field_bounds": np.asarray(tube["field_bounds"], dtype=np.float64),
        "path_tube_radius": np.asarray(tube["radius"], dtype=np.float64),
        "path_tube_quantile": np.asarray([tube["quantile"]], dtype=np.float64),
        "path_tube_rank": np.asarray([tube["rank"]], dtype=np.int64),
        "path_tube_selected_calibration_indices": np.asarray(
            tube["selected_calibration_indices"], dtype=np.int64
        ),
        "uncertainty_subsample_seed": np.asarray(
            [int(uncertainty_seed)], dtype=np.int64
        ),
    }
    return metrics, pd.concat([calibration_frame, test_frame], ignore_index=True), arrays


def run_cell(
    runtime: TaskRuntime,
    task_spec: TaskSpec | Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    repeat: int,
    n_train: int,
    model_id: str,
    branch: str = "fixed_main",
    ablation_id: str | None = None,
    sensitivity_id: str | None = None,
    neural_selector_core: Mapping[str, Any] | None = None,
) -> CellComputation:
    """Fit, calibrate where applicable, and evaluate one immutable cell."""

    task = runtime.task
    setup = _resolve_cell_setup(
        runtime,
        task_spec,
        design,
        repeat=repeat,
        n_train=n_train,
        model_id=model_id,
        branch=branch,
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
    )
    split = setup.split
    train_games = list(setup.train_games)
    calibration_games = list(setup.calibration_games)
    test_games = list(setup.test_games)
    train_index = setup.train_index
    calibration_index = setup.calibration_index
    test_index = setup.test_index
    family = setup.family
    config = setup.config
    seeds = setup.seeds
    if neural_selector_core is not None:
        if family in {"glm", "lightgbm"}:
            raise ValueError("tabular cells cannot consume a neural selector receipt")
        neural_selector_core = _validate_neural_selector_core(
            neural_selector_core,
            task,
            setup,
            repeat=repeat,
            n_train=n_train,
            model_id=model_id,
            branch=branch,
            ablation_id=ablation_id,
            sensitivity_id=sensitivity_id,
        )
    uncertainty_seed = int(seeds["uncertainty_subsample"])
    (calibration_prediction, test_prediction), fit_history = _fit_and_predict(
        runtime,
        family,
        config,
        train_index,
        (calibration_index, test_index),
        fit_seed=int(seeds["fit"]),
        validation_split_seed=int(seeds["validation_split"]),
        selection_seed=int(seeds["epoch_selection"]),
        refit_seed=int(seeds["refit"]),
        ablation_id=ablation_id,
        sensitivity_id=sensitivity_id,
        neural_selector_core=neural_selector_core,
    )
    outcome = _modeled_outcome(task)
    if outcome == "binary":
        metrics, predictions, arrays = _binary_result(
            task, train_index, calibration_index, test_index,
            calibration_prediction, test_prediction,
            uncertainty_seed=uncertainty_seed,
        )
    elif outcome == "distribution":
        metrics, predictions, arrays = _distribution_result(
            task, train_index, calibration_index, test_index,
            calibration_prediction, test_prediction,
            uncertainty_seed=uncertainty_seed,
        )
    elif outcome == "trajectory":
        metrics, predictions, arrays = _trajectory_result(
            task, calibration_index, test_index,
            calibration_prediction, test_prediction,
            horizon_scale=config.get("dev_horizon_scale"),
            uncertainty_seed=uncertainty_seed,
        )
    else:
        raise ValueError(f"unsupported outcome {outcome!r}")
    metrics.update(
        {
            "task_id": task.task_id,
            "branch": str(branch),
            "ablation_id": ablation_id,
            "sensitivity_id": sensitivity_id,
            "repeat": int(repeat),
            "n_train": int(n_train),
            "model": model_id,
            "family": family,
            "skill": skill_score(float(metrics["primary_loss"]), float(metrics["null_loss"])),
            "n_train_examples": int(len(train_index)),
            "n_calibration_examples": int(len(calibration_index)),
            "n_test_examples": int(len(test_index)),
            "outer_split_hash": split["outer_split_hash"],
            "nested_split_hash": split["nested_split_hash"],
            "model_config": config,
            "seeds": seeds,
            "uncertainty_subsample_seed": uncertainty_seed,
        }
    )
    fit_history.update(
        {
            "model_config": config,
            "seeds": seeds,
            "train_game_ids": train_games,
            "calibration_game_ids": calibration_games,
            "test_game_ids": test_games,
            "branch": str(branch),
            "ablation_id": ablation_id,
            "sensitivity_id": sensitivity_id,
            "uncertainty_subsample_seed": uncertainty_seed,
        }
    )
    return CellComputation(metrics, predictions, fit_history, arrays)


def development_evaluator(runtime: TaskRuntime):
    """Build the callback consumed by :func:`bdb_study.devcv.run_development_cv`."""

    def evaluate(
        family: str,
        config: Mapping[str, Any],
        train_index: np.ndarray,
        validation_index: np.ndarray,
        fit_seed: int,
        prediction_seed: int,
    ) -> tuple[float, Mapping[str, Any]]:
        (prediction,), history = _fit_and_predict(
            runtime,
            family,
            config,
            train_index,
            (validation_index,),
            fit_seed=fit_seed,
            validation_split_seed=fit_seed,
            selection_seed=fit_seed,
            refit_seed=prediction_seed,
        )
        task = runtime.task
        outcome = _modeled_outcome(task)
        if outcome == "binary":
            loss = float(brier_contributions(task.y[validation_index], prediction).mean())
        elif outcome == "distribution":
            loss = float(crps_contributions(task.y[validation_index], prediction).mean())
        elif outcome == "trajectory":
            residual = np.asarray(prediction).reshape(np.asarray(task.target_values[validation_index]).shape)
            absolute = np.asarray(task.target_baseline[validation_index]) + residual
            loss = trajectory_rmse(
                task.target_values[validation_index], absolute, task.target_mask[validation_index]
            )
            error = np.asarray(task.target_values[validation_index]) - absolute
            radial = np.sqrt(np.sum(np.square(error), axis=-1))
            mask = np.asarray(task.target_mask[validation_index], dtype=bool)
            horizon_median: list[float | None] = []
            horizon_count: list[int] = []
            for horizon in range(mask.shape[1]):
                observed = mask[:, horizon]
                count = int(np.sum(observed))
                horizon_count.append(count)
                horizon_median.append(
                    None if count == 0 else float(np.median(radial[observed, horizon]))
                )
            history["horizon_scale_median"] = horizon_median
            history["horizon_scale_count"] = horizon_count
        else:
            raise ValueError(f"unsupported outcome {outcome!r}")
        return loss, history

    return evaluate
