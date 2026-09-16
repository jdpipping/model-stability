"""Immutable design and provenance utilities for the rushing confirmatory study.

The public entry point is :func:`build_study_manifest`.  Its ``game_table``
argument must describe exactly one row per NFL game.  It may be a pandas-like
table, a sequence of mappings, or ``{season: [game_id, ...]}``.  For tabular
inputs, column names come from ``config['data_assumptions']`` (``GameId`` and
``Season`` in v1).  Game IDs are serialized as non-empty strings, must be
globally unique, and may not end in ``_aug``.  If a play-ID column is present,
augmented play IDs are rejected as well.

All split membership is materialized in JSON-friendly records.  Runners should
consume ``manifest['split_manifests']`` and obtain randomness only through
``manifest_seed``; they should not reproduce the key encoding themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable

from packaging.requirements import InvalidRequirement, Requirement


BASE_SEED = 20260817
SEED_ALGORITHM = "hmac-sha256-31bit-v1"
LOCKED_FINAL_CONFIG_SHA256 = (
    "cbaabbe51ba976059e5ea744937bd281e3d1e06ff479fe34e3f3e70b8efd8aab"
)
SMOKE_DESCRIPTION_PREFIX = "SMOKE ONLY: "
SEASONS = (2017, 2018, 2019)
OUTER_COUNTS = {
    "2017": {"train": 154, "calibration": 51, "test": 51},
    "2018": {"train": 154, "calibration": 51, "test": 51},
    "2019": {"train": 106, "calibration": 35, "test": 35},
}
TUNE_COUNTS = {"2017": 15, "2018": 15, "2019": 10}
MAIN_ANCHORS = (20, 40, 80, 160, 240, 360)
SENSITIVITY_ANCHORS = (20, 160, 360)
DEFAULT_CONFIRMATORY_REPEATS = 50
FULL_CONFIRMATORY_REPEATS = 100
ALLOWED_CONFIRMATORY_REPEATS = (
    DEFAULT_CONFIRMATORY_REPEATS,
    FULL_CONFIRMATORY_REPEATS,
)
SEQUENTIAL_EXECUTION_PROFILE = "sequential_gpu1"
HYBRID_EXECUTION_PROFILE = "hybrid_cpu12_gpu1"
TABULAR_MODEL_IDS = ("ridge_sgd_l2", "lightgbm_multiclass")
NEURAL_MODEL_IDS = ("zoo_cnn", "set_transformer")
HYBRID_EXECUTION_QUEUES = {
    "cpu_tabular": {
        "device": "cpu",
        "workers": 12,
        "models": list(TABULAR_MODEL_IDS),
    },
    "gpu_neural": {
        "device": "gpu",
        "workers": 1,
        "models": list(NEURAL_MODEL_IDS),
    },
}
CANONICAL_MODEL_IDS = (
    "ridge_sgd_l2",
    "lightgbm_multiclass",
    "zoo_cnn",
    "set_transformer",
)
CELL_STAGES = ("fit", "epoch_selection", "refit", "prediction_rebuild")
DETERMINISTIC_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": str(BASE_SEED),
    "TF_DETERMINISTIC_OPS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}
REQUIRED_CODE_PROVENANCE = {
    "neural_models": "rushing_study/neural_models.py",
    "prepare_data": "scripts/prepare_data.py",
    "requirements_lock": "requirements-lock.txt",
    "rushing_study___init__": "rushing_study/__init__.py",
    "rushing_study___main__": "rushing_study/__main__.py",
    "rushing_study_analysis": "rushing_study/analysis.py",
    "rushing_study_cli": "rushing_study/cli.py",
    "rushing_study_data": "rushing_study/data.py",
    "rushing_study_design": "rushing_study/design.py",
    "rushing_study_execution": "rushing_study/execution.py",
    "rushing_study_intervals": "rushing_study/intervals.py",
    "rushing_study_metrics": "rushing_study/metrics.py",
    "rushing_study_models": "rushing_study/models.py",
    "rushing_study_runner": "rushing_study/runner.py",
    "rushing_study_storage": "rushing_study/storage.py",
    "study_config": "configs/rushing_confirmatory_v1.json",
}


class DesignError(ValueError):
    """Base class for study-design failures."""


class ConfigValidationError(DesignError):
    """The immutable configuration is missing or internally inconsistent."""


class IdentifierValidationError(DesignError):
    """A game/play identifier is augmented, duplicated, or malformed."""


class SeedCollisionError(DesignError):
    """Two semantic seed keys mapped to the same bounded integer seed."""


class ProvenanceError(DesignError):
    """Required code, data, environment, or repository provenance is invalid."""


class ManifestCollisionError(DesignError):
    """An immutable manifest bundle path already contains different bytes."""


def _canonicalize(value: Any) -> Any:
    """Convert supported values to an unambiguous JSON-compatible structure."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DesignError("Canonical JSON does not permit NaN or infinity.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DesignError(f"Canonical JSON mapping keys must be strings; got {key!r}.")
            if key in out:
                raise DesignError(f"Duplicate canonical JSON key: {key!r}.")
            out[key] = _canonicalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    # NumPy scalar values expose item(), but importing NumPy here would make the
    # design layer unnecessarily dependent on the modeling environment.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        item = item_method()
        if item is not value:
            return _canonicalize(item)
    raise DesignError(f"Unsupported canonical JSON value: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    """Return stable UTF-8 JSON text with sorted keys and no insignificant space."""

    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Return the full SHA-256 hex digest of canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_run_hash(value: Any, length: int = 24) -> str:
    """Return a stable, prefix-safe SHA-256 identifier for a run payload."""

    if not isinstance(length, int) or not 12 <= length <= 64:
        raise DesignError("Run-hash length must be an integer from 12 through 64.")
    return sha256_json(value)[:length]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{path} must be a JSON object.")
    return value


def _exact(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise ConfigValidationError(f"{path} must equal {expected!r}; got {value!r}.")


def _validate_locked_config_identity(config: Mapping[str, Any]) -> None:
    """Accept the frozen 50/100 profiles or their explicit CLI smoke projections."""

    def default_profile_hash(value: Mapping[str, Any]) -> str:
        projected = deepcopy(_canonicalize(value))
        execution = projected.get("execution")
        if not isinstance(execution, dict):
            return sha256_json(projected)
        repeats = execution.get("confirmatory_repeats")
        if repeats == FULL_CONFIRMATORY_REPEATS:
            execution["confirmatory_repeats"] = DEFAULT_CONFIRMATORY_REPEATS
        if execution.get("profile") == HYBRID_EXECUTION_PROFILE:
            execution.pop("profile", None)
            execution.pop("queues", None)
        return sha256_json(projected)

    actual_hash = sha256_json(config)
    execution = config.get("execution")
    plan = execution.get("plan") if isinstance(execution, Mapping) else None
    if plan == "final":
        if default_profile_hash(config) != LOCKED_FINAL_CONFIG_SHA256:
            raise ConfigValidationError(
                "Final rushing_confirmatory_v1 configuration differs from its immutable "
                "50/100-repeat contracts: expected the locked default SHA-256 "
                f"{LOCKED_FINAL_CONFIG_SHA256} or its exact 100-repeat projection; got "
                f"{actual_hash}."
            )
        return

    if plan != "smoke":
        return  # The structural validation reports the invalid plan.
    projected = deepcopy(_canonicalize(config))
    neural_training = projected.get("neural_training")
    description = projected.get("description")
    if not isinstance(neural_training, dict):
        raise ConfigValidationError("neural_training must be a JSON object.")
    max_epochs = neural_training.get("max_epochs")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ConfigValidationError(
            "Smoke neural_training.max_epochs must be a positive integer."
        )
    if neural_training.get("early_stopping_patience") != 0:
        raise ConfigValidationError(
            "Smoke neural_training.early_stopping_patience must equal 0."
        )
    if not isinstance(description, str) or not description.startswith(
        SMOKE_DESCRIPTION_PREFIX
    ):
        raise ConfigValidationError(
            f"Smoke description must begin with {SMOKE_DESCRIPTION_PREFIX!r}."
        )

    # Reverse exactly the four assignments performed by cli.command_plan.  Any
    # additional change, including an added field, leaves a different hash.
    projected["execution"]["plan"] = "final"
    projected["neural_training"]["max_epochs"] = 50
    projected["neural_training"]["early_stopping_patience"] = 10
    projected["description"] = description[len(SMOKE_DESCRIPTION_PREFIX) :]
    projected_hash = default_profile_hash(projected)
    if projected_hash != LOCKED_FINAL_CONFIG_SHA256:
        raise ConfigValidationError(
            "Smoke configuration may differ from frozen v1 only by execution.plan, "
            "the positive neural epoch limit, zero early-stopping patience, and the "
            f"{SMOKE_DESCRIPTION_PREFIX!r} description prefix, with either the locked "
            "50-repeat default or exact 100-repeat full profile; projected SHA-256 was "
            f"{projected_hash}."
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the complete v1 confirmatory-design contract."""

    if not isinstance(config, Mapping):
        raise ConfigValidationError("Configuration root must be a JSON object.")
    _exact(config.get("schema_version"), 1, "schema_version")
    _exact(config.get("study_id"), "rushing_confirmatory_v1", "study_id")
    _exact(config.get("base_seed"), BASE_SEED, "base_seed")
    _exact(config.get("seed_algorithm"), SEED_ALGORITHM, "seed_algorithm")

    assumptions = _mapping(config.get("data_assumptions"), "data_assumptions")
    for field in ("game_id_column", "season_column", "play_id_column"):
        if not isinstance(assumptions.get(field), str) or not assumptions[field]:
            raise ConfigValidationError(f"data_assumptions.{field} must be a non-empty string.")
    _exact(assumptions.get("one_row_per_game"), True, "data_assumptions.one_row_per_game")
    _exact(assumptions.get("reject_augmented_suffix"), "_aug", "data_assumptions.reject_augmented_suffix")
    _exact(assumptions.get("expected_seasons"), list(SEASONS), "data_assumptions.expected_seasons")

    data = _mapping(config.get("data"), "data")
    for field in ("raw_train_csv", "processed_dir", "train_x", "train_x_set", "train_y", "df_season"):
        if not isinstance(data.get(field), str) or not data[field].startswith("data/"):
            raise ConfigValidationError(f"data.{field} must be a repository-relative path under data/.")

    split = _mapping(config.get("splits"), "splits")
    _exact(split.get("season_order"), list(SEASONS), "splits.season_order")
    _exact(split.get("outer_counts"), OUTER_COUNTS, "splits.outer_counts")
    _exact(split.get("tune_reserve_counts"), TUNE_COUNTS, "splits.tune_reserve_counts")
    _exact(split.get("nested_train_anchors"), list(MAIN_ANCHORS), "splits.nested_train_anchors")
    _exact(
        split.get("anchor_allocation"),
        "largest_remainder_proportional_by_sweep_capacity",
        "splits.anchor_allocation",
    )
    _exact(
        split.get("largest_remainder_tie_break"),
        "repeat_seeded_season_order",
        "splits.largest_remainder_tie_break",
    )
    inner = _mapping(split.get("neural_inner_split"), "splits.neural_inner_split")
    if Fraction(str(inner.get("train_fraction"))) != Fraction(4, 5):
        raise ConfigValidationError("Neural train fraction must be exactly 0.8.")
    if Fraction(str(inner.get("validation_fraction"))) != Fraction(1, 5):
        raise ConfigValidationError("Neural validation fraction must be exactly 0.2.")
    _exact(inner.get("stratify_by"), "Season", "splits.neural_inner_split.stratify_by")
    _exact(inner.get("shared_across_neural_models"), True, "splits.neural_inner_split.shared_across_neural_models")

    model_map = _mapping(config.get("model_id_map"), "model_id_map")
    if set(model_map) != set(CANONICAL_MODEL_IDS):
        raise ConfigValidationError(f"model_id_map keys must equal {CANONICAL_MODEL_IDS!r}.")
    if len(set(model_map.values())) != len(CANONICAL_MODEL_IDS):
        raise ConfigValidationError("Legacy model IDs must be unique.")
    models = _mapping(config.get("models"), "models")
    if set(models) != set(CANONICAL_MODEL_IDS):
        raise ConfigValidationError(f"models keys must equal {CANONICAL_MODEL_IDS!r}.")
    for model_id in CANONICAL_MODEL_IDS:
        spec = _mapping(models[model_id], f"models.{model_id}")
        _exact(spec.get("frozen"), True, f"models.{model_id}.frozen")
    _exact(models["ridge_sgd_l2"].get("alpha"), 1.0 / 3.0, "models.ridge_sgd_l2.alpha")
    _exact(models["ridge_sgd_l2"].get("eta0"), 0.01, "models.ridge_sgd_l2.eta0")
    expected_tree = {
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_samples": 50,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
    }
    for field, expected in expected_tree.items():
        _exact(models["lightgbm_multiclass"].get(field), expected, f"models.lightgbm_multiclass.{field}")
    _exact(models["zoo_cnn"].get("expected_parameters"), 145584, "models.zoo_cnn.expected_parameters")
    _exact(
        models["set_transformer"].get("expected_parameters"),
        157416,
        "models.set_transformer.expected_parameters",
    )

    uncertainty = _mapping(config.get("uncertainty"), "uncertainty")
    _exact(uncertainty.get("alpha"), 0.1, "uncertainty.alpha")
    _exact(uncertainty.get("local_k"), 200, "uncertainty.local_k")

    execution = _mapping(config.get("execution"), "execution")
    if execution.get("plan") not in {"final", "smoke"}:
        raise ConfigValidationError("execution.plan must be 'final' or 'smoke'.")
    confirmatory_repeats = execution.get("confirmatory_repeats")
    if confirmatory_repeats not in ALLOWED_CONFIRMATORY_REPEATS:
        raise ConfigValidationError(
            "execution.confirmatory_repeats must select the locked 50-repeat default "
            f"or 100-repeat full profile; got {confirmatory_repeats!r}."
        )
    _exact(execution.get("workers"), 1, "execution.workers")
    _exact(execution.get("retune_hyperparameters"), False, "execution.retune_hyperparameters")
    _exact(
        execution.get("require_tensorflow_deterministic_ops"),
        True,
        "execution.require_tensorflow_deterministic_ops",
    )
    _exact(execution.get("lightgbm_n_jobs"), 1, "execution.lightgbm_n_jobs")
    profile = execution.get("profile", SEQUENTIAL_EXECUTION_PROFILE)
    if profile not in {SEQUENTIAL_EXECUTION_PROFILE, HYBRID_EXECUTION_PROFILE}:
        raise ConfigValidationError(
            "execution.profile must select the locked sequential_gpu1 or "
            f"hybrid_cpu12_gpu1 topology; got {profile!r}."
        )
    if profile == HYBRID_EXECUTION_PROFILE:
        _exact(
            execution.get("queues"),
            HYBRID_EXECUTION_QUEUES,
            "execution.queues",
        )
    elif "queues" in execution:
        raise ConfigValidationError(
            "execution.queues is permitted only for the locked hybrid profile."
        )
    deterministic_environment = _mapping(execution.get("environment"), "execution.environment")
    _exact(dict(deterministic_environment), DETERMINISTIC_ENVIRONMENT, "execution.environment")
    sensitivity = _mapping(config.get("sensitivity"), "sensitivity")
    _exact(sensitivity.get("reuse_main_split_manifests"), True, "sensitivity.reuse_main_split_manifests")
    stage1 = _mapping(sensitivity.get("stage1"), "sensitivity.stage1")
    _exact(stage1.get("repeats"), 20, "sensitivity.stage1.repeats")
    _exact(stage1.get("repeat_ids"), list(range(1, 21)), "sensitivity.stage1.repeat_ids")
    _exact(stage1.get("anchors"), list(SENSITIVITY_ANCHORS), "sensitivity.stage1.anchors")
    extension = _mapping(sensitivity.get("extension"), "sensitivity.extension")
    _exact(extension.get("target_total_repeats"), 50, "sensitivity.extension.target_total_repeats")
    _exact(extension.get("additional_repeat_ids"), list(range(21, 51)), "sensitivity.extension.additional_repeat_ids")
    grids = _mapping(sensitivity.get("grids"), "sensitivity.grids")
    if set(grids) != set(CANONICAL_MODEL_IDS):
        raise ConfigValidationError(f"sensitivity.grids keys must equal {CANONICAL_MODEL_IDS!r}.")
    _exact(
        grids["ridge_sgd_l2"].get("values"),
        [1.0 / 3.0, 10.0, 10.0 / 3.0, 1.0, 0.1],
        "sensitivity.grids.ridge_sgd_l2.values",
    )
    _exact(grids["lightgbm_multiclass"][0], expected_tree, "sensitivity.grids.lightgbm_multiclass[0]")
    neural_main = {"learning_rate": 0.001, "dropout": 0.3}
    _exact(grids["zoo_cnn"][0], neural_main, "sensitivity.grids.zoo_cnn[0]")
    _exact(grids["set_transformer"][0], neural_main, "sensitivity.grids.set_transformer[0]")

    analysis = _mapping(config.get("analysis"), "analysis")
    _exact(analysis.get("bootstrap_draws"), 10000, "analysis.bootstrap_draws")
    _exact(analysis.get("bootstrap_seed_key"), ["analysis", "bootstrap"], "analysis.bootstrap_seed_key")
    _exact(analysis.get("coverage_target"), 0.9, "analysis.coverage_target")

    provenance = _mapping(config.get("provenance"), "provenance")
    for field in ("data_files", "code_files"):
        declared = _mapping(provenance.get(field), f"provenance.{field}")
        if execution["plan"] == "final" and not declared:
            raise ConfigValidationError(f"provenance.{field} cannot be empty for a final plan.")
        for name, path in declared.items():
            if not name or not isinstance(path, str) or not path:
                raise ConfigValidationError(f"Invalid provenance.{field} entry {name!r}: {path!r}.")
    code_files = provenance["code_files"]
    missing_or_changed = {
        name: path
        for name, path in REQUIRED_CODE_PROVENANCE.items()
        if code_files.get(name) != path
    }
    if missing_or_changed:
        raise ConfigValidationError(
            "provenance.code_files must bind every definitive local dependency; "
            f"missing or changed: {missing_or_changed!r}."
        )
    if not isinstance(provenance.get("packages"), list):
        raise ConfigValidationError("provenance.packages must be a list.")
    environment_names = provenance.get("environment_variables")
    if not isinstance(environment_names, list) or any(
        not isinstance(name, str) or not name for name in environment_names
    ):
        raise ConfigValidationError("provenance.environment_variables must be a list of names.")
    if len(environment_names) != len(set(environment_names)):
        raise ConfigValidationError("provenance.environment_variables cannot contain duplicates.")
    missing_environment = sorted(set(DETERMINISTIC_ENVIRONMENT) - set(environment_names))
    if missing_environment:
        raise ConfigValidationError(
            "provenance.environment_variables must observe every declared deterministic setting; "
            f"missing: {missing_environment!r}."
        )
    _validate_locked_config_identity(config)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a study JSON file, returning an independent dictionary."""

    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(f"Could not load study config {config_path}: {exc}") from exc
    validate_config(value)
    return json.loads(canonical_json(value))


def _seed_key(parts: Sequence[Any]) -> str:
    return canonical_json(list(parts))


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a positive 31-bit seed with HMAC-SHA256 and a semantic key."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise DesignError("base_seed must be a positive integer.")
    message = _seed_key(parts).encode("utf-8")
    digest = hmac.new(str(base_seed).encode("ascii"), message, hashlib.sha256).digest()
    return 1 + (int.from_bytes(digest[:8], "big") % ((1 << 31) - 1))


class SeedRegistry:
    """Collision-checking registry of pre-materialized semantic random seeds."""

    def __init__(
        self,
        base_seed: int = BASE_SEED,
        seed_function: Callable[..., int] = stable_seed,
        namespace: str | None = None,
    ) -> None:
        self.base_seed = base_seed
        self.seed_function = seed_function
        self.namespace = namespace
        self._by_key: dict[str, int] = {}
        self._by_seed: dict[int, str] = {}

    def get(self, *parts: Any) -> int:
        key = _seed_key(parts)
        if key in self._by_key:
            return self._by_key[key]
        seed_parts = (self.namespace, *parts) if self.namespace is not None else parts
        seed = int(self.seed_function(self.base_seed, *seed_parts))
        if not 1 <= seed < (1 << 31):
            raise DesignError(f"Derived seed for {key} is outside the positive 31-bit range: {seed}.")
        prior_key = self._by_seed.get(seed)
        if prior_key is not None and prior_key != key:
            raise SeedCollisionError(f"Seed collision {seed}: {prior_key} and {key}.")
        self._by_key[key] = seed
        self._by_seed[seed] = key
        return seed

    def snapshot(self) -> dict[str, int]:
        return {key: self._by_key[key] for key in sorted(self._by_key)}


def manifest_seed(manifest: Mapping[str, Any], *parts: Any) -> int:
    """Return a pre-materialized manifest seed or raise on an unknown semantic key."""

    registry = manifest.get("seed_registry")
    if not isinstance(registry, Mapping):
        raise DesignError("Manifest has no keyed seed_registry.")
    key = _seed_key(parts)
    if key not in registry:
        raise KeyError(f"Seed key was not materialized: {key}")
    seed = registry[key]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DesignError(f"Manifest seed for {key} is not an integer.")
    return seed


def _normalize_identifier(value: Any, suffix: str, kind: str) -> str:
    item_method = getattr(value, "item", None)
    if callable(item_method) and not isinstance(value, (str, bytes)):
        value = item_method()
    if value is None or isinstance(value, bool):
        raise IdentifierValidationError(f"{kind} must be a non-empty scalar; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise IdentifierValidationError(f"{kind} float must be a finite integer value; got {value!r}.")
        value = int(value)
    if not isinstance(value, (str, int)):
        raise IdentifierValidationError(f"{kind} must be a string or integer; got {type(value).__name__}.")
    normalized = str(value).strip()
    if not normalized:
        raise IdentifierValidationError(f"{kind} cannot be empty.")
    if normalized.lower().endswith(suffix.lower()):
        raise IdentifierValidationError(f"Augmented {kind} is forbidden in study design: {normalized!r}.")
    return normalized


def reject_augmented_ids(ids: Sequence[Any], suffix: str = "_aug", kind: str = "identifier") -> None:
    """Raise if any supplied ID has the augmentation suffix."""

    for value in ids:
        _normalize_identifier(value, suffix=suffix, kind=kind)


def _normalize_season(value: Any) -> int:
    item_method = getattr(value, "item", None)
    if callable(item_method) and not isinstance(value, (str, bytes)):
        value = item_method()
    try:
        season = int(value)
    except (TypeError, ValueError) as exc:
        raise IdentifierValidationError(f"Season must be integer-like; got {value!r}.") from exc
    if str(season) != str(value).strip() and not isinstance(value, int):
        # Accept NumPy/Python integral values and ordinary numeric strings, but
        # reject lossy conversions such as 2017.5.
        try:
            if float(value) != season:
                raise IdentifierValidationError(f"Season must be integral; got {value!r}.")
        except (TypeError, ValueError):
            raise IdentifierValidationError(f"Season must be integral; got {value!r}.")
    return season


def _normalize_game_table(config: Mapping[str, Any], game_table: Any) -> dict[str, list[str]]:
    assumptions = config["data_assumptions"]
    game_col = assumptions["game_id_column"]
    season_col = assumptions["season_column"]
    play_col = assumptions["play_id_column"]
    suffix = assumptions["reject_augmented_suffix"]

    records: list[Mapping[str, Any]] = []
    if isinstance(game_table, Mapping) and game_col not in game_table and season_col not in game_table:
        for season, ids in game_table.items():
            if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
                raise IdentifierValidationError(f"Game IDs for season {season!r} must be a sequence.")
            records.extend({season_col: season, game_col: game_id} for game_id in ids)
    elif hasattr(game_table, "to_dict") and hasattr(game_table, "columns"):
        try:
            records = list(game_table.to_dict(orient="records"))
        except TypeError as exc:
            raise IdentifierValidationError("game_table.to_dict must support orient='records'.") from exc
    elif isinstance(game_table, Sequence) and not isinstance(game_table, (str, bytes)):
        records = list(game_table)
    else:
        raise IdentifierValidationError(
            "game_table must be a pandas-like table, a sequence of records, or {season: game_ids}."
        )

    expected_seasons = set(SEASONS)
    by_season = {str(season): [] for season in SEASONS}
    seen: dict[str, int] = {}
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise IdentifierValidationError(f"game_table row {row_number} is not a mapping.")
        if game_col not in record or season_col not in record:
            raise IdentifierValidationError(
                f"game_table row {row_number} must contain {game_col!r} and {season_col!r}."
            )
        season = _normalize_season(record[season_col])
        if season not in expected_seasons:
            raise IdentifierValidationError(f"Unexpected season {season} in row {row_number}.")
        game_id = _normalize_identifier(record[game_col], suffix=suffix, kind="game ID")
        if play_col in record and record[play_col] is not None:
            _normalize_identifier(record[play_col], suffix=suffix, kind="play ID")
        if game_id in seen:
            raise IdentifierValidationError(
                f"Game ID collision {game_id!r} in seasons {seen[game_id]} and {season}; "
                "pass exactly one row per game."
            )
        seen[game_id] = season
        by_season[str(season)].append(game_id)

    for season in SEASONS:
        season_key = str(season)
        expected = sum(OUTER_COUNTS[season_key].values())
        actual = len(by_season[season_key])
        if actual != expected:
            raise IdentifierValidationError(
                f"Season {season} must contain exactly {expected} unique games; got {actual}."
            )
    return by_season


def largest_remainder_allocation(
    total: int,
    capacities: Mapping[str, int],
    tie_order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Allocate ``total`` proportionally with exact integer sum and stable ties."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise DesignError("Allocation total must be a non-negative integer.")
    if not capacities:
        raise DesignError("Allocation capacities cannot be empty.")
    keys = list(tie_order) if tie_order is not None else list(capacities)
    if set(keys) != set(capacities) or len(keys) != len(capacities):
        raise DesignError("tie_order must contain every capacity key exactly once.")
    normalized: dict[str, int] = {}
    for key in keys:
        value = capacities[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignError(f"Capacity for {key!r} must be a non-negative integer.")
        normalized[key] = value
    denominator = sum(normalized.values())
    if total > denominator:
        raise DesignError(f"Cannot allocate {total} units into total capacity {denominator}.")
    if denominator == 0:
        if total == 0:
            return {key: 0 for key in keys}
        raise DesignError("Positive allocation requires positive capacity.")

    allocation: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for key in keys:
        numerator = total * normalized[key]
        allocation[key] = numerator // denominator
        remainders[key] = numerator % denominator
    remaining = total - sum(allocation.values())
    order_index = {key: index for index, key in enumerate(keys)}
    ranked = sorted(keys, key=lambda key: (-remainders[key], order_index[key]))
    for key in ranked[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != total or any(allocation[key] > normalized[key] for key in keys):
        raise DesignError("Largest-remainder allocation invariant failed.")
    return allocation


def proportional_nested_allocations(
    capacities: Mapping[str, int],
    anchors: Sequence[int],
    tie_order: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return proportional cumulative allocations and reject non-nested paradoxes."""

    if list(anchors) != sorted(set(anchors)):
        raise DesignError("Nested anchors must be strictly increasing.")
    result: dict[str, dict[str, int]] = {}
    prior = {key: 0 for key in tie_order}
    for anchor in anchors:
        allocation = largest_remainder_allocation(anchor, capacities, tie_order=tie_order)
        for key in tie_order:
            if allocation[key] < prior[key]:
                raise DesignError(
                    f"Independent largest-remainder allocation is not nested at anchor {anchor} for {key}."
                )
        result[str(anchor)] = allocation
        prior = allocation
    return result


def _rank_ids(ids: Sequence[str], seed: int, namespace: str) -> list[str]:
    def rank(game_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            canonical_json({"namespace": namespace, "seed": seed, "game_id": game_id}).encode("utf-8")
        ).hexdigest()
        return digest, game_id

    return sorted(ids, key=rank)


def _combined(by_season: Mapping[str, Sequence[str]], season_order: Sequence[str]) -> list[str]:
    return [game_id for season in season_order for game_id in by_season[season]]


def _assert_disjoint(named: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in named.items():
        for game_id in ids:
            if game_id in seen:
                raise DesignError(f"Split collision: game {game_id} occurs in {seen[game_id]} and {name}.")
            seen[game_id] = name


def _build_one_split(
    config: Mapping[str, Any],
    games_by_season: Mapping[str, Sequence[str]],
    repeat_id: int,
    registry: SeedRegistry,
    anchor_allocations: Mapping[str, Mapping[str, int]],
    anchor_tie_seed: int,
    anchor_tie_order: Sequence[str],
) -> dict[str, Any]:
    season_order = [str(season) for season in config["splits"]["season_order"]]
    outer_by_season: dict[str, dict[str, list[str]]] = {}
    sweep_by_season: dict[str, list[str]] = {}
    tune_by_season: dict[str, list[str]] = {}

    for season in season_order:
        outer_seed = registry.get("split", "outer", repeat_id, int(season))
        ordered = _rank_ids(games_by_season[season], outer_seed, f"outer/{repeat_id}/{season}")
        counts = config["splits"]["outer_counts"][season]
        train_end = counts["train"]
        cal_end = train_end + counts["calibration"]
        outer_train = ordered[:train_end]
        calibration = ordered[train_end:cal_end]
        test = ordered[cal_end:]

        tune_seed = registry.get("split", "tune", repeat_id, int(season))
        tune_order = _rank_ids(outer_train, tune_seed, f"tune/{repeat_id}/{season}")
        n_tune = config["splits"]["tune_reserve_counts"][season]
        tune = tune_order[:n_tune]
        tune_set = set(tune)
        sweep_candidates = [game_id for game_id in outer_train if game_id not in tune_set]
        nested_seed = registry.get("split", "nested", repeat_id, int(season))
        sweep_order = _rank_ids(sweep_candidates, nested_seed, f"nested/{repeat_id}/{season}")

        outer_by_season[season] = {
            "outer_train": outer_train,
            "train": sweep_order,
            "tune": tune,
            "calibration": calibration,
            "test": test,
        }
        sweep_by_season[season] = sweep_order
        tune_by_season[season] = tune

    anchors: dict[str, Any] = {}
    inner_fraction = Fraction(str(config["splits"]["neural_inner_split"]["validation_fraction"]))
    for anchor in config["splits"]["nested_train_anchors"]:
        anchor_key = str(anchor)
        season_counts = dict(anchor_allocations[anchor_key])
        anchor_by_season = {
            season: sweep_by_season[season][: season_counts[season]] for season in season_order
        }
        target_validation = Fraction(anchor) * inner_fraction
        if target_validation.denominator != 1:
            raise DesignError(f"Anchor {anchor} does not permit an exact 80/20 inner split.")
        validation_counts = largest_remainder_allocation(
            int(target_validation), season_counts, tie_order=season_order
        )
        neural_fit_by_season: dict[str, list[str]] = {}
        neural_validation_by_season: dict[str, list[str]] = {}
        for season in season_order:
            inner_seed = registry.get("split", "neural_inner", repeat_id, anchor, int(season))
            inner_order = _rank_ids(
                anchor_by_season[season], inner_seed, f"neural_inner/{repeat_id}/{anchor}/{season}"
            )
            n_validation = validation_counts[season]
            neural_validation_by_season[season] = inner_order[:n_validation]
            neural_fit_by_season[season] = inner_order[n_validation:]

        anchors[anchor_key] = {
            "n_train": anchor,
            "season_counts": season_counts,
            "train_games": _combined(anchor_by_season, season_order),
            "neural_fit_games": _combined(neural_fit_by_season, season_order),
            "neural_validation_games": _combined(neural_validation_by_season, season_order),
            "by_season": {
                season: {
                    "train_games": anchor_by_season[season],
                    "neural_fit_games": neural_fit_by_season[season],
                    "neural_validation_games": neural_validation_by_season[season],
                }
                for season in season_order
            },
        }

    split_manifest = {
        "repeat_id": repeat_id,
        "anchor_allocation_tie_seed": anchor_tie_seed,
        "anchor_allocation_tie_order": list(anchor_tie_order),
        "train_games": _combined(sweep_by_season, season_order),
        "tune_games": _combined(tune_by_season, season_order),
        "calibration_games": _combined(
            {season: outer_by_season[season]["calibration"] for season in season_order}, season_order
        ),
        "test_games": _combined(
            {season: outer_by_season[season]["test"] for season in season_order}, season_order
        ),
        "anchors": anchors,
        "by_season": {
            season: {
                "train_games": outer_by_season[season]["train"],
                "tune_games": outer_by_season[season]["tune"],
                "calibration_games": outer_by_season[season]["calibration"],
                "test_games": outer_by_season[season]["test"],
            }
            for season in season_order
        },
    }
    _validate_one_split(config, games_by_season, split_manifest, anchor_allocations)
    split_manifest["split_hash"] = sha256_json(split_manifest)
    return split_manifest


def _validate_one_split(
    config: Mapping[str, Any],
    games_by_season: Mapping[str, Sequence[str]],
    split_manifest: Mapping[str, Any],
    anchor_allocations: Mapping[str, Mapping[str, int]],
) -> None:
    season_order = [str(season) for season in config["splits"]["season_order"]]
    tie_seed = split_manifest.get("anchor_allocation_tie_seed")
    tie_order = split_manifest.get("anchor_allocation_tie_order")
    expected_tie_order = _rank_ids(
        season_order,
        tie_seed,
        f"anchor_allocation_tie_order/{split_manifest['repeat_id']}",
    )
    if tie_order != expected_tie_order:
        raise DesignError(
            f"Repeat {split_manifest['repeat_id']} anchor-allocation tie order is not seed-derived."
        )
    partitions = {
        "train": split_manifest["train_games"],
        "tune": split_manifest["tune_games"],
        "calibration": split_manifest["calibration_games"],
        "test": split_manifest["test_games"],
    }
    _assert_disjoint(partitions)
    universe = set(_combined(games_by_season, season_order))
    observed = set(game_id for ids in partitions.values() for game_id in ids)
    if observed != universe:
        raise DesignError("Outer split does not partition the full game universe exactly.")

    for season in season_order:
        season_split = split_manifest["by_season"][season]
        counts = config["splits"]["outer_counts"][season]
        tune_count = config["splits"]["tune_reserve_counts"][season]
        expected = {
            "train_games": counts["train"] - tune_count,
            "tune_games": tune_count,
            "calibration_games": counts["calibration"],
            "test_games": counts["test"],
        }
        for field, expected_count in expected.items():
            if len(season_split[field]) != expected_count:
                raise DesignError(
                    f"Repeat {split_manifest['repeat_id']} season {season} {field} has "
                    f"{len(season_split[field])} games, expected {expected_count}."
                )

    prior_anchor: set[str] = set()
    for anchor in config["splits"]["nested_train_anchors"]:
        anchor_key = str(anchor)
        record = split_manifest["anchors"][anchor_key]
        train = record["train_games"]
        neural_fit = record["neural_fit_games"]
        neural_validation = record["neural_validation_games"]
        if len(train) != anchor or len(set(train)) != anchor:
            raise DesignError(f"Anchor {anchor} must contain exactly {anchor} unique games.")
        if not prior_anchor.issubset(train):
            raise DesignError(f"Anchor {anchor} is not nested over the prior anchor.")
        prior_anchor = set(train)
        _assert_disjoint({"neural_fit": neural_fit, "neural_validation": neural_validation})
        if set(neural_fit) | set(neural_validation) != set(train):
            raise DesignError(f"Neural inner split does not partition anchor {anchor}.")
        if len(neural_fit) * 5 != anchor * 4 or len(neural_validation) * 5 != anchor:
            raise DesignError(f"Neural inner split at anchor {anchor} is not exactly 80/20.")
        for season in season_order:
            season_record = record["by_season"][season]
            if len(season_record["train_games"]) != anchor_allocations[anchor_key][season]:
                raise DesignError(f"Anchor {anchor} season {season} violates largest-remainder allocation.")


def _materialize_cell_seeds(config: Mapping[str, Any], registry: SeedRegistry) -> None:
    repeats = range(1, config["execution"]["confirmatory_repeats"] + 1)
    # The scientific seed tuple is deliberately branch-free so a sensitivity
    # refit and its paired frozen-primary cell share optimization randomness:
    # (study namespace, repeat, stage, model, training size).
    for repeat_id in repeats:
        for n_train in config["splits"]["nested_train_anchors"]:
            for model_id in CANONICAL_MODEL_IDS:
                for stage in CELL_STAGES:
                    registry.get("cell", repeat_id, stage, model_id, n_train)
    registry.get(*config["analysis"]["bootstrap_seed_key"])


def build_split_manifests(
    config: Mapping[str, Any],
    game_table: Any,
    registry: SeedRegistry | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build all configured outer/nested/inner split records and seed keys."""

    validate_config(config)
    games_by_season = _normalize_game_table(config, game_table)
    season_order = [str(season) for season in config["splits"]["season_order"]]
    capacities = {
        season: config["splits"]["outer_counts"][season]["train"]
        - config["splits"]["tune_reserve_counts"][season]
        for season in season_order
    }
    if registry is not None and registry.namespace != config["study_id"]:
        raise DesignError(
            f"Supplied seed registry namespace {registry.namespace!r} does not match "
            f"study_id {config['study_id']!r}."
        )
    seed_registry = registry or SeedRegistry(
        config["base_seed"], namespace=config["study_id"]
    )
    split_manifests = []
    for repeat_id in range(1, config["execution"]["confirmatory_repeats"] + 1):
        tie_seed = seed_registry.get("split", "anchor_allocation_tie_order", repeat_id)
        tie_order = _rank_ids(
            season_order,
            tie_seed,
            f"anchor_allocation_tie_order/{repeat_id}",
        )
        anchor_allocations = proportional_nested_allocations(
            capacities,
            config["splits"]["nested_train_anchors"],
            tie_order=tie_order,
        )
        split_manifests.append(
            _build_one_split(
                config,
                games_by_season,
                repeat_id,
                seed_registry,
                anchor_allocations,
                tie_seed,
                tie_order,
            )
        )
    _materialize_cell_seeds(config, seed_registry)
    return split_manifests, seed_registry.snapshot()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ProvenanceError(f"Could not hash {target}: {exc}") from exc
    return digest.hexdigest()


def parse_requirements_lock(
    config: Mapping[str, Any], repo_root: str | Path = "."
) -> dict[str, str]:
    """Parse the definitive exact-version lock declared by the study config."""

    try:
        relative_text = config["provenance"]["code_files"]["requirements_lock"]
    except (KeyError, TypeError) as exc:
        raise ProvenanceError(
            "Configuration does not declare provenance.code_files.requirements_lock."
        ) from exc
    if not isinstance(relative_text, str) or not relative_text:
        raise ProvenanceError("The declared requirements-lock path must be non-empty.")
    root = Path(repo_root).resolve()
    relative = Path(relative_text)
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(
            f"Declared requirements lock escapes repository root: {relative_text}"
        ) from exc
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProvenanceError(f"Could not read requirements lock {target}: {exc}") from exc

    expected: dict[str, str] = {}
    normalized_names: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ProvenanceError(
                f"Invalid exact requirement on line {line_number}: {raw_line!r}."
            ) from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or not specifiers[0].version
            or "*" in specifiers[0].version
        ):
            raise ProvenanceError(
                f"Requirements lock line {line_number} must be an exact name==version "
                f"requirement with an optional environment marker: {raw_line!r}."
            )
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = requirement.name
        version = specifiers[0].version
        normalized = name.lower().replace("_", "-").replace(".", "-")
        if normalized in normalized_names:
            raise ProvenanceError(
                "Duplicate distribution in requirements lock: "
                f"{normalized_names[normalized]!r} and {name!r}."
            )
        normalized_names[normalized] = name
        expected[name] = version
    if not expected:
        raise ProvenanceError(f"Requirements lock contains no exact requirements: {target}")
    return expected


def _declared_file_provenance(
    repo_root: Path,
    declared: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    root = repo_root.resolve()
    for logical_name in sorted(declared):
        relative = Path(declared[logical_name])
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(f"Declared file escapes repository root: {relative}") from exc
        if not target.is_file():
            raise ProvenanceError(f"Declared provenance file is missing: {relative}")
        output[logical_name] = {
            "path": relative.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    return output


def collect_code_provenance(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Hash every declared code/config file directly from its bytes."""

    root = Path(repo_root).resolve()
    declared = config["provenance"]["code_files"]
    code_files = _declared_file_provenance(root, declared)
    return {
        "repo_root": str(root),
        "files": code_files,
    }


def collect_data_provenance(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Hash all immutable input/pilot artifacts declared by the configuration."""

    return {"files": _declared_file_provenance(Path(repo_root), config["provenance"]["data_files"])}


def declared_execution_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scheduler-ready deterministic settings bound by the config."""

    validate_config(config)
    execution = config["execution"]
    settings = {
        "worker_count": execution["workers"],
        "lightgbm_n_jobs": execution["lightgbm_n_jobs"],
        "environment_variables": {
            name: execution["environment"][name]
            for name in sorted(execution["environment"])
        },
    }
    profile = execution.get("profile", SEQUENTIAL_EXECUTION_PROFILE)
    if profile == HYBRID_EXECUTION_PROFILE:
        settings["profile"] = profile
        settings["queues"] = deepcopy(HYBRID_EXECUTION_QUEUES)
    return settings


def execution_profile(config: Mapping[str, Any]) -> str:
    """Return the locked execution-topology identifier."""

    validate_config(config)
    return str(
        config["execution"].get("profile", SEQUENTIAL_EXECUTION_PROFILE)
    )


def execution_queues(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the exact model/device queues for the selected topology."""

    profile = execution_profile(config)
    if profile == HYBRID_EXECUTION_PROFILE:
        return deepcopy(HYBRID_EXECUTION_QUEUES)
    return {
        "sequential": {
            "device": "gpu_preferred",
            "workers": 1,
            "models": list(CANONICAL_MODEL_IDS),
        }
    }


def _tensorflow_environment_provenance() -> dict[str, Any]:
    """Collect TensorFlow build and device details without making them mandatory."""

    try:
        tensorflow = importlib.import_module("tensorflow")
    except Exception as exc:  # pragma: no cover - exact import failures are environment-specific
        return {
            "available": False,
            "import_error": {"type": type(exc).__name__, "message": str(exc)},
        }

    result: dict[str, Any] = {
        "available": True,
        "version": str(getattr(tensorflow, "__version__", "unknown")),
    }
    try:
        devices = tensorflow.config.list_physical_devices()
        result["physical_devices"] = sorted(
            [
                {
                    "name": str(getattr(device, "name", "unknown")),
                    "device_type": str(getattr(device, "device_type", "unknown")),
                }
                for device in devices
            ],
            key=lambda item: (item["device_type"], item["name"]),
        )
    except Exception as exc:  # pragma: no cover - depends on installed runtime/drivers
        result["physical_devices"] = []
        result["physical_devices_error"] = {"type": type(exc).__name__, "message": str(exc)}
    try:
        result["build_info"] = _canonicalize(tensorflow.sysconfig.get_build_info())
    except Exception as exc:  # pragma: no cover - depends on TensorFlow version/build
        result["build_info"] = None
        result["build_info_error"] = {"type": type(exc).__name__, "message": str(exc)}
    try:
        result["runtime_threading"] = {
            "inter_op_threads": int(
                tensorflow.config.threading.get_inter_op_parallelism_threads()
            ),
            "intra_op_threads": int(
                tensorflow.config.threading.get_intra_op_parallelism_threads()
            ),
        }
    except Exception as exc:  # pragma: no cover - depends on TensorFlow version/runtime state
        result["runtime_threading"] = None
        result["runtime_threading_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def collect_environment_provenance(
    config: Mapping[str, Any],
    repo_root: str | Path = ".",
    *,
    include_tensorflow: bool = True,
) -> dict[str, Any]:
    """Capture and enforce the complete locked runtime environment."""

    settings = declared_execution_settings(config)
    expected_packages = parse_requirements_lock(config, repo_root)
    declared_packages = config["provenance"]["packages"]
    active_declared_packages = [
        name for name in declared_packages if name in expected_packages
    ]
    inactive_declared_packages = [
        name for name in declared_packages if name not in expected_packages
    ]
    allowed_inactive_packages = (
        ["tensorflow-metal"] if sys.platform != "darwin" else []
    )
    if (
        active_declared_packages != list(expected_packages)
        or inactive_declared_packages != allowed_inactive_packages
    ):
        raise ProvenanceError(
            "provenance.packages must list every requirements-lock distribution in "
            "the same order, including only marker-inactive platform packages."
        )
    packages: dict[str, str | None] = {}
    package_mismatches: list[str] = []
    for package_name, expected_version in expected_packages.items():
        try:
            observed_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            observed_version = None
        packages[package_name] = observed_version
        if observed_version != expected_version:
            package_mismatches.append(
                f"{package_name}=={expected_version} (installed {observed_version!r})"
            )
    if package_mismatches:
        raise ProvenanceError(
            "Installed environment does not match requirements-lock.txt: "
            + "; ".join(package_mismatches)
        )
    lock_relative = Path(
        config["provenance"]["code_files"]["requirements_lock"]
    )
    lock_path = (Path(repo_root).resolve() / lock_relative).resolve()
    environment_variables = {
        name: os.environ.get(name) for name in sorted(config["provenance"].get("environment_variables", []))
    }
    tensorflow = (
        _tensorflow_environment_provenance()
        if include_tensorflow
        else {"verification_skipped": True}
    )
    if include_tensorflow and execution_profile(config) == HYBRID_EXECUTION_PROFILE and not any(
        device.get("device_type") == "GPU"
        for device in tensorflow.get("physical_devices", [])
        if isinstance(device, Mapping)
    ):
        raise ProvenanceError(
            "The hybrid_cpu12_gpu1 profile requires a TensorFlow-visible GPU on "
            "the planning and execution host."
        )
    return {
        "declared_execution_settings": settings,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "byteorder": sys.byteorder,
        },
        "packages": packages,
        "requirements_lock": {
            "path": lock_relative.as_posix(),
            "size_bytes": lock_path.stat().st_size,
            "sha256": sha256_file(lock_path),
            "expected_packages": expected_packages,
            "package_count": len(expected_packages),
        },
        "environment_variables": environment_variables,
        "tensorflow": tensorflow,
    }


def _runtime_file_mismatches(
    repo_root: Path,
    category: str,
    frozen_files: Any,
) -> list[str]:
    """Compare frozen file records without consulting repository status."""

    if not isinstance(frozen_files, Mapping):
        return [f"{category} file provenance is missing or malformed"]
    root = repo_root.resolve()
    mismatches: list[str] = []
    for logical_name in sorted(frozen_files):
        record = frozen_files[logical_name]
        if not isinstance(record, Mapping):
            mismatches.append(f"{category} file {logical_name!r} record is malformed")
            continue
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            mismatches.append(f"{category} file {logical_name!r} has no frozen path")
            continue
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            mismatches.append(f"{category} file {logical_name!r} escapes repository root: {relative}")
            continue
        if not target.is_file():
            mismatches.append(f"{category} file {logical_name!r} is missing: {relative}")
            continue
        current_size = target.stat().st_size
        expected_size = record.get("size_bytes")
        if current_size != expected_size:
            mismatches.append(
                f"{category} file {logical_name!r} size changed: "
                f"expected {expected_size!r}, got {current_size!r}"
            )
        current_hash = sha256_file(target)
        expected_hash = record.get("sha256")
        if current_hash != expected_hash:
            mismatches.append(
                f"{category} file {logical_name!r} SHA-256 changed: "
                f"expected {expected_hash!r}, got {current_hash!r}"
            )
    return mismatches


def _selected_environment_mismatches(
    frozen: Any,
    current: Mapping[str, Any],
    *,
    verify_tensorflow: bool = True,
    verify_hardware: bool = True,
) -> list[str]:
    """Compare only runtime properties that define the execution environment."""

    if not isinstance(frozen, Mapping):
        return ["environment provenance is missing or malformed"]
    mismatches: list[str] = []
    frozen_sections: dict[str, Mapping[str, Any]] = {}
    current_sections: dict[str, Mapping[str, Any]] = {}
    section_names = ["python"]
    if verify_hardware:
        section_names.append("hardware")
    if verify_tensorflow:
        section_names.append("tensorflow")
    for section_name in section_names:
        frozen_section = frozen.get(section_name)
        current_section = current.get(section_name)
        if not isinstance(frozen_section, Mapping):
            mismatches.append(f"frozen {section_name} provenance is missing or malformed")
            frozen_section = {}
        if not isinstance(current_section, Mapping):
            mismatches.append(f"current {section_name} provenance is missing or malformed")
            current_section = {}
        frozen_sections[section_name] = frozen_section
        current_sections[section_name] = current_section
    comparisons = {
        "Python version": (
            frozen_sections["python"].get("version"),
            current_sections["python"].get("version"),
        ),
        "Python implementation": (
            frozen_sections["python"].get("implementation"),
            current_sections["python"].get("implementation"),
        ),
        "package versions": (frozen.get("packages"), current.get("packages")),
        "requirements lock details": (
            frozen.get("requirements_lock"),
            current.get("requirements_lock"),
        ),
        "OS": (frozen.get("os"), current.get("os")),
    }
    if verify_hardware:
        frozen_hardware = frozen_sections["hardware"]
        current_hardware = current_sections["hardware"]
        for field, label in (
            ("machine", "hardware machine"),
            ("processor", "hardware processor"),
            ("cpu_count", "hardware CPU count"),
            ("byteorder", "hardware byte order"),
        ):
            comparisons[label] = (frozen_hardware.get(field), current_hardware.get(field))
    if verify_tensorflow:
        frozen_tensorflow = frozen_sections["tensorflow"]
        current_tensorflow = current_sections["tensorflow"]
        comparisons["TensorFlow availability"] = (
            frozen_tensorflow.get("available"),
            current_tensorflow.get("available"),
        )
        comparisons["TensorFlow version"] = (
            frozen_tensorflow.get("version"),
            current_tensorflow.get("version"),
        )
        comparisons["TensorFlow physical device inventory"] = (
            frozen_tensorflow.get("physical_devices", []),
            current_tensorflow.get("physical_devices", []),
        )
        comparisons["TensorFlow build info"] = (
            frozen_tensorflow.get("build_info"),
            current_tensorflow.get("build_info"),
        )
        comparisons["TensorFlow runtime threading"] = (
            frozen_tensorflow.get("runtime_threading"),
            current_tensorflow.get("runtime_threading"),
        )
    for label, (expected, observed) in comparisons.items():
        if expected != observed:
            mismatches.append(f"{label} changed: expected {expected!r}, got {observed!r}")
    return mismatches


def verify_runtime_provenance(
    manifest: Mapping[str, Any],
    repo_root: str | Path = ".",
    *,
    require_environment: bool = False,
    verify_tensorflow: bool = True,
    verify_hardware: bool = True,
) -> dict[str, Any]:
    """Verify a frozen manifest against the current read-only runtime state.

    ``require_environment=False`` is intended for a parent scheduler before it
    creates the deterministic worker environment.  Workers must pass
    ``require_environment=True`` so every declared environment variable is
    checked exactly. ``verify_tensorflow=False`` is reserved for locked CPU-only
    children after their parent has verified the full TensorFlow/GPU inventory.
    ``verify_hardware=False`` supports a manifest planned on its GPU queue and
    executed by a separately frozen CPU-only queue; code, data, Python, package,
    lock, OS, and declared environment checks remain enforced.
    Version-control state is not consulted; only declared file hashes are part
    of the runtime gate.
    """

    if not isinstance(manifest, Mapping):
        raise ProvenanceError("Runtime verification requires a manifest mapping.")
    expected_manifest_hash = sha256_json(_manifest_identity_payload(manifest))
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise ProvenanceError("Manifest payload does not match its frozen manifest_hash.")
    if manifest.get("run_hash") != expected_manifest_hash[:24]:
        raise ProvenanceError("Manifest run_hash does not match its frozen manifest_hash.")

    provenance = manifest.get("provenance")
    config = manifest.get("config")
    if not isinstance(provenance, Mapping) or not isinstance(config, Mapping):
        raise ProvenanceError("Manifest must contain config and provenance mappings.")
    code = provenance.get("code")
    data = provenance.get("data")
    if not isinstance(code, Mapping) or not isinstance(data, Mapping):
        raise ProvenanceError("Manifest code/data provenance is missing or malformed.")

    root = Path(repo_root).resolve()
    mismatches: list[str] = []
    mismatches.extend(_runtime_file_mismatches(root, "code", code.get("files")))
    mismatches.extend(_runtime_file_mismatches(root, "data", data.get("files")))
    if mismatches:
        raise ProvenanceError("Runtime provenance mismatch: " + "; ".join(mismatches))

    current_environment = collect_environment_provenance(
        config,
        root,
        include_tensorflow=verify_tensorflow,
    )
    mismatches.extend(
        _selected_environment_mismatches(
            provenance.get("environment"),
            current_environment,
            verify_tensorflow=verify_tensorflow,
            verify_hardware=verify_hardware,
        )
    )
    if require_environment:
        frozen_environment = provenance.get("environment", {})
        if not isinstance(frozen_environment, Mapping):
            frozen_environment = {}
        declared = frozen_environment.get("declared_execution_settings", {})
        if not isinstance(declared, Mapping):
            declared = {}
        expected_variables = declared.get("environment_variables")
        if not isinstance(expected_variables, Mapping):
            mismatches.append("declared deterministic environment is missing or malformed")
        else:
            for name in sorted(expected_variables):
                expected = expected_variables[name]
                observed = os.environ.get(name)
                if observed != expected:
                    mismatches.append(
                        f"environment variable {name} changed: "
                        f"expected {expected!r}, got {observed!r}"
                    )
    if mismatches:
        raise ProvenanceError("Runtime provenance mismatch: " + "; ".join(mismatches))
    return {
        "verified": True,
        "manifest_hash": manifest["manifest_hash"],
        "code_files_verified": len(code["files"]),
        "data_files_verified": len(data["files"]),
        "environment_required": require_environment,
        "tensorflow_verified": verify_tensorflow,
        "hardware_verified": verify_hardware,
    }


def _manifest_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in manifest.items()
        if key not in {"manifest_hash", "run_hash"}
    }


def build_study_manifest(
    config: Mapping[str, Any],
    game_table: Any,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the runner-ready immutable study manifest.

    The returned contract has top-level ``config``, ``manifest_hash``,
    ``run_hash``, a keyed ``seed_registry``, and 50 or 100 ``split_manifests``.
    Version-control metadata is not consulted. Declared code and data files
    are enforced by exact size and SHA-256 hashes.
    """

    validate_config(config)
    normalized_config = json.loads(canonical_json(config))
    code = collect_code_provenance(normalized_config, repo_root)
    split_manifests, seed_registry = build_split_manifests(normalized_config, game_table)
    provenance = {
        "code": code,
        "data": collect_data_provenance(normalized_config, repo_root),
        "environment": collect_environment_provenance(normalized_config, repo_root),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study_id": normalized_config["study_id"],
        "config": normalized_config,
        "config_hash": sha256_json(normalized_config),
        "seed_derivation": {
            "algorithm": normalized_config["seed_algorithm"],
            "base_seed": normalized_config["base_seed"],
            "namespace": normalized_config["study_id"],
        },
        "seed_registry": seed_registry,
        "split_manifests": split_manifests,
        "provenance": provenance,
    }
    manifest_hash = sha256_json(_manifest_identity_payload(manifest))
    manifest["manifest_hash"] = manifest_hash
    manifest["run_hash"] = manifest_hash[:24]
    return manifest


def _atomic_immutable_write(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ManifestCollisionError(f"Refusing to overwrite immutable bundle file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ManifestCollisionError(f"Concurrent immutable bundle collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest_bundle(run_dir: str | Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Atomically write an immutable, runner-friendly manifest bundle.

    Calling this again with identical bytes is idempotent.  Any differing file
    at a bundle path raises :class:`ManifestCollisionError`.
    """

    required = {
        "config",
        "manifest_hash",
        "run_hash",
        "seed_derivation",
        "seed_registry",
        "split_manifests",
        "provenance",
    }
    missing = required - set(manifest)
    if missing:
        raise DesignError(f"Manifest is missing required fields: {sorted(missing)}")
    expected_hash = sha256_json(_manifest_identity_payload(manifest))
    if manifest["manifest_hash"] != expected_hash or manifest["run_hash"] != expected_hash[:24]:
        raise DesignError("Manifest hash fields do not match the canonical manifest payload.")

    root = Path(run_dir)
    files: dict[str, tuple[str, Any]] = {
        "manifest": ("manifest.json", manifest),
        "config": ("config.json", manifest["config"]),
        "splits": ("split_manifests.json", manifest["split_manifests"]),
        "seeds": ("seed_registry.json", manifest["seed_registry"]),
        "provenance": ("provenance.json", manifest["provenance"]),
    }
    written: dict[str, Path] = {}
    for logical_name, (filename, value) in files.items():
        target = root / filename
        _atomic_immutable_write(target, (canonical_json(value) + "\n").encode("utf-8"))
        written[logical_name] = target
    hash_path = root / "MANIFEST_SHA256"
    _atomic_immutable_write(hash_path, (manifest["manifest_hash"] + "\n").encode("ascii"))
    written["hash"] = hash_path
    return written


__all__ = [
    "BASE_SEED",
    "CANONICAL_MODEL_IDS",
    "CELL_STAGES",
    "DETERMINISTIC_ENVIRONMENT",
    "FULL_CONFIRMATORY_REPEATS",
    "HYBRID_EXECUTION_PROFILE",
    "HYBRID_EXECUTION_QUEUES",
    "ConfigValidationError",
    "DesignError",
    "IdentifierValidationError",
    "ManifestCollisionError",
    "ProvenanceError",
    "REQUIRED_CODE_PROVENANCE",
    "SEQUENTIAL_EXECUTION_PROFILE",
    "SeedCollisionError",
    "SeedRegistry",
    "build_split_manifests",
    "build_study_manifest",
    "canonical_json",
    "collect_code_provenance",
    "collect_data_provenance",
    "collect_environment_provenance",
    "declared_execution_settings",
    "execution_profile",
    "execution_queues",
    "largest_remainder_allocation",
    "load_config",
    "manifest_seed",
    "proportional_nested_allocations",
    "reject_augmented_ids",
    "sha256_file",
    "sha256_json",
    "stable_run_hash",
    "stable_seed",
    "validate_config",
    "verify_runtime_provenance",
    "write_manifest_bundle",
]
