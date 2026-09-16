"""Deterministic grouped designs for the harmonized BDB task suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
from typing import Any, Callable

from .contracts import TaskSpec, TaskSpecError, canonical_json, sha256_json, validate_task_spec


BASE_SEED = 20260817
DESIGN_SCHEMA_VERSION = "bdb-task-design-v1"
SEED_ALGORITHM = "hmac-sha256-31bit-v1"
CONFIRMATORY_REPEATS = 50
DEFINITIVE_REPEATS = 100
PILOT_REPEATS = 10
DEVELOPMENT_FOLDS = 5
CELL_STAGES = (
    "fit",
    "epoch_selection",
    "validation",
    "refit",
    "prediction",
    "uncertainty_subsample",
)
CELL_SEED_FIELDS = CELL_STAGES + ("validation_split",)


class DesignError(ValueError):
    """Base class for deterministic-design failures."""


class GameRegistryError(DesignError):
    """The supplied game table does not match its task contract."""


class SeedCollisionError(DesignError):
    """Two different semantic keys produced the same bounded seed."""


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    repeats: int
    include_primary: bool
    include_structural_ablation: bool
    include_frozen_sensitivities: bool
    expected_primary_cells: int
    expected_ablation_cells: int
    expected_sensitivity_cells: int

    @property
    def expected_required_cells(self) -> int:
        return (
            self.expected_primary_cells
            + self.expected_ablation_cells
            + self.expected_sensitivity_cells
        )


EXECUTION_PROFILES: dict[str, ExecutionProfile] = {
    "full50": ExecutionProfile("full50", 50, True, False, False, 1_200, 0, 0),
    "pilot10": ExecutionProfile(
        "pilot10", PILOT_REPEATS, True, False, False, 300, 0, 0
    ),
    "full100": ExecutionProfile("full100", 100, True, True, False, 3_000, 80, 0),
    "sensitivity20": ExecutionProfile(
        "sensitivity20", 20, False, False, True, 0, 0, 200
    ),
}


def profile_seed_namespace(profile: str | None) -> tuple[str, ...]:
    """Return the immutable semantic seed namespace for an execution profile.

    The historical/full profiles deliberately keep their original empty
    namespace. Pilot cells and splits therefore cannot accidentally reuse a
    full100 seed, while adding the pilot does not perturb any frozen full100
    seed identity.
    """

    return ("pilot10",) if profile == "pilot10" else ()


def analysis_seed_parts(task_id: str, profile: str | None) -> tuple[str, ...]:
    """Semantic key for a profile-bound task-analysis seed."""

    return (
        "task",
        str(task_id),
        *profile_seed_namespace(profile),
        "analysis",
        "bootstrap",
    )


@dataclass(frozen=True)
class TaskDesignProfile:
    task_id: str
    release_year: int
    outcome_type: str
    total_games: int
    development_games: int
    confirmatory_games: int
    train_games: int
    calibration_games: int
    test_games: int
    anchors: tuple[int, ...]
    shared_game_registry_with: str | None = None

    @property
    def outer_counts(self) -> dict[str, int]:
        return {
            "train": self.train_games,
            "calibration": self.calibration_games,
            "test": self.test_games,
        }


TASK_DESIGNS: dict[str, TaskDesignProfile] = {
    "bdb2020_rushing_harmonized": TaskDesignProfile(
        "bdb2020_rushing_harmonized", 2020, "distribution", 688, 40, 648,
        374, 137, 137, (20, 40, 80, 160, 240, 360),
    ),
    "bdb2021_completion": TaskDesignProfile(
        "bdb2021_completion", 2021, "binary", 253, 30, 223, 134, 44, 45,
        (10, 20, 40, 60, 100, 130),
    ),
    "bdb2022_punt_returns": TaskDesignProfile(
        "bdb2022_punt_returns", 2022, "distribution", 712, 90, 622, 373, 124, 125,
        (10, 20, 40, 60, 160, 360),
    ),
    "bdb2023_sack": TaskDesignProfile(
        "bdb2023_sack", 2023, "binary", 122, 20, 102, 61, 20, 21,
        (10, 20, 30, 40, 50, 60),
    ),
    "bdb2024_tackle": TaskDesignProfile(
        "bdb2024_tackle", 2024, "binary", 136, 30, 106, 64, 21, 21,
        (10, 20, 30, 40, 50, 60),
    ),
    "bdb2025_man_zone": TaskDesignProfile(
        "bdb2025_man_zone", 2025, "binary", 136, 30, 106, 64, 21, 21,
        (10, 20, 30, 40, 50, 60), "bdb2024_tackle",
    ),
    "bdb2026_trajectory": TaskDesignProfile(
        "bdb2026_trajectory", 2026, "trajectory", 272, 36, 236, 142, 47, 47,
        (10, 20, 40, 60, 100, 140),
    ),
}


def _seed_key(parts: Sequence[Any]) -> str:
    return canonical_json(list(parts))


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a reproducible positive 31-bit integer from a semantic tuple."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise DesignError("base_seed must be a positive integer")
    digest = hmac.new(
        str(base_seed).encode("ascii"),
        _seed_key(parts).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return 1 + (int.from_bytes(digest[:8], "big") % ((1 << 31) - 1))


class SeedRegistry:
    """Collision-check and materialize every semantic seed used by a design."""

    def __init__(
        self,
        base_seed: int = BASE_SEED,
        *,
        namespace: str = "bdb-suite",
        seed_function: Callable[..., int] = stable_seed,
    ) -> None:
        self.base_seed = base_seed
        self.namespace = namespace
        self.seed_function = seed_function
        self._by_key: dict[str, int] = {}
        self._by_seed: dict[int, str] = {}

    def get(self, *parts: Any) -> int:
        key = _seed_key(parts)
        if key in self._by_key:
            return self._by_key[key]
        seed = int(self.seed_function(self.base_seed, self.namespace, *parts))
        if not 1 <= seed < (1 << 31):
            raise DesignError(f"derived seed for {key} is outside the positive 31-bit range")
        prior = self._by_seed.get(seed)
        if prior is not None and prior != key:
            raise SeedCollisionError(f"seed collision {seed}: {prior} and {key}")
        self._by_key[key] = seed
        self._by_seed[seed] = key
        return seed

    def snapshot(self) -> dict[str, int]:
        return {key: self._by_key[key] for key in sorted(self._by_key)}


def design_seed(design: Mapping[str, Any], *parts: Any) -> int:
    registry = design.get("seed_registry")
    if not isinstance(registry, Mapping):
        raise DesignError("design has no seed_registry")
    key = _seed_key(parts)
    if key not in registry:
        raise KeyError(f"seed key was not materialized: {key}")
    seed = registry[key]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DesignError(f"materialized seed is not an integer: {key}")
    return seed


def _normalize_scalar(value: Any, path: str) -> str:
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        value = item()
    if value is None or isinstance(value, bool):
        raise GameRegistryError(f"{path} must be a nonempty scalar")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise GameRegistryError(f"{path} must be integer-like or a string")
        value = int(value)
    if not isinstance(value, (str, int)):
        raise GameRegistryError(f"{path} must be integer-like or a string")
    text = str(value).strip()
    if not text:
        raise GameRegistryError(f"{path} cannot be empty")
    return text


def _normalize_int(value: Any, path: str) -> int:
    text = _normalize_scalar(value, path)
    try:
        integer = int(text)
    except ValueError as exc:
        raise GameRegistryError(f"{path} must be integer-like") from exc
    try:
        if float(text) != integer:
            raise GameRegistryError(f"{path} must be integral")
    except ValueError as exc:
        raise GameRegistryError(f"{path} must be integral") from exc
    return integer


def _records(value: Any) -> list[Mapping[str, Any]]:
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        try:
            records = list(value.to_dict(orient="records"))
        except TypeError as exc:
            raise GameRegistryError("game table must support to_dict(orient='records')") from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = list(value)
    else:
        raise GameRegistryError("game records must be a table or sequence of mappings")
    if any(not isinstance(record, Mapping) for record in records):
        raise GameRegistryError("every game record must be a mapping")
    return records


def normalize_game_records(
    game_records: Any,
    *,
    game_id_column: str = "game_id",
    season_column: str = "season",
    week_column: str = "week",
) -> list[dict[str, Any]]:
    """Normalize exactly one record per game and reject augmented identifiers."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, record in enumerate(_records(game_records), start=1):
        missing = [
            name for name in (game_id_column, season_column, week_column) if name not in record
        ]
        if missing:
            raise GameRegistryError(f"game row {row_number} is missing columns: {missing}")
        game_id = _normalize_scalar(record[game_id_column], f"row {row_number} game_id")
        if game_id.lower().endswith("_aug"):
            raise GameRegistryError(f"augmented game is forbidden: {game_id}")
        if game_id in seen:
            raise GameRegistryError(f"duplicate game_id: {game_id}")
        seen.add(game_id)
        normalized.append(
            {
                "game_id": game_id,
                "season": _normalize_int(record[season_column], f"row {row_number} season"),
                "week": _normalize_int(record[week_column], f"row {row_number} week"),
            }
        )
    return sorted(normalized, key=lambda row: row["game_id"])


def largest_remainder_allocation(
    total: int,
    capacities: Mapping[str, int],
    tie_order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Allocate an integer total proportionally without exceeding capacity."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise DesignError("allocation total must be a nonnegative integer")
    if not capacities:
        raise DesignError("capacities cannot be empty")
    keys = list(tie_order) if tie_order is not None else sorted(capacities)
    if len(keys) != len(set(keys)) or set(keys) != set(capacities):
        raise DesignError("tie_order must contain every capacity key exactly once")
    normalized: dict[str, int] = {}
    for key in keys:
        value = capacities[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignError(f"capacity for {key!r} must be nonnegative")
        normalized[key] = value
    denominator = sum(normalized.values())
    if total > denominator:
        raise DesignError(f"cannot allocate {total} into capacity {denominator}")
    if denominator == 0:
        return {key: 0 for key in keys}
    allocation = {
        key: (total * normalized[key]) // denominator
        for key in keys
    }
    remainders = {
        key: (total * normalized[key]) % denominator
        for key in keys
    }
    remaining = total - sum(allocation.values())
    rank = {key: index for index, key in enumerate(keys)}
    for key in sorted(keys, key=lambda item: (-remainders[item], rank[item]))[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != total:
        raise DesignError("allocation sum invariant failed")
    if any(allocation[key] > normalized[key] for key in keys):
        raise DesignError("allocation capacity invariant failed")
    return allocation


def _rank_items(items: Sequence[str], seed: int, namespace: str) -> list[str]:
    def key(item: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            canonical_json({"namespace": namespace, "seed": seed, "item": item}).encode("utf-8")
        ).hexdigest()
        return digest, item

    return sorted(items, key=key)


def _group_records(records: Sequence[Mapping[str, Any]], stratify_by: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for record in records:
        if stratify_by == "season":
            key = f"season:{record['season']}"
        elif stratify_by == "week":
            key = f"week:{int(record['week']):02d}"
        else:
            raise DesignError(f"unsupported stratum: {stratify_by}")
        output.setdefault(key, []).append(str(record["game_id"]))
    return {key: sorted(output[key]) for key in sorted(output)}


def _stratification(records: Sequence[Mapping[str, Any]]) -> str:
    seasons = {record["season"] for record in records}
    return "season" if len(seasons) > 1 else "week"


def _tie_order(keys: Sequence[str], seed: int, namespace: str) -> list[str]:
    return _rank_items(list(keys), seed, namespace)


def _split_development_folds(
    development_by_stratum: Mapping[str, Sequence[str]],
    *,
    registry: SeedRegistry,
    registry_id: str,
) -> list[list[str]]:
    total = sum(len(ids) for ids in development_by_stratum.values())
    fold_seed = registry.get("registry", registry_id, "development_folds", "fold_order")
    fold_names = [str(index) for index in range(1, DEVELOPMENT_FOLDS + 1)]
    tie = _tie_order(fold_names, fold_seed, f"{registry_id}/development/fold-order")
    base, remainder = divmod(total, DEVELOPMENT_FOLDS)
    desired = {name: base for name in fold_names}
    for name in tie[:remainder]:
        desired[name] += 1
    remaining = dict(desired)
    folds = {name: [] for name in fold_names}
    stratum_seed = registry.get("registry", registry_id, "development_folds", "strata")
    stratum_order = _tie_order(
        list(development_by_stratum), stratum_seed, f"{registry_id}/development/strata"
    )
    for stratum in stratum_order:
        rank_seed = registry.get("registry", registry_id, "development_folds", stratum)
        ids = _rank_items(
            list(development_by_stratum[stratum]),
            rank_seed,
            f"{registry_id}/development/folds/{stratum}",
        )
        allocation = largest_remainder_allocation(len(ids), remaining, tie_order=tie)
        cursor = 0
        for fold_name in tie:
            count = allocation[fold_name]
            folds[fold_name].extend(ids[cursor : cursor + count])
            remaining[fold_name] -= count
            cursor += count
    if any(remaining.values()):
        raise DesignError("development fold allocation failed to fill every fold")
    return [sorted(folds[str(index)]) for index in range(1, DEVELOPMENT_FOLDS + 1)]


def build_game_registry(
    task_spec: TaskSpec | Mapping[str, Any],
    game_records: Any,
    *,
    base_seed: int = BASE_SEED,
    registry: SeedRegistry | None = None,
) -> dict[str, Any]:
    """Sequester deterministic development games and five grouped folds."""

    spec = validate_task_spec(task_spec)
    profile = validate_task_profile(spec)
    records = normalize_game_records(game_records)
    if len(records) != spec.total_games:
        raise GameRegistryError(
            f"{spec.task_id} requires exactly {spec.total_games} games; got {len(records)}"
        )
    registry_id = spec.shared_game_registry_with or spec.task_id
    seeds = registry or SeedRegistry(base_seed)
    stratify_by = _stratification(records)
    by_stratum = _group_records(records, stratify_by)
    capacities = {key: len(ids) for key, ids in by_stratum.items()}
    tie_seed = seeds.get("registry", registry_id, "development", "allocation_tie")
    tie = _tie_order(list(by_stratum), tie_seed, f"{registry_id}/development/tie")
    allocations = largest_remainder_allocation(
        spec.development_games, capacities, tie_order=tie
    )
    development_by_stratum: dict[str, list[str]] = {}
    confirmatory_by_stratum: dict[str, list[str]] = {}
    for stratum in tie:
        seed = seeds.get("registry", registry_id, "development", stratum)
        ordered = _rank_items(
            by_stratum[stratum], seed, f"{registry_id}/development/{stratum}"
        )
        stop = allocations[stratum]
        development_by_stratum[stratum] = ordered[:stop]
        confirmatory_by_stratum[stratum] = ordered[stop:]
    development_ids = sorted(
        game_id for ids in development_by_stratum.values() for game_id in ids
    )
    confirmatory_ids = sorted(
        game_id for ids in confirmatory_by_stratum.values() for game_id in ids
    )
    folds = _split_development_folds(
        development_by_stratum, registry=seeds, registry_id=registry_id
    )
    record_map = {record["game_id"]: dict(record) for record in records}
    payload = {
        "schema_version": "bdb-game-registry-v1",
        "registry_id": registry_id,
        "stratify_by": stratify_by,
        "records": [record_map[game_id] for game_id in sorted(record_map)],
        "stratum_counts": capacities,
        "development_allocations": allocations,
        "development_game_ids": development_ids,
        "confirmatory_game_ids": confirmatory_ids,
        "development_folds": [
            {"fold": index, "validation_game_ids": fold}
            for index, fold in enumerate(folds, start=1)
        ],
    }
    payload["registry_hash"] = sha256_json(payload)
    # Referencing profile here is intentional: it catches accidental use of a
    # same-sized but scientifically different task contract.
    if profile.shared_game_registry_with != spec.shared_game_registry_with:
        raise DesignError("task registry namespace differs from its locked profile")
    return payload


def _balanced_order(
    by_stratum: Mapping[str, Sequence[str]], tie_order: Sequence[str]
) -> list[str]:
    """Interleave pre-ranked stratum lists with nested proportional prefixes."""

    capacities = {key: len(by_stratum[key]) for key in tie_order}
    total = sum(capacities.values())
    used = {key: 0 for key in tie_order}
    cursor = {key: 0 for key in tie_order}
    rank = {key: index for index, key in enumerate(tie_order)}
    output: list[str] = []
    for position in range(1, total + 1):
        available = [key for key in tie_order if cursor[key] < capacities[key]]
        chosen = max(
            available,
            key=lambda key: (
                Fraction(position * capacities[key], total) - used[key],
                -rank[key],
            ),
        )
        output.append(str(by_stratum[chosen][cursor[chosen]]))
        cursor[chosen] += 1
        used[chosen] += 1
    return output


def _build_outer_split(
    spec: TaskSpec,
    game_registry: Mapping[str, Any],
    repeat: int,
    seeds: SeedRegistry,
    *,
    seed_namespace: Sequence[str] = (),
) -> dict[str, Any]:
    registry_id = str(game_registry["registry_id"])
    namespace = tuple(str(value) for value in seed_namespace)
    rank_namespace = "/".join((registry_id, *namespace, "outer", str(repeat)))
    confirmatory_set = set(game_registry["confirmatory_game_ids"])
    records = [
        record for record in game_registry["records"] if record["game_id"] in confirmatory_set
    ]
    by_stratum = _group_records(records, str(game_registry["stratify_by"]))
    capacities = {key: len(ids) for key, ids in by_stratum.items()}
    tie_seed = seeds.get(
        "registry", registry_id, *namespace, "outer", repeat, "allocation_tie"
    )
    tie = _tie_order(list(by_stratum), tie_seed, f"{rank_namespace}/tie")
    train_allocation = largest_remainder_allocation(
        spec.outer_counts["train"], capacities, tie_order=tie
    )
    after_train = {
        stratum: capacities[stratum] - train_allocation[stratum] for stratum in tie
    }
    calibration_allocation = largest_remainder_allocation(
        spec.outer_counts["calibration"], after_train, tie_order=tie
    )
    train_by: dict[str, list[str]] = {}
    calibration_by: dict[str, list[str]] = {}
    test_by: dict[str, list[str]] = {}
    stratum_seeds: dict[str, int] = {}
    for stratum in tie:
        seed = seeds.get("registry", registry_id, *namespace, "outer", repeat, stratum)
        stratum_seeds[stratum] = seed
        ordered = _rank_items(
            by_stratum[stratum], seed, f"{rank_namespace}/{stratum}"
        )
        train_stop = train_allocation[stratum]
        calibration_stop = train_stop + calibration_allocation[stratum]
        train_by[stratum] = ordered[:train_stop]
        calibration_by[stratum] = ordered[train_stop:calibration_stop]
        test_by[stratum] = ordered[calibration_stop:]

    order_seed = seeds.get(
        "registry", registry_id, *namespace, "outer", repeat, "nested_order"
    )
    nested_tie = _tie_order(tie, order_seed, f"{rank_namespace}/nested-tie")
    # Re-rank within each already selected train stratum independently of the
    # selection rank so the learning curve has its own explicit random stage.
    ranked_train_by: dict[str, list[str]] = {}
    nested_stratum_seeds: dict[str, int] = {}
    for stratum in nested_tie:
        seed = seeds.get(
            "registry", registry_id, *namespace, "outer", repeat, "nested", stratum
        )
        nested_stratum_seeds[stratum] = seed
        ranked_train_by[stratum] = _rank_items(
            train_by[stratum], seed, f"{rank_namespace}/nested/{stratum}"
        )
    ordered_train = _balanced_order(ranked_train_by, nested_tie)
    train_games = sorted(game_id for ids in train_by.values() for game_id in ids)
    calibration_games = sorted(
        game_id for ids in calibration_by.values() for game_id in ids
    )
    test_games = sorted(game_id for ids in test_by.values() for game_id in ids)
    nested = {str(anchor): ordered_train[:anchor] for anchor in spec.anchors}
    split_payload = {
        "repeat": repeat,
        "train_game_ids": train_games,
        "calibration_game_ids": calibration_games,
        "test_game_ids": test_games,
        "ordered_train_game_ids": ordered_train,
        "nested_train_game_ids": nested,
        "allocations": {
            "train": train_allocation,
            "calibration": calibration_allocation,
            "test": {stratum: len(test_by[stratum]) for stratum in tie},
        },
        "seeds": {
            "allocation_tie": tie_seed,
            "strata": stratum_seeds,
            "nested_order": order_seed,
            "nested_strata": nested_stratum_seeds,
        },
    }
    split_payload["outer_split_hash"] = sha256_json(
        {
            "train": train_games,
            "calibration": calibration_games,
            "test": test_games,
        }
    )
    split_payload["nested_split_hash"] = sha256_json(nested)
    return split_payload


def validate_task_profile(spec: TaskSpec | Mapping[str, Any]) -> TaskDesignProfile:
    parsed = validate_task_spec(spec)
    try:
        profile = TASK_DESIGNS[parsed.task_id]
    except KeyError as exc:
        raise TaskSpecError(f"no locked suite design exists for {parsed.task_id!r}") from exc
    actual = {
        "release_year": parsed.release_year,
        "outcome_type": parsed.outcome_type,
        "total_games": parsed.total_games,
        "development_games": parsed.development_games,
        "confirmatory_games": parsed.confirmatory_games,
        "outer_counts": dict(parsed.outer_counts),
        "anchors": tuple(parsed.anchors),
        "shared_game_registry_with": parsed.shared_game_registry_with,
    }
    expected = {
        "release_year": profile.release_year,
        "outcome_type": profile.outcome_type,
        "total_games": profile.total_games,
        "development_games": profile.development_games,
        "confirmatory_games": profile.confirmatory_games,
        "outer_counts": profile.outer_counts,
        "anchors": profile.anchors,
        "shared_game_registry_with": profile.shared_game_registry_with,
    }
    if actual != expected:
        differing = [key for key in expected if actual[key] != expected[key]]
        raise TaskSpecError(
            f"{parsed.task_id} differs from its locked design fields: {differing}"
        )
    return profile


def _cell_seed_model_key(spec: TaskSpec, model_id: str) -> str:
    """Use one semantic seed key for all prospectively paired neural roles."""

    family = str(spec.models[model_id]["family"])
    return (
        "shared_neural"
        if family in {"relnet", "attn_relnet", "set_transformer"}
        else model_id
    )


def _primary_model_ids(
    spec: TaskSpec, execution_profile: ExecutionProfile | None
) -> tuple[str, ...]:
    """Return the profile-bound primary roles, grandfathering immutable full50."""

    if execution_profile is not None and execution_profile.profile_id == "full50":
        legacy = (
            "linear_structure",
            "boosted_structure",
            "relnet",
            "attn_relnet",
        )
        if not set(legacy) <= set(spec.models):
            raise DesignError("full50 lacks its four grandfathered primary roles")
        return tuple(sorted(legacy))
    return tuple(sorted(spec.models))


def build_task_design(
    task_spec: TaskSpec | Mapping[str, Any],
    game_records: Any,
    *,
    repeats: int = CONFIRMATORY_REPEATS,
    base_seed: int = BASE_SEED,
    profile: str | None = None,
    seed_profile: str | None = None,
) -> dict[str, Any]:
    """Build the full dev/main registry, split manifests, and seed registry."""

    spec = validate_task_spec(task_spec)
    validate_task_profile(spec)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise DesignError("repeats must be a positive integer")
    execution_profile: ExecutionProfile | None = None
    if profile is not None:
        try:
            execution_profile = EXECUTION_PROFILES[profile]
        except KeyError as exc:
            raise DesignError(f"unknown execution profile {profile!r}") from exc
        if repeats != execution_profile.repeats:
            raise DesignError(
                f"profile {profile!r} requires exactly {execution_profile.repeats} repeats"
            )
    if seed_profile is not None:
        if seed_profile != "pilot10":
            raise DesignError("only pilot10 may override a custom design seed namespace")
        if profile not in {None, "pilot10"}:
            raise DesignError("seed_profile cannot differ from an execution profile")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise DesignError("base_seed must be a positive integer")
    seeds = SeedRegistry(base_seed)
    effective_seed_profile = seed_profile if seed_profile is not None else profile
    seed_namespace = profile_seed_namespace(effective_seed_profile)
    game_registry = build_game_registry(spec, game_records, base_seed=base_seed, registry=seeds)
    split_manifests = [
        _build_outer_split(
            spec,
            game_registry,
            repeat,
            seeds,
            seed_namespace=seed_namespace,
        )
        for repeat in range(1, repeats + 1)
    ]

    primary_cells: list[dict[str, Any]] = []
    primary_model_ids = _primary_model_ids(spec, execution_profile)
    primary_repeats = (
        range(1, repeats + 1)
        if execution_profile is None or execution_profile.include_primary
        else ()
    )
    for repeat in primary_repeats:
        split = split_manifests[repeat - 1]
        for anchor in spec.anchors:
            shared_validation_split_seed = seeds.get(
                "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                "validation_split", "shared_neural", anchor,
            )
            for model_id in primary_model_ids:
                family = spec.models[model_id]["family"]
                seed_model_key = _cell_seed_model_key(spec, model_id)
                cell_seeds = {
                    stage: seeds.get(
                        "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                        stage, seed_model_key, anchor,
                    )
                    for stage in CELL_STAGES
                }
                cell_seeds["uncertainty_subsample"] = seeds.get(
                    "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                    "uncertainty_subsample", "shared_all_models_and_anchors",
                )
                cell_seeds["validation_split"] = shared_validation_split_seed
                primary_cells.append(
                    {
                        "branch": "fixed_main",
                        "ablation_id": None,
                        "sensitivity_id": None,
                        "repeat": repeat,
                        "n_train": anchor,
                        "model": model_id,
                        "role": spec.models[model_id]["role"],
                        "queue": "cpu_tabular" if family in {"glm", "lightgbm"} else "gpu_neural",
                        "outer_split_hash": split["outer_split_hash"],
                        "nested_split_hash": split["nested_split_hash"],
                        "seeds": cell_seeds,
                    }
                )
    ablation_cells: list[dict[str, Any]] = []
    if execution_profile is not None and execution_profile.include_structural_ablation:
        ablation = spec.structural_ablation
        ablation_id = str(ablation["id"])
        for repeat in range(1, int(ablation["repeats"]) + 1):
            split = split_manifests[repeat - 1]
            for anchor in map(int, ablation["anchors"]):
                shared_validation_split_seed = seeds.get(
                    "task", spec.task_id, "cell", "structural_ablation", ablation_id,
                    repeat, "validation_split", "shared_neural", anchor,
                )
                for model_id in ablation["models"]:
                    seed_model_key = _cell_seed_model_key(spec, model_id)
                    cell_seeds = {
                        stage: seeds.get(
                            "task", spec.task_id, "cell", "structural_ablation",
                            ablation_id, repeat, stage, seed_model_key, anchor,
                        )
                        for stage in CELL_STAGES
                    }
                    cell_seeds["uncertainty_subsample"] = seeds.get(
                        "task", spec.task_id, "cell", "fixed_main", repeat,
                        "uncertainty_subsample", "shared_all_models_and_anchors",
                    )
                    cell_seeds["validation_split"] = shared_validation_split_seed
                    ablation_cells.append(
                        {
                            "branch": "structural_ablation",
                            "ablation_id": ablation_id,
                            "sensitivity_id": None,
                            "repeat": repeat,
                            "n_train": anchor,
                            "model": model_id,
                            "role": spec.models[model_id]["role"],
                            "queue": "gpu_neural",
                            "outer_split_hash": split["outer_split_hash"],
                            "nested_split_hash": split["nested_split_hash"],
                            "seeds": cell_seeds,
                        }
                    )
    sensitivity_cells: list[dict[str, Any]] = []
    if execution_profile is not None and execution_profile.include_frozen_sensitivities:
        frozen = [
            item for item in spec.sensitivities
            if item["selection"] == "frozen_prespecified"
        ]
        if len(frozen) != 1:
            raise DesignError(
                "sensitivity20 requires exactly one frozen prespecified sensitivity"
            )
        sensitivity = frozen[0]
        execution = sensitivity["execution"]
        sensitivity_id = str(sensitivity["id"])
        for repeat in range(1, int(execution["repeats"]) + 1):
            split = split_manifests[repeat - 1]
            for anchor in map(int, execution["anchors"]):
                shared_validation_split_seed = seeds.get(
                    "task", spec.task_id, "cell", "fixed_main", repeat,
                    "validation_split", "shared_neural", anchor,
                )
                for model_id in execution["models"]:
                    family = spec.models[model_id]["family"]
                    seed_model_key = _cell_seed_model_key(spec, model_id)
                    cell_seeds = {
                        stage: seeds.get(
                            "task", spec.task_id, "cell", "fixed_main", repeat,
                            stage, seed_model_key, anchor,
                        )
                        for stage in CELL_STAGES
                    }
                    cell_seeds["uncertainty_subsample"] = seeds.get(
                        "task", spec.task_id, "cell", "fixed_main", repeat,
                        "uncertainty_subsample", "shared_all_models_and_anchors",
                    )
                    cell_seeds["validation_split"] = shared_validation_split_seed
                    sensitivity_cells.append(
                        {
                            "branch": "frozen_sensitivity",
                            "ablation_id": None,
                            "sensitivity_id": sensitivity_id,
                            "repeat": repeat,
                            "n_train": anchor,
                            "model": model_id,
                            "role": spec.models[model_id]["role"],
                            "queue": (
                                "cpu_tabular"
                                if family in {"glm", "lightgbm"}
                                else "gpu_neural"
                            ),
                            "outer_split_hash": split["outer_split_hash"],
                            "nested_split_hash": split["nested_split_hash"],
                            "seeds": cell_seeds,
                        }
                    )
    required_cells = primary_cells + ablation_cells + sensitivity_cells
    for fold in range(1, DEVELOPMENT_FOLDS + 1):
        for model_id in sorted(spec.models):
            seed_model_key = _cell_seed_model_key(spec, model_id)
            seeds.get(
                "task", spec.task_id, "development_cv", fold, "fit", seed_model_key
            )
            seeds.get(
                "task", spec.task_id, "development_cv", fold, "prediction",
                seed_model_key,
            )
    seeds.get(*analysis_seed_parts(spec.task_id, effective_seed_profile))

    payload = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "task_spec_hash": spec.spec_hash,
        "base_seed": base_seed,
        "seed_algorithm": SEED_ALGORITHM,
        "profile": execution_profile.profile_id if execution_profile is not None else "custom",
        **({"seed_profile": seed_profile} if seed_profile is not None else {}),
        "repeats": repeats,
        "anchors": list(spec.anchors),
        "models": sorted(spec.models),
        "game_registry": game_registry,
        "split_manifests": split_manifests,
        "primary_cells": primary_cells,
        "ablation_cells": ablation_cells,
        "sensitivity_cells": sensitivity_cells,
        "required_cells": required_cells,
        "cell_counts": {
            "primary": len(primary_cells),
            "structural_ablation": len(ablation_cells),
            "frozen_sensitivity": len(sensitivity_cells),
            "required": len(required_cells),
        },
        "seed_registry": seeds.snapshot(),
    }
    payload["design_hash"] = sha256_json(payload)
    validate_task_design(payload, spec, profile=profile)
    return payload


def _assert_unique_disjoint(named: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in named.items():
        if len(ids) != len(set(ids)):
            raise DesignError(f"{name} contains duplicate game IDs")
        for game_id in ids:
            prior = seen.get(game_id)
            if prior is not None:
                raise DesignError(f"game {game_id} occurs in both {prior} and {name}")
            seen[game_id] = name


def validate_task_design(
    design: Mapping[str, Any],
    task_spec: TaskSpec | Mapping[str, Any],
    *,
    profile: str | None = None,
) -> None:
    """Reject wrong counts, seeds, hashes, cells, leakage, or non-nesting."""

    spec = validate_task_spec(task_spec)
    validate_task_profile(spec)
    if not isinstance(design, Mapping) or design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise DesignError("unsupported task design")
    if design.get("task_id") != spec.task_id or design.get("task_spec_hash") != spec.spec_hash:
        raise DesignError("task design is bound to another TaskSpec")
    observed_profile = design.get("profile", "custom")
    if profile is not None and observed_profile != profile:
        raise DesignError("task design execution profile differs from the requested profile")
    execution_profile: ExecutionProfile | None = None
    if observed_profile != "custom":
        try:
            execution_profile = EXECUTION_PROFILES[str(observed_profile)]
        except KeyError as exc:
            raise DesignError(f"unknown task design profile {observed_profile!r}") from exc
        if profile is not None and execution_profile.profile_id != profile:
            raise DesignError("task design profile mismatch")
    observed_seed_profile = design.get(
        "seed_profile", None if observed_profile == "custom" else observed_profile
    )
    if "seed_profile" in design and (
        observed_seed_profile != "pilot10" or observed_profile not in {"custom", "pilot10"}
    ):
        raise DesignError("task design seed_profile override is invalid")
    payload = {key: value for key, value in design.items() if key != "design_hash"}
    if design.get("design_hash") != sha256_json(payload):
        raise DesignError("task design hash does not match its payload")
    game_registry = design.get("game_registry")
    if not isinstance(game_registry, Mapping):
        raise DesignError("game_registry is missing")
    registry_payload = {
        key: value for key, value in game_registry.items() if key != "registry_hash"
    }
    if game_registry.get("registry_hash") != sha256_json(registry_payload):
        raise DesignError("game registry hash does not match its payload")
    frozen_game_registry = spec.cohort.get("game_registry")
    if frozen_game_registry is not None:
        if not isinstance(frozen_game_registry, Mapping):
            raise DesignError("frozen TaskSpec cohort.game_registry must be an object")
        if canonical_json(frozen_game_registry) != canonical_json(game_registry):
            raise DesignError(
                "task design game registry differs from the exact registry frozen in the TaskSpec"
            )
    development = game_registry.get("development_game_ids")
    confirmatory = game_registry.get("confirmatory_game_ids")
    if not isinstance(development, list) or not isinstance(confirmatory, list):
        raise DesignError("game registry cohorts must be arrays")
    if len(development) != spec.development_games or len(confirmatory) != spec.confirmatory_games:
        raise DesignError("game registry cohort counts are wrong")
    _assert_unique_disjoint({"development": development, "confirmatory": confirmatory})
    folds = game_registry.get("development_folds")
    if not isinstance(folds, list) or len(folds) != DEVELOPMENT_FOLDS:
        raise DesignError("game registry must contain five development folds")
    fold_ids: list[str] = []
    for expected_fold, fold in enumerate(folds, start=1):
        if not isinstance(fold, Mapping) or fold.get("fold") != expected_fold:
            raise DesignError("development folds are not numbered 1 through 5")
        ids = fold.get("validation_game_ids")
        if not isinstance(ids, list):
            raise DesignError("development fold game IDs must be arrays")
        fold_ids.extend(ids)
    if sorted(fold_ids) != sorted(development) or len(fold_ids) != len(set(fold_ids)):
        raise DesignError("development folds must partition development games exactly")
    if max(map(len, [fold["validation_game_ids"] for fold in folds])) - min(
        map(len, [fold["validation_game_ids"] for fold in folds])
    ) > 1:
        raise DesignError("development fold sizes differ by more than one game")

    repeats = design.get("repeats")
    splits = design.get("split_manifests")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise DesignError("design repeats is invalid")
    if execution_profile is not None and repeats != execution_profile.repeats:
        raise DesignError(
            f"profile {execution_profile.profile_id!r} requires "
            f"{execution_profile.repeats} repeats"
        )
    if not isinstance(splits, list) or len(splits) != repeats:
        raise DesignError("split manifest count does not match repeats")
    seed_namespace = profile_seed_namespace(
        None if observed_seed_profile is None else str(observed_seed_profile)
    )
    split_hashes: set[str] = set()
    for repeat, split in enumerate(splits, start=1):
        if not isinstance(split, Mapping) or split.get("repeat") != repeat:
            raise DesignError("split repeats must be consecutive from 1")
        train = split.get("train_game_ids")
        calibration = split.get("calibration_game_ids")
        test = split.get("test_game_ids")
        ordered_train = split.get("ordered_train_game_ids")
        if not all(isinstance(ids, list) for ids in (train, calibration, test, ordered_train)):
            raise DesignError("outer split memberships must be arrays")
        if (len(train), len(calibration), len(test)) != (
            spec.outer_counts["train"],
            spec.outer_counts["calibration"],
            spec.outer_counts["test"],
        ):
            raise DesignError(f"repeat {repeat} has wrong outer counts")
        _assert_unique_disjoint({"train": train, "calibration": calibration, "test": test})
        if set(train) | set(calibration) | set(test) != set(confirmatory):
            raise DesignError(f"repeat {repeat} does not partition confirmatory games")
        if set(ordered_train) != set(train) or len(ordered_train) != len(train):
            raise DesignError(f"repeat {repeat} ordered train list is invalid")
        nested = split.get("nested_train_game_ids")
        if not isinstance(nested, Mapping) or set(nested) != {str(a) for a in spec.anchors}:
            raise DesignError(f"repeat {repeat} nested anchors are wrong")
        prior: list[str] = []
        for anchor in spec.anchors:
            ids = nested[str(anchor)]
            if not isinstance(ids, list) or len(ids) != anchor:
                raise DesignError(f"repeat {repeat} anchor {anchor} has wrong size")
            if ids != ordered_train[:anchor] or ids[: len(prior)] != prior:
                raise DesignError(f"repeat {repeat} anchor {anchor} is not strictly nested")
            prior = ids
        expected_outer_hash = sha256_json(
            {"train": train, "calibration": calibration, "test": test}
        )
        if split.get("outer_split_hash") != expected_outer_hash:
            raise DesignError(f"repeat {repeat} outer split hash is wrong")
        if split.get("nested_split_hash") != sha256_json(nested):
            raise DesignError(f"repeat {repeat} nested split hash is wrong")
        if expected_outer_hash in split_hashes:
            raise DesignError("two repeats contain the same complete outer partition")
        split_hashes.add(expected_outer_hash)
        replay_registry = SeedRegistry(int(design.get("base_seed", 0)))
        expected_split = _build_outer_split(
            spec,
            game_registry,
            repeat,
            replay_registry,
            seed_namespace=seed_namespace,
        )
        if canonical_json(split) != canonical_json(expected_split):
            raise DesignError(
                f"repeat {repeat} split does not replay from its profile-bound seeds"
            )

    primary_cells = design.get("primary_cells")
    ablation_cells = design.get("ablation_cells")
    sensitivity_cells = design.get("sensitivity_cells")
    cells = design.get("required_cells")
    expected_primary_count = (
        repeats * len(spec.anchors) * len(
            _primary_model_ids(spec, execution_profile)
        )
        if execution_profile is None or execution_profile.include_primary
        else 0
    )
    if not isinstance(primary_cells, list) or len(primary_cells) != expected_primary_count:
        raise DesignError(
            f"primary grid must contain exactly {expected_primary_count} cells"
        )
    expected_ablation_count = (
        int(spec.structural_ablation["repeats"])
        * len(spec.structural_ablation["anchors"])
        * len(spec.structural_ablation["models"])
        if execution_profile is not None
        and execution_profile.include_structural_ablation
        else 0
    )
    if not isinstance(ablation_cells, list) or len(ablation_cells) != expected_ablation_count:
        raise DesignError(
            f"structural-ablation grid must contain exactly {expected_ablation_count} cells"
        )
    expected_sensitivity_count = (
        20 * 2 * len(spec.models)
        if execution_profile is not None
        and execution_profile.include_frozen_sensitivities
        else 0
    )
    if (
        not isinstance(sensitivity_cells, list)
        or len(sensitivity_cells) != expected_sensitivity_count
    ):
        raise DesignError(
            f"frozen-sensitivity grid must contain exactly {expected_sensitivity_count} cells"
        )
    if (
        not isinstance(cells, list)
        or cells != primary_cells + ablation_cells + sensitivity_cells
    ):
        raise DesignError(
            "required cells must be primary, ablation, then sensitivity cells"
        )
    cell_counts = design.get("cell_counts")
    expected_counts = {
        "primary": expected_primary_count,
        "structural_ablation": expected_ablation_count,
        "frozen_sensitivity": expected_sensitivity_count,
        "required": (
            expected_primary_count
            + expected_ablation_count
            + expected_sensitivity_count
        ),
    }
    if cell_counts != expected_counts:
        raise DesignError("task design cell_counts are invalid")
    if execution_profile is not None and (
        expected_primary_count != execution_profile.expected_primary_cells
        or expected_ablation_count != execution_profile.expected_ablation_cells
        or expected_sensitivity_count != execution_profile.expected_sensitivity_cells
        or len(cells) != execution_profile.expected_required_cells
    ):
        raise DesignError(
            f"profile {execution_profile.profile_id!r} grid cardinality is invalid"
        )
    identities: set[tuple[Any, ...]] = set()
    seed_values: dict[int, str] = {}
    registry = design.get("seed_registry")
    if not isinstance(registry, Mapping):
        raise DesignError("seed registry is missing")
    base_seed = design.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed <= 0:
        raise DesignError("task design base_seed is invalid")
    for semantic_key, seed in registry.items():
        if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < (1 << 31):
            raise DesignError(f"invalid materialized seed: {semantic_key}")
        prior = seed_values.get(seed)
        if prior is not None and prior != semantic_key:
            raise DesignError(f"materialized seed collision {seed}")
        seed_values[seed] = semantic_key
        try:
            parts = json.loads(str(semantic_key))
        except (TypeError, json.JSONDecodeError) as exc:
            raise DesignError(f"materialized seed key is invalid: {semantic_key}") from exc
        if not isinstance(parts, list):
            raise DesignError(f"materialized seed key is not a tuple: {semantic_key}")
        expected_materialized = stable_seed(
            base_seed, "bdb-suite", *parts
        )
        if seed != expected_materialized:
            raise DesignError(f"materialized seed does not replay: {semantic_key}")
    for fold in range(1, DEVELOPMENT_FOLDS + 1):
        for model_id in sorted(spec.models):
            seed_model_key = _cell_seed_model_key(spec, model_id)
            for stage in ("fit", "prediction"):
                design_seed(
                    design,
                    "task",
                    spec.task_id,
                    "development_cv",
                    fold,
                    stage,
                    seed_model_key,
                )
    design_seed(
        design,
        *analysis_seed_parts(
            spec.task_id,
            None if observed_seed_profile is None else str(observed_seed_profile),
        ),
    )
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise DesignError("required cells must be objects")
        identity = (
            cell.get("branch"), cell.get("ablation_id"), cell.get("sensitivity_id"), cell.get("repeat"),
            cell.get("n_train"), cell.get("model"),
        )
        if identity in identities:
            raise DesignError(f"duplicate required cell: {identity}")
        identities.add(identity)
        branch = cell.get("branch")
        ablation_id = cell.get("ablation_id")
        sensitivity_id = cell.get("sensitivity_id")
        model = cell.get("model")
        if branch == "fixed_main":
            if (
                ablation_id is not None
                or sensitivity_id is not None
                or cell not in primary_cells
                or model not in _primary_model_ids(spec, execution_profile)
            ):
                raise DesignError("fixed-main cell has an ablation identity")
        elif branch == "structural_ablation":
            if (
                ablation_id != spec.structural_ablation["id"]
                or sensitivity_id is not None
                or cell not in ablation_cells
            ):
                raise DesignError("structural-ablation cell has a wrong identity")
        elif branch == "frozen_sensitivity":
            frozen_ids = {
                str(item["id"])
                for item in spec.sensitivities
                if item["selection"] == "frozen_prespecified"
            }
            if (
                ablation_id is not None
                or sensitivity_id not in frozen_ids
                or cell not in sensitivity_cells
            ):
                raise DesignError("frozen-sensitivity cell has a wrong identity")
        else:
            raise DesignError("unexpected cell branch")
        repeat = cell.get("repeat")
        anchor = cell.get("n_train")
        if repeat not in range(1, repeats + 1) or anchor not in spec.anchors or model not in spec.models:
            raise DesignError(f"unexpected required cell: {identity}")
        if cell.get("role") != spec.models[model]["role"]:
            raise DesignError(f"wrong role for model {model}")
        if branch == "structural_ablation" and (
            repeat > int(spec.structural_ablation["repeats"])
            or anchor not in spec.structural_ablation["anchors"]
            or model not in spec.structural_ablation["models"]
        ):
            raise DesignError(f"unexpected structural-ablation cell: {identity}")
        if branch == "frozen_sensitivity":
            sensitivity = next(
                item for item in spec.sensitivities
                if item["id"] == sensitivity_id
            )
            execution = sensitivity["execution"]
            if (
                repeat > int(execution["repeats"])
                or anchor not in execution["anchors"]
                or model not in execution["models"]
            ):
                raise DesignError(f"unexpected frozen-sensitivity cell: {identity}")
        split = splits[repeat - 1]
        if cell.get("outer_split_hash") != split["outer_split_hash"]:
            raise DesignError("model cell does not share its repeat outer split")
        if cell.get("nested_split_hash") != split["nested_split_hash"]:
            raise DesignError("model cell does not share its repeat nested split")
        expected_queue = (
            "cpu_tabular"
            if spec.models[model]["family"] in {"glm", "lightgbm"}
            else "gpu_neural"
        )
        if cell.get("queue") != expected_queue:
            raise DesignError(f"wrong queue for model {model}")
        cell_seeds = cell.get("seeds")
        if not isinstance(cell_seeds, Mapping) or set(cell_seeds) != set(CELL_SEED_FIELDS):
            raise DesignError("cell stage seeds are incomplete")
        for stage in CELL_STAGES:
            seed_model_key = _cell_seed_model_key(spec, str(model))
            seed_parts: tuple[Any, ...]
            if stage == "uncertainty_subsample":
                seed_parts = (
                    "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                    "uncertainty_subsample", "shared_all_models_and_anchors",
                )
            elif branch == "fixed_main":
                seed_parts = (
                    "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                    stage, seed_model_key, anchor,
                )
            elif branch == "structural_ablation":
                seed_parts = (
                    "task", spec.task_id, "cell", "structural_ablation",
                    ablation_id, repeat, stage, seed_model_key, anchor,
                )
            else:
                seed_parts = (
                    "task", spec.task_id, "cell", "fixed_main", repeat,
                    stage, seed_model_key, anchor,
                )
            expected_seed = design_seed(design, *seed_parts)
            if cell_seeds[stage] != expected_seed:
                raise DesignError(f"wrong {stage} seed for cell {identity}")
        if branch == "fixed_main":
            validation_parts = (
                "task", spec.task_id, *seed_namespace, "cell", "fixed_main", repeat,
                "validation_split", "shared_neural", anchor,
            )
        elif branch == "structural_ablation":
            validation_parts = (
                "task", spec.task_id, "cell", "structural_ablation",
                ablation_id, repeat, "validation_split", "shared_neural", anchor,
            )
        else:
            validation_parts = (
                "task", spec.task_id, "cell", "fixed_main", repeat,
                "validation_split", "shared_neural", anchor,
            )
        expected_validation_split = design_seed(design, *validation_parts)
        if cell_seeds["validation_split"] != expected_validation_split:
            raise DesignError(f"wrong shared validation split seed for cell {identity}")

    uq_seeds_by_repeat: dict[int, set[int]] = {}
    for cell in cells:
        uq_seeds_by_repeat.setdefault(int(cell["repeat"]), set()).add(
            int(cell["seeds"]["uncertainty_subsample"])
        )
    if any(len(values) != 1 for values in uq_seeds_by_repeat.values()):
        raise DesignError(
            "uncertainty subsampling must share one seed across every repeat panel"
        )

    neural_pairs: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        if cell["model"] not in {"relnet", "attn_relnet"}:
            continue
        pair_key = (
            cell["branch"], cell.get("ablation_id"), cell.get("sensitivity_id"),
            int(cell["repeat"]), int(cell["n_train"]),
        )
        neural_pairs.setdefault(pair_key, {})[str(cell["model"])] = cell["seeds"]
    for pair_key, pair in neural_pairs.items():
        if set(pair) != {"relnet", "attn_relnet"}:
            raise DesignError(f"matched neural pair is incomplete: {pair_key}")
        if dict(pair["relnet"]) != dict(pair["attn_relnet"]):
            raise DesignError(f"matched neural pair does not share all seeds: {pair_key}")


def shared_split_hashes(design: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return repeat-indexed hashes used to assert cross-task shared partitions."""

    splits = design.get("split_manifests")
    if not isinstance(splits, list):
        raise DesignError("design has no split manifests")
    return tuple(
        (str(split["outer_split_hash"]), str(split["nested_split_hash"]))
        for split in splits
    )
