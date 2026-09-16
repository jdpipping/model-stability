"""Split-local feature encoders and lazy neural representations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEAM_IDENTITY_COLUMNS = frozenset(
    {
        "possessionTeam",
        "defensiveTeam",
        "kickingTeam",
        "returnTeam",
        "homeTeamAbbr",
        "visitorTeamAbbr",
        "club",
        "team",
    }
)

RELATIONAL_FAMILIES = frozenset({"relnet", "attn_relnet"})
GLOBAL_SET_FAMILIES = frozenset({"transformer", "set_transformer"})
TOKEN_NEURAL_FAMILIES = frozenset({*GLOBAL_SET_FAMILIES, *RELATIONAL_FAMILIES})

# Edge IDs are shared across every task so the two relational families always
# consume an identical tensor. Task structure changes the causal edge mask,
# not the input API or learned parameter count.
EDGE_TYPE_NAMES = (
    "none",
    "quarterback_receiver",
    "receiver_defender",
    "receiver_receiver",
    "returner_blocker",
    "returner_coverer",
    "blocker_coverer",
    "same_team_lane",
    "protector_rusher",
    "quarterback_protector",
    "quarterback_rusher",
    "receiver_coverage",
    "candidate_carrier",
    "blocker_candidate",
    "blocker_carrier",
    "help_carrier",
    "same_team_influence",
    "defender_defender",
    "focal_teammate",
    "focal_opponent",
    "same_side_social",
    "opposing_side_social",
    "generic_relation",
)


def primary_context_frame(
    frame: pd.DataFrame, *, include_team_identity: bool = False
) -> pd.DataFrame:
    """Return context permitted by the primary protocol identity policy."""

    result = frame.reset_index(drop=True).copy()
    if include_team_identity:
        return result
    excluded = [column for column in result if str(column) in TEAM_IDENTITY_COLUMNS]
    return result.drop(columns=excluded)


def _indicator(tokens: np.ndarray, names: tuple[str, ...], name: str) -> np.ndarray:
    if name not in names:
        return np.zeros(tokens.shape[:-1], dtype=bool)
    return tokens[..., names.index(name)] > 0.5


def task_structural_player_mask(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    channel_names: Iterable[str],
    *,
    task_id: str,
) -> np.ndarray:
    """Restrict inputs to the prospectively declared node roles for a task."""

    tokens = np.asarray(player_tokens, dtype=np.float32)
    valid = np.asarray(player_mask, dtype=bool)
    names = tuple(str(value) for value in channel_names)
    if tokens.ndim != 4 or valid.shape != tokens.shape[:3]:
        raise ValueError("structural tokens and masks are incompatible")
    offense = _indicator(tokens, names, "offense") & valid
    defense = _indicator(tokens, names, "defense") & valid
    football = _indicator(tokens, names, "football") & valid
    focal = _indicator(tokens, names, "focal") & valid
    eligible = np.zeros_like(valid)
    for role in ("position_wr", "position_te", "position_rb"):
        eligible |= _indicator(tokens, names, role) & valid
    protector = np.zeros_like(valid)
    for role in ("position_ol", "position_rb", "position_te"):
        protector |= _indicator(tokens, names, role) & valid

    if task_id in {"bdb2021_completion", "bdb2025_man_zone"}:
        active = focal | eligible | defense | football
    elif task_id in {"bdb2020_rushing_harmonized", "bdb2022_punt_returns"}:
        active = focal | offense | defense | football
    elif task_id == "bdb2023_sack":
        active = focal | protector | eligible | defense | football
    elif task_id == "bdb2024_tackle":
        if "carrier" not in names:
            raise ValueError("BDB2024 structural inputs require the official carrier role")
        carrier = _indicator(tokens, names, "carrier") & valid
        active = focal | carrier | offense | defense | football
    elif task_id == "bdb2026_trajectory":
        active = valid & ~football
    else:
        active = valid
    return valid & active


def relational_edge_types(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    channel_names: Iterable[str],
    *,
    task_id: str,
    ablation_id: str | None = None,
) -> np.ndarray:
    """Build deterministic task-structured directed edge types.

    Axis order is ``[example,time,target,source]``. Player identity is absent:
    the graph uses only causal position/group/focal indicators and geometry.
    """

    tokens = np.asarray(player_tokens, dtype=np.float32)
    valid = np.asarray(player_mask, dtype=bool)
    names = tuple(str(value) for value in channel_names)
    if tokens.ndim != 4 or valid.shape != tokens.shape[:3]:
        raise ValueError("relational tokens and masks are incompatible")
    pair = valid[..., :, None] & valid[..., None, :]
    diagonal = np.eye(tokens.shape[2], dtype=bool)[None, None, :, :]
    pair &= ~diagonal
    offense = _indicator(tokens, names, "offense")
    defense = _indicator(tokens, names, "defense")
    football = _indicator(tokens, names, "football")
    focal = _indicator(tokens, names, "focal")
    eligible = np.zeros_like(valid)
    for role in ("position_wr", "position_te", "position_rb"):
        eligible |= _indicator(tokens, names, role)
    protector = np.zeros_like(valid)
    for role in ("position_ol", "position_rb", "position_te"):
        protector |= _indicator(tokens, names, role)

    edges = np.zeros(pair.shape, dtype=np.uint8)

    def between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (
            (left[..., :, None] & right[..., None, :])
            | (right[..., :, None] & left[..., None, :])
        ) & pair

    def within(group: np.ndarray) -> np.ndarray:
        return group[..., :, None] & group[..., None, :] & pair

    def edge(name: str) -> int:
        return EDGE_TYPE_NAMES.index(name)

    if task_id == "bdb2021_completion":
        active = focal | eligible | defense | football
        pair &= active[..., :, None] & active[..., None, :]
        edges[within(eligible)] = edge("receiver_receiver")
        edges[between(eligible, defense)] = edge("receiver_defender")
        edges[between(focal, eligible)] = edge("quarterback_receiver")
    elif task_id in {"bdb2020_rushing_harmonized", "bdb2022_punt_returns"}:
        blocker = offense & ~focal
        coverer = defense
        active = focal | blocker | coverer | football
        pair &= active[..., :, None] & active[..., None, :]
        same_team_lane = within(blocker) | within(coverer)
        returner_blocker = between(focal, blocker)
        returner_coverer = between(focal, coverer)
        blocker_coverer = between(blocker, coverer)
        edges[same_team_lane] = edge("same_team_lane")
        edges[returner_blocker] = edge("returner_blocker")
        edges[returner_coverer] = edge("returner_coverer")
        edges[blocker_coverer] = edge("blocker_coverer")
        if ablation_id in {
            "remove_blocker_coverer_edges",
            "remove_blocker_defender_edges",
        }:
            pair &= ~blocker_coverer
    elif task_id == "bdb2023_sack":
        position_dl = _indicator(tokens, names, "position_dl")
        position_lb = _indicator(tokens, names, "position_lb")
        rusher = defense & (position_dl | position_lb)
        coverage_defender = defense & ~rusher
        active = focal | protector | rusher | eligible | coverage_defender | football
        pair &= active[..., :, None] & active[..., None, :]
        edges[between(protector, rusher)] = edge("protector_rusher")
        edges[between(focal, protector)] = edge("quarterback_protector")
        edges[between(focal, rusher)] = edge("quarterback_rusher")
        edges[between(eligible, coverage_defender)] = edge("receiver_coverage")
        if ablation_id == "collapse_typed_edges":
            edges[pair] = edge("generic_relation")
    elif task_id == "bdb2024_tackle":
        if "carrier" not in names:
            raise ValueError("BDB2024 relational inputs require the official carrier role")
        carrier = _indicator(tokens, names, "carrier") & valid
        blocker = offense & ~carrier
        help_defender = defense & ~focal
        active = focal | carrier | blocker | help_defender | football
        pair &= active[..., :, None] & active[..., None, :]
        edges[within(blocker) | within(focal | help_defender)] = edge(
            "same_team_influence"
        )
        edges[between(focal, blocker)] = edge("blocker_candidate")
        edges[between(blocker, carrier)] = edge("blocker_carrier")
        edges[between(help_defender, carrier)] = edge("help_carrier")
        edges[between(focal, carrier)] = edge("candidate_carrier")
        if ablation_id == "candidate_carrier_pair_only":
            active = focal | carrier
            pair &= active[..., :, None] & active[..., None, :]
    elif task_id == "bdb2025_man_zone":
        active = focal | eligible | defense | football
        pair &= active[..., :, None] & active[..., None, :]
        edges[within(defense)] = edge("defender_defender")
        edges[within(eligible)] = edge("receiver_receiver")
        edges[between(eligible, defense)] = edge("receiver_defender")
        edges[between(focal, eligible)] = edge("quarterback_receiver")
    elif task_id == "bdb2026_trajectory":
        focal_offense = np.any(focal & offense, axis=-1, keepdims=True)
        teammate = np.where(focal_offense, offense, defense) & ~focal
        opponent = np.where(focal_offense, defense, offense)
        active = focal | teammate | opponent
        pair &= active[..., :, None] & active[..., None, :]
        edges[within(teammate) | within(opponent)] = edge("same_side_social")
        edges[between(teammate, opponent)] = edge("opposing_side_social")
        edges[between(focal, teammate)] = edge("focal_teammate")
        edges[between(focal, opponent)] = edge("focal_opponent")
        if ablation_id == "remove_social_player_messages":
            pair &= False
    else:
        # Synthetic/legacy tasks receive a task-neutral causal complete graph.
        edges[pair] = edge("generic_relation")
    edges[~pair] = 0
    return edges


def relative_time_to_event(frame_mask: np.ndarray, *, hz: float = 10.0) -> np.ndarray:
    """Encode seconds relative to each example's final observed input frame."""

    mask = np.asarray(frame_mask, dtype=bool)
    if mask.ndim != 2 or np.any(mask.sum(axis=1) == 0) or hz <= 0:
        raise ValueError("frame mask/time frequency is invalid")
    result = np.zeros(mask.shape + (1,), dtype=np.float32)
    for row in range(len(mask)):
        positions = np.flatnonzero(mask[row])
        result[row, positions, 0] = (positions - positions[-1]) / float(hz)
    return result


def structural_ablation_player_mask(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    channel_names: Iterable[str],
    *,
    task_id: str,
    ablation_id: str | None,
) -> np.ndarray:
    """Remove nodes that are outside a node-level structural ablation."""

    tokens = np.asarray(player_tokens, dtype=np.float32)
    mask = task_structural_player_mask(
        tokens,
        player_mask,
        channel_names,
        task_id=task_id,
    )
    names = tuple(str(value) for value in channel_names)
    if ablation_id is None:
        return mask
    focal = _indicator(tokens, names, "focal") & mask
    if task_id == "bdb2026_trajectory" and ablation_id == "remove_social_player_messages":
        return mask & focal
    if task_id == "bdb2024_tackle" and ablation_id == "candidate_carrier_pair_only":
        if "carrier" not in names:
            raise ValueError("BDB2024 ablation inputs require the official carrier role")
        carrier = _indicator(tokens, names, "carrier") & mask
        return mask & (focal | carrier)
    return mask


def _channel_index(channel_names: Iterable[str], *candidates: str) -> int:
    names = list(channel_names)
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    raise KeyError(f"none of the required channels {candidates!r} are present")


def token_summary_frame(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    frame_mask: np.ndarray,
    channel_names: Iterable[str],
) -> pd.DataFrame:
    """Create deterministic, unscaled tracking summaries for tabular models."""

    # Keep a float32 prepared-task memory map as a view.  Upcasting the full
    # BDB2026 tensor would allocate roughly 17 GiB in every CPU worker; the
    # individual reductions below still accumulate in float64.
    tokens = np.asarray(player_tokens)
    valid_player = np.asarray(player_mask, dtype=bool)
    valid_frame = np.asarray(frame_mask, dtype=bool)
    names = list(channel_names)
    if tokens.ndim != 4 or valid_player.shape != tokens.shape[:3] or valid_frame.shape != tokens.shape[:2]:
        raise ValueError("token and mask shapes are incompatible")
    continuous = [
        name
        for name in (
            "x_rel",
            "y_rel",
            "vx",
            "vy",
            "speed",
            "acceleration",
            "dir_sin",
            "dir_cos",
            "orientation_sin",
            "orientation_cos",
        )
        if name in names
    ]
    group_channels = {
        group: names.index(group)
        for group in ("offense", "defense", "football", "focal")
        if group in names
    }
    rows: list[dict[str, float]] = []
    for example in range(tokens.shape[0]):
        frame_positions = np.flatnonzero(valid_frame[example])
        if len(frame_positions) == 0:
            raise ValueError(f"example {example} has no valid frame")
        final = int(frame_positions[-1])
        row: dict[str, float] = {
            "tracking_frames": float(len(frame_positions)),
            "tracking_players_final": float(valid_player[example, final].sum()),
        }
        for group, group_index in group_channels.items():
            member = valid_player[example, final] & (tokens[example, final, :, group_index] > 0.5)
            row[f"{group}_count"] = float(member.sum())
            for name in continuous:
                values = tokens[example, final, member, names.index(name)]
                if len(values):
                    row[f"{group}_{name}_mean"] = float(
                        np.nanmean(values, dtype=np.float64)
                    )
                    row[f"{group}_{name}_std"] = float(
                        np.nanstd(values, dtype=np.float64)
                    )
                    row[f"{group}_{name}_min"] = float(np.nanmin(values))
                    row[f"{group}_{name}_max"] = float(np.nanmax(values))
                else:
                    for statistic in ("mean", "std", "min", "max"):
                        row[f"{group}_{name}_{statistic}"] = np.nan
        # Causal temporal summaries use only frames supplied by the adapter.
        for name in ("x_rel", "y_rel", "speed", "acceleration"):
            if name not in names:
                continue
            channel = names.index(name)
            values = tokens[example, :, :, channel][valid_player[example]]
            row[f"temporal_{name}_mean"] = (
                float(np.nanmean(values, dtype=np.float64)) if len(values) else np.nan
            )
            row[f"temporal_{name}_std"] = (
                float(np.nanstd(values, dtype=np.float64)) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def relational_summary_frame(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    frame_mask: np.ndarray,
    channel_names: Iterable[str],
) -> pd.DataFrame:
    """Create deterministic permutation-invariant geometric summaries."""

    tokens = np.asarray(player_tokens, dtype=np.float32)
    players = np.asarray(player_mask, dtype=bool)
    frames = np.asarray(frame_mask, dtype=bool)
    names = tuple(str(value) for value in channel_names)
    if "x_rel" not in names or "y_rel" not in names:
        return pd.DataFrame(index=np.arange(len(tokens)))
    x_index, y_index = names.index("x_rel"), names.index("y_rel")
    offense = _indicator(tokens, names, "offense") & players
    defense = _indicator(tokens, names, "defense") & players
    focal = _indicator(tokens, names, "focal") & players
    rows: list[dict[str, float]] = []
    for example in range(len(tokens)):
        observed_frames = np.flatnonzero(frames[example])
        final = int(observed_frames[-1])
        row: dict[str, float] = {}
        temporal_nearest: list[float] = []
        temporal_cross_mean: list[float] = []
        for frame in observed_frames:
            xy = tokens[example, frame, :, [x_index, y_index]].T
            off_index = np.flatnonzero(offense[example, frame])
            def_index = np.flatnonzero(defense[example, frame])
            if len(off_index) and len(def_index):
                delta = xy[off_index, None, :] - xy[def_index, :][None, :, :]
                distances = np.sqrt(np.sum(np.square(delta), axis=-1))
                temporal_nearest.append(float(np.min(distances)))
                temporal_cross_mean.append(float(np.mean(distances)))
                if int(frame) == final:
                    row["cross_side_distance_min"] = float(np.min(distances))
                    row["cross_side_distance_mean"] = float(np.mean(distances))
                    row["cross_side_distance_sd"] = float(np.std(distances))
            focal_index = np.flatnonzero(focal[example, frame])
            if len(focal_index) == 1:
                distance = np.sqrt(
                    np.sum(np.square(xy - xy[focal_index[0]]), axis=-1)
                )
                for label, member in (("offense", off_index), ("defense", def_index)):
                    member = member[member != focal_index[0]]
                    if len(member) and int(frame) == final:
                        row[f"focal_{label}_distance_min"] = float(np.min(distance[member]))
                        row[f"focal_{label}_distance_mean"] = float(np.mean(distance[member]))
        if temporal_nearest:
            row["temporal_cross_distance_min_mean"] = float(np.mean(temporal_nearest))
            row["temporal_cross_distance_min_change"] = float(
                temporal_nearest[-1] - temporal_nearest[0]
            )
        if temporal_cross_mean:
            row["temporal_cross_distance_mean"] = float(np.mean(temporal_cross_mean))
        rows.append(row)
    return pd.DataFrame(rows)


def sorted_token_frame(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    frame_mask: np.ndarray,
    channel_names: Iterable[str],
) -> pd.DataFrame:
    """Flatten the adapter's deterministic final-frame player ordering.

    Adapters order football, focal player, offense, and defense stably before
    padding.  Including both the numeric token and an explicit slot mask lets
    classical models use the same causal objects as the neural families.
    """

    tokens = np.asarray(player_tokens, dtype=np.float32)
    players = np.asarray(player_mask, dtype=bool)
    frames = np.asarray(frame_mask, dtype=bool)
    names = tuple(str(value) for value in channel_names)
    if tokens.ndim != 4 or players.shape != tokens.shape[:3] or frames.shape != tokens.shape[:2]:
        raise ValueError("token and mask shapes are incompatible")
    rows: list[dict[str, float]] = []
    for example in range(len(tokens)):
        valid_frames = np.flatnonzero(frames[example])
        if not len(valid_frames):
            raise ValueError(f"example {example} has no valid frame")
        final = int(valid_frames[-1])
        row: dict[str, float] = {}
        for slot in range(tokens.shape[2]):
            valid = bool(players[example, final, slot])
            row[f"token_{slot:02d}_valid"] = float(valid)
            for channel, name in enumerate(names):
                row[f"token_{slot:02d}_{name}"] = (
                    float(tokens[example, final, slot, channel]) if valid else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)


def classical_feature_frame(
    raw_tabular: pd.DataFrame,
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    frame_mask: np.ndarray,
    channel_names: Iterable[str],
    *,
    include_team_identity: bool = False,
    task_id: str | None = None,
) -> pd.DataFrame:
    """Combine allowed context and permutation-invariant structural summaries.

    The primary protocol intentionally does not flatten stable player slots:
    IDs may align histories but cannot induce a tabular ordering feature.
    Linear-Structure and Boosted-Structure receive this exact same frame.
    """

    raw = primary_context_frame(
        raw_tabular, include_team_identity=include_team_identity
    )
    structural_mask = (
        np.asarray(player_mask, dtype=bool)
        if task_id is None
        else task_structural_player_mask(
            player_tokens,
            player_mask,
            channel_names,
            task_id=task_id,
        )
    )
    summaries = token_summary_frame(
        player_tokens, structural_mask, frame_mask, channel_names
    )
    relations = relational_summary_frame(
        player_tokens, structural_mask, frame_mask, channel_names
    )
    if not (len(raw) == len(summaries) == len(relations)):
        raise ValueError("classical feature components are not row-aligned")
    duplicate = (set(raw.columns) & set(summaries.columns)) | (
        set(raw.columns) & set(relations.columns)
    ) | (set(summaries.columns) & set(relations.columns))
    if duplicate:
        raise ValueError(f"classical feature columns collide: {sorted(duplicate)}")
    return pd.concat([raw, summaries, relations], axis=1)


@dataclass
class TabularEncoder:
    """A train-only imputation/scaling/vocabulary artifact."""

    transformer: ColumnTransformer | None = None
    input_columns: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        if frame.empty:
            raise ValueError("cannot fit a tabular encoder on an empty frame")
        self.input_columns = tuple(str(column) for column in frame.columns)
        numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        categorical = [column for column in frame.columns if column not in numeric]
        transformers: list[tuple[str, Any, list[Any]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "one_hot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical,
                )
            )
        self.transformer = ColumnTransformer(transformers, remainder="drop")
        self.transformer.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("tabular encoder is not fitted")
        if tuple(str(column) for column in frame.columns) != self.input_columns:
            raise ValueError("tabular feature order differs from the fitted schema")
        result = np.asarray(self.transformer.transform(frame), dtype=np.float32)
        if result.ndim != 2 or not np.all(np.isfinite(result)):
            raise RuntimeError("tabular encoding returned invalid values")
        return result


@dataclass
class NeuralContextEncoder:
    """Fixed-width context encoding with train-only numeric statistics.

    Categorical values use deterministic feature hashing rather than a
    corpus-fitted vocabulary.  This keeps neural capacity identical at every
    training size while providing an explicit bucket for every unseen value.
    """

    hash_bins: int = 8
    input_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "NeuralContextEncoder":
        if frame.empty or self.hash_bins <= 0:
            raise ValueError("cannot fit neural context encoder on empty data")
        self.input_columns = tuple(str(column) for column in frame.columns)
        self.numeric_columns = tuple(
            str(column) for column in frame if pd.api.types.is_numeric_dtype(frame[column])
        )
        self.categorical_columns = tuple(
            str(column) for column in frame if str(column) not in self.numeric_columns
        )
        if self.numeric_columns:
            values = frame.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            medians = np.nanmedian(values, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            filled = np.where(np.isfinite(values), values, medians)
            means = filled.mean(axis=0)
            scales = filled.std(axis=0)
            self.medians = medians.astype(np.float32)
            self.means = means.astype(np.float32)
            self.scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0).astype(np.float32)
        else:
            self.medians = self.means = self.scales = np.empty(0, dtype=np.float32)
        return self

    @property
    def output_dimension(self) -> int:
        return len(self.numeric_columns) + len(self.categorical_columns) * self.hash_bins

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or tuple(str(column) for column in frame.columns) != self.input_columns:
            raise RuntimeError("neural context encoder is unfitted or schema changed")
        output = np.zeros((len(frame), self.output_dimension), dtype=np.float32)
        offset = 0
        if self.numeric_columns:
            values = frame.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.medians)
            output[:, : len(self.numeric_columns)] = (values - self.means) / self.scales
            offset = len(self.numeric_columns)
        for column_index, column in enumerate(self.categorical_columns):
            base = offset + column_index * self.hash_bins
            for row_index, value in enumerate(frame[column].fillna("<MISSING>").astype(str)):
                digest = hashlib.sha256(f"{column}\0{value}".encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.hash_bins
                output[row_index, base + bucket] = 1.0
        if not np.all(np.isfinite(output)):
            raise RuntimeError("neural context encoding returned non-finite values")
        return output

    def audit(self) -> dict[str, Any]:
        if self.medians is None:
            raise RuntimeError("neural context encoder is not fitted")
        return {
            "input_columns": list(self.input_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "hash_bins_per_categorical": self.hash_bins,
            "output_dimension": self.output_dimension,
            "numeric_medians": self.medians.tolist(),
            "numeric_means": self.means.tolist(),
            "numeric_scales": self.scales.tolist(),
            "fit_scope": "selected_training_games_only",
        }


@dataclass
class MaskedTokenScaler:
    """Train-only scaling for continuous token channels.

    Indicator channels and padding remain byte-identical.  Statistics are
    computed only from valid players in the selected training games.
    """

    channel_names: tuple[str, ...]
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    continuous_indices: tuple[int, ...] = ()

    def fit(self, tokens: np.ndarray, player_mask: np.ndarray) -> "MaskedTokenScaler":
        values = np.asarray(tokens, dtype=np.float32)
        mask = np.asarray(player_mask, dtype=bool)
        if values.ndim != 4 or mask.shape != values.shape[:3]:
            raise ValueError("token and player-mask shapes are incompatible")
        continuous = tuple(
            index
            for index, name in enumerate(self.channel_names)
            if name
            in {
                "x_rel",
                "y_rel",
                "vx",
                "vy",
                "speed",
                "acceleration",
                "dir_sin",
                "dir_cos",
                "orientation_sin",
                "orientation_cos",
            }
        )
        if not continuous or not mask.any():
            raise ValueError("cannot fit token scaler without valid continuous tokens")
        observed = values[mask][:, continuous].astype(np.float64)
        means = np.nanmean(observed, axis=0)
        scales = np.nanstd(observed, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
        self.continuous_indices = continuous
        self.means = means.astype(np.float32)
        self.scales = scales.astype(np.float32)
        return self

    def fit_indexed(
        self,
        tokens: np.ndarray,
        player_mask: np.ndarray,
        indices: np.ndarray,
        *,
        batch_size: int = 128,
        frame_mask: np.ndarray | None = None,
        final_frame_only: bool = False,
        task_id: str | None = None,
        ablation_id: str | None = None,
    ) -> "MaskedTokenScaler":
        index = np.asarray(indices, dtype=int)
        continuous = tuple(
            position
            for position, name in enumerate(self.channel_names)
            if name in {
                "x_rel", "y_rel", "vx", "vy", "speed", "acceleration",
                "dir_sin", "dir_cos", "orientation_sin", "orientation_cos",
            }
        )
        sums = np.zeros(len(continuous), dtype=np.float64)
        squares = np.zeros(len(continuous), dtype=np.float64)
        count = 0
        for start in range(0, len(index), int(batch_size)):
            selected = index[start : start + int(batch_size)]
            values = np.asarray(tokens[selected], dtype=np.float32)
            mask = np.asarray(player_mask[selected], dtype=bool)
            if task_id is not None:
                mask = structural_ablation_player_mask(
                    values,
                    mask,
                    self.channel_names,
                    task_id=task_id,
                    ablation_id=ablation_id,
                )
            elif ablation_id is not None:
                raise ValueError("ablation-aware scaling requires a task ID")
            if final_frame_only:
                if frame_mask is None:
                    raise ValueError("final-frame scaling requires a frame mask")
                observed_frames = np.asarray(frame_mask[selected], dtype=bool)
                keep = np.zeros_like(observed_frames)
                for row in range(len(keep)):
                    observed = np.flatnonzero(observed_frames[row])
                    if not len(observed):
                        raise ValueError("snapshot scaling found an empty history")
                    keep[row, observed[-1]] = True
                mask &= keep[..., None]
            observed = values[mask][:, continuous].astype(np.float64)
            if len(observed):
                sums += observed.sum(axis=0)
                squares += np.square(observed).sum(axis=0)
                count += len(observed)
        if not continuous or count == 0:
            raise ValueError("cannot fit token scaler without valid continuous tokens")
        means = sums / count
        scales = np.sqrt(np.maximum(squares / count - np.square(means), 0.0))
        self.continuous_indices = continuous
        self.means = means.astype(np.float32)
        self.scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0).astype(np.float32)
        return self

    def transform(self, tokens: np.ndarray, player_mask: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("token scaler is not fitted")
        values = np.asarray(tokens, dtype=np.float32).copy()
        mask = np.asarray(player_mask, dtype=bool)
        if values.ndim != 4 or mask.shape != values.shape[:3]:
            raise ValueError("token and player-mask shapes are incompatible")
        for local, channel in enumerate(self.continuous_indices):
            channel_values = values[..., channel]
            channel_values[mask] = (
                channel_values[mask] - self.means[local]
            ) / self.scales[local]
            channel_values[~mask] = 0.0
        values[~mask] = 0.0
        if not np.all(np.isfinite(values)):
            raise RuntimeError("scaled token values are non-finite")
        return values


@dataclass
class RasterValueScaler:
    """Train-only scaling of raster kinematic channels (occupancy is fixed)."""

    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, raster: np.ndarray) -> "RasterValueScaler":
        values = np.asarray(raster, dtype=np.float32)
        if values.ndim != 5 or values.shape[-1] < 8:
            raise ValueError("raster must have shape [N,T,X,Y,>=8]")
        occupied = np.any(values[..., :4] != 0.0, axis=-1)
        observed = values[..., 4:8][occupied]
        if len(observed) == 0:
            raise ValueError("cannot fit raster scaler without occupied cells")
        self.means = np.nanmean(observed, axis=0).astype(np.float32)
        scales = np.nanstd(observed, axis=0)
        self.scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0).astype(np.float32)
        return self

    def fit_tokens(
        self,
        player_tokens: np.ndarray,
        player_mask: np.ndarray,
        channel_names: Iterable[str],
        *,
        batch_size: int = 32,
    ) -> "RasterValueScaler":
        """Fit raster-cell statistics without materializing dense zero rasters."""

        return self._fit_sparse_tokens(
            player_tokens,
            player_mask,
            np.arange(len(player_tokens), dtype=int),
            channel_names,
            batch_size=batch_size,
        )

    def fit_indexed_tokens(
        self,
        player_tokens: np.ndarray,
        player_mask: np.ndarray,
        indices: np.ndarray,
        channel_names: Iterable[str],
        *,
        batch_size: int = 16,
    ) -> "RasterValueScaler":
        return self._fit_sparse_tokens(
            player_tokens,
            player_mask,
            np.asarray(indices, dtype=int),
            channel_names,
            batch_size=batch_size,
        )

    def _fit_sparse_tokens(
        self,
        player_tokens: np.ndarray,
        player_mask: np.ndarray,
        indices: np.ndarray,
        channel_names: Iterable[str],
        *,
        batch_size: int,
        x_bins: int = 60,
        y_bins: int = 27,
    ) -> "RasterValueScaler":
        """Accumulate the exact occupied-cell values used by rasterization.

        A dense BDB2026 raster batch of 16 examples is about 97 MiB because
        each example is padded to 123 frames.  Only a few hundred cells are
        occupied, so constructing the dense array solely to fit four scalar
        moments is unnecessary.  This routine applies the same binning and
        within-cell averaging as :func:`rasterize_tokens`, then accumulates
        those occupied-cell means in float64.
        """

        if int(batch_size) <= 0:
            raise ValueError("raster-scaler batch size must be positive")
        tokens = np.asarray(player_tokens, dtype=np.float32)
        masks = np.asarray(player_mask, dtype=bool)
        selected = np.asarray(indices, dtype=int).reshape(-1)
        names = list(channel_names)
        if tokens.ndim != 4 or masks.shape != tokens.shape[:3]:
            raise ValueError("token and mask shapes are incompatible")
        if np.any((selected < 0) | (selected >= len(tokens))):
            raise IndexError("raster-scaler example index is out of bounds")
        x_index = _channel_index(names, "x_rel", "relative_x")
        y_index = _channel_index(names, "y_rel", "relative_y")
        group_indices = [
            names.index(name) if name in names else None
            for name in ("offense", "defense", "football", "focal")
        ]
        value_indices = [
            names.index(name) if name in names else None
            for name in ("vx", "vy", "speed", "acceleration")
        ]
        sums = np.zeros(4, dtype=np.float64)
        squares = np.zeros(4, dtype=np.float64)
        count = 0
        for start in range(0, len(selected), int(batch_size)):
            for example in selected[start : start + int(batch_size)]:
                frame_index, player_index = np.nonzero(masks[example])
                if not len(frame_index):
                    continue
                observed = tokens[example, frame_index, player_index]
                x = np.clip(
                    ((observed[:, x_index] + 60.0) / 120.0 * x_bins).astype(int),
                    0,
                    x_bins - 1,
                )
                y = np.clip(
                    ((observed[:, y_index] + 27.0) / 54.0 * y_bins).astype(int),
                    0,
                    y_bins - 1,
                )
                cell = (frame_index * x_bins + x) * y_bins + y
                _, inverse = np.unique(cell, return_inverse=True)
                cell_counts = np.bincount(inverse).astype(np.float64)
                occupied = np.zeros(len(cell_counts), dtype=bool)
                for source in group_indices:
                    if source is not None:
                        occupied |= np.bincount(
                            inverse, weights=observed[:, source], minlength=len(cell_counts)
                        ) != 0.0
                if not np.any(occupied):
                    continue
                cell_values = np.zeros((len(cell_counts), 4), dtype=np.float32)
                for output, source in enumerate(value_indices):
                    if source is not None:
                        cell_values[:, output] = (
                            np.bincount(
                                inverse,
                                weights=observed[:, source],
                                minlength=len(cell_counts),
                            )
                            / cell_counts
                        ).astype(np.float32)
                values = cell_values[occupied].astype(np.float64)
                sums += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                count += len(values)
        if count == 0:
            raise ValueError("cannot fit raster scaler without occupied cells")
        means = sums / count
        scales = np.sqrt(np.maximum(squares / count - np.square(means), 0.0))
        self.means = means.astype(np.float32)
        self.scales = np.where(
            np.isfinite(scales) & (scales > 1e-8), scales, 1.0
        ).astype(np.float32)
        return self

    def transform(self, raster: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("raster scaler is not fitted")
        values = np.asarray(raster, dtype=np.float32).copy()
        occupied = np.any(values[..., :4] != 0.0, axis=-1)
        kinematics = values[..., 4:8]
        kinematics[occupied] = (kinematics[occupied] - self.means) / self.scales
        kinematics[~occupied] = 0.0
        if not np.all(np.isfinite(values)):
            raise RuntimeError("scaled raster values are non-finite")
        return values


@dataclass
class LazyNeuralInputs:
    """Batch-materialized neural inputs backed by prepared memory maps."""

    family: str
    player_tokens: np.ndarray
    player_mask: np.ndarray
    frame_mask: np.ndarray
    global_context: np.ndarray
    example_indices: np.ndarray
    channel_names: tuple[str, ...]
    token_scaler: MaskedTokenScaler | None = None
    raster_scaler: RasterValueScaler | None = None
    task_id: str = "legacy_task"
    ablation_id: str | None = None
    _frame_slice: slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.example_indices)
        if self.family not in {"cnn", *TOKEN_NEURAL_FAMILIES}:
            raise ValueError("lazy neural family is unsupported")
        if len(self.global_context) != n:
            raise ValueError("lazy neural inputs are not example-aligned")
        if self.player_mask.shape != self.player_tokens.shape[:3] or self.frame_mask.shape != self.player_tokens.shape[:2]:
            raise ValueError("prepared token masks are not aligned")
        if self.family == "cnn" and self.raster_scaler is None:
            raise ValueError("CNN lazy inputs require a raster scaler")
        if self.family in TOKEN_NEURAL_FAMILIES and self.token_scaler is None:
            raise ValueError("token neural inputs require a token scaler")
        source_index = np.asarray(self.example_indices, dtype=int)
        if (
            source_index.ndim != 1
            or not len(source_index)
            or np.any((source_index < 0) | (source_index >= len(self.player_tokens)))
        ):
            raise ValueError("lazy neural source indices are invalid")
        source_frame_mask = self._selected_frame_mask(source_index)
        used_columns = np.flatnonzero(np.any(source_frame_mask, axis=0))
        if not len(used_columns):
            raise ValueError("lazy neural source has no valid frame")
        # Keras 3 samples only the first two PyDataset batches when deriving its
        # TensorSpec. Batch-local trimming can therefore freeze a coincidental
        # short history (for example T=6) before a later batch needs the task's
        # complete history (T=20). Cache one causal slice for this split-local
        # source so every batch has a stable physical width. The model time
        # dimension remains dynamic and source-global all-padding columns are
        # still omitted.
        self._frame_slice = slice(
            int(used_columns[0]), int(used_columns[-1]) + 1
        )

    def _selected_frame_mask(self, source_index: np.ndarray) -> np.ndarray:
        selected = np.asarray(self.frame_mask[source_index], dtype=bool)
        if self.ablation_id not in {"release_snapshot_only", "final_snapshot_only"}:
            return selected
        snapshot = np.zeros_like(selected)
        observed = np.any(selected, axis=1)
        if np.any(observed):
            last = selected.shape[1] - 1 - np.argmax(selected[:, ::-1], axis=1)
            rows = np.flatnonzero(observed)
            snapshot[rows, last[rows]] = True
        return snapshot

    def __len__(self) -> int:
        return len(self.example_indices)

    @property
    def input_shapes(self) -> dict[str, tuple[int | None, ...]]:
        # Time remains model-dynamic because selector, validation, and refit
        # sources can have different source-global causal widths. Every batch
        # within one source uses its cached width, while player/raster and
        # channel dimensions remain frozen, so model capacity is unchanged.
        common = {
            "frame_mask": (None,),
            "global_context": tuple(self.global_context.shape[1:]),
        }
        if self.family == "cnn":
            return {"raster": (None, 60, 27, 8), **common}
        result = {
            "player_tokens": (None, *self.player_tokens.shape[2:]),
            "player_mask": (None, self.player_mask.shape[2]),
            "time_to_event": (None, 1),
            **common,
        }
        if self.family in RELATIONAL_FAMILIES:
            result["edge_type"] = (
                None,
                self.player_mask.shape[2],
                self.player_mask.shape[2],
            )
        return result

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        index = np.asarray(indices, dtype=int)
        if np.any((index < 0) | (index >= len(self))):
            raise IndexError("lazy neural batch index is out of bounds")
        source_index = np.asarray(self.example_indices, dtype=int)[index]
        selected_frame_mask = self._selected_frame_mask(source_index)
        frame_slice = self._frame_slice
        if self.family == "cnn":
            raster = rasterize_tokens(
                self.player_tokens[source_index, frame_slice],
                self.player_mask[source_index, frame_slice],
                self.channel_names,
            )
            return {
                "raster": self.raster_scaler.transform(raster),
                "frame_mask": selected_frame_mask[:, frame_slice],
                "global_context": np.asarray(self.global_context[index], dtype=np.float32),
            }
        raw_tokens = np.asarray(
            self.player_tokens[source_index, frame_slice], dtype=np.float32
        )
        raw_player_mask = np.asarray(
            self.player_mask[source_index, frame_slice], dtype=bool
        ) & selected_frame_mask[:, frame_slice, None]
        raw_player_mask = structural_ablation_player_mask(
            raw_tokens,
            raw_player_mask,
            self.channel_names,
            task_id=self.task_id,
            ablation_id=self.ablation_id,
        )
        result = {
            "player_tokens": self.token_scaler.transform(raw_tokens, raw_player_mask),
            "player_mask": raw_player_mask,
            "frame_mask": selected_frame_mask[:, frame_slice],
            "time_to_event": relative_time_to_event(
                selected_frame_mask[:, frame_slice]
            ),
            "global_context": np.asarray(self.global_context[index], dtype=np.float32),
        }
        if self.family in RELATIONAL_FAMILIES:
            result["edge_type"] = relational_edge_types(
                raw_tokens,
                raw_player_mask,
                self.channel_names,
                task_id=self.task_id,
                ablation_id=self.ablation_id,
            )
        return result


def representative_neural_input_signature(
    source: LazyNeuralInputs,
    *,
    max_examples: int = 4,
) -> dict[str, Any]:
    """Hash one deterministic bounded batch after every input transformation."""

    if int(max_examples) < 1 or len(source) < 1:
        raise ValueError("representative neural input signature requires examples")
    positions = np.arange(min(int(max_examples), len(source)), dtype=int)
    batch = source.batch(positions)
    tensors: list[dict[str, Any]] = []
    for name in sorted(batch):
        array = np.ascontiguousarray(np.asarray(batch[name]))
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        tensors.append(
            {
                "name": str(name),
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "sha256": digest,
            }
        )
    payload = {
        "schema_version": "bdb-representative-neural-input-v1",
        "batch_positions": positions.tolist(),
        "source_example_indices": np.asarray(source.example_indices, dtype=int)[
            positions
        ].tolist(),
        "examples": int(len(positions)),
        "tensors": tensors,
        "scope": "after_split_local_preprocessing_and_structural_ablation",
    }
    signature = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "signature_sha256": signature}


def representative_shared_neural_input_signature(
    source: LazyNeuralInputs,
    *,
    max_examples: int = 4,
) -> dict[str, Any]:
    """Hash the tensors shared exactly by all protocol-v2 neural families.

    RelNet and AttnRelNet additionally consume the task's typed edge tensor;
    Global Set Transformer intentionally does not. This receipt therefore
    proves equality of tokenization, causal history, masks, time-to-event, and
    context without falsely asserting equality of graph aggregation inputs.
    """

    if source.family not in {*RELATIONAL_FAMILIES, *GLOBAL_SET_FAMILIES}:
        raise ValueError(
            "shared protocol-v2 signature requires a primary neural family"
        )
    if int(max_examples) < 1 or len(source) < 1:
        raise ValueError("representative shared neural signature requires examples")
    positions = np.arange(min(int(max_examples), len(source)), dtype=int)
    batch = source.batch(positions)
    names = (
        "frame_mask",
        "global_context",
        "player_mask",
        "player_tokens",
        "time_to_event",
    )
    if not set(names).issubset(batch):
        raise ValueError("primary neural input is missing a shared tensor")
    tensors: list[dict[str, Any]] = []
    for name in names:
        array = np.ascontiguousarray(np.asarray(batch[name]))
        tensors.append(
            {
                "name": name,
                "shape": [int(value) for value in array.shape],
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
            }
        )
    payload = {
        "schema_version": "bdb-representative-neural-input-v1",
        "batch_positions": positions.tolist(),
        "source_example_indices": np.asarray(source.example_indices, dtype=int)[
            positions
        ].tolist(),
        "examples": int(len(positions)),
        "tensors": tensors,
        "scope": "shared_tokens_history_masks_time_context_after_split_local_preprocessing",
    }
    signature = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "signature_sha256": signature}


def rasterize_tokens(
    player_tokens: np.ndarray,
    player_mask: np.ndarray,
    channel_names: Iterable[str],
    *,
    x_bins: int = 60,
    y_bins: int = 27,
) -> np.ndarray:
    """Rasterize causal player tokens on a two-yard, focal-centered field grid.

    Output channels are offense/defense/football/focal occupancy followed by
    mean vx, vy, speed, and acceleration.  Rasterization is deterministic and
    contains no learned corpus statistics.
    """

    tokens = np.asarray(player_tokens, dtype=np.float32)
    valid = np.asarray(player_mask, dtype=bool)
    names = list(channel_names)
    if tokens.ndim != 4 or valid.shape != tokens.shape[:3]:
        raise ValueError("token and mask shapes are incompatible")
    x_index = _channel_index(names, "x_rel", "relative_x")
    y_index = _channel_index(names, "y_rel", "relative_y")
    group_indices = [
        names.index(name) if name in names else None
        for name in ("offense", "defense", "football", "focal")
    ]
    value_indices = [
        names.index(name) if name in names else None
        for name in ("vx", "vy", "speed", "acceleration")
    ]
    output = np.zeros((tokens.shape[0], tokens.shape[1], x_bins, y_bins, 8), dtype=np.float32)
    counts = np.zeros(output.shape[:-1] + (1,), dtype=np.float32)
    for example in range(tokens.shape[0]):
        for frame in range(tokens.shape[1]):
            indices = np.flatnonzero(valid[example, frame])
            if not len(indices):
                continue
            x = np.clip(((tokens[example, frame, indices, x_index] + 60.0) / 120.0 * x_bins).astype(int), 0, x_bins - 1)
            y = np.clip(((tokens[example, frame, indices, y_index] + 27.0) / 54.0 * y_bins).astype(int), 0, y_bins - 1)
            for local, player in enumerate(indices):
                cell = (example, frame, x[local], y[local])
                counts[cell + (0,)] += 1.0
                for channel, source in enumerate(group_indices):
                    if source is not None:
                        output[cell + (channel,)] += tokens[example, frame, player, source]
                for offset, source in enumerate(value_indices, start=4):
                    if source is not None:
                        output[cell + (offset,)] += tokens[example, frame, player, source]
    nonzero = counts[..., 0] > 0
    for channel in range(4, 8):
        output[..., channel][nonzero] /= counts[..., 0][nonzero]
    return output
