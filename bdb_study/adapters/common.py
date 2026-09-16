"""Common, causal data structures and tracking transforms for BDB adapters.

Adapters deliberately stop at deterministic, physically meaningful features.
Imputation, scaling, categorical vocabularies, and every other learned
transformation belong to the model's training partition (see
``bdb_study.representations``).  This boundary is what prevents a prepared
whole-release artifact from leaking calibration or test information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0
MAX_TRACKED_OBJECTS = 23

POSITION_GROUPS = (
    "qb",
    "rb",
    "wr",
    "te",
    "ol",
    "dl",
    "lb",
    "db",
    "special",
    "other",
)

CANONICAL_PLAYER_CHANNELS = (
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
    "offense",
    "defense",
    "football",
    "focal",
    *(f"position_{group}" for group in POSITION_GROUPS),
)

TRACKING_TOKEN_COLUMNS = (
    "gameId",
    "playId",
    "frameId",
    "nflId",
    "displayName",
    "position",
    "team",
    "club",
    "playDirection",
    "x",
    "y",
    "s",
    "a",
    "o",
    "dir",
    "event",
    "frameType",
)


@dataclass(frozen=True)
class CohortAudit:
    """Machine-readable accounting from the play table to retained examples."""

    task_id: str
    total_play_rows: int
    outcome_eligible: int
    cutoff_eligible: int
    retained_examples: int
    retained_games: int
    exclusions: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = (
            self.total_play_rows,
            self.outcome_eligible,
            self.cutoff_eligible,
            self.retained_examples,
            self.retained_games,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("cohort audit counts cannot be negative")
        if self.retained_examples > self.cutoff_eligible:
            raise ValueError("retained examples exceed cutoff-eligible examples")
        if self.cutoff_eligible > self.outcome_eligible:
            raise ValueError("cutoff-eligible examples exceed outcome eligibility")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_play_rows": int(self.total_play_rows),
            "outcome_eligible": int(self.outcome_eligible),
            "cutoff_eligible": int(self.cutoff_eligible),
            "retained_examples": int(self.retained_examples),
            "retained_games": int(self.retained_games),
            "exclusions": {str(key): int(value) for key, value in self.exclusions.items()},
            "details": dict(self.details),
        }

    def require(self, *, examples: int, games: int) -> "CohortAudit":
        if self.retained_examples != int(examples) or self.retained_games != int(games):
            raise ValueError(
                f"{self.task_id} cohort differs from its frozen receipt: "
                f"expected {examples} examples/{games} games, got "
                f"{self.retained_examples}/{self.retained_games}"
            )
        return self


@dataclass(frozen=True)
class PreparedTask:
    """Task-neutral adapter output consumed by split-local model code.

    Player tensors always use ``[example, time, tracked_object, channel]``.
    A one-frame event task therefore has a time dimension of one.  Distribution
    labels in ``y`` are zero-based support indices; the unencoded outcome is
    retained in ``examples['target']``.  Trajectory adapters store absolute
    observed coordinates in ``target_values``, their reconstructable null path
    in ``target_baseline``, and validity in ``target_mask``; runners derive
    residual training targets from those arrays.
    """

    task_id: str
    outcome_type: str
    primary_metric: str
    examples: pd.DataFrame
    tabular: pd.DataFrame
    player_tokens: np.ndarray
    player_mask: np.ndarray
    frame_mask: np.ndarray
    channel_names: tuple[str, ...]
    y: np.ndarray | None = None
    support: tuple[float, ...] | None = None
    audit: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    target_values: np.ndarray | None = None
    target_mask: np.ndarray | None = None
    target_baseline: np.ndarray | None = None

    def __post_init__(self) -> None:
        required = {"example_id", "game_id", "stratum", "target"}
        missing = required.difference(self.examples.columns)
        if missing:
            raise ValueError(f"examples table is missing {sorted(missing)}")
        if self.examples["example_id"].duplicated().any():
            raise ValueError("example IDs must be unique")
        n_examples = len(self.examples)
        if len(self.tabular) != n_examples:
            raise ValueError("tabular rows do not align with examples")
        tokens = np.asarray(self.player_tokens)
        player_mask = np.asarray(self.player_mask)
        frame_mask = np.asarray(self.frame_mask)
        if tokens.ndim != 4 or tokens.shape[0] != n_examples:
            raise ValueError("player_tokens must have shape [N,T,P,C]")
        if tokens.shape[-1] != len(self.channel_names):
            raise ValueError("channel_names do not align with player_tokens")
        if player_mask.shape != tokens.shape[:3] or frame_mask.shape != tokens.shape[:2]:
            raise ValueError("player/frame masks do not align with player_tokens")
        if not np.issubdtype(player_mask.dtype, np.bool_) or not np.issubdtype(frame_mask.dtype, np.bool_):
            raise ValueError("player_mask and frame_mask must be boolean")
        # BDB2026's frozen dense token tensor is 8.6 GiB and may be a memmap.
        # Validate bounded chunks so construction never creates an equally
        # large temporary boolean array or faults the whole mapping into RAM.
        for start in range(0, n_examples, 256):
            stop = min(start + 256, n_examples)
            if np.any(
                player_mask[start:stop]
                & ~frame_mask[start:stop, :, None]
            ):
                raise ValueError("a masked frame cannot contain valid tracked objects")
            if not np.all(np.isfinite(tokens[start:stop])):
                raise ValueError("canonical player tokens must be finite")
        if self.y is not None and np.asarray(self.y).reshape(-1).shape[0] != n_examples:
            raise ValueError("y does not align with examples")
        if self.outcome_type == "binary" and self.y is not None:
            labels = np.asarray(self.y).reshape(-1)
            if not np.all(np.isin(labels, [0, 1])):
                raise ValueError("binary y must contain only zero and one")
        if self.outcome_type == "distribution" and self.y is not None:
            labels = np.asarray(self.y).reshape(-1)
            if not self.support or np.any((labels < 0) | (labels >= len(self.support))):
                raise ValueError("distribution y is outside the declared support")
        if (self.target_values is None) != (self.target_mask is None):
            raise ValueError("target_values and target_mask must be supplied together")
        if self.target_values is None and self.target_baseline is not None:
            raise ValueError("target_baseline requires trajectory targets")
        if self.target_values is not None:
            values = np.asarray(self.target_values)
            mask = np.asarray(self.target_mask)
            if values.ndim != 3 or values.shape[0] != n_examples or values.shape[-1] != 2:
                raise ValueError("trajectory targets must have shape [N,H,2]")
            if mask.shape != values.shape[:2] or not np.issubdtype(mask.dtype, np.bool_):
                raise ValueError("trajectory target mask must have shape [N,H] and boolean dtype")
            if np.any(mask.sum(axis=1) == 0) or not np.all(np.isfinite(values[mask])):
                raise ValueError("each trajectory needs at least one finite target frame")
            if self.target_baseline is None or np.asarray(self.target_baseline).shape != values.shape:
                raise ValueError("trajectory baseline must share target shape [N,H,2]")
            if not np.all(np.isfinite(np.asarray(self.target_baseline)[mask])):
                raise ValueError("trajectory baseline must be finite on target frames")

    @property
    def example_ids(self) -> np.ndarray:
        return self.examples["example_id"].astype(str).to_numpy()

    @property
    def game_ids(self) -> np.ndarray:
        return self.examples["game_id"].to_numpy(dtype=np.int64)

    @property
    def strata(self) -> np.ndarray:
        return self.examples["stratum"].astype(str).to_numpy()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_adapter_task_spec(task_id: str):
    """Load and validate a tracked adapter task receipt lazily.

    Keeping the contracts import lazy lets data-inventory tools inspect source
    files without importing execution/model dependencies.
    """

    path = repository_root() / "configs" / "bdb_suite" / "tasks" / f"{task_id}.json"
    from bdb_study.contracts import load_task_spec

    # Adapters can inspect receipt-free draft configs during data development;
    # the core freeze/design paths deliberately keep strict receipt validation.
    return load_task_spec(path, require_source_receipts=False)


def _natural_number(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else -1, path.name)


def weekly_tracking_paths(raw_dir: Path, *patterns: str) -> tuple[Path, ...]:
    """Return weekly/season tracking files without accidentally loading an aggregate twice."""

    root = Path(raw_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.name != "tracking_all.csv")
    paths = tuple(sorted(set(matches), key=_natural_number))
    if not paths:
        raise FileNotFoundError(f"no tracking files matching {patterns!r} in {root}")
    return paths


def read_tracking_chunks(
    path: Path,
    requested_columns: Iterable[str] = TRACKING_TOKEN_COLUMNS,
    *,
    required_columns: Iterable[str] = ("gameId", "playId", "frameId"),
    chunksize: int = 500_000,
) -> Iterator[pd.DataFrame]:
    available = tuple(pd.read_csv(path, nrows=0).columns)
    required = set(required_columns)
    missing = required.difference(available)
    if missing:
        raise ValueError(f"{path} is missing tracking columns {sorted(missing)}")
    usecols = [column for column in requested_columns if column in available]
    yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["gameId", "playId"]].drop_duplicates().copy()
    result["gameId"] = result["gameId"].astype(np.int64)
    result["playId"] = result["playId"].astype(np.int64)
    return result


def discover_cutoffs(
    paths: Sequence[Path],
    eligible: pd.DataFrame,
    *,
    event_names: Iterable[str] = (),
    frame_types: Iterable[str] = (),
    require_unique_frame: bool = False,
    require_valid_football_coordinate: bool = False,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover causal cutoff frames and return both retained frames and an audit table."""

    keys = _key_frame(eligible)
    event_set = {str(value).lower() for value in event_names}
    frame_type_set = {str(value).upper() for value in frame_types}
    if not event_set and not frame_type_set:
        raise ValueError("at least one event name or frame type is required")
    markers: list[pd.DataFrame] = []
    valid_football_frames: list[pd.DataFrame] = []
    requested = ["gameId", "playId", "frameId", "event", "frameType"]
    if require_valid_football_coordinate:
        requested.extend(["displayName", "team", "club", "x", "y"])
    for path in paths:
        for chunk in read_tracking_chunks(path, requested, chunksize=chunksize):
            selected = chunk.merge(keys, on=["gameId", "playId"], how="inner")
            if selected.empty:
                continue
            if require_valid_football_coordinate:
                club = (
                    selected["club"].astype(str).str.lower()
                    if "club" in selected
                    else pd.Series("", index=selected.index)
                )
                team = (
                    selected["team"].astype(str).str.lower()
                    if "team" in selected
                    else pd.Series("", index=selected.index)
                )
                display_name = (
                    selected["displayName"].astype(str).str.lower()
                    if "displayName" in selected
                    else pd.Series("", index=selected.index)
                )
                valid_football = (
                    (club.eq("football") | team.eq("football") | display_name.eq("football"))
                    & selected["x"].notna()
                    & selected["y"].notna()
                )
                if valid_football.any():
                    valid_football_frames.append(
                        selected.loc[valid_football, ["gameId", "playId", "frameId"]]
                        .drop_duplicates()
                    )
            match = np.zeros(len(selected), dtype=bool)
            if event_set and "event" in selected:
                match |= selected["event"].astype(str).str.lower().isin(event_set).to_numpy()
            if frame_type_set and "frameType" in selected:
                match |= selected["frameType"].astype(str).str.upper().isin(frame_type_set).to_numpy()
            selected = selected.loc[match].copy()
            if selected.empty:
                continue
            if "event" not in selected:
                selected["event"] = ""
            selected = selected[["gameId", "playId", "frameId", "event"]].drop_duplicates()
            markers.append(selected)
    if markers:
        marker = pd.concat(markers, ignore_index=True).drop_duplicates()
    else:
        marker = pd.DataFrame(columns=["gameId", "playId", "frameId", "event"])
    distinct = marker[["gameId", "playId", "frameId"]].drop_duplicates()
    frame_counts = (
        distinct.groupby(["gameId", "playId"], as_index=False)
        .size()
        .rename(columns={"size": "cutoff_frame_count"})
    )
    audit = keys.merge(frame_counts, on=["gameId", "playId"], how="left")
    audit["cutoff_frame_count"] = audit["cutoff_frame_count"].fillna(0).astype(int)
    valid = audit["cutoff_frame_count"].eq(1) if require_unique_frame else audit["cutoff_frame_count"].gt(0)
    retained = audit.loc[valid, ["gameId", "playId"]]
    cutoff = (
        distinct.merge(retained, on=["gameId", "playId"], how="inner")
        .groupby(["gameId", "playId"], as_index=False)["frameId"]
        .min()
        .rename(columns={"frameId": "cutoff_frame_id"})
    )
    if require_valid_football_coordinate:
        football_frames = (
            pd.concat(valid_football_frames, ignore_index=True).drop_duplicates()
            if valid_football_frames
            else pd.DataFrame(columns=["gameId", "playId", "frameId"])
        )
        cutoff_validity = cutoff.merge(
            football_frames.assign(cutoff_has_valid_football=True),
            left_on=["gameId", "playId", "cutoff_frame_id"],
            right_on=["gameId", "playId", "frameId"],
            how="left",
        )
        cutoff_validity["cutoff_has_valid_football"] = (
            cutoff_validity["cutoff_has_valid_football"].fillna(False).astype(bool)
        )
        audit = audit.merge(
            cutoff_validity[
                ["gameId", "playId", "cutoff_has_valid_football"]
            ],
            on=["gameId", "playId"],
            how="left",
        )
        audit["cutoff_has_valid_football"] = (
            audit["cutoff_has_valid_football"].fillna(False).astype(bool)
        )
        cutoff = cutoff_validity.loc[
            cutoff_validity["cutoff_has_valid_football"],
            ["gameId", "playId", "cutoff_frame_id"],
        ]
    if not cutoff.empty:
        chosen_events = marker.merge(cutoff, on=["gameId", "playId"], how="inner")
        chosen_events = chosen_events.loc[chosen_events["frameId"] == chosen_events["cutoff_frame_id"]]
        event_value = (
            chosen_events.groupby(["gameId", "playId"])["event"]
            .agg(lambda values: "+".join(sorted({str(value) for value in values if pd.notna(value)})))
            .rename("cutoff_event")
            .reset_index()
        )
        cutoff = cutoff.merge(event_value, on=["gameId", "playId"], how="left")
    else:
        cutoff["cutoff_event"] = pd.Series(dtype=object)
    return cutoff.sort_values(["gameId", "playId"]).reset_index(drop=True), audit


def load_rows_at_cutoffs(
    paths: Sequence[Path],
    cutoffs: pd.DataFrame,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load all tracked objects at previously discovered cutoff frames."""

    cutoff = cutoffs[["gameId", "playId", "cutoff_frame_id"]].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for path in paths:
        for chunk in read_tracking_chunks(path, chunksize=chunksize):
            selected = chunk.merge(cutoff, on=["gameId", "playId"], how="inner")
            selected = selected.loc[selected["frameId"] == selected["cutoff_frame_id"]]
            if not selected.empty:
                rows.append(selected.drop(columns="cutoff_frame_id"))
    if not rows:
        return pd.DataFrame(columns=TRACKING_TOKEN_COLUMNS)
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["gameId", "playId", "frameId", "nflId"], keep="first"
    )


def discover_frame_windows(
    paths: Sequence[Path],
    cutoffs: pd.DataFrame,
    *,
    window: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Find the last ``window`` observed frame IDs through each cutoff."""

    cutoff = cutoffs[["gameId", "playId", "cutoff_frame_id"]].drop_duplicates()
    observed: list[pd.DataFrame] = []
    requested = ("gameId", "playId", "frameId")
    for path in paths:
        for chunk in read_tracking_chunks(path, requested, chunksize=chunksize):
            selected = chunk.merge(cutoff, on=["gameId", "playId"], how="inner")
            selected = selected.loc[selected["frameId"] <= selected["cutoff_frame_id"]]
            if not selected.empty:
                observed.append(selected[["gameId", "playId", "frameId"]].drop_duplicates())
    if not observed:
        return pd.DataFrame(columns=["gameId", "playId", "frameId", "time_index"])
    frames = pd.concat(observed, ignore_index=True).drop_duplicates()
    frames = frames.sort_values(["gameId", "playId", "frameId"])
    frames = frames.groupby(["gameId", "playId"], group_keys=False).tail(int(window)).copy()
    frames["time_index"] = frames.groupby(["gameId", "playId"])["frameId"].transform(
        lambda values: np.arange(window - len(values), window, dtype=int)
    )
    return frames.reset_index(drop=True)


def load_rows_at_selected_frames(
    paths: Sequence[Path],
    selected_frames: pd.DataFrame,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    frame_keys = selected_frames[["gameId", "playId", "frameId", "time_index"]].drop_duplicates()
    rows: list[pd.DataFrame] = []
    for path in paths:
        for chunk in read_tracking_chunks(path, chunksize=chunksize):
            selected = chunk.merge(frame_keys, on=["gameId", "playId", "frameId"], how="inner")
            if not selected.empty:
                rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=(*TRACKING_TOKEN_COLUMNS, "time_index"))
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["gameId", "playId", "frameId", "nflId"], keep="first"
    )


def position_group(position: Any) -> str:
    value = "" if pd.isna(position) else str(position).upper().strip()
    mapping = {
        "QB": "qb",
        "RB": "rb",
        "FB": "rb",
        "HB": "rb",
        "WR": "wr",
        "TE": "te",
        "C": "ol",
        "G": "ol",
        "OG": "ol",
        "OT": "ol",
        "T": "ol",
        "OL": "ol",
        "DE": "dl",
        "DT": "dl",
        "NT": "dl",
        "DL": "dl",
        "LB": "lb",
        "ILB": "lb",
        "MLB": "lb",
        "OLB": "lb",
        "CB": "db",
        "DB": "db",
        "FS": "db",
        "SS": "db",
        "S": "db",
        "K": "special",
        "P": "special",
        "LS": "special",
    }
    return mapping.get(value, "other")


def parse_first_nfl_id(value: Any) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def clock_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(":")
    try:
        if len(parts) >= 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return np.nan


def _angle_components(value: Any, mirror: bool) -> tuple[float, float]:
    if pd.isna(value):
        return 0.0, 0.0
    radians = math.radians((90.0 - float(value)) % 360.0)
    x_component, y_component = math.cos(radians), math.sin(radians)
    if mirror:
        x_component, y_component = -x_component, -y_component
    return y_component, x_component  # sin, cos in standardized coordinates


def _club(value: Any, game: Mapping[str, Any]) -> str:
    club = "" if pd.isna(value) else str(value)
    if club.lower() == "home":
        return str(game.get("homeTeamAbbr", ""))
    if club.lower() == "away":
        return str(game.get("visitorTeamAbbr", ""))
    return club


def _tracking_object_key(row: pd.Series) -> tuple[str, str]:
    """Return an alignment-only key that is never emitted as a model feature."""

    club = str(row.get("_club", row.get("club", row.get("team", "")))).lower()
    name = str(row.get("displayName", "")).lower()
    if club == "football" or name == "football":
        return ("football", "football")
    nfl_id = pd.to_numeric(pd.Series([row.get("nflId")]), errors="coerce").iloc[0]
    if pd.notna(nfl_id):
        return ("player", str(int(float(nfl_id))))
    # This fallback is only an alignment key for malformed/legacy rows.  It is
    # deliberately not encoded in the returned tensor.
    return ("anonymous", str(row.get("displayName", "<missing>")))


def stable_player_slot_map(
    history: pd.DataFrame,
    *,
    play: Mapping[str, Any],
    game: Mapping[str, Any],
    player_positions: Mapping[int, str] | None = None,
    focal_nfl_id: int | None = None,
    infer_offensive_qb_as_focal: bool = False,
    max_objects: int = MAX_TRACKED_OBJECTS,
) -> dict[tuple[str, str], int]:
    """Allocate one deterministic tracked-object slot map for a whole play.

    NFL IDs and names are used only to keep an object in the same slot over
    time.  Neither value is returned by this function's consumers as a model
    channel.  The allocation is deliberately independent of the focal player:
    football is followed by offense and defense, with numeric ID used solely
    as a deterministic tie-breaker.  ``focal_nfl_id`` and QB inference remain
    accepted for API compatibility, but affect only frame encoding downstream.
    """

    if history.empty:
        raise ValueError("cannot allocate stable slots from empty tracking history")
    rows = history.copy()
    for column in TRACKING_TOKEN_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    team_column = "club" if rows["club"].notna().any() else "team"
    rows["_club"] = rows[team_column].map(lambda value: _club(value, game))
    rows["_football"] = (
        rows["_club"].astype(str).str.lower().eq("football")
        | rows["displayName"].astype(str).str.lower().eq("football")
    )
    possession = str(play.get("possessionTeam", ""))
    positions = player_positions or {}

    def row_position(row: pd.Series) -> str:
        if pd.notna(row.get("position")):
            return str(row["position"])
        if pd.notna(row.get("nflId")):
            return str(positions.get(int(float(row["nflId"])), ""))
        return ""

    rows["_position"] = rows.apply(row_position, axis=1)
    rows["_offense"] = (~rows["_football"]) & rows["_club"].eq(possession)
    rows["_key"] = rows.apply(_tracking_object_key, axis=1)
    identities: dict[tuple[str, str], tuple[int, int, float, str]] = {}
    for _, row in rows.iterrows():
        key = row["_key"]
        numeric_id = (
            float(row["nflId"])
            if pd.notna(row.get("nflId"))
            else float("inf")
        )
        rank = (
            0 if bool(row["_football"]) else 1,
            0 if bool(row["_offense"]) else 1,
            numeric_id,
            str(key),
        )
        identities[key] = min(identities.get(key, rank), rank)
    ordered = sorted(identities, key=lambda key: identities[key])
    if len(ordered) > int(max_objects):
        raise ValueError(
            f"play history contains {len(ordered)} tracked objects; maximum is {max_objects}"
        )
    return {key: index for index, key in enumerate(ordered)}


def canonicalize_tracking_frame(
    frame: pd.DataFrame,
    *,
    play: Mapping[str, Any],
    game: Mapping[str, Any],
    player_positions: Mapping[int, str] | None = None,
    focal_nfl_id: int | None = None,
    infer_offensive_qb_as_focal: bool = False,
    stable_slots: Mapping[tuple[str, str], int] | None = None,
    max_objects: int = MAX_TRACKED_OBJECTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one frame without using any corpus-fitted statistics."""

    if frame.empty:
        raise ValueError("cannot canonicalize an empty tracking frame")
    rows = frame.copy()
    for column in TRACKING_TOKEN_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    team_column = "club" if rows["club"].notna().any() else "team"
    rows["_club"] = rows[team_column].map(lambda value: _club(value, game))
    rows["_football"] = (
        rows["_club"].astype(str).str.lower().eq("football")
        | rows["displayName"].astype(str).str.lower().eq("football")
    )
    ball = rows.loc[rows["_football"] & rows["x"].notna() & rows["y"].notna()]
    if ball.empty:
        raise ValueError("tracking cutoff frame has no football coordinate")
    ball_row = ball.sort_values(["frameId"], kind="mergesort").iloc[0]
    direction = str(rows["playDirection"].dropna().iloc[0]).lower() if rows["playDirection"].notna().any() else "right"
    mirror = direction == "left"

    def standardized_xy(row: pd.Series) -> tuple[float, float]:
        x, y = float(row["x"]), float(row["y"])
        return (FIELD_LENGTH - x, FIELD_WIDTH - y) if mirror else (x, y)

    ball_x, ball_y = standardized_xy(ball_row)
    possession = str(play.get("possessionTeam", ""))
    positions = player_positions or {}
    valid_rows = rows.loc[rows["x"].notna() & rows["y"].notna()].copy()

    def row_position(row: pd.Series) -> str:
        if pd.notna(row.get("position")):
            return str(row["position"])
        if pd.notna(row.get("nflId")):
            return str(positions.get(int(float(row["nflId"])), ""))
        return ""

    valid_rows["_position"] = valid_rows.apply(row_position, axis=1)
    valid_rows["_offense"] = (~valid_rows["_football"]) & valid_rows["_club"].eq(possession)
    valid_rows["_defense"] = (~valid_rows["_football"]) & (~valid_rows["_offense"])
    if infer_offensive_qb_as_focal and focal_nfl_id is None:
        qb = valid_rows.loc[valid_rows["_offense"] & valid_rows["_position"].eq("QB") & valid_rows["nflId"].notna()]
        if not qb.empty:
            focal_nfl_id = int(float(qb.sort_values("nflId").iloc[0]["nflId"]))
    valid_rows["_focal"] = False
    if focal_nfl_id is not None:
        valid_rows["_focal"] = valid_rows["nflId"].fillna(-1).astype(float).eq(float(focal_nfl_id))
    valid_rows["_key"] = valid_rows.apply(_tracking_object_key, axis=1)
    if valid_rows["_key"].duplicated().any():
        raise ValueError("tracking frame contains a duplicate tracked-object identity")
    if stable_slots is None:
        valid_rows["_sort_nfl"] = valid_rows["nflId"].fillna(-1).astype(float)
        valid_rows = valid_rows.sort_values(
            ["_football", "_focal", "_offense", "_sort_nfl"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        row_slots = {key: index for index, key in enumerate(valid_rows["_key"])}
    else:
        row_slots = {tuple(key): int(value) for key, value in stable_slots.items()}
        missing = [key for key in valid_rows["_key"] if key not in row_slots]
        if missing:
            raise ValueError(f"stable slot map omits tracked objects: {missing[:3]}")
    if len(valid_rows) > max_objects:
        raise ValueError(f"frame contains {len(valid_rows)} objects; maximum is {max_objects}")

    tokens = np.zeros((max_objects, len(CANONICAL_PLAYER_CHANNELS)), dtype=np.float32)
    mask = np.zeros(max_objects, dtype=bool)
    channel = {name: index for index, name in enumerate(CANONICAL_PLAYER_CHANNELS)}
    for _, row in valid_rows.iterrows():
        index = row_slots[row["_key"]]
        if not 0 <= index < max_objects or mask[index]:
            raise ValueError("stable slot map is out of range or not one-to-one")
        x_std, y_std = standardized_xy(row)
        speed = 0.0 if pd.isna(row["s"]) else float(row["s"])
        acceleration = 0.0 if pd.isna(row["a"]) else float(row["a"])
        dir_sin, dir_cos = _angle_components(row["dir"], mirror)
        orientation_sin, orientation_cos = _angle_components(row["o"], mirror)
        tokens[index, channel["x_rel"]] = x_std - ball_x
        tokens[index, channel["y_rel"]] = y_std - ball_y
        tokens[index, channel["vx"]] = speed * dir_cos
        tokens[index, channel["vy"]] = speed * dir_sin
        tokens[index, channel["speed"]] = speed
        tokens[index, channel["acceleration"]] = acceleration
        tokens[index, channel["dir_sin"]] = dir_sin
        tokens[index, channel["dir_cos"]] = dir_cos
        tokens[index, channel["orientation_sin"]] = orientation_sin
        tokens[index, channel["orientation_cos"]] = orientation_cos
        tokens[index, channel["offense"]] = float(row["_offense"])
        tokens[index, channel["defense"]] = float(row["_defense"])
        tokens[index, channel["football"]] = float(row["_football"])
        tokens[index, channel["focal"]] = float(row["_focal"])
        if not row["_football"]:
            tokens[index, channel[f"position_{position_group(row['_position'])}"]] = 1.0
        mask[index] = True
    return tokens, mask


def player_position_map(players: pd.DataFrame) -> dict[int, str]:
    if "nflId" not in players:
        return {}
    position_column = "officialPosition" if "officialPosition" in players else "position"
    if position_column not in players:
        return {}
    rows = players.loc[players["nflId"].notna(), ["nflId", position_column]].drop_duplicates("nflId")
    return {int(float(row.nflId)): str(getattr(row, position_column)) for row in rows.itertuples()}


def game_records(games: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row["gameId"]): row.to_dict()
        for _, row in games.drop_duplicates("gameId").iterrows()
    }


def play_records(plays: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["gameId"]), int(row["playId"])): row.to_dict()
        for _, row in plays.drop_duplicates(["gameId", "playId"]).iterrows()
    }


def assemble_player_tensors(
    examples: pd.DataFrame,
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    *,
    time_steps: int,
    focal_by_example: Mapping[str, int | None] | None = None,
    infer_offensive_qb_as_focal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble canonical fixed-shape tensors in exact example-table order."""

    n = len(examples)
    tokens = np.zeros(
        (n, int(time_steps), MAX_TRACKED_OBJECTS, len(CANONICAL_PLAYER_CHANNELS)),
        dtype=np.float32,
    )
    player_mask = np.zeros((n, int(time_steps), MAX_TRACKED_OBJECTS), dtype=bool)
    frame_mask = np.zeros((n, int(time_steps)), dtype=bool)
    positions = player_position_map(players)
    game_map = game_records(games)
    play_map = play_records(plays)
    focal = focal_by_example or {}
    time_column = "time_index" if "time_index" in tracking else None
    group_columns = ["gameId", "playId"] + (["time_index"] if time_column else [])
    groups = {
        tuple(int(value) for value in key) if isinstance(key, tuple) else (int(key),): group
        for key, group in tracking.groupby(group_columns, sort=False)
    }
    play_groups = {
        (int(game_id), int(play_id)): group
        for (game_id, play_id), group in tracking.groupby(["gameId", "playId"], sort=False)
    }
    stable_slots_by_play = {
        key: stable_player_slot_map(
            play_history,
            play=play_map[key],
            game=game_map[key[0]],
            player_positions=positions,
        )
        for key, play_history in play_groups.items()
        if not play_history.empty
    }
    for example_index, row in examples.reset_index(drop=True).iterrows():
        game_id, play_id = int(row["game_id"]), int(row["play_id"])
        focal_id = focal.get(str(row["example_id"]))
        play_history = play_groups.get((game_id, play_id))
        if play_history is None or play_history.empty:
            continue
        stable_slots = stable_slots_by_play[(game_id, play_id)]
        slots = range(time_steps) if time_column else (0,)
        for slot in slots:
            key = (game_id, play_id, int(slot)) if time_column else (game_id, play_id)
            frame = groups.get(key)
            if frame is None or frame.empty:
                continue
            encoded, valid = canonicalize_tracking_frame(
                frame,
                play=play_map[(game_id, play_id)],
                game=game_map[game_id],
                player_positions=positions,
                focal_nfl_id=focal_id,
                infer_offensive_qb_as_focal=infer_offensive_qb_as_focal,
                stable_slots=stable_slots,
            )
            tokens[example_index, slot] = encoded
            player_mask[example_index, slot] = valid
            frame_mask[example_index, slot] = True
    if np.any(frame_mask.sum(axis=1) == 0):
        missing = examples.loc[frame_mask.sum(axis=1) == 0, "example_id"].tolist()[:5]
        raise ValueError(f"prepared examples lack tracking frames: {missing}")
    return tokens, player_mask, frame_mask


def enrich_examples_with_games(
    cohort: pd.DataFrame,
    games: pd.DataFrame,
    *,
    single_season: bool,
) -> pd.DataFrame:
    game_columns = [column for column in ("gameId", "season", "week") if column in games]
    result = cohort.merge(games[game_columns].drop_duplicates("gameId"), on="gameId", how="left")
    if "season" not in result:
        result["season"] = result["gameId"].astype(np.int64) // 1_000_000
    if "week" not in result:
        result["week"] = -1
    result["example_id"] = result["gameId"].astype(str) + ":" + result["playId"].astype(str)
    result["game_id"] = result["gameId"].astype(np.int64)
    result["play_id"] = result["playId"].astype(np.int64)
    if single_season:
        result["stratum"] = "week_" + result["week"].astype(int).astype(str).str.zfill(2)
    else:
        result["stratum"] = "season_" + result["season"].astype(int).astype(str)
    return result.sort_values(["gameId", "playId"], kind="mergesort").reset_index(drop=True)


def select_examples(
    frame: pd.DataFrame,
    *,
    game_ids: Iterable[int] | None,
    max_examples: int | None,
) -> pd.DataFrame:
    result = frame
    if game_ids is not None:
        selected_games = {int(value) for value in game_ids}
        result = result.loc[result["gameId"].astype(int).isin(selected_games)]
    result = result.sort_values(["gameId", "playId"], kind="mergesort")
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError("max_examples must be positive")
        result = result.head(int(max_examples))
    return result.reset_index(drop=True)


def causal_tabular_frame(examples: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a stable raw feature schema, filling absent source fields with NaN."""

    result = pd.DataFrame(index=np.arange(len(examples)))
    for column in columns:
        if column == "gameClockSeconds":
            result[column] = examples.get("gameClock", pd.Series(np.nan, index=examples.index)).map(clock_seconds).to_numpy()
        elif column in examples:
            result[column] = examples[column].to_numpy()
        else:
            result[column] = np.nan
    return result


def dump_audit(path: Path, audit: CohortAudit) -> None:
    """Write an audit atomically when a caller explicitly requests persistence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
