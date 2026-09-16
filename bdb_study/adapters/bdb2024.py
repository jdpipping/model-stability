"""BDB 2024 tackle-candidate adapter and winner-fidelity cohort builder."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_PLAYER_CHANNELS,
    MAX_TRACKED_OBJECTS,
    CohortAudit,
    PreparedTask,
    canonicalize_tracking_frame,
    load_adapter_task_spec,
    player_position_map,
    stable_player_slot_map,
)
from bdb_study.fidelity.bdb2024 import (
    FEATURE_NAMES,
    fidelity_receipt,
    infer_candidate_event_frame,
    quantize_winner_geometry,
    rotate_direction_and_orientation,
    strict_penalty_play_mask,
    validate_fidelity_counts,
    winner_feature_vector,
)


TASK_ID = "bdb2024_tackle"
KEY = ["gameId", "playId"]
TIME_STEPS = 10
TASK_PLAYER_CHANNELS = (*CANONICAL_PLAYER_CHANNELS, "carrier")
TRACKING_NUMERIC_SENTINEL_COLUMNS = ("nflId", "jerseyNumber", "o", "dir")


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported BDB2024 table format: {path}")


def _find_one(root: Path, basename: str) -> Path:
    matches = sorted(
        path
        for path in root.rglob(f"{basename}.*")
        if path.suffix.lower() in {".csv", ".parquet", ".gz"}
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {basename}.csv/parquet below {root}, found {len(matches)}"
        )
    return matches[0]


def _tracking_paths(root: Path) -> list[Path]:
    matches = sorted(
        [
            *root.rglob("tracking_week_*.csv"),
            *root.rglob("tracking_week_*.parquet"),
        ],
        key=lambda path: int(re.search(r"week_(\d+)", path.name).group(1)),
    )
    if len(matches) != 9:
        raise FileNotFoundError(f"expected nine BDB2024 weekly tracking files, found {len(matches)}")
    return matches


def _normalize_tracking_types(tracking: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric tracking columns from lossless lake string storage.

    The imported Parquet receipt preserves NFL ``NA`` sentinels.  Columns
    containing that sentinel (notably ``nflId``, ``o``, and ``dir`` because
    of the football row) consequently arrive as strings, unlike the winner's
    ``read_csv`` representation.  Numeric coercion restores the source
    notebook semantics and turns the football sentinel into a real null.
    """

    result = tracking.copy()
    for column in TRACKING_NUMERIC_SENTINEL_COLUMNS:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_raw(
    raw_dir: Path,
    *,
    game_ids: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four official tables and nine week files from either layout."""

    root = Path(raw_dir)
    games = _read_table(_find_one(root, "games"))
    plays = _read_table(_find_one(root, "plays"))
    players = _read_table(_find_one(root, "players"))
    tackles = _read_table(_find_one(root, "tackles"))
    selected = None if game_ids is None else {int(value) for value in game_ids}
    if selected is not None:
        games = games.loc[games["gameId"].astype(int).isin(selected)].copy()
        plays = plays.loc[plays["gameId"].astype(int).isin(selected)].copy()
        tackles = tackles.loc[tackles["gameId"].astype(int).isin(selected)].copy()
    tracking_parts: list[pd.DataFrame] = []
    for path in _tracking_paths(root):
        week = int(re.search(r"week_(\d+)", path.name).group(1))
        part = _read_table(path)
        if selected is not None:
            part = part.loc[part["gameId"].astype(int).isin(selected)]
        if not part.empty:
            part = part.copy()
            part["week"] = week
            tracking_parts.append(part)
    if not tracking_parts:
        raise ValueError("BDB2024 selection contains no tracking rows")
    tracking = pd.concat(tracking_parts, ignore_index=True)
    return games, plays, players, tackles, tracking


def _candidate_labels(tackles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"gameId", "playId", "nflId", "tackle", "assist", "pff_missedTackle"}
    missing = required.difference(tackles.columns)
    if missing:
        raise ValueError(f"tackles are missing columns: {sorted(missing)}")
    frame = tackles.copy()
    # The winner drops assist-only rows rather than requiring ``assist == 0``:
    # a defender who is also charted with a PFF miss remains a negative.  The
    # rare tackle+miss conflict is resolved by the notebook's tackle-first
    # label assignment and is therefore positive.
    made = frame["tackle"].eq(1)
    missed = frame["pff_missedTackle"].eq(1) & frame["tackle"].eq(0)
    candidates = frame.loc[made | missed].copy()
    candidates["target"] = made.loc[candidates.index].astype(np.int8)
    candidates = candidates.sort_values(["gameId", "playId", "nflId"], kind="stable")
    if candidates.duplicated(["gameId", "playId", "nflId"]).any():
        raise ValueError("BDB2024 has duplicate candidate defender labels")
    return candidates.reset_index(drop=True), {
        "assist_only_or_non_candidate": int(len(frame) - len(candidates)),
        "conflicting_made_and_missed": int(
            (frame["tackle"].eq(1) & frame["pff_missedTackle"].eq(1)).sum()
        ),
    }


def _merge_player_tracking(
    tracking: pd.DataFrame, plays: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    play_columns = [
        column
        for column in (
            "gameId",
            "playId",
            "ballCarrierId",
            "possessionTeam",
            "defensiveTeam",
            "passResult",
            "playDirection",
        )
        if column in plays
    ]
    result = tracking.merge(
        plays[play_columns].drop_duplicates(KEY), on=KEY, how="inner", suffixes=("", "_play")
    )
    if "playDirection_play" in result:
        if "playDirection" not in result:
            result["playDirection"] = result["playDirection_play"]
        result = result.drop(columns="playDirection_play")
    position_column = "position" if "position" in players else "officialPosition"
    player_columns = ["nflId", position_column]
    if "displayName" in players:
        player_columns.append("displayName")
    player_rows = players[player_columns].drop_duplicates("nflId")
    if position_column != "position":
        player_rows = player_rows.rename(columns={position_column: "position"})
    # This inner join intentionally removes the football, matching aggregate_data
    # in the pinned winner source for feature engineering.
    result = result.loc[result["nflId"].notna()].merge(
        player_rows, on="nflId", how="inner", suffixes=("", "_player")
    )
    if "club" not in result and "team" in result:
        result["club"] = result["team"]
    return quantize_winner_geometry(rotate_direction_and_orientation(result))


def _active_frames(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the winner's snap+5 through tackle/OOB preprocessing window."""

    retained: list[pd.DataFrame] = []
    for key, play in frame.groupby(KEY, sort=False):
        snaps = np.sort(play.loc[play["event"].eq("ball_snap"), "frameId"].dropna().unique())
        tackles = np.sort(play.loc[play["event"].eq("tackle"), "frameId"].dropna().unique())
        out_of_bounds = np.sort(
            play.loc[play["event"].eq("out_of_bounds"), "frameId"].dropna().unique()
        )
        if len(snaps) > 1 or len(tackles) > 1 or len(out_of_bounds) > 1:
            raise ValueError(f"multiple active-window marker frames for play {key}")
        minimum = int(snaps[0]) + 5 if len(snaps) else int(play["frameId"].min())
        tackle_frame = int(tackles[0]) if len(tackles) else int(play["frameId"].max())
        oob_frame = int(out_of_bounds[0]) if len(out_of_bounds) else int(play["frameId"].max())
        maximum = min(tackle_frame, oob_frame)
        retained.append(play.loc[play["frameId"].between(minimum, maximum)])
    if not retained:
        return frame.iloc[:0].copy()
    return pd.concat(retained, ignore_index=True)


def _game_mapping(games: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row.gameId): row._asdict()
        for row in games.drop_duplicates("gameId").itertuples(index=False)
    }


def _play_mapping(plays: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row.gameId), int(row.playId)): row._asdict()
        for row in plays.drop_duplicates(KEY).itertuples(index=False)
    }


def _winner_is_run(pass_result: Any) -> bool:
    """Match pandas ``read_csv`` handling of the NFL ``NA`` sentinel."""

    return bool(
        pd.isna(pass_result)
        or str(pass_result).strip().upper() in {"", "NA", "R"}
    )


def prepare_bdb2024_frames(
    games: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
    tackles: pd.DataFrame,
    tracking: pd.DataFrame,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
    require_fidelity_counts: bool = False,
    cutoff_policy: str = "primary_candidate_event",
) -> PreparedTask:
    """Construct one event-minus-ten-frame example per solo tackle/charted miss."""

    if cutoff_policy not in {
        "primary_candidate_event",
        "common_closest_approach_minus_ten",
    }:
        raise ValueError("unsupported BDB2024 cutoff policy")
    selected = None if game_ids is None else {int(value) for value in game_ids}
    if selected is not None:
        games = games.loc[games["gameId"].astype(int).isin(selected)].copy()
        plays = plays.loc[plays["gameId"].astype(int).isin(selected)].copy()
        tackles = tackles.loc[tackles["gameId"].astype(int).isin(selected)].copy()
        tracking = tracking.loc[tracking["gameId"].astype(int).isin(selected)].copy()
    tracking = _normalize_tracking_types(tracking)
    required_play = {
        "gameId",
        "playId",
        "ballCarrierId",
        "possessionTeam",
        "defensiveTeam",
        "passResult",
        "foulName1",
        "playNullifiedByPenalty",
    }
    missing = required_play.difference(plays.columns)
    if missing:
        raise ValueError(f"plays are missing winner columns: {sorted(missing)}")
    if "week" not in tracking:
        tracking = tracking.copy()
        tracking["week"] = -1
    candidate, initial_exclusions = _candidate_labels(tackles)
    outcome_eligible = len(candidate)
    valid_plays = plays.loc[strict_penalty_play_mask(plays), KEY]
    before_penalty = len(candidate)
    candidate = candidate.merge(valid_plays.assign(_valid=True), on=KEY, how="inner")
    penalty_excluded = before_penalty - len(candidate)
    if max_examples is not None and int(max_examples) <= 0:
        raise ValueError("max_examples must be positive")

    keys = candidate[KEY].drop_duplicates()
    tracking = tracking.merge(keys.assign(_selected=True), on=KEY, how="inner").drop(
        columns="_selected"
    )
    player_tracking = _active_frames(_merge_player_tracking(tracking, plays, players))
    player_groups = {
        (int(game), int(play)): group.sort_values(["frameId", "nflId"], kind="stable")
        for (game, play), group in player_tracking.groupby(KEY, sort=False)
    }
    raw_groups = {
        (int(game), int(play)): group.sort_values(["frameId", "nflId"], kind="stable", na_position="last")
        for (game, play), group in tracking.groupby(KEY, sort=False)
    }
    play_map = _play_mapping(plays)
    game_map = _game_mapping(games)
    positions = player_position_map(players)
    # Allocate one focal-independent registry per play.  The full tracked
    # participant identity set is alignment metadata only; coordinates and
    # event state are still taken exclusively from each candidate's causal
    # history through its cutoff.
    stable_slots_by_play = {
        key: stable_player_slot_map(
            raw_play,
            play=play_map[key],
            game=game_map[key[0]],
            player_positions=positions,
            max_objects=MAX_TRACKED_OBJECTS,
        )
        for key, raw_play in raw_groups.items()
        if not raw_play.empty
    }

    example_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    frame_mask_rows: list[np.ndarray] = []
    exclusion = {
        **initial_exclusions,
        "strict_penalty": int(penalty_excluded),
        "missing_tracking": 0,
        "missing_event_frame": 0,
        "event_too_early_or_missing_state": 0,
        "invalid_feature_or_token": 0,
    }
    cutoff_eligible = 0
    for row in candidate.itertuples(index=False):
        if max_examples is not None and len(example_rows) >= int(max_examples):
            break
        key = (int(row.gameId), int(row.playId))
        play_tracking = player_groups.get(key)
        raw_play = raw_groups.get(key)
        if play_tracking is None or raw_play is None or play_tracking.empty:
            exclusion["missing_tracking"] += 1
            continue
        play_record = play_map[key]
        carrier = int(play_record["ballCarrierId"])
        made = int(row.target) == 1
        try:
            event_frame = infer_candidate_event_frame(
                play_tracking,
                tackler_id=int(row.nflId),
                ballcarrier_id=carrier,
                made=(made if cutoff_policy == "primary_candidate_event" else False),
            )
        except ValueError:
            exclusion["missing_event_frame"] += 1
            continue
        if event_frame is None:
            exclusion["missing_event_frame"] += 1
            continue
        cutoff_eligible += 1
        state_frame_id = int(event_frame) - 10
        if state_frame_id <= 0:
            exclusion["event_too_early_or_missing_state"] += 1
            continue
        state = play_tracking.loc[play_tracking["frameId"].eq(state_frame_id)]
        raw_state = raw_play.loc[raw_play["frameId"].eq(state_frame_id)]
        if state.empty or raw_state.empty:
            exclusion["event_too_early_or_missing_state"] += 1
            continue
        is_run = _winner_is_run(play_record.get("passResult"))
        try:
            features = winner_feature_vector(
                state,
                tackler_id=int(row.nflId),
                ballcarrier_id=carrier,
                is_run=bool(is_run),
                offset_frames=10,
            )
            history_ids = np.sort(
                raw_play.loc[raw_play["frameId"].le(state_frame_id), "frameId"].unique()
            )[-TIME_STEPS:]
            raw_history = raw_play.loc[raw_play["frameId"].isin(history_ids)]
            stable_slots = stable_slots_by_play[key]
            token = np.zeros(
                (TIME_STEPS, MAX_TRACKED_OBJECTS, len(TASK_PLAYER_CHANNELS)),
                dtype=np.float32,
            )
            token_mask = np.zeros((TIME_STEPS, MAX_TRACKED_OBJECTS), dtype=bool)
            history_mask = np.zeros(TIME_STEPS, dtype=bool)
            offset = TIME_STEPS - len(history_ids)
            for history_index, frame_id in enumerate(history_ids):
                encoded, valid = canonicalize_tracking_frame(
                    raw_history.loc[raw_history["frameId"].eq(frame_id)],
                    play=play_record,
                    game=game_map[int(row.gameId)],
                    player_positions=positions,
                    focal_nfl_id=int(row.nflId),
                    stable_slots=stable_slots,
                    max_objects=MAX_TRACKED_OBJECTS,
                )
                token[offset + history_index, :, : len(CANONICAL_PLAYER_CHANNELS)] = encoded
                # The official ballCarrierId is used only to mark the already
                # aligned stable slot.  The identifier itself is never exposed
                # to a model.  If the carrier is momentarily absent from a
                # tracking frame, the role channel remains zero for that frame.
                carrier_slot = stable_slots.get(("player", str(carrier)))
                if carrier_slot is None:
                    raise ValueError("official ball carrier is absent from the play history")
                if valid[carrier_slot]:
                    token[
                        offset + history_index,
                        carrier_slot,
                        len(CANONICAL_PLAYER_CHANNELS),
                    ] = 1.0
                token_mask[offset + history_index] = valid
                history_mask[offset + history_index] = True
        except (KeyError, IndexError, TypeError, ValueError, FloatingPointError):
            exclusion["invalid_feature_or_token"] += 1
            continue
        week_values = raw_state["week"].dropna().astype(int).unique()
        week = int(week_values[0]) if len(week_values) == 1 else -1
        example_rows.append(
            {
                "example_id": f"{int(row.gameId)}:{int(row.playId)}:{int(row.nflId)}",
                "game_id": int(row.gameId),
                "stratum": f"2022-w{week:02d}",
                "target": int(row.target),
                "play_id": int(row.playId),
                "nfl_id": int(row.nflId),
                "week": week,
                "event_frame_id": int(event_frame),
                "state_frame_id": state_frame_id,
                "history_frames": int(len(history_ids)),
                "left_pad_frames": int(offset),
                "event_source": (
                    "minimum_distance"
                    if cutoff_policy == "common_closest_approach_minus_ten"
                    else ("tackle_or_out_of_bounds" if made else "minimum_distance")
                ),
            }
        )
        feature_rows.append(features)
        token_rows.append(token)
        mask_rows.append(token_mask)
        frame_mask_rows.append(history_mask)

    examples = pd.DataFrame(example_rows)
    if examples.empty:
        raise ValueError("BDB2024 preparation retained no tackle candidates")
    labels = examples["target"].to_numpy(dtype=np.int8)
    if require_fidelity_counts:
        if selected is not None or max_examples is not None:
            raise ValueError("fidelity counts require the complete unfiltered release")
        validate_fidelity_counts(
            int(((examples["week"] <= 8) & examples["target"].eq(1)).sum()),
            int(((examples["week"] <= 8) & examples["target"].eq(0)).sum()),
            int((examples["week"].eq(9) & examples["target"].eq(1)).sum()),
            int((examples["week"].eq(9) & examples["target"].eq(0)).sum()),
        )
    tokens = np.stack(token_rows).astype(np.float32)
    player_mask = np.stack(mask_rows).astype(bool)
    frame_mask = np.stack(frame_mask_rows).astype(bool)
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=int(len(plays)),
        outcome_eligible=int(outcome_eligible),
        cutoff_eligible=int(cutoff_eligible),
        retained_examples=int(len(examples)),
        retained_games=int(examples["game_id"].nunique()),
        exclusions=exclusion,
        details={
            "positive": "solo tackle",
            "negative": "PFF charted missed tackle",
            "assists": "assist-only rows excluded; PFF misses remain negative",
            "state_offset_frames": 10,
            "history_frames": TIME_STEPS,
            "score_interpretation": "case-control score",
            "cutoff_policy": cutoff_policy,
        },
    )
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="binary",
        primary_metric="brier",
        support=None,
        examples=examples.reset_index(drop=True),
        tabular=pd.DataFrame(feature_rows, columns=FEATURE_NAMES),
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=TASK_PLAYER_CHANNELS,
        y=labels,
        audit=audit.as_dict(),
        metadata={
            "winner_fidelity": fidelity_receipt(),
            "tabular_geometry": "winner x-only reflection; y retained",
            "neural_token_geometry": "suite-canonical 180-degree x/y rotation centered on football",
            "history_frames": TIME_STEPS,
            "padding": "left, masked",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
            "graph": "candidate defender, official ball carrier, blockers, and defensive-help relations",
            "carrier_role": "official ballCarrierId used for stable-slot alignment only; identifier never encoded",
            "structural_ablation": "candidate_carrier_pair_only",
            "team_identity_primary": "excluded",
            "cutoff_policy": cutoff_policy,
            "cutoff_sensitivity": "common_closest_approach_minus_ten",
        },
    )


def audit_cohort(raw_dir: Path) -> CohortAudit:
    prepared = prepare_bdb2024(raw_dir)
    return CohortAudit(**prepared.audit)


def prepare_bdb2024(
    raw_dir: Path,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
    require_fidelity_counts: bool = False,
    cutoff_policy: str = "primary_candidate_event",
) -> PreparedTask:
    games, plays, players, tackles, tracking = load_raw(raw_dir, game_ids=game_ids)
    return prepare_bdb2024_frames(
        games,
        plays,
        players,
        tackles,
        tracking,
        game_ids=game_ids,
        max_examples=max_examples,
        require_fidelity_counts=require_fidelity_counts,
        cutoff_policy=cutoff_policy,
    )


prepare = prepare_bdb2024
