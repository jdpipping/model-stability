"""Harmonized BDB 2020 rushing-yard distribution adapter.

The competition file is already a row-per-player handoff snapshot, so this
adapter deliberately creates one observed time step and never invents a
football row.  NFL IDs are used only to allocate deterministic slots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_PLAYER_CHANNELS,
    FIELD_LENGTH,
    FIELD_WIDTH,
    MAX_TRACKED_OBJECTS,
    CohortAudit,
    PreparedTask,
    clock_seconds,
    load_adapter_task_spec,
    position_group,
    select_examples,
)


TASK_ID = "bdb2020_rushing_harmonized"
EXPECTED_EXAMPLES = 31_007
EXPECTED_GAMES = 688
SUPPORT = tuple(range(-28, 52))
TIME_STEPS = 1
TABULAR_COLUMNS = (
    "quarter",
    "down",
    "yardsToGo",
    "yardsFromOwnGoal",
    "gameClockSeconds",
    "offenseScoreBeforePlay",
    "defenseScoreBeforePlay",
    "scoreDifferentialBeforePlay",
    "defendersInTheBox",
    "offenseFormation",
    "offensePersonnel",
    "defensePersonnel",
)

_RAW_COLUMNS = (
    "GameId",
    "PlayId",
    "Team",
    "X",
    "Y",
    "S",
    "A",
    "Dis",
    "Orientation",
    "Dir",
    "NflId",
    "Season",
    "YardLine",
    "Quarter",
    "GameClock",
    "PossessionTeam",
    "Down",
    "Distance",
    "FieldPosition",
    "HomeScoreBeforePlay",
    "VisitorScoreBeforePlay",
    "NflIdRusher",
    "OffenseFormation",
    "OffensePersonnel",
    "DefendersInTheBox",
    "DefensePersonnel",
    "PlayDirection",
    "TimeHandoff",
    "Yards",
    "Position",
    "Week",
)
_RENAMES = {
    "GameId": "gameId",
    "PlayId": "playId",
    "Team": "team",
    "X": "x",
    "Y": "y",
    "S": "s",
    "A": "a",
    "Dis": "dis",
    "Orientation": "orientation",
    "Dir": "dir",
    "NflId": "nflId",
    "Season": "season",
    "YardLine": "yardLine",
    "Quarter": "quarter",
    "GameClock": "gameClock",
    "PossessionTeam": "possessionTeam",
    "Down": "down",
    "Distance": "yardsToGo",
    "FieldPosition": "fieldPosition",
    "HomeScoreBeforePlay": "preSnapHomeScore",
    "VisitorScoreBeforePlay": "preSnapVisitorScore",
    "NflIdRusher": "nflIdRusher",
    "OffenseFormation": "offenseFormation",
    "OffensePersonnel": "offensePersonnel",
    "DefendersInTheBox": "defendersInTheBox",
    "DefensePersonnel": "defensePersonnel",
    "PlayDirection": "playDirection",
    "TimeHandoff": "timeHandoff",
    "Yards": "yards",
    "Position": "position",
    "Week": "week",
}
_KEYS = ["gameId", "playId"]


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _bad_keys(condition: pd.Series) -> list[tuple[int, int]]:
    return [tuple(int(item) for item in key) for key in condition.index[~condition][:3]]


def _require_groups(condition: pd.Series, message: str) -> None:
    if not bool(condition.all()):
        raise ValueError(f"{message}; examples: {_bad_keys(condition)}")


def _load_validated_rows(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"BDB 2020 source is missing: {path}")
    try:
        rows = pd.read_csv(path, usecols=list(_RAW_COLUMNS), low_memory=False).rename(
            columns=_RENAMES
        )
    except ValueError as exc:
        raise ValueError("BDB 2020 train.csv lacks required columns") from exc

    numeric = (
        "gameId",
        "playId",
        "x",
        "y",
        "s",
        "a",
        "dis",
        "orientation",
        "dir",
        "nflId",
        "nflIdRusher",
        "season",
        "yards",
    )
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    required_finite = ("gameId", "playId", "x", "y", "s", "a", "dis", "nflId", "nflIdRusher", "season", "yards")
    for column in required_finite:
        if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
            raise ValueError(f"BDB 2020 {column} must be finite on every player row")
    for column in ("orientation", "dir"):
        observed = rows[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"BDB 2020 {column} must be finite when observed")
    for column in ("gameId", "playId", "nflId", "nflIdRusher", "season", "yards"):
        values = rows[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"BDB 2020 {column} must be integer-valued")
        rows[column] = values.astype(np.int64)

    rows["team"] = rows["team"].astype(str).str.lower()
    rows["playDirection"] = rows["playDirection"].astype(str).str.lower()
    grouped = rows.groupby(_KEYS, sort=False, observed=True)
    _require_groups(grouped.size().eq(22), "each BDB 2020 handoff must contain exactly 22 player rows")
    _require_groups(
        grouped["nflId"].nunique().eq(22),
        "each BDB 2020 handoff must contain 22 unique player identities",
    )
    for column in (
        "nflIdRusher",
        "season",
        "week",
        "yards",
        "playDirection",
        "timeHandoff",
    ):
        _require_groups(
            grouped[column].nunique(dropna=False).eq(1),
            f"BDB 2020 {column} must be constant within a play",
        )
    if rows["timeHandoff"].isna().any():
        raise ValueError("BDB 2020 handoff timestamp must be observed on every player row")
    if not rows["playDirection"].isin(("left", "right")).all():
        raise ValueError("BDB 2020 playDirection must be left or right")
    if not rows["season"].isin((2017, 2018, 2019)).all():
        raise ValueError("BDB 2020 season must be 2017, 2018, or 2019")

    rows["_is_rusher"] = rows["nflId"].eq(rows["nflIdRusher"])
    _require_groups(
        rows.groupby(_KEYS, observed=True)["_is_rusher"].sum().eq(1),
        "each BDB 2020 handoff must contain exactly one rusher",
    )
    rusher_team = rows.loc[rows["_is_rusher"], _KEYS + ["team"]].set_index(_KEYS)["team"]
    rows = rows.join(rusher_team.rename("_rusher_team"), on=_KEYS)
    rows["_offense"] = rows["team"].eq(rows["_rusher_team"])
    offense_count = rows.groupby(_KEYS, observed=True)["_offense"].sum()
    _require_groups(
        offense_count.eq(11),
        "each BDB 2020 handoff must contain 11 offense and 11 defense players",
    )
    if not rows["team"].isin(("home", "away")).all():
        raise ValueError("BDB 2020 Team must identify only home or away players")
    return rows


def _cohort(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, CohortAudit]:
    rows = _load_validated_rows(raw_dir)
    plays = (
        rows.sort_values(_KEYS + ["nflId"], kind="mergesort")
        .drop_duplicates(_KEYS, keep="first")
        .sort_values(_KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=len(plays),
        outcome_eligible=len(plays),
        cutoff_eligible=len(plays),
        retained_examples=len(plays),
        retained_games=plays["gameId"].nunique(),
        exclusions={
            "missing_handoff_snapshot": 0,
            "invalid_player_structure": 0,
            "missing_rushing_yards": 0,
        },
        details={
            "source_player_rows": len(rows),
            "cutoff_rule": "competition handoff snapshot",
            "tracked_players_per_play": 22,
            "support": [SUPPORT[0], SUPPORT[-1]],
            "expected_examples": EXPECTED_EXAMPLES,
            "expected_games": EXPECTED_GAMES,
        },
    )
    return rows, plays, audit


def audit_cohort(raw_dir: Path) -> CohortAudit:
    *_, audit = _cohort(Path(raw_dir))
    return audit


def _tabular_frame(plays: pd.DataFrame) -> pd.DataFrame:
    yard_line = pd.to_numeric(plays["yardLine"], errors="coerce").to_numpy(dtype=float)
    own_side = plays["fieldPosition"].astype(str).eq(plays["possessionTeam"].astype(str)).to_numpy()
    yards_from_own_goal = np.where(yard_line == 50.0, 50.0, np.where(own_side, yard_line, 100.0 - yard_line))
    home_offense = plays["_rusher_team"].eq("home").to_numpy()
    home_score = pd.to_numeric(plays["preSnapHomeScore"], errors="coerce").to_numpy(dtype=float)
    visitor_score = pd.to_numeric(plays["preSnapVisitorScore"], errors="coerce").to_numpy(dtype=float)
    offense_score = np.where(home_offense, home_score, visitor_score)
    defense_score = np.where(home_offense, visitor_score, home_score)
    result = pd.DataFrame(
        {
            "quarter": plays["quarter"].to_numpy(),
            "down": plays["down"].to_numpy(),
            "yardsToGo": plays["yardsToGo"].to_numpy(),
            "yardsFromOwnGoal": yards_from_own_goal,
            "gameClockSeconds": plays["gameClock"].map(clock_seconds).to_numpy(),
            "offenseScoreBeforePlay": offense_score,
            "defenseScoreBeforePlay": defense_score,
            "scoreDifferentialBeforePlay": offense_score - defense_score,
            "defendersInTheBox": plays["defendersInTheBox"].to_numpy(),
            "offenseFormation": plays["offenseFormation"].to_numpy(),
            "offensePersonnel": plays["offensePersonnel"].to_numpy(),
            "defensePersonnel": plays["defensePersonnel"].to_numpy(),
        }
    )
    return result.loc[:, TABULAR_COLUMNS]


def _angle_components(
    values: pd.Series,
    *,
    mirror: bool,
    missing_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = values.to_numpy(dtype=float)
    missing = np.isnan(numeric)
    radians = np.deg2rad(np.mod(90.0 - np.nan_to_num(numeric, nan=90.0), 360.0))
    x_component = np.cos(radians)
    y_component = np.sin(radians)
    if mirror:
        x_component *= -1.0
        y_component *= -1.0
    if missing_x is None:
        x_component[missing] = 0.0
        y_component[missing] = 0.0
    else:
        x_component[missing] = missing_x[missing]
        y_component[missing] = 0.0
    return y_component, x_component


def _assemble_snapshot_tensors(
    plays: pd.DataFrame, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_examples = len(plays)
    tokens = np.zeros(
        (n_examples, TIME_STEPS, MAX_TRACKED_OBJECTS, len(CANONICAL_PLAYER_CHANNELS)),
        dtype=np.float32,
    )
    player_mask = np.zeros((n_examples, TIME_STEPS, MAX_TRACKED_OBJECTS), dtype=bool)
    frame_mask = np.ones((n_examples, TIME_STEPS), dtype=bool)
    channel = {name: index for index, name in enumerate(CANONICAL_PLAYER_CHANNELS)}
    example_index = {
        (int(row.gameId), int(row.playId)): index
        for index, row in enumerate(plays.itertuples(index=False))
    }
    selected_keys = plays.loc[:, _KEYS]
    selected_rows = rows.merge(selected_keys, on=_KEYS, how="inner", validate="many_to_one")

    for key, frame in selected_rows.groupby(_KEYS, sort=False, observed=True):
        index = example_index[(int(key[0]), int(key[1]))]
        ordered = frame.sort_values(
            ["_offense", "nflId"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        offense = ordered["_offense"].to_numpy(dtype=bool)
        focal = ordered["_is_rusher"].to_numpy(dtype=bool)
        mirror = str(ordered.loc[0, "playDirection"]) == "left"
        x = ordered["x"].to_numpy(dtype=float)
        y = ordered["y"].to_numpy(dtype=float)
        x_std = FIELD_LENGTH - x if mirror else x
        y_std = FIELD_WIDTH - y if mirror else y
        rusher_x = float(x_std[focal][0])
        rusher_y = float(y_std[focal][0])
        speed = ordered["s"].to_numpy(dtype=float, copy=True)
        season_2017 = ordered["season"].to_numpy(dtype=int) == 2017
        speed[season_2017] = 10.0 * ordered.loc[season_2017, "dis"].to_numpy(dtype=float)
        dir_sin, dir_cos = _angle_components(
            ordered["dir"], mirror=mirror, missing_x=np.where(offense, 1.0, -1.0)
        )
        orientation_sin, orientation_cos = _angle_components(
            ordered["orientation"], mirror=mirror
        )

        out = tokens[index, 0]
        out[:22, channel["x_rel"]] = x_std - rusher_x
        out[:22, channel["y_rel"]] = y_std - rusher_y
        out[:22, channel["vx"]] = speed * dir_cos
        out[:22, channel["vy"]] = speed * dir_sin
        out[:22, channel["speed"]] = speed
        out[:22, channel["acceleration"]] = ordered["a"].to_numpy(dtype=float)
        out[:22, channel["dir_sin"]] = dir_sin
        out[:22, channel["dir_cos"]] = dir_cos
        out[:22, channel["orientation_sin"]] = orientation_sin
        out[:22, channel["orientation_cos"]] = orientation_cos
        out[:22, channel["offense"]] = offense.astype(np.float32)
        out[:22, channel["defense"]] = (~offense).astype(np.float32)
        out[:22, channel["focal"]] = focal.astype(np.float32)
        for slot, value in enumerate(ordered["position"]):
            out[slot, channel[f"position_{position_group(value)}"]] = 1.0
        player_mask[index, 0, :22] = True
    return tokens, player_mask, frame_mask


def prepare_bdb2020(
    raw_dir: Path,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
) -> PreparedTask:
    rows, full_cohort, audit = _cohort(Path(raw_dir))
    if game_ids is None and max_examples is None:
        audit.require(examples=EXPECTED_EXAMPLES, games=EXPECTED_GAMES)
    selected = select_examples(
        full_cohort, game_ids=game_ids, max_examples=max_examples
    )
    if selected.empty:
        raise ValueError("BDB 2020 selection contains no rushing plays")
    raw_target = selected["yards"].to_numpy(dtype=np.int64)
    clipped_target = np.clip(raw_target, SUPPORT[0], SUPPORT[-1]).astype(np.int16)
    selected["raw_target"] = raw_target
    selected["target"] = clipped_target
    selected["target_index"] = clipped_target - SUPPORT[0]
    selected["example_id"] = (
        selected["gameId"].astype(str) + ":" + selected["playId"].astype(str)
    )
    selected["game_id"] = selected["gameId"].astype(np.int64)
    selected["play_id"] = selected["playId"].astype(np.int64)
    selected["stratum"] = "season_" + selected["season"].astype(str)
    tabular = _tabular_frame(selected)
    tokens, player_mask, frame_mask = _assemble_snapshot_tensors(selected, rows)
    examples = selected.loc[
        :,
        [
            "example_id",
            "game_id",
            "play_id",
            "stratum",
            "season",
            "week",
            "target",
            "raw_target",
            "target_index",
            "timeHandoff",
        ],
    ].reset_index(drop=True)
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="distribution",
        primary_metric="crps",
        examples=examples,
        tabular=tabular.reset_index(drop=True),
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CANONICAL_PLAYER_CHANNELS,
        y=selected["target_index"].to_numpy(dtype=np.int16),
        support=tuple(float(value) for value in SUPPORT),
        audit=audit.as_dict(),
        metadata={
            "cutoff": "competition handoff snapshot",
            "history_frames": TIME_STEPS,
            "padding": "one observed frame; one masked tracked-object slot",
            "stable_player_slots": (
                "play_global_focal_independent_v1; "
                "offense_then_defense_numeric_nflId_alignment_only_v1; "
                "nflId never encoded"
            ),
            "coordinate_normalization": (
                "orient offensive progress toward increasing x and center on rusher"
            ),
            "graph": "rusher, blocker, and defender relations at handoff",
            "output_head": "punt_cdf_residual_v1",
            "structural_ablation": "remove_blocker_defender_edges",
            "team_identity_primary": "excluded",
            "support_min": SUPPORT[0],
            "support_max": SUPPORT[-1],
            "target_clipping": "clip raw rushing yards to frozen support -28..51",
            "speed_correction": "season 2017 speed equals 10 times displacement",
            "missing_direction": (
                "offense forward and defense backward after direction normalization"
            ),
            "missing_orientation": "zero vector",
            "tabular_columns": list(TABULAR_COLUMNS),
            "learned_preprocessing": "split-local only",
        },
    )


prepare = prepare_bdb2020
