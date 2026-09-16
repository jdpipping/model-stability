"""BDB 2022 returned-punt yard distribution adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_PLAYER_CHANNELS,
    CohortAudit,
    PreparedTask,
    assemble_player_tensors,
    causal_tabular_frame,
    discover_cutoffs,
    discover_frame_windows,
    enrich_examples_with_games,
    load_adapter_task_spec,
    load_rows_at_selected_frames,
    parse_first_nfl_id,
    select_examples,
    weekly_tracking_paths,
)


TASK_ID = "bdb2022_punt_returns"
EXPECTED_EXAMPLES = 2_273
EXPECTED_GAMES = 712
SUPPORT = tuple(range(-20, 111))
TIME_STEPS = 40
TABULAR_COLUMNS = (
    "quarter",
    "down",
    "yardsToGo",
    "absoluteYardlineNumber",
    "gameClockSeconds",
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "kickingTeam",
    "returnTeam",
    "kickLength",
)


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _return_team(row: MappingLike, game: MappingLike) -> str:
    kicking = str(row.get("possessionTeam", ""))
    home = str(game.get("homeTeamAbbr", ""))
    visitor = str(game.get("visitorTeamAbbr", ""))
    if kicking == home:
        return visitor
    if kicking == visitor:
        return home
    raise ValueError(f"kicking team {kicking!r} is not in game {row.get('gameId')}")


MappingLike = dict[str, Any] | pd.Series


def _cohort(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[Path, ...], pd.DataFrame, CohortAudit]:
    root = Path(raw_dir)
    plays = pd.read_csv(root / "plays.csv", low_memory=False)
    games = pd.read_csv(root / "games.csv", low_memory=False)
    players = pd.read_csv(root / "players.csv", low_memory=False)
    paths = weekly_tracking_paths(root, "tracking*.csv")
    eligible = plays.loc[
        plays["specialTeamsPlayType"].eq("Punt")
        & plays["specialTeamsResult"].eq("Return")
        & plays["kickReturnYardage"].notna()
    ].copy()
    cutoffs, cutoff_audit = discover_cutoffs(
        paths,
        eligible,
        event_names=("punt_received",),
        require_unique_frame=True,
    )
    cohort = eligible.merge(cutoffs, on=["gameId", "playId"], how="inner")
    counts = cutoff_audit["cutoff_frame_count"]
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=len(plays),
        outcome_eligible=len(eligible),
        cutoff_eligible=len(cohort),
        retained_examples=len(cohort),
        retained_games=cohort["gameId"].nunique(),
        exclusions={
            "not_returned_punt_with_yards": len(plays) - len(eligible),
            "missing_punt_received": int(counts.eq(0).sum()),
            "non_unique_punt_received_frame": int(counts.gt(1).sum()),
        },
        details={
            "cutoff_rule": "earliest and unique punt_received frame",
            "support": [SUPPORT[0], SUPPORT[-1]],
            "expected_examples": EXPECTED_EXAMPLES,
            "expected_games": EXPECTED_GAMES,
        },
    )
    return plays, games, players, paths, cohort, audit


def audit_cohort(raw_dir: Path) -> CohortAudit:
    *_, audit = _cohort(Path(raw_dir))
    return audit


def prepare_bdb2022(
    raw_dir: Path,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
) -> PreparedTask:
    plays, games, players, paths, full_cohort, audit = _cohort(Path(raw_dir))
    if game_ids is None and max_examples is None:
        audit.require(examples=EXPECTED_EXAMPLES, games=EXPECTED_GAMES)
    selected = select_examples(full_cohort, game_ids=game_ids, max_examples=max_examples)
    if selected.empty:
        raise ValueError("BDB 2022 selection contains no returned punts")
    if not selected["kickReturnYardage"].between(SUPPORT[0], SUPPORT[-1]).all():
        raise ValueError("punt-return target lies outside frozen support -20..110")
    cutoffs = selected[["gameId", "playId", "cutoff_frame_id", "cutoff_event"]]
    selected_frames = discover_frame_windows(paths, cutoffs, window=TIME_STEPS)
    tracking = load_rows_at_selected_frames(paths, selected_frames)
    # NFL ``playDirection`` follows the kick.  The modeled motion begins with
    # the return, so invert it before the shared transform to orient return
    # progress (not kick progress) toward increasing standardized x.
    tracking_for_model = tracking.copy()
    if "playDirection" in tracking_for_model:
        tracking_for_model["playDirection"] = tracking_for_model["playDirection"].map(
            {"left": "right", "right": "left"}
        ).fillna(tracking_for_model["playDirection"])
    prepared = enrich_examples_with_games(selected, games, single_season=False)
    game_map = {int(row.gameId): row._asdict() for row in games.itertuples(index=False)}
    prepared["kickingTeam"] = prepared["possessionTeam"]
    prepared["returnTeam"] = [
        _return_team(row, game_map[int(row["gameId"])]) for _, row in prepared.iterrows()
    ]
    prepared["target"] = prepared["kickReturnYardage"].astype(int)
    focal = {
        str(row["example_id"]): parse_first_nfl_id(row["returnerId"])
        for _, row in prepared.iterrows()
    }
    # For this target, the receiving side is the modeled offense.  Preserve the
    # original kicking team separately in the raw tabular feature table.
    token_plays = selected.copy()
    return_lookup = prepared.set_index(["gameId", "playId"])["returnTeam"]
    token_plays["possessionTeam"] = [
        return_lookup.loc[(int(row.gameId), int(row.playId))]
        for row in token_plays.itertuples(index=False)
    ]
    tabular = causal_tabular_frame(prepared, TABULAR_COLUMNS)
    tokens, player_mask, frame_mask = assemble_player_tensors(
        prepared,
        tracking_for_model,
        token_plays,
        games,
        players,
        time_steps=TIME_STEPS,
        focal_by_example=focal,
    )
    counts = (
        tracking.groupby(["gameId", "playId", "frameId"]).size()
        .groupby(["gameId", "playId"]).max()
        .rename("tracked_objects").reset_index()
    )
    prepared = prepared.merge(counts, on=["gameId", "playId"], how="left")
    frame_counts = selected_frames.groupby(["gameId", "playId"]).size().rename("history_frames").reset_index()
    prepared = prepared.merge(frame_counts, on=["gameId", "playId"], how="left")
    prepared["left_pad_frames"] = TIME_STEPS - prepared["history_frames"].astype(int)
    prepared["target_index"] = prepared["target"] - SUPPORT[0]
    example_columns = (
        "example_id",
        "game_id",
        "play_id",
        "stratum",
        "season",
        "week",
        "target",
        "target_index",
        "cutoff_frame_id",
        "cutoff_event",
        "kickingTeam",
        "returnTeam",
        "tracked_objects",
        "history_frames",
        "left_pad_frames",
    )
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="distribution",
        primary_metric="crps",
        examples=prepared.loc[:, example_columns],
        tabular=tabular,
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CANONICAL_PLAYER_CHANNELS,
        y=prepared["target_index"].to_numpy(dtype=np.int16),
        support=tuple(float(value) for value in SUPPORT),
        audit=audit.as_dict(),
        metadata={
            "cutoff": "unique punt_received frame",
            "history_frames": TIME_STEPS,
            "padding": "left, masked",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
            "graph": "returner, blocker, coverage, and lane relations",
            "output_head": "punt_cdf_residual_v1",
            "structural_ablation": "remove_blocker_coverer_edges",
            "team_identity_primary": "excluded",
            "support_min": SUPPORT[0],
            "support_max": SUPPORT[-1],
            "tabular_columns": list(TABULAR_COLUMNS),
            "learned_preprocessing": "split-local only",
        },
    )


prepare = prepare_bdb2022
