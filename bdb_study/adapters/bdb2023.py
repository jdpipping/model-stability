"""Causal BDB 2023 sack-at-snap adapter.

Unlike the legacy sack preparation script, this module never reads
``pffScoutingData.csv`` and never uses charted role, route, action, dropback,
or coverage fields as inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
    select_examples,
    weekly_tracking_paths,
)


TASK_ID = "bdb2023_sack"
EXPECTED_EXAMPLES = 8_533
EXPECTED_GAMES = 122
PASS_RESULTS = ("C", "I", "IN", "R", "S")
SNAP_EVENTS = ("ball_snap", "autoevent_ballsnap")
TIME_STEPS = 20
TABULAR_COLUMNS = (
    "quarter",
    "down",
    "yardsToGo",
    "absoluteYardlineNumber",
    "gameClockSeconds",
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "possessionTeam",
    "defensiveTeam",
    "offenseFormation",
    "personnelO",
    "defendersInBox",
    "personnelD",
)
FORBIDDEN_MODEL_COLUMNS = (
    "passResult",
    "playDescription",
    "dropBackType",
    "pff_playAction",
    "pff_passCoverage",
    "pff_passCoverageType",
    "penaltyYards",
    "prePenaltyPlayResult",
    "playResult",
    "foulName1",
    "foulNFLId1",
    "foulName2",
    "foulNFLId2",
    "foulName3",
    "foulNFLId3",
    "pff_role",
    "pff_positionLinedUp",
    "pff_sack",
    "pff_blockType",
    "pff_backFieldBlock",
)


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _cohort(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[Path, ...], pd.DataFrame, CohortAudit]:
    root = Path(raw_dir)
    plays = pd.read_csv(root / "plays.csv", low_memory=False)
    games = pd.read_csv(root / "games.csv", low_memory=False)
    players = pd.read_csv(root / "players.csv", low_memory=False)
    paths = weekly_tracking_paths(root, "week*.csv")
    eligible = plays.loc[plays["passResult"].isin(PASS_RESULTS)].copy()
    cutoffs, cutoff_audit = discover_cutoffs(paths, eligible, event_names=SNAP_EVENTS)
    cohort = eligible.merge(cutoffs, on=["gameId", "playId"], how="inner")
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=len(plays),
        outcome_eligible=len(eligible),
        cutoff_eligible=len(cohort),
        retained_examples=len(cohort),
        retained_games=cohort["gameId"].nunique(),
        exclusions={
            "pass_result_not_C_I_IN_R_S": len(plays) - len(eligible),
            "missing_snap_event": int(cutoff_audit["cutoff_frame_count"].eq(0).sum()),
        },
        details={
            "snap_events": list(SNAP_EVENTS),
            "cutoff_rule": "earliest observed snap event",
            "scramble_label": 0,
            "source_files_never_read": ["pffScoutingData.csv"],
            "forbidden_model_columns": list(FORBIDDEN_MODEL_COLUMNS),
            "expected_examples": EXPECTED_EXAMPLES,
            "expected_games": EXPECTED_GAMES,
        },
    )
    return plays, games, players, paths, cohort, audit


def audit_cohort(raw_dir: Path) -> CohortAudit:
    *_, audit = _cohort(Path(raw_dir))
    return audit


def prepare_bdb2023(
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
        raise ValueError("BDB 2023 selection contains no snap-observed dropbacks")
    cutoffs = selected[["gameId", "playId", "cutoff_frame_id", "cutoff_event"]]
    selected_frames = discover_frame_windows(paths, cutoffs, window=TIME_STEPS)
    tracking = load_rows_at_selected_frames(paths, selected_frames)
    prepared = enrich_examples_with_games(selected, games, single_season=True)
    prepared["target"] = prepared["passResult"].eq("S").astype(np.int8)
    counts = (
        tracking.groupby(["gameId", "playId", "frameId"]).size()
        .groupby(["gameId", "playId"]).max()
        .rename("tracked_objects").reset_index()
    )
    prepared = prepared.merge(counts, on=["gameId", "playId"], how="left")
    frame_counts = selected_frames.groupby(["gameId", "playId"]).size().rename("history_frames").reset_index()
    prepared = prepared.merge(frame_counts, on=["gameId", "playId"], how="left")
    prepared["left_pad_frames"] = TIME_STEPS - prepared["history_frames"].astype(int)
    tabular = causal_tabular_frame(prepared, TABULAR_COLUMNS)
    forbidden_present = set(tabular.columns).intersection(FORBIDDEN_MODEL_COLUMNS)
    if forbidden_present:
        raise RuntimeError(f"post-cutoff features reached BDB 2023 model inputs: {sorted(forbidden_present)}")
    tokens, player_mask, frame_mask = assemble_player_tensors(
        prepared,
        tracking,
        selected,
        games,
        players,
        time_steps=TIME_STEPS,
        infer_offensive_qb_as_focal=True,
    )
    example_columns = (
        "example_id",
        "game_id",
        "play_id",
        "stratum",
        "season",
        "week",
        "target",
        "passResult",
        "cutoff_frame_id",
        "cutoff_event",
        "tracked_objects",
        "history_frames",
        "left_pad_frames",
    )
    examples = prepared.loc[:, example_columns].rename(columns={"passResult": "outcome_label"})
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="binary",
        primary_metric="brier",
        examples=examples,
        tabular=tabular,
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CANONICAL_PLAYER_CHANNELS,
        y=prepared["target"].to_numpy(dtype=np.int8),
        audit=audit.as_dict(),
        metadata={
            "cutoff": "earliest ball_snap/autoevent_ballsnap frame",
            "history_frames": TIME_STEPS,
            "padding": "left, masked",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
            "graph": "typed pass-protection and coverage relations inferred from causal positions",
            "structural_ablation": "collapse_typed_edges",
            "team_identity_primary": "excluded",
            "positive_label": "S",
            "scramble_label": 0,
            "tabular_columns": list(TABULAR_COLUMNS),
            "forbidden_model_columns": list(FORBIDDEN_MODEL_COLUMNS),
            "pff_scouting_loaded": False,
            "learned_preprocessing": "split-local only",
        },
    )


prepare = prepare_bdb2023
