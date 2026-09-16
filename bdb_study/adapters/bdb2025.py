"""BDB 2025 pre-snap Man-versus-Zone adapter."""

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


TASK_ID = "bdb2025_man_zone"
EXPECTED_EXAMPLES = 9_229
EXPECTED_GAMES = 136
TIME_STEPS = 20
TABULAR_COLUMNS = (
    "quarter",
    "down",
    "yardsToGo",
    "absoluteYardlineNumber",
    "gameClockSeconds",
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "preSnapHomeTeamWinProbability",
    "preSnapVisitorTeamWinProbability",
    "expectedPoints",
    "possessionTeam",
    "defensiveTeam",
    "offenseFormation",
    "receiverAlignment",
    "playClockAtSnap",
)
FORBIDDEN_MODEL_COLUMNS = (
    "pff_manZone",
    "pff_passCoverage",
    "passResult",
    "passLength",
    "targetX",
    "targetY",
    "playAction",
    "dropbackType",
    "dropbackDistance",
    "passLocationType",
    "timeToThrow",
    "timeInTackleBox",
    "timeToSack",
    "passTippedAtLine",
    "unblockedPressure",
    "rushLocationType",
    "yardsGained",
    "expectedPointsAdded",
    "pff_runConceptPrimary",
    "pff_runConceptSecondary",
    "pff_runPassOption",
)


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _dropback_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(("true", "1", "1.0"))


def _cohort(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[Path, ...], pd.DataFrame, CohortAudit]:
    root = Path(raw_dir)
    plays = pd.read_csv(root / "plays.csv", low_memory=False)
    games = pd.read_csv(root / "games.csv", low_memory=False)
    players = pd.read_csv(root / "players.csv", low_memory=False)
    paths = weekly_tracking_paths(root, "tracking_week_*.csv")
    eligible = plays.loc[_dropback_mask(plays["isDropback"]) & plays["pff_manZone"].isin(("Man", "Zone"))].copy()
    cutoffs, cutoff_audit = discover_cutoffs(paths, eligible, frame_types=("SNAP",))
    cohort = eligible.merge(cutoffs, on=["gameId", "playId"], how="inner")
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=len(plays),
        outcome_eligible=len(eligible),
        cutoff_eligible=len(cohort),
        retained_examples=len(cohort),
        retained_games=cohort["gameId"].nunique(),
        exclusions={
            "not_labeled_dropback_man_or_zone": len(plays) - len(eligible),
            "missing_snap_frame": int(cutoff_audit["cutoff_frame_count"].eq(0).sum()),
        },
        details={
            "cutoff_rule": "frameType SNAP",
            "history_frames": TIME_STEPS,
            "left_padding": True,
            "positive_label": "Man",
            "forbidden_model_columns": list(FORBIDDEN_MODEL_COLUMNS),
            "expected_examples": EXPECTED_EXAMPLES,
            "expected_games": EXPECTED_GAMES,
        },
    )
    return plays, games, players, paths, cohort, audit


def audit_cohort(raw_dir: Path) -> CohortAudit:
    *_, audit = _cohort(Path(raw_dir))
    return audit


def prepare_bdb2025(
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
        raise ValueError("BDB 2025 selection contains no labeled dropbacks")
    cutoffs = selected[["gameId", "playId", "cutoff_frame_id", "cutoff_event"]]
    selected_frames = discover_frame_windows(paths, cutoffs, window=TIME_STEPS)
    tracking = load_rows_at_selected_frames(paths, selected_frames)
    prepared = enrich_examples_with_games(selected, games, single_season=True)
    prepared["target"] = prepared["pff_manZone"].eq("Man").astype(np.int8)
    frame_counts = selected_frames.groupby(["gameId", "playId"]).size().rename("history_frames").reset_index()
    prepared = prepared.merge(frame_counts, on=["gameId", "playId"], how="left")
    prepared["left_pad_frames"] = TIME_STEPS - prepared["history_frames"].astype(int)
    tabular = causal_tabular_frame(prepared, TABULAR_COLUMNS)
    forbidden_present = set(tabular.columns).intersection(FORBIDDEN_MODEL_COLUMNS)
    if forbidden_present:
        raise RuntimeError(f"post-snap features reached BDB 2025 model inputs: {sorted(forbidden_present)}")
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
        "pff_manZone",
        "cutoff_frame_id",
        "history_frames",
        "left_pad_frames",
    )
    examples = prepared.loc[:, example_columns].rename(columns={"pff_manZone": "outcome_label"})
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
            "cutoff": "SNAP frame",
            "history_frames": TIME_STEPS,
            "padding": "left, masked",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
            "graph": "identity-preserving quarterback, receiver, and defender temporal relations",
            "infer_offensive_qb_as_focal": True,
            "time_to_snap": "derived from the masked 10 Hz frame index",
            "team_identity_primary": "excluded",
            "team_identity_sensitivity": "prespecified include_team_identity",
            "structural_ablation": "final_snapshot_only",
            "positive_label": "Man",
            "tabular_columns": list(TABULAR_COLUMNS),
            "forbidden_model_columns": list(FORBIDDEN_MODEL_COLUMNS),
            "learned_preprocessing": "split-local only",
        },
    )


prepare = prepare_bdb2025
