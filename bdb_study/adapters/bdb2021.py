"""BDB 2021 completion-at-release adapter."""

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


TASK_ID = "bdb2021_completion"
EXPECTED_EXAMPLES = 17_846
EXPECTED_GAMES = 253
RELEASE_EVENTS = ("pass_forward", "pass_shovel")
PASS_RESULTS = ("C", "I", "IN")
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
    "offenseFormation",
    "personnelO",
    "personnelD",
    "defendersInTheBox",
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
    cutoffs, cutoff_audit = discover_cutoffs(
        paths,
        eligible,
        event_names=RELEASE_EVENTS,
        require_valid_football_coordinate=True,
    )
    cohort = eligible.merge(cutoffs, on=["gameId", "playId"], how="inner")
    frame_counts = cutoff_audit["cutoff_frame_count"].value_counts().sort_index()
    invalid_football_cutoffs = cutoff_audit.loc[
        (cutoff_audit["cutoff_frame_count"] > 0)
        & ~cutoff_audit["cutoff_has_valid_football"],
        ["gameId", "playId"],
    ].sort_values(["gameId", "playId"])
    audit = CohortAudit(
        task_id=TASK_ID,
        total_play_rows=len(plays),
        outcome_eligible=len(eligible),
        cutoff_eligible=int((cutoff_audit["cutoff_frame_count"] > 0).sum()),
        retained_examples=len(cohort),
        retained_games=cohort["gameId"].nunique(),
        exclusions={
            "pass_result_not_C_I_IN": len(plays) - len(eligible),
            "missing_pass_forward_or_pass_shovel": int((cutoff_audit["cutoff_frame_count"] == 0).sum()),
            "release_frame_missing_valid_football_coordinate": int(
                (
                    (cutoff_audit["cutoff_frame_count"] > 0)
                    & ~cutoff_audit["cutoff_has_valid_football"]
                ).sum()
            ),
        },
        details={
            "release_events": list(RELEASE_EVENTS),
            "cutoff_rule": "earliest observed release event",
            "tracking_quality_rule": (
                "require a valid football coordinate at the release frame; "
                "exclude rather than impute"
            ),
            "tracking_quality_excluded_play_keys": [
                f"{int(row.gameId)}:{int(row.playId)}"
                for row in invalid_football_cutoffs.itertuples(index=False)
            ],
            "plays_by_distinct_release_frames": {str(int(key)): int(value) for key, value in frame_counts.items()},
            "expected_examples": EXPECTED_EXAMPLES,
            "expected_games": EXPECTED_GAMES,
        },
    )
    return plays, games, players, paths, cohort, audit


def audit_cohort(raw_dir: Path) -> CohortAudit:
    *_, audit = _cohort(Path(raw_dir))
    return audit


def prepare_bdb2021(
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
        raise ValueError("BDB 2021 selection contains no completion examples")
    cutoffs = selected[["gameId", "playId", "cutoff_frame_id", "cutoff_event"]]
    selected_frames = discover_frame_windows(paths, cutoffs, window=TIME_STEPS)
    tracking = load_rows_at_selected_frames(paths, selected_frames)
    prepared = enrich_examples_with_games(selected, games, single_season=True)
    prepared["target"] = prepared["passResult"].eq("C").astype(np.int8)
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
            "cutoff": "first pass_forward/pass_shovel frame",
            "history_frames": TIME_STEPS,
            "padding": "left, masked",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nflId alignment only and never encoded"
            ),
            "graph": "quarterback, eligible-receiver, and defender relations",
            "structural_ablation": "release_snapshot_only",
            "team_identity_primary": "excluded",
            "positive_label": "C",
            "tabular_columns": list(TABULAR_COLUMNS),
            "learned_preprocessing": "split-local only",
            "tracking_quality": (
                "require valid football coordinate at release; no coordinate imputation"
            ),
            "tracking_quality_exclusions": [
                {
                    "game_id": int(key.split(":", 1)[0]),
                    "play_id": int(key.split(":", 1)[1]),
                    "reason": "football unavailable at the release frame",
                }
                for key in audit.details["tracking_quality_excluded_play_keys"]
            ],
        },
    )


prepare = prepare_bdb2021
