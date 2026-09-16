from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters import bdb2021
from bdb_study.adapters.common import (
    discover_cutoffs,
    discover_frame_windows,
    load_rows_at_selected_frames,
)


def _frame(game: int, play: int, frame: int, event: str | None, direction: str = "right") -> list[dict]:
    common = {"gameId": game, "playId": play, "frameId": frame, "event": event, "playDirection": direction}
    return [
        {**common, "nflId": np.nan, "displayName": "Football", "position": np.nan, "team": "football", "x": 40.0, "y": 26.0, "s": 0.0, "a": 0.0, "o": np.nan, "dir": np.nan},
        {**common, "nflId": 10, "displayName": "Quarterback", "position": "QB", "team": "home", "x": 35.0, "y": 26.0, "s": 1.0, "a": 0.2, "o": 90.0, "dir": 90.0},
        {**common, "nflId": 20, "displayName": "Corner", "position": "CB", "team": "away", "x": 43.0, "y": 29.0, "s": 2.0, "a": 0.1, "o": 270.0, "dir": 270.0},
    ]


def _fixture(root: Path) -> None:
    pd.DataFrame(
        [{"gameId": 2018000001, "week": 1, "homeTeamAbbr": "AAA", "visitorTeamAbbr": "BBB"}]
    ).to_csv(root / "games.csv", index=False)
    pd.DataFrame(
        [
            {"nflId": 10, "officialPosition": "QB"},
            {"nflId": 20, "officialPosition": "CB"},
        ]
    ).to_csv(root / "players.csv", index=False)
    pd.DataFrame(
        [
            {"gameId": 2018000001, "playId": 1, "passResult": "C", "possessionTeam": "AAA", "quarter": 1, "down": 1, "yardsToGo": 10, "absoluteYardlineNumber": 40, "gameClock": "12:00:00", "route": "GO"},
            {"gameId": 2018000001, "playId": 2, "passResult": "I", "possessionTeam": "AAA", "quarter": 1, "down": 2, "yardsToGo": 5, "absoluteYardlineNumber": 50, "gameClock": "11:00:00", "route": "OUT"},
            {"gameId": 2018000001, "playId": 3, "passResult": "IN", "possessionTeam": "AAA"},
            {"gameId": 2018000001, "playId": 4, "passResult": "S", "possessionTeam": "AAA"},
        ]
    ).to_csv(root / "plays.csv", index=False)
    rows = []
    rows += _frame(2018000001, 1, 4, "pass_shovel")
    rows += _frame(2018000001, 1, 5, "pass_forward")
    rows += _frame(2018000001, 2, 3, "pass_forward", direction="left")
    rows += _frame(2018000001, 3, 2, None)
    rows += _frame(2018000001, 4, 2, "pass_forward")
    # A released pass with no football row at the endpoint is excluded rather
    # than receiving an invented coordinate.
    rows += _frame(2018000001, 5, 2, "pass_forward")[1:]
    plays = pd.read_csv(root / "plays.csv")
    plays.loc[len(plays)] = {
        "gameId": 2018000001,
        "playId": 5,
        "passResult": "C",
        "possessionTeam": "AAA",
    }
    plays.to_csv(root / "plays.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "week1.csv", index=False)


class BDB2021AdapterTests(unittest.TestCase):
    def test_completion_cohort_and_earliest_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            audit = bdb2021.audit_cohort(root)
            self.assertEqual(audit.outcome_eligible, 4)
            self.assertEqual(audit.cutoff_eligible, 3)
            self.assertEqual(audit.retained_examples, 2)
            self.assertEqual(audit.exclusions["missing_pass_forward_or_pass_shovel"], 1)
            self.assertEqual(
                audit.exclusions["release_frame_missing_valid_football_coordinate"], 1
            )
            self.assertEqual(
                audit.details["tracking_quality_excluded_play_keys"],
                ["2018000001:5"],
            )

            bundle = bdb2021.prepare(root, max_examples=10)
            self.assertEqual(bundle.task_id, "bdb2021_completion")
            self.assertEqual(bundle.y.tolist(), [1, 0])
            self.assertEqual(bundle.examples["cutoff_frame_id"].tolist(), [4, 3])
            self.assertEqual(bundle.player_tokens.shape, (2, 20, 23, len(bundle.channel_names)))
            self.assertTrue(np.all(bundle.frame_mask[:, :-1] == 0))
            self.assertTrue(bundle.frame_mask[:, -1].all())
            self.assertNotIn("route", bundle.tabular.columns)
            self.assertNotIn("passResult", bundle.tabular.columns)
            self.assertEqual(
                bundle.metadata["tracking_quality_exclusions"],
                [
                    {
                        "game_id": 2018000001,
                        "play_id": 5,
                        "reason": "football unavailable at the release frame",
                    }
                ],
            )
            focal = bundle.channel_names.index("focal")
            self.assertTrue(np.all(bundle.player_tokens[:, -1, :, focal].sum(axis=1) == 1.0))

    def test_football_cutoff_filter_is_chunk_invariant_and_uses_earliest_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week1.csv"

            def row(
                play_id: int,
                frame_id: int,
                *,
                event: str | None,
                display_name: str,
                team: str,
                x: float,
                y: float,
            ) -> dict:
                return {
                    "gameId": 1,
                    "playId": play_id,
                    "frameId": frame_id,
                    "event": event,
                    "displayName": display_name,
                    "team": team,
                    "x": x,
                    "y": y,
                }

            rows = [
                # The release marker and football are deliberately separate
                # rows, and therefore separate chunks when chunksize=1.
                row(1, 4, event="pass_forward", display_name="QB", team="home", x=30.0, y=20.0),
                row(1, 4, event=None, display_name="Football", team="football", x=31.0, y=20.0),
                # A later valid release must not rescue an invalid earliest
                # release, because the endpoint is frozen as the earliest one.
                row(2, 5, event="pass_forward", display_name="QB", team="home", x=32.0, y=20.0),
                row(2, 6, event="pass_forward", display_name="QB", team="home", x=33.0, y=20.0),
                row(2, 6, event=None, display_name="Football", team="football", x=34.0, y=20.0),
                # A football identity with a missing coordinate is invalid.
                row(3, 7, event="pass_shovel", display_name="QB", team="home", x=35.0, y=20.0),
                row(3, 7, event=None, display_name="Football", team="football", x=np.nan, y=20.0),
                row(4, 3, event=None, display_name="Football", team="football", x=36.0, y=20.0),
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            eligible = pd.DataFrame({"gameId": [1, 1, 1, 1], "playId": [1, 2, 3, 4]})

            signatures = []
            for chunksize in (1, 2, 3, 500_000):
                cutoffs, audit = discover_cutoffs(
                    (path,),
                    eligible,
                    event_names=("pass_forward", "pass_shovel"),
                    require_valid_football_coordinate=True,
                    chunksize=chunksize,
                )
                signatures.append(
                    (
                        cutoffs.to_dict("records"),
                        audit.sort_values(["gameId", "playId"]).to_dict("records"),
                    )
                )

            self.assertTrue(all(signature == signatures[0] for signature in signatures[1:]))
            self.assertEqual(
                signatures[0][0],
                [
                    {
                        "gameId": 1,
                        "playId": 1,
                        "cutoff_frame_id": 4,
                        "cutoff_event": "pass_forward",
                    }
                ],
            )
            audit_by_play = {row["playId"]: row for row in signatures[0][1]}
            self.assertFalse(audit_by_play[2]["cutoff_has_valid_football"])
            self.assertFalse(audit_by_play[3]["cutoff_has_valid_football"])
            self.assertEqual(audit_by_play[2]["cutoff_frame_count"], 2)
            self.assertEqual(audit_by_play[4]["cutoff_frame_count"], 0)

    def test_twenty_frame_window_never_loads_post_release_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week1.csv"
            rows = []
            for frame_id in range(1, 26):
                rows.extend(
                    [
                        {
                            "gameId": 1,
                            "playId": 9,
                            "frameId": frame_id,
                            "nflId": np.nan,
                            "displayName": "Football",
                            "team": "football",
                            "x": float(frame_id),
                            "y": 20.0,
                        },
                        {
                            "gameId": 1,
                            "playId": 9,
                            "frameId": frame_id,
                            "nflId": 10,
                            "displayName": "QB",
                            "team": "home",
                            "x": float(frame_id + 1),
                            "y": 20.0,
                        },
                    ]
                )
            pd.DataFrame(rows).to_csv(path, index=False)
            cutoffs = pd.DataFrame(
                {
                    "gameId": [1],
                    "playId": [9],
                    "cutoff_frame_id": [22],
                }
            )

            selected = discover_frame_windows((path,), cutoffs, window=20, chunksize=3)
            self.assertEqual(selected["frameId"].tolist(), list(range(3, 23)))
            self.assertEqual(selected["time_index"].tolist(), list(range(20)))
            loaded = load_rows_at_selected_frames((path,), selected, chunksize=4)
            self.assertEqual(sorted(loaded["frameId"].unique()), list(range(3, 23)))
            self.assertFalse(loaded["frameId"].isin([23, 24, 25]).any())

    def test_completion_task_spec_is_preimport_valid(self) -> None:
        spec = bdb2021.task_spec()
        self.assertEqual(spec.total_games, 253)
        self.assertEqual(spec.outer_counts, {"train": 134, "calibration": 44, "test": 45})
        self.assertEqual(spec.anchors, (10, 20, 40, 60, 100, 130))
        expected_exclusion = [
            {
                "game_id": 2018101404,
                "play_id": 2003,
                "reason": "football unavailable at the release frame",
            }
        ]
        self.assertEqual(spec.cohort["tracking_quality_exclusions"], expected_exclusion)
        self.assertEqual(
            spec.prepared_contract["adapter_metadata"]["tracking_quality_exclusions"],
            expected_exclusion,
        )


if __name__ == "__main__":
    unittest.main()
