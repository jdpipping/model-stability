from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters import bdb2022
from bdb_study.prepared import build_prepared_semantic_receipt


def _frame(play: int, frame: int, event: str | None) -> list[dict]:
    common = {"gameId": 2018000001, "playId": play, "frameId": frame, "event": event, "playDirection": "right"}
    return [
        {**common, "nflId": np.nan, "displayName": "Football", "position": np.nan, "team": "football", "x": 60.0, "y": 26.0, "s": 0.0, "a": 0.0, "o": np.nan, "dir": np.nan},
        {**common, "nflId": 20, "displayName": "Returner", "position": "WR", "team": "away", "x": 60.5, "y": 26.0, "s": 1.0, "a": 0.1, "o": 90.0, "dir": 90.0},
        {**common, "nflId": 30, "displayName": "Gunner", "position": "CB", "team": "home", "x": 62.0, "y": 27.0, "s": 2.0, "a": 0.2, "o": 270.0, "dir": 270.0},
    ]


def _fixture(root: Path) -> None:
    pd.DataFrame(
        [{"gameId": 2018000001, "season": 2018, "week": 1, "homeTeamAbbr": "AAA", "visitorTeamAbbr": "BBB"}]
    ).to_csv(root / "games.csv", index=False)
    pd.DataFrame(
        [
            {"nflId": 20, "officialPosition": "WR"},
            {"nflId": 30, "officialPosition": "CB"},
        ]
    ).to_csv(root / "players.csv", index=False)
    pd.DataFrame(
        [
            {"gameId": 2018000001, "playId": 1, "specialTeamsPlayType": "Punt", "specialTeamsResult": "Return", "kickReturnYardage": -3, "returnerId": "20;21", "possessionTeam": "AAA", "kickLength": 45},
            {"gameId": 2018000001, "playId": 2, "specialTeamsPlayType": "Punt", "specialTeamsResult": "Return", "kickReturnYardage": 4, "returnerId": "20", "possessionTeam": "AAA", "kickLength": 40},
            {"gameId": 2018000001, "playId": 3, "specialTeamsPlayType": "Punt", "specialTeamsResult": "Fair Catch", "kickReturnYardage": np.nan, "returnerId": "20", "possessionTeam": "AAA"},
        ]
    ).to_csv(root / "plays.csv", index=False)
    rows = _frame(1, 6, "punt_received")
    rows += _frame(2, 5, "punt_received")
    rows += _frame(2, 6, "punt_received")
    pd.DataFrame(rows).to_csv(root / "tracking2018.csv", index=False)


class BDB2022AdapterTests(unittest.TestCase):
    def test_returned_punts_require_unique_reception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            audit = bdb2022.audit_cohort(root)
            self.assertEqual(audit.outcome_eligible, 2)
            self.assertEqual(audit.retained_examples, 1)
            self.assertEqual(audit.exclusions["non_unique_punt_received_frame"], 1)

            bundle = bdb2022.prepare(root, max_examples=10)
            self.assertEqual(bundle.task_id, "bdb2022_punt_returns")
            self.assertEqual(bundle.examples["target"].tolist(), [-3])
            self.assertEqual(bundle.y.tolist(), [17])
            self.assertEqual(bundle.support[0], -20.0)
            self.assertEqual(bundle.support[-1], 110.0)
            offense = bundle.channel_names.index("offense")
            focal = bundle.channel_names.index("focal")
            self.assertEqual(bundle.player_tokens.shape[1], 40)
            self.assertEqual(bundle.player_tokens[0, -1, :, focal].sum(), 1.0)
            self.assertEqual(bundle.player_tokens[0, -1, :, offense].sum(), 1.0)
            self.assertEqual(bundle.examples.loc[0, "returnTeam"], "BBB")

    def test_complete_40_frame_history_emits_all_observed_mask_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            rows = []
            for frame in range(1, 41):
                rows += _frame(
                    1,
                    frame,
                    "punt_received" if frame == 40 else None,
                )
            # Preserve the second play's ambiguous reception so the fixture's
            # retained cohort still consists only of play 1.
            rows += _frame(2, 5, "punt_received")
            rows += _frame(2, 6, "punt_received")
            pd.DataFrame(rows).to_csv(root / "tracking2018.csv", index=False)

            bundle = bdb2022.prepare(root, max_examples=10)
            self.assertTrue(bundle.frame_mask.all())
            semantic = build_prepared_semantic_receipt(bundle)
            self.assertEqual(
                semantic["mask_contract"]["frame_mask"]["layout"],
                "all_observed",
            )

    def test_punt_task_spec_locks_support_and_counts(self) -> None:
        spec = bdb2022.task_spec()
        self.assertEqual(spec.task_id, "bdb2022_punt_returns")
        self.assertEqual(spec.outcome["support"], [-20, 110])
        self.assertEqual(spec.outer_counts, {"train": 373, "calibration": 124, "test": 125})
        self.assertEqual(
            spec.prepared_contract["mask_contract"]["frame_mask"]["layout"],
            "all_observed",
        )
        self.assertEqual(
            spec.prepared_contract["semantic_hash"],
            "4b74413582ea73defb973ee7590ebbfd25c3a32645cbb57f3d6ee76ad285569a",
        )


if __name__ == "__main__":
    unittest.main()
