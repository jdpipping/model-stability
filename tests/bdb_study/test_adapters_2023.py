from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters import bdb2023


def _frame(play: int, event: str) -> list[dict]:
    common = {"gameId": 2021000001, "playId": play, "frameId": 8, "event": event, "playDirection": "right"}
    return [
        {**common, "nflId": np.nan, "team": "football", "x": 40.0, "y": 26.0, "s": 0.0, "a": 0.0, "o": np.nan, "dir": np.nan},
        {**common, "nflId": 10, "team": "AAA", "x": 35.0, "y": 26.0, "s": 0.5, "a": 0.1, "o": 90.0, "dir": 90.0},
        {**common, "nflId": 20, "team": "BBB", "x": 42.0, "y": 27.0, "s": 1.0, "a": 0.2, "o": 270.0, "dir": 270.0},
    ]


def _fixture(root: Path) -> None:
    pd.DataFrame(
        [{"gameId": 2021000001, "season": 2021, "week": 1, "homeTeamAbbr": "AAA", "visitorTeamAbbr": "BBB"}]
    ).to_csv(root / "games.csv", index=False)
    pd.DataFrame(
        [
            {"nflId": 10, "officialPosition": "QB"},
            {"nflId": 20, "officialPosition": "DE"},
        ]
    ).to_csv(root / "players.csv", index=False)
    pd.DataFrame(
        [
            {"gameId": 2021000001, "playId": 1, "passResult": "C", "possessionTeam": "AAA", "defensiveTeam": "BBB", "pff_playAction": 1, "pff_passCoverage": "Cover-1", "dropBackType": "TRADITIONAL"},
            {"gameId": 2021000001, "playId": 2, "passResult": "R", "possessionTeam": "AAA", "defensiveTeam": "BBB", "pff_playAction": 0, "pff_passCoverage": "Cover-3", "dropBackType": "SCRAMBLE"},
            {"gameId": 2021000001, "playId": 3, "passResult": "S", "possessionTeam": "AAA", "defensiveTeam": "BBB", "pff_playAction": 0, "pff_passCoverage": "Cover-2", "dropBackType": "TRADITIONAL"},
            {"gameId": 2021000001, "playId": 4, "passResult": "X", "possessionTeam": "AAA", "defensiveTeam": "BBB"},
        ]
    ).to_csv(root / "plays.csv", index=False)
    pd.DataFrame([{"gameId": 2021000001, "playId": 3, "nflId": 20, "pff_role": "Pass Rush"}]).to_csv(
        root / "pffScoutingData.csv", index=False
    )
    rows = _frame(1, "ball_snap") + _frame(2, "autoevent_ballsnap") + _frame(3, "ball_snap")
    pd.DataFrame(rows).to_csv(root / "week1.csv", index=False)


class BDB2023AdapterTests(unittest.TestCase):
    def test_sacks_are_snap_only_and_scrambles_are_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            bundle = bdb2023.prepare(root, max_examples=10)
            self.assertEqual(bundle.y.tolist(), [0, 0, 1])
            self.assertEqual(bundle.examples["outcome_label"].tolist(), ["C", "R", "S"])
            self.assertTrue(set(bundle.tabular).isdisjoint(bdb2023.FORBIDDEN_MODEL_COLUMNS))
            self.assertIs(bundle.metadata["pff_scouting_loaded"], False)
            self.assertEqual(bundle.audit["retained_examples"], 3)
            focal = bundle.channel_names.index("focal")
            self.assertEqual(bundle.player_tokens.shape[1], 20)
            self.assertTrue(np.all(bundle.player_tokens[:, -1, :, focal].sum(axis=1) == 1.0))

    def test_sack_task_spec_denies_oracle_columns(self) -> None:
        spec = bdb2023.task_spec()
        self.assertIn("pff_role", spec.features["denylist"])
        self.assertIn("pff_passCoverage", spec.features["denylist"])
        self.assertIn("dropBackType", spec.features["denylist"])
        self.assertEqual(spec.outcome["negative_labels"], ["C", "I", "IN", "R"])


if __name__ == "__main__":
    unittest.main()
