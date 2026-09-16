from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters import bdb2025
from bdb_study.prepared import build_prepared_semantic_receipt


def _frame(play: int, frame: int, frame_type: str) -> list[dict]:
    common = {
        "gameId": 2022000001,
        "playId": play,
        "frameId": frame,
        "frameType": frame_type,
        "event": "ball_snap" if frame_type == "SNAP" else None,
        "playDirection": "right",
    }
    return [
        {**common, "nflId": np.nan, "displayName": "football", "club": "football", "x": 40.0, "y": 26.0, "s": 0.0, "a": 0.0, "o": np.nan, "dir": np.nan},
        {**common, "nflId": 10, "displayName": "Quarterback", "club": "AAA", "x": 35.0 + frame / 10, "y": 26.0, "s": 0.2, "a": 0.1, "o": 90.0, "dir": 90.0},
        {**common, "nflId": 20, "displayName": "Corner", "club": "BBB", "x": 45.0, "y": 28.0, "s": 0.3, "a": 0.1, "o": 270.0, "dir": 270.0},
    ]


def _fixture(root: Path) -> None:
    pd.DataFrame(
        [{"gameId": 2022000001, "season": 2022, "week": 1, "homeTeamAbbr": "AAA", "visitorTeamAbbr": "BBB"}]
    ).to_csv(root / "games.csv", index=False)
    pd.DataFrame([{"nflId": 10, "position": "QB"}, {"nflId": 20, "position": "CB"}]).to_csv(
        root / "players.csv", index=False
    )
    pd.DataFrame(
        [
            {"gameId": 2022000001, "playId": 1, "isDropback": True, "pff_manZone": "Man", "possessionTeam": "AAA", "defensiveTeam": "BBB", "pff_passCoverage": "Cover-1", "timeToThrow": 3.0},
            {"gameId": 2022000001, "playId": 2, "isDropback": True, "pff_manZone": "Zone", "possessionTeam": "AAA", "defensiveTeam": "BBB", "pff_passCoverage": "Cover-3", "timeToThrow": 2.5},
            {"gameId": 2022000001, "playId": 3, "isDropback": True, "pff_manZone": "Other", "possessionTeam": "AAA", "defensiveTeam": "BBB"},
            {"gameId": 2022000001, "playId": 4, "isDropback": False, "pff_manZone": "Man", "possessionTeam": "AAA", "defensiveTeam": "BBB"},
        ]
    ).to_csv(root / "plays.csv", index=False)
    rows = []
    for play in (1, 2, 3, 4):
        for frame in range(1, 5):
            rows += _frame(play, frame, "SNAP" if frame == 4 else "BEFORE_SNAP")
    pd.DataFrame(rows).to_csv(root / "tracking_week_1.csv", index=False)


class BDB2025AdapterTests(unittest.TestCase):
    def test_man_zone_uses_left_padded_presnap_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            audit = bdb2025.audit_cohort(root)
            self.assertEqual(audit.outcome_eligible, 2)
            self.assertEqual(audit.retained_examples, 2)
            bundle = bdb2025.prepare(root, max_examples=10)
            self.assertEqual(bundle.y.tolist(), [1, 0])
            self.assertEqual(bundle.examples["outcome_label"].tolist(), ["Man", "Zone"])
            self.assertEqual(bundle.player_tokens.shape[:3], (2, 20, 23))
            self.assertTrue(np.all(bundle.frame_mask[:, :16] == 0))
            self.assertTrue(np.all(bundle.frame_mask[:, 16:] == 1))
            self.assertEqual(bundle.examples["left_pad_frames"].tolist(), [16, 16])
            self.assertTrue(set(bundle.tabular).isdisjoint(bdb2025.FORBIDDEN_MODEL_COLUMNS))
            focal = bundle.player_tokens[..., bundle.channel_names.index("focal")] > 0.5
            quarterback = (
                bundle.player_tokens[..., bundle.channel_names.index("position_qb")] > 0.5
            )
            np.testing.assert_array_equal(focal.sum(axis=2), bundle.frame_mask.astype(int))
            np.testing.assert_array_equal(focal, quarterback & bundle.player_mask)
            receipt = build_prepared_semantic_receipt(bundle)
            self.assertTrue(receipt["adapter_metadata"]["infer_offensive_qb_as_focal"])
            self.assertEqual(
                receipt["adapter_metadata"],
                bdb2025.task_spec().prepared_contract["adapter_metadata"],
            )

    def test_man_zone_task_spec_shares_2022_registry(self) -> None:
        spec = bdb2025.task_spec()
        self.assertEqual(spec.shared_game_registry_with, "bdb2024_tackle")
        self.assertEqual(spec.outcome["positive_label"], "Man")
        self.assertEqual(spec.anchors, (10, 20, 30, 40, 50, 60))


if __name__ == "__main__":
    unittest.main()
