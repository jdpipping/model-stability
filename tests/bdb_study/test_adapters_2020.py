from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters import bdb2020
from bdb_study.adapters.common import CANONICAL_PLAYER_CHANNELS, FIELD_WIDTH
from bdb_study.prepared import build_prepared_semantic_receipt


def _play(
    game: int,
    play: int,
    *,
    direction: str = "right",
    season: int = 2018,
    yards: int = 4,
    speed: float = 3.0,
    displacement: float = 0.3,
) -> list[dict]:
    rows = []
    for slot in range(22):
        team = "home" if slot < 11 else "away"
        x_right = 50.0 + slot / 2.0
        y_right = 10.0 + slot
        mirrored = direction == "left"
        rows.append(
            {
                "GameId": game,
                "PlayId": play,
                "Team": team,
                "X": 120.0 - x_right if mirrored else x_right,
                "Y": FIELD_WIDTH - y_right if mirrored else y_right,
                "S": speed,
                "A": 0.2,
                "Dis": displacement,
                "Orientation": 270.0 if mirrored else 90.0,
                "Dir": (270.0 if mirrored else 90.0) if team == "home" else (90.0 if mirrored else 270.0),
                "NflId": game + 100 + slot,
                "Season": season,
                "YardLine": 30,
                "Quarter": 2,
                "GameClock": "08:15",
                "PossessionTeam": "AAA",
                "Down": 2,
                "Distance": 7,
                "FieldPosition": "AAA",
                "HomeScoreBeforePlay": 14,
                "VisitorScoreBeforePlay": 10,
                "NflIdRusher": game + 100,
                "OffenseFormation": "SINGLEBACK",
                "OffensePersonnel": "1 RB, 1 TE, 3 WR",
                "DefendersInTheBox": 7,
                "DefensePersonnel": "4 DL, 3 LB, 4 DB",
                "PlayDirection": direction,
                "TimeHandoff": "2019-01-01T00:00:00.000Z",
                "Yards": yards,
                "Position": "RB" if slot == 0 else ("G" if slot < 11 else "CB"),
                "Week": 1,
            }
        )
    return rows


def _write(root: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


class BDB2020AdapterTests(unittest.TestCase):
    def test_handoff_snapshot_is_direction_normalized_clipped_and_identity_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _play(2018000001, 10, yards=-40)
            rows += _play(2018000002, 20, direction="left", yards=80)
            _write(root, rows)
            bundle = bdb2020.prepare(root, max_examples=10)

            self.assertEqual(bundle.task_id, "bdb2020_rushing_harmonized")
            self.assertEqual(bundle.player_tokens.shape, (2, 1, 23, 24))
            self.assertEqual(tuple(bundle.channel_names), CANONICAL_PLAYER_CHANNELS)
            self.assertTrue(bundle.frame_mask.all())
            self.assertTrue((bundle.player_mask.sum(axis=2) == 22).all())
            self.assertTrue((bundle.player_tokens[:, :, 22, :] == 0.0).all())
            channel = {name: bundle.channel_names.index(name) for name in bundle.channel_names}
            self.assertTrue((bundle.player_tokens[..., channel["football"]] == 0.0).all())
            self.assertTrue((bundle.player_tokens[..., channel["focal"]].sum(axis=2) == 1.0).all())
            self.assertTrue((bundle.player_tokens[..., channel["offense"]].sum(axis=2) == 11.0).all())
            self.assertTrue((bundle.player_tokens[..., channel["defense"]].sum(axis=2) == 11.0).all())
            focal = bundle.player_tokens[..., channel["focal"]] > 0.5
            np.testing.assert_allclose(bundle.player_tokens[..., channel["x_rel"]][focal], 0.0)
            np.testing.assert_allclose(bundle.player_tokens[..., channel["y_rel"]][focal], 0.0)
            np.testing.assert_allclose(
                bundle.player_tokens[0], bundle.player_tokens[1], atol=1e-6
            )
            self.assertEqual(bundle.examples["raw_target"].tolist(), [-40, 80])
            self.assertEqual(bundle.examples["target"].tolist(), [-28, 51])
            self.assertEqual(bundle.y.tolist(), [0, 79])
            self.assertEqual((bundle.support[0], bundle.support[-1]), (-28.0, 51.0))
            self.assertEqual(tuple(bundle.tabular.columns), bdb2020.TABULAR_COLUMNS)
            self.assertFalse(any("team" in name.lower() for name in bundle.tabular.columns))
            receipt = build_prepared_semantic_receipt(bundle)
            self.assertEqual(receipt["arrays"]["player_tokens"]["shape"], [2, 1, 23, 24])
            self.assertEqual(receipt["adapter_metadata"]["output_head"], "punt_cdf_residual_v1")
            self.assertEqual(receipt["adapter_metadata"]["structural_ablation"], "remove_blocker_defender_edges")

    def test_2017_speed_fix_missing_angles_and_slot_order_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _play(
                2017000001,
                10,
                season=2017,
                speed=99.0,
                displacement=0.3,
            )
            rows[1]["Dir"] = np.nan
            rows[1]["Orientation"] = np.nan
            _write(root, rows)
            first = bdb2020.prepare(root, max_examples=1)
            _write(root, list(reversed(rows)))
            second = bdb2020.prepare(root, max_examples=1)
            np.testing.assert_array_equal(first.player_tokens, second.player_tokens)
            speed = first.channel_names.index("speed")
            vx = first.channel_names.index("vx")
            orientation_sin = first.channel_names.index("orientation_sin")
            orientation_cos = first.channel_names.index("orientation_cos")
            np.testing.assert_allclose(first.player_tokens[0, 0, :22, speed], 3.0)
            self.assertAlmostEqual(float(first.player_tokens[0, 0, 1, vx]), 3.0)
            self.assertEqual(float(first.player_tokens[0, 0, 1, orientation_sin]), 0.0)
            self.assertEqual(float(first.player_tokens[0, 0, 1, orientation_cos]), 0.0)
            self.assertFalse(any("id" in name.lower() for name in first.channel_names))

    def test_structurally_invalid_handoffs_fail_closed(self) -> None:
        base = _play(2018000001, 10)
        cases = {
            "22 player rows": base[:-1],
            "exactly one rusher": [
                {**row, "NflIdRusher": 99999999} for row in base
            ],
            "11 offense and 11 defense": [
                {**row, "Team": "home" if index == 11 else row["Team"]}
                for index, row in enumerate(base)
            ],
            "x must be finite": [
                {**row, "X": np.nan if index == 0 else row["X"]}
                for index, row in enumerate(base)
            ],
        }
        for message, rows in cases.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write(root, rows)
                with self.assertRaisesRegex(ValueError, message):
                    bdb2020.audit_cohort(root)

    def test_frozen_full_cohort_constants(self) -> None:
        self.assertEqual(bdb2020.EXPECTED_EXAMPLES, 31_007)
        self.assertEqual(bdb2020.EXPECTED_GAMES, 688)
        self.assertEqual(len(bdb2020.SUPPORT), 80)


if __name__ == "__main__":
    unittest.main()
