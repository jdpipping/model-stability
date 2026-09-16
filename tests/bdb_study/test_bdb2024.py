from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters.bdb2024 import _candidate_labels, prepare_bdb2024_frames
from bdb_study.contracts import load_task_spec
from bdb_study.fidelity.bdb2024 import (
    ExecutedXGBoostFit,
    FEATURE_NAMES,
    executed_xgboost_parameters,
    fidelity_receipt,
    infer_candidate_event_frame,
    opportunity_state_machine,
    persist_reference_artifacts,
    predict_fidelity_score,
    process_framewise_tracking_week,
    quantize_winner_geometry,
    rotate_direction_and_orientation,
    strict_penalty_play_mask,
    validate_fidelity_counts,
    winner_feature_matrix_for_defenders,
    winner_feature_vector,
)


def _synthetic_release() -> tuple[pd.DataFrame, ...]:
    games = pd.DataFrame(
        {
            "gameId": [2022090100],
            "season": [2022],
            "week": [1],
            "homeTeamAbbr": ["O"],
            "visitorTeamAbbr": ["D"],
        }
    )
    plays = pd.DataFrame(
        {
            "gameId": [2022090100],
            "playId": [7],
            "ballCarrierId": [1],
            "possessionTeam": ["O"],
            "defensiveTeam": ["D"],
            "passResult": [np.nan],
            "foulName1": [np.nan],
            "playNullifiedByPenalty": ["N"],
        }
    )
    players = pd.DataFrame(
        {
            "nflId": [1, 2, 3, 4],
            "displayName": ["carrier", "blocker", "tackler", "defender"],
            "position": ["RB", "WR", "ILB", "CB"],
        }
    )
    tackles = pd.DataFrame(
        {
            "gameId": [2022090100],
            "playId": [7],
            "nflId": [3],
            "tackle": [1],
            "assist": [0],
            "forcedFumble": [0],
            "pff_missedTackle": [0],
        }
    )
    positions = {
        1: (30.0, 20.0),
        2: (31.0, 25.0),
        3: (25.0, 20.0),
        4: (25.0, 26.0),
    }
    names = players.set_index("nflId")["displayName"]
    rows: list[dict[str, object]] = []
    for frame_id, event in ((1, "ball_snap"), (10, ""), (20, "tackle")):
        for nfl_id, (x_value, y_value) in positions.items():
            rows.append(
                {
                    "gameId": 2022090100,
                    "playId": 7,
                    "nflId": nfl_id,
                    "displayName": names.loc[nfl_id],
                    "frameId": frame_id,
                    "time": "",
                    "jerseyNumber": nfl_id,
                    "club": "O" if nfl_id in (1, 2) else "D",
                    "playDirection": "right",
                    "x": x_value,
                    "y": y_value,
                    "s": 1.0 if nfl_id == 1 else 2.0,
                    "a": 0.0,
                    "dis": 0.0,
                    "o": 90.0,
                    "dir": 90.0,
                    "event": event,
                    "week": 1,
                }
            )
        rows.append(
            {
                "gameId": 2022090100,
                "playId": 7,
                "nflId": np.nan,
                "displayName": "football",
                "frameId": frame_id,
                "time": "",
                "jerseyNumber": np.nan,
                "club": "football",
                "playDirection": "right",
                "x": 30.0,
                "y": 20.0,
                "s": 0.0,
                "a": 0.0,
                "dis": 0.0,
                "o": np.nan,
                "dir": np.nan,
                "event": event,
                "week": 1,
            }
        )
    return games, plays, players, tackles, pd.DataFrame(rows)


def _framewise_release() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plays = pd.DataFrame(
        {
            "gameId": [2022090100],
            "playId": [7],
            "ballCarrierId": [1],
            "possessionTeam": ["O"],
            "defensiveTeam": ["D"],
            # The lake preserves this literal; winner read_csv parsed it null.
            "passResult": ["NA"],
            "foulName1": ["NA"],
            "playNullifiedByPenalty": ["N"],
        }
    )
    players = pd.DataFrame({"nflId": [1, 2, 3, 4]})
    positions = {
        1: (30.0, 20.0),
        2: (31.0, 25.0),
        3: (25.0, 20.0),
        4: (25.0, 26.0),
    }
    rows: list[dict[str, object]] = []
    for frame_id in range(1, 16):
        event = "ball_snap" if frame_id == 1 else "tackle" if frame_id == 15 else ""
        for nfl_id, (x_value, y_value) in positions.items():
            rows.append(
                {
                    "gameId": 2022090100,
                    "playId": 7,
                    "nflId": nfl_id,
                    "frameId": frame_id,
                    "club": "O" if nfl_id in (1, 2) else "D",
                    "playDirection": "right",
                    "x": x_value,
                    "y": y_value,
                    "s": 1.0 if nfl_id == 1 else 2.0,
                    "a": 0.0,
                    "o": 90.0,
                    "dir": 90.0,
                    "event": event,
                }
            )
        rows.append(
            {
                "gameId": 2022090100,
                "playId": 7,
                "nflId": np.nan,
                "frameId": frame_id,
                "club": "football",
                "playDirection": "right",
                "x": 30.0,
                "y": 20.0,
                "s": 0.0,
                "a": 0.0,
                "o": np.nan,
                "dir": np.nan,
                "event": event,
            }
        )
    return plays, players, pd.DataFrame(rows)


class _LazyFidelityModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def save_model(self, path: str) -> None:
        Path(path).write_bytes(b"synthetic-xgboost-150-tree-artifact\n")

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(len(values))
        # Frame-major input has two defenders. Frames 6--10 are high and
        # frames 11--15 are low, creating exact events at frames 10 and 15.
        positive = np.concatenate(
            [np.full(len(values) // 2, 0.8), np.full(len(values) - len(values) // 2, 0.2)]
        )
        return np.column_stack([1.0 - positive, positive])


class TestBDB2024(unittest.TestCase):
    def test_task_contract_binds_current_executable_fidelity_receipt(self) -> None:
        root = Path(__file__).resolve().parents[2]
        spec = load_task_spec(
            root / "configs" / "bdb_suite" / "tasks" / "bdb2024_tackle.json",
            repo_root=root,
            require_source=False,
        )
        self.assertEqual(
            spec.prepared_contract["adapter_metadata"]["winner_fidelity"],
            fidelity_receipt(),
        )

    def test_fidelity_receipt_records_executed_not_appendix_tree_count(self) -> None:
        receipt = fidelity_receipt()
        self.assertEqual(
            receipt["commit"], "8b3de97f1e42351d14e5b69d8cb03f51f244806a"
        )
        self.assertEqual(receipt["executed_notebook_model"]["n_estimators"], 150)
        self.assertEqual(receipt["submission_appendix_model"]["n_estimators"], 250)
        self.assertEqual(tuple(receipt["feature_names"]), FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 9)
        parameters = executed_xgboost_parameters([0, 0, 1, 1, 1])
        self.assertEqual(parameters["n_estimators"], 150)
        self.assertAlmostEqual(parameters["scale_pos_weight"], 2.0 / 3.0)

        class _ReferenceStub:
            @staticmethod
            def predict_proba(values: np.ndarray) -> np.ndarray:
                return np.tile([0.25, 0.75], (len(values), 1))

        ordered = pd.DataFrame(np.zeros((2, 9)), columns=FEATURE_NAMES)
        np.testing.assert_allclose(
            predict_fidelity_score(_ReferenceStub(), ordered), [0.75, 0.75]
        )
        with self.assertRaisesRegex(ValueError, "exact feature order"):
            predict_fidelity_score(_ReferenceStub(), ordered.iloc[:, ::-1])

    def test_opportunity_machine_is_strict_and_requires_five_up_then_five_down(self) -> None:
        summary = opportunity_state_machine([0.8] * 5 + [0.2] * 5)
        self.assertEqual(summary.opportunities, 1)
        self.assertEqual(summary.missed_opportunities, 1)
        self.assertEqual(summary.opportunity_indices, (4,))
        self.assertEqual(summary.missed_indices, (9,))

        values = [0.75] * 5 + [0.8] * 5 + [0.2] * 4 + [0.8] + [0.2] * 5
        interrupted = opportunity_state_machine(values)
        self.assertEqual(interrupted.opportunities, 1)
        self.assertEqual(interrupted.missed_opportunities, 1)
        gap = opportunity_state_machine(
            [0.8] * 5, frame_ids=[1, 2, 3, 5, 6]
        )
        self.assertEqual(gap.opportunities, 0)

    def test_winner_geometry_reflects_x_and_angles_but_not_y(self) -> None:
        raw = pd.DataFrame(
            {
                "x": [25.0],
                "y": [12.0],
                "s": [3.0],
                "a": [0.5],
                "dir": [90.0],
                "o": [0.0],
                "playDirection": ["left"],
            }
        )
        clean = rotate_direction_and_orientation(raw)
        self.assertAlmostEqual(clean.loc[0, "x_clean"], 95.0)
        self.assertAlmostEqual(clean.loc[0, "y_clean"], 12.0)
        self.assertAlmostEqual(clean.loc[0, "dir_clean"], 180.0)
        self.assertAlmostEqual(clean.loc[0, "o_clean"], 90.0)

        quantized = quantize_winner_geometry(
            clean.assign(x_clean=95.004, y_clean=12.006, s_clean=3.004, a_clean=0.506)
        )
        self.assertAlmostEqual(quantized.loc[0, "x_clean"], 95.0)
        self.assertAlmostEqual(quantized.loc[0, "y_clean"], 12.01)
        self.assertAlmostEqual(quantized.loc[0, "a_clean"], 0.51)

    def test_missed_frame_uses_first_minimum_defender_carrier_distance(self) -> None:
        raw = pd.DataFrame(
            {
                "frameId": [10, 11, 12, 10, 11, 12],
                "nflId": [1, 1, 1, 2, 2, 2],
                "x_clean": [0.0, 1.0, 2.0, 2.0, 1.0, 4.0],
                "y_clean": [0.0] * 6,
            }
        )
        self.assertEqual(
            infer_candidate_event_frame(
                raw, tackler_id=2, ballcarrier_id=1, made=False
            ),
            11,
        )

    def test_fidelity_counts_fail_closed(self) -> None:
        validate_fidelity_counts(8000, 1583, 836, 180)
        with self.assertRaisesRegex(ValueError, "weeks_1_8.made"):
            validate_fidelity_counts(7999, 1583, 836, 180)

    def test_strict_penalty_filter_accepts_csv_nan_and_parquet_na_sentinel(self) -> None:
        plays = pd.DataFrame(
            {
                "foulName1": [np.nan, "NA", "", "Offensive Holding", "NA"],
                "playNullifiedByPenalty": ["N", "N", "N", "N", "Y"],
            }
        )
        self.assertEqual(
            strict_penalty_play_mask(plays).tolist(),
            [True, True, True, False, False],
        )

    def test_candidate_labels_match_notebook_tackle_priority_and_assist_handling(self) -> None:
        tackles = pd.DataFrame(
            {
                "gameId": [1, 1, 1],
                "playId": [1, 2, 3],
                "nflId": [10, 20, 30],
                "tackle": [1, 0, 0],
                "assist": [0, 1, 1],
                "pff_missedTackle": [1, 1, 0],
            }
        )
        candidates, audit = _candidate_labels(tackles)
        self.assertEqual(
            candidates[["playId", "target"]].values.tolist(), [[1, 1], [2, 0]]
        )
        self.assertEqual(audit["conflicting_made_and_missed"], 1)
        self.assertEqual(audit["assist_only_or_non_candidate"], 1)

    def test_tackle_adapter_preserves_exact_feature_order_and_event_offset(self) -> None:
        prepared = prepare_bdb2024_frames(*_synthetic_release())
        self.assertEqual(prepared.task_id, "bdb2024_tackle")
        self.assertEqual(tuple(prepared.tabular.columns), FEATURE_NAMES)
        self.assertEqual(prepared.examples.loc[0, "target"], 1)
        self.assertEqual(prepared.examples.loc[0, "event_frame_id"], 20)
        self.assertEqual(prepared.examples.loc[0, "state_frame_id"], 10)
        self.assertAlmostEqual(prepared.tabular.loc[0, "ballcarrier_speed"], 1.0)
        self.assertAlmostEqual(prepared.tabular.loc[0, "relative_x_speed"], 1.0)
        self.assertAlmostEqual(
            prepared.tabular.loc[0, "absolute_relative_y_speed"], 0.0
        )
        self.assertAlmostEqual(prepared.tabular.loc[0, "euclidean_distance"], 5.0)
        self.assertAlmostEqual(
            prepared.tabular.loc[0, "angle_of_attack_cosine"], 1.0
        )
        self.assertEqual(prepared.player_tokens.shape[1:], (10, 23, 25))
        carrier_channel = prepared.channel_names.index("carrier")
        np.testing.assert_array_equal(
            prepared.player_tokens[0, :, :, carrier_channel].sum(axis=1) > 0,
            prepared.frame_mask[0],
        )
        self.assertEqual(prepared.player_mask[0, -1].sum(), 5)
        self.assertEqual(prepared.examples.loc[0, "history_frames"], 2)
        self.assertTrue(
            prepared.metadata["winner_fidelity"]["score_interpretation"].startswith(
                "case-control"
            )
        )

    def test_multi_candidate_play_reuses_focal_independent_slots(self) -> None:
        games, plays, players, tackles, tracking = _synthetic_release()
        tackles = pd.concat(
            [
                tackles,
                pd.DataFrame(
                    {
                        "gameId": [2022090100],
                        "playId": [7],
                        "nflId": [4],
                        "tackle": [0],
                        "assist": [0],
                        "forcedFumble": [0],
                        "pff_missedTackle": [1],
                    }
                ),
            ],
            ignore_index=True,
        )
        # Make defender 4's unique closest approach occur at frame 20, so its
        # missed-tackle state is also evaluated at frame 10.
        tracking.loc[
            tracking["nflId"].eq(4) & tracking["frameId"].eq(1), "x"
        ] = 10.0
        tracking.loc[
            tracking["nflId"].eq(4) & tracking["frameId"].eq(10), "x"
        ] = 20.0
        tracking.loc[
            tracking["nflId"].eq(4) & tracking["frameId"].eq(20), "x"
        ] = 29.0
        prepared = prepare_bdb2024_frames(
            games, plays, players, tackles, tracking
        )
        self.assertEqual(prepared.examples["nfl_id"].tolist(), [3, 4])
        focal = prepared.channel_names.index("focal")
        carrier = prepared.channel_names.index("carrier")
        # Football, offense IDs 1/2, then defense IDs 3/4 are play-global.
        self.assertEqual(prepared.player_tokens[0, -1, 3, focal], 1.0)
        self.assertEqual(prepared.player_tokens[0, -1, 4, focal], 0.0)
        self.assertEqual(prepared.player_tokens[1, -1, 3, focal], 0.0)
        self.assertEqual(prepared.player_tokens[1, -1, 4, focal], 1.0)
        self.assertEqual(prepared.player_tokens[0, -1, 1, carrier], 1.0)
        self.assertEqual(prepared.player_tokens[1, -1, 1, carrier], 1.0)

    def test_parquet_na_pass_result_matches_winner_csv_run_semantics(self) -> None:
        games, plays, players, tackles, tracking = _synthetic_release()
        plays["passResult"] = pd.Series(["NA"], dtype=object)
        prepared = prepare_bdb2024_frames(games, plays, players, tackles, tracking)
        self.assertEqual(prepared.tabular.loc[0, "is_run"], 1.0)

    def test_shared_frame_features_equal_scalar_winner_features(self) -> None:
        plays, _, tracking = _framewise_release()
        state = tracking.loc[
            tracking["frameId"].eq(6) & tracking["nflId"].notna()
        ].merge(
            plays[
                [
                    "gameId",
                    "playId",
                    "ballCarrierId",
                    "possessionTeam",
                    "defensiveTeam",
                ]
            ],
            on=["gameId", "playId"],
        )
        state = quantize_winner_geometry(rotate_direction_and_orientation(state))
        matrix = winner_feature_matrix_for_defenders(state, is_run=True)
        for row in matrix.itertuples(index=False):
            expected = winner_feature_vector(
                state,
                tackler_id=int(row.nfl_id),
                ballcarrier_id=1,
                is_run=True,
            )
            np.testing.assert_allclose(
                np.asarray(row[1:], dtype=np.float32), expected, rtol=0, atol=1e-6
            )

    def test_all_defender_framewise_pipeline_is_streamed_and_checksummed(self) -> None:
        plays, players, tracking = _framewise_release()
        model = _LazyFidelityModel()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = process_framewise_tracking_week(
                model,
                tracking,
                plays,
                players,
                root,
                week=1,
                source_sha256="a" * 64,
                model_sha256="b" * 64,
                artifact_format="csv",
                chunk_rows=3,
            )
            self.assertEqual(receipt["audit"]["retained_frames"], 10)
            self.assertEqual(receipt["audit"]["scored_defender_frames"], 20)
            self.assertEqual(model.batch_sizes, [20])
            score_record = receipt["artifacts"]["frame_scores"]
            event_record = receipt["artifacts"]["opportunity_summaries"]
            scores_path = root / score_record["path"]
            events_path = root / event_record["path"]
            self.assertEqual(
                hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                score_record["sha256"],
            )
            scores = pd.read_csv(scores_path)
            self.assertEqual(
                tuple(scores.columns),
                tuple(receipt["artifacts"]["frame_scores"]["columns"]),
            )
            self.assertEqual(scores["frame_id"].min(), 6)
            self.assertEqual(scores["frame_id"].max(), 15)
            self.assertTrue(scores["is_run"].eq(1.0).all())
            events = pd.read_csv(events_path)
            self.assertEqual(len(events), 2)
            self.assertTrue(events["opportunities"].eq(1).all())
            self.assertTrue(events["missed_opportunities"].eq(1).all())
            self.assertTrue(events["opportunity_frame_ids_json"].eq("[10]").all())
            self.assertTrue(events["missed_frame_ids_json"].eq("[15]").all())

            before = scores_path.read_bytes()
            again = process_framewise_tracking_week(
                model,
                tracking,
                plays,
                players,
                root,
                week=1,
                source_sha256="a" * 64,
                model_sha256="b" * 64,
                artifact_format="csv",
                chunk_rows=3,
            )
            self.assertEqual(again["receipt_hash"], receipt["receipt_hash"])
            self.assertEqual(scores_path.read_bytes(), before)
            self.assertEqual(model.batch_sizes, [20])
            scores_path.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                process_framewise_tracking_week(
                    model,
                    tracking,
                    plays,
                    players,
                    root,
                    week=1,
                    source_sha256="a" * 64,
                    model_sha256="b" * 64,
                    artifact_format="csv",
                    chunk_rows=3,
                )

    def test_reference_model_and_week9_candidate_scores_are_immutable(self) -> None:
        model = _LazyFidelityModel()
        fitted = ExecutedXGBoostFit(
            model=model,
            train_indices=np.array([0, 2]),
            validation_indices=np.array([1]),
            parameters={"n_estimators": 150, "max_depth": 7},
        )
        examples = pd.DataFrame(
            {
                "example_id": ["1:10:100", "1:11:101"],
                "game_id": [1, 1],
                "play_id": [10, 11],
                "nfl_id": [100, 101],
                "week": [9, 9],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = persist_reference_artifacts(
                fitted,
                examples,
                [0, 1],
                [0.2, 0.8],
                root,
                prepared_hash="c" * 64,
            )
            self.assertEqual(receipt["parameters"]["n_estimators"], 150)
            self.assertEqual(receipt["training"]["configured_tree_budget"], 150)
            self.assertEqual(receipt["week9_examples"], 2)
            model_path = root / receipt["artifacts"]["model"]["path"]
            self.assertEqual(
                hashlib.sha256(model_path.read_bytes()).hexdigest(),
                receipt["artifacts"]["model"]["sha256"],
            )
            scores = pd.read_csv(
                root / receipt["artifacts"]["week9_candidate_scores"]["path"]
            )
            self.assertEqual(scores["case_control_score"].tolist(), [0.2, 0.8])
            again = persist_reference_artifacts(
                fitted,
                examples,
                [0, 1],
                [0.2, 0.8],
                root,
                prepared_hash="c" * 64,
            )
            self.assertEqual(again["receipt_hash"], receipt["receipt_hash"])


if __name__ == "__main__":
    unittest.main()
