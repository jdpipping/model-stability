from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bdb_study.adapters.bdb2026 import (
    _prepare_bdb2026_frames_reference,
    add_residual_predictions,
    bdb2026_prepare_resource_estimate,
    game_equal_rmse,
    horizon_rmse,
    pooled_coordinate_rmse,
    prepare_bdb2026_frames,
)
from bdb_study.prepared import (
    build_prepared_semantic_receipt,
    load_prepared_task,
    write_prepared_task,
)


def _input_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_id in (1, 2):
        for nfl_id, target, x_value, side, position, role in (
            (10, True, 10.0 + frame_id / 2.0, "Offense", "WR", "Targeted Receiver"),
            (5, False, 15.0, "Defense", "CB", "Defensive Coverage"),
        ):
            rows.append(
                {
                    "game_id": 2023090100,
                    "play_id": 9,
                    "player_to_predict": target,
                    "nfl_id": nfl_id,
                    "frame_id": frame_id,
                    "play_direction": "right",
                    "absolute_yardline_number": 40,
                    "player_name": str(nfl_id),
                    "player_height": "6-0",
                    "player_weight": 200,
                    "player_birth_date": "2000-01-01",
                    "player_position": position,
                    "player_side": side,
                    "player_role": role,
                    "x": x_value,
                    "y": 20.0,
                    "s": 2.0 if target else 0.0,
                    "a": 0.0,
                    "dir": 90.0,
                    "o": 90.0,
                    "num_frames_output": 2,
                    "ball_land_x": 30.0,
                    "ball_land_y": 20.0,
                    "week": 1,
                }
            )
    return pd.DataFrame(rows)


def _output_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [2023090100, 2023090100],
            "play_id": [9, 9],
            "nfl_id": [10, 10],
            "frame_id": [1, 2],
            "x": [11.3, 11.7],
            "y": [20.0, 20.0],
            "week": [1, 1],
        }
    )


class TestBDB2026(unittest.TestCase):
    def test_play_centric_adapter_exactly_matches_rowwise_multi_focal_oracle(self) -> None:
        inputs = _input_rows().assign(player_to_predict=True)
        defense_output = _output_rows().assign(
            nfl_id=5,
            x=[15.0, 15.0],
            y=[20.0, 20.0],
        )
        outputs = pd.concat([_output_rows(), defense_output], ignore_index=True)
        reference = _prepare_bdb2026_frames_reference(inputs, outputs)
        actual = prepare_bdb2026_frames(inputs, outputs)

        pd.testing.assert_frame_equal(actual.examples, reference.examples)
        pd.testing.assert_frame_equal(actual.tabular, reference.tabular)
        self.assertEqual(actual.audit, reference.audit)
        self.assertEqual(actual.metadata, reference.metadata)
        self.assertEqual(actual.channel_names, reference.channel_names)
        for name in (
            "player_tokens",
            "player_mask",
            "frame_mask",
            "target_values",
            "target_mask",
            "target_baseline",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(reference, name)
            )
        self.assertEqual(
            build_prepared_semantic_receipt(actual),
            build_prepared_semantic_receipt(reference),
        )

    def test_large_token_path_is_anonymous_mapped_and_row_iteration_free(self) -> None:
        # The launch path must remain play-vectorized even if a future edit
        # accidentally reintroduces the original pandas loop.
        self.assertNotIn("iterrows", inspect.getsource(prepare_bdb2026_frames))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            prepared = prepare_bdb2026_frames(
                _input_rows(),
                _output_rows(),
                scratch_dir=scratch,
                memmap_threshold_bytes=1,
            )
            self.assertIsInstance(prepared.player_tokens, np.memmap)
            self.assertIsNone(prepared.player_tokens.filename)
            self.assertEqual(list(scratch.iterdir()), [])
            write_prepared_task(prepared, root / "prepared")
            replay = load_prepared_task(root / "prepared")
            np.testing.assert_array_equal(
                replay.player_tokens, prepared.player_tokens
            )

    def test_full_resource_estimate_maps_8_gib_tokens_and_bounds_heap_arrays(self) -> None:
        estimate = bdb2026_prepare_resource_estimate(
            examples=46_045,
            input_frames=123,
            players=17,
            output_horizon=94,
        )
        self.assertEqual(estimate["player_tokens_bytes"], 9_242_889_120)
        self.assertEqual(estimate["prepared_array_bytes"], 9_418_412_660)
        self.assertLess(
            estimate["resident_array_bytes_excluding_mapped_tokens"],
            180 * 1024 * 1024,
        )

    def test_trajectory_adapter_stores_absolute_targets_and_residual_baseline(self) -> None:
        prepared = prepare_bdb2026_frames(_input_rows(), _output_rows())
        self.assertEqual(prepared.task_id, "bdb2026_trajectory")
        self.assertEqual(prepared.examples["nfl_id"].tolist(), [10])
        self.assertEqual(prepared.target_values.shape, (1, 2, 2))
        self.assertEqual(prepared.target_mask.tolist(), [[True, True]])
        # Final input x=11, speed=2 yd/s toward +x. Baseline is 11.2, 11.4.
        np.testing.assert_allclose(prepared.target_baseline[0, :, 0], [11.2, 11.4])
        np.testing.assert_allclose(
            prepared.target_values[0, :, 0], [11.3, 11.7], atol=1e-6
        )
        residual = prepared.target_values - prepared.target_baseline
        np.testing.assert_allclose(residual[0, :, 0], [0.1, 0.3], atol=1e-6)
        reconstructed = add_residual_predictions(prepared.target_baseline, residual)
        np.testing.assert_allclose(reconstructed[0, :, 0], [11.3, 11.7], atol=1e-6)
        self.assertEqual(prepared.player_tokens.shape, (1, 2, 2, 24))
        self.assertTrue(prepared.frame_mask.all())
        self.assertTrue(prepared.player_mask.all())
        self.assertEqual(prepared.tabular.loc[0, "absolute_yardline_number"], 40.0)
        self.assertEqual(
            prepared.metadata["tabular_feature_order"], prepared.tabular.columns.tolist()
        )
        self.assertEqual(
            prepared.audit["training_target"],
            "development_selected_residual_or_absolute_from_stored_absolute_xy_and_baseline",
        )
        self.assertEqual(
            prepared.metadata["graph"],
            "focal-player social graph with horizon-conditioned development-selected decoder",
        )
        self.assertEqual(
            prepared.metadata["output_head"],
            "shared_horizon_conditioned_development_selected_v1",
        )
        semantic = build_prepared_semantic_receipt(prepared)
        self.assertEqual(
            semantic["target_contract"],
            {
                "encoding": "absolute_xy",
                "examples_target": "nan_placeholder",
                "training_target": (
                    "development_selected_residual_or_absolute_from_stored_"
                    "absolute_xy_and_baseline"
                ),
                "model_output": (
                    "masked_xy_in_development_selected_residual_or_absolute_"
                    "coordinates"
                ),
                "coordinate_order": ["x", "y"],
                "reconstruction": (
                    "absolute_identity_or_target_baseline_plus_residual_as_frozen"
                ),
            },
        )
        focal_channel = prepared.channel_names.index("focal")
        self.assertEqual(
            prepared.player_tokens[0, :, 0, focal_channel].tolist(), [1.0, 1.0]
        )

    def test_pooled_and_game_equal_rmse_obey_masks(self) -> None:
        truth = np.asarray(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [99.0, 99.0]],
            ]
        )
        prediction = np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[2.0, 2.0], [-99.0, -99.0]],
            ]
        )
        mask = np.asarray([[True, True], [True, False]])
        # Six valid scalar coordinates: four errors of 1 and two errors of 2.
        self.assertAlmostEqual(
            pooled_coordinate_rmse(truth, prediction, mask), np.sqrt(12.0 / 6.0)
        )
        self.assertAlmostEqual(
            game_equal_rmse(truth, prediction, mask, [1, 2]), (1.0 + 2.0) / 2.0
        )
        np.testing.assert_allclose(
            horizon_rmse(truth, prediction, mask), [np.sqrt(2.5), 1.0]
        )

    def test_output_keys_must_exactly_match_player_to_predict(self) -> None:
        bad = _output_rows().assign(nfl_id=20)
        with self.assertRaisesRegex(ValueError, "player_to_predict keys"):
            prepare_bdb2026_frames(_input_rows(), bad)

    def test_multi_focal_examples_reuse_one_play_global_slot_map(self) -> None:
        inputs = _input_rows().assign(player_to_predict=True)
        defense_output = _output_rows().assign(
            nfl_id=5,
            x=[15.0, 15.0],
            y=[20.0, 20.0],
        )
        prepared = prepare_bdb2026_frames(
            inputs, pd.concat([_output_rows(), defense_output], ignore_index=True)
        )
        self.assertEqual(prepared.examples["nfl_id"].tolist(), [5, 10])
        focal = prepared.channel_names.index("focal")
        offense = prepared.channel_names.index("offense")
        # Offense ID 10 is always slot 0 and defense ID 5 always slot 1.
        self.assertTrue(np.all(prepared.player_tokens[:, :, 0, offense] == 1.0))
        self.assertTrue(np.all(prepared.player_tokens[:, :, 1, offense] == 0.0))
        self.assertTrue(np.all(prepared.player_tokens[0, :, 1, focal] == 1.0))
        self.assertTrue(np.all(prepared.player_tokens[0, :, 0, focal] == 0.0))
        self.assertTrue(np.all(prepared.player_tokens[1, :, 0, focal] == 1.0))
        self.assertTrue(np.all(prepared.player_tokens[1, :, 1, focal] == 0.0))


if __name__ == "__main__":
    unittest.main()
