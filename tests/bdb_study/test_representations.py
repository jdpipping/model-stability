from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bdb_study.adapters.common import (
    CANONICAL_PLAYER_CHANNELS,
    canonicalize_tracking_frame,
    stable_player_slot_map,
)
from bdb_study.models import RELATIONAL_EDGE_VOCAB_SIZE

from bdb_study.representations import (
    EDGE_TYPE_NAMES,
    LazyNeuralInputs,
    MaskedTokenScaler,
    RasterValueScaler,
    TabularEncoder,
    primary_context_frame,
    representative_neural_input_signature,
    representative_shared_neural_input_signature,
    rasterize_tokens,
    relational_edge_types,
    structural_ablation_player_mask,
    token_summary_frame,
)


class RepresentationTests(unittest.TestCase):
    def test_relational_edge_vocabulary_matches_embedding_capacity(self) -> None:
        self.assertEqual(len(EDGE_TYPE_NAMES), RELATIONAL_EDGE_VOCAB_SIZE)
        self.assertEqual(len(set(EDGE_TYPE_NAMES)), RELATIONAL_EDGE_VOCAB_SIZE)

    def test_stable_player_slots_do_not_shift_when_a_player_disappears(self) -> None:
        rows = []
        for frame, players in ((1, [(10, "QB", "H"), (20, "CB", "A")]), (2, [(20, "CB", "A")])):
            rows.append({
                "gameId": 1, "playId": 2, "frameId": frame, "nflId": np.nan,
                "displayName": "football", "team": "football", "playDirection": "right",
                "x": 40.0, "y": 26.0, "s": 0.0, "a": 0.0, "o": np.nan, "dir": np.nan,
            })
            for nfl_id, position, team in players:
                rows.append({
                    "gameId": 1, "playId": 2, "frameId": frame, "nflId": nfl_id,
                    "displayName": str(nfl_id), "position": position, "team": team,
                    "playDirection": "right", "x": 35.0 + nfl_id / 10,
                    "y": 25.0, "s": 1.0, "a": 0.0, "o": 90.0, "dir": 90.0,
                })
        history = pd.DataFrame(rows)
        play, game = {"possessionTeam": "H"}, {"homeTeamAbbr": "H", "visitorTeamAbbr": "A"}
        slots = stable_player_slot_map(
            history, play=play, game=game, focal_nfl_id=10
        )
        first, first_mask = canonicalize_tracking_frame(
            history.loc[history.frameId.eq(1)], play=play, game=game,
            focal_nfl_id=10, stable_slots=slots,
        )
        second, second_mask = canonicalize_tracking_frame(
            history.loc[history.frameId.eq(2)], play=play, game=game,
            focal_nfl_id=10, stable_slots=slots,
        )
        defender_slot = slots[("player", "20")]
        self.assertTrue(first_mask[defender_slot] and second_mask[defender_slot])
        self.assertEqual(first[defender_slot].tolist(), second[defender_slot].tolist())
        self.assertFalse(second_mask[slots[("player", "10")]])

    def test_play_global_slots_are_independent_of_focal_identity(self) -> None:
        rows = []
        for nfl_id, position, team, x_value in (
            (10, "QB", "H", 35.0),
            (20, "CB", "A", 45.0),
        ):
            rows.append(
                {
                    "gameId": 1,
                    "playId": 2,
                    "frameId": 1,
                    "nflId": nfl_id,
                    "displayName": str(nfl_id),
                    "position": position,
                    "team": team,
                    "playDirection": "right",
                    "x": x_value,
                    "y": 25.0,
                    "s": 1.0,
                    "a": 0.0,
                    "o": 90.0,
                    "dir": 90.0,
                }
            )
        rows.append(
            {
                "gameId": 1,
                "playId": 2,
                "frameId": 1,
                "nflId": np.nan,
                "displayName": "football",
                "team": "football",
                "playDirection": "right",
                "x": 40.0,
                "y": 26.0,
                "s": 0.0,
                "a": 0.0,
                "o": np.nan,
                "dir": np.nan,
            }
        )
        history = pd.DataFrame(rows)
        play = {"possessionTeam": "H"}
        game = {"homeTeamAbbr": "H", "visitorTeamAbbr": "A"}
        first_slots = stable_player_slot_map(
            history, play=play, game=game, focal_nfl_id=10
        )
        second_slots = stable_player_slot_map(
            history, play=play, game=game, focal_nfl_id=20
        )
        self.assertEqual(first_slots, second_slots)
        first, _ = canonicalize_tracking_frame(
            history,
            play=play,
            game=game,
            focal_nfl_id=10,
            stable_slots=first_slots,
        )
        second, _ = canonicalize_tracking_frame(
            history,
            play=play,
            game=game,
            focal_nfl_id=20,
            stable_slots=second_slots,
        )
        focal_channel = CANONICAL_PLAYER_CHANNELS.index("focal")
        self.assertEqual(first[first_slots[("player", "10")], focal_channel], 1.0)
        self.assertEqual(first[first_slots[("player", "20")], focal_channel], 0.0)
        self.assertEqual(second[first_slots[("player", "10")], focal_channel], 0.0)
        self.assertEqual(second[first_slots[("player", "20")], focal_channel], 1.0)

    def test_split_local_encoder_and_unknown_category(self) -> None:
        train = pd.DataFrame({"x": [0.0, 2.0], "formation": ["A", "B"]})
        encoder = TabularEncoder().fit(train)
        transformed = encoder.transform(pd.DataFrame({"x": [1000.0], "formation": ["NEW"]}))
        self.assertTrue(np.all(np.isfinite(transformed)))
        # The train-only mean/std are 1/1; a holdout outlier must not change them.
        self.assertAlmostEqual(float(transformed[0, 0]), 999.0)

    def test_token_summary_and_raster_shapes(self) -> None:
        names = [
            "x_rel",
            "y_rel",
            "vx",
            "vy",
            "speed",
            "acceleration",
            "dir_sin",
            "dir_cos",
            "orientation_sin",
            "orientation_cos",
            "offense",
            "defense",
            "football",
            "focal",
        ]
        tokens = np.zeros((2, 2, 3, len(names)), dtype=np.float32)
        mask = np.ones((2, 2, 3), dtype=bool)
        frame_mask = np.ones((2, 2), dtype=bool)
        tokens[:, :, 0, names.index("offense")] = 1.0
        tokens[:, :, 1, names.index("defense")] = 1.0
        tokens[:, :, 2, names.index("football")] = 1.0
        summary = token_summary_frame(tokens, mask, frame_mask, names)
        self.assertEqual(len(summary), 2)
        raster = rasterize_tokens(tokens, mask, names)
        self.assertEqual(raster.shape, (2, 2, 60, 27, 8))
        self.assertGreater(float(raster.sum()), 0.0)

    def test_token_summary_does_not_upcast_the_complete_token_tensor(self) -> None:
        names = ["x_rel", "y_rel", "speed", "acceleration", "offense", "focal"]
        tokens = np.zeros((2, 3, 2, len(names)), dtype=np.float32)
        tokens[..., names.index("offense")] = 1.0
        tokens[:, :, 0, names.index("focal")] = 1.0
        mask = np.ones(tokens.shape[:3], dtype=bool)
        frame_mask = np.ones(tokens.shape[:2], dtype=bool)
        original_asarray = np.asarray

        def guarded_asarray(value, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else None)
            if value is tokens and dtype is not None and np.dtype(dtype) == np.dtype(np.float64):
                raise AssertionError("whole token tensor was upcast")
            return original_asarray(value, *args, **kwargs)

        with patch("bdb_study.representations.np.asarray", side_effect=guarded_asarray):
            summary = token_summary_frame(tokens, mask, frame_mask, names)
        self.assertEqual(len(summary), 2)

    def test_sparse_raster_scaler_matches_dense_raster_moments(self) -> None:
        names = [
            "x_rel", "y_rel", "vx", "vy", "speed", "acceleration",
            "offense", "defense", "football", "focal",
        ]
        tokens = np.zeros((3, 4, 3, len(names)), dtype=np.float32)
        mask = np.zeros(tokens.shape[:3], dtype=bool)
        for example in range(3):
            for frame in range(1, 4):
                mask[example, frame, :2] = True
                # Both players collide in one cell so the sparse path must use
                # the same within-cell kinematic mean as rasterize_tokens.
                tokens[example, frame, :, names.index("x_rel")] = 2.1 + example
                tokens[example, frame, :, names.index("y_rel")] = -1.1 + frame
                tokens[example, frame, 0, names.index("offense")] = 1.0
                tokens[example, frame, 1, names.index("defense")] = 1.0
                tokens[example, frame, 0, names.index("focal")] = 1.0
                tokens[example, frame, 0, names.index("vx")] = example + frame
                tokens[example, frame, 1, names.index("vx")] = example + frame + 2
                tokens[example, frame, :, names.index("vy")] = frame / 2
                tokens[example, frame, :, names.index("speed")] = frame + 1
                tokens[example, frame, :, names.index("acceleration")] = example / 3
        dense = RasterValueScaler().fit(rasterize_tokens(tokens, mask, names))
        sparse = RasterValueScaler().fit_tokens(tokens, mask, names, batch_size=2)
        indexed = RasterValueScaler().fit_indexed_tokens(
            tokens, mask, np.arange(len(tokens)), names, batch_size=1
        )
        np.testing.assert_allclose(sparse.means, dense.means, rtol=0, atol=1e-7)
        np.testing.assert_allclose(sparse.scales, dense.scales, rtol=0, atol=1e-7)
        np.testing.assert_allclose(indexed.means, dense.means, rtol=0, atol=1e-7)
        np.testing.assert_allclose(indexed.scales, dense.scales, rtol=0, atol=1e-7)

    def test_lazy_neural_batches_use_one_source_global_causal_width(self) -> None:
        names = (
            "x_rel", "y_rel", "vx", "vy", "speed", "acceleration",
            "offense", "defense", "football", "focal",
        )
        tokens = np.zeros((2, 12, 2, len(names)), dtype=np.float32)
        mask = np.zeros(tokens.shape[:3], dtype=bool)
        frame_mask = np.zeros(tokens.shape[:2], dtype=bool)
        mask[0, -4:, :] = True
        mask[1, -7:, :] = True
        frame_mask[0, -4:] = True
        frame_mask[1, -7:] = True
        tokens[..., names.index("offense")] = 1.0
        tokens[:, :, 0, names.index("focal")] = 1.0
        context = np.zeros((2, 1), dtype=np.float32)
        token_scaler = MaskedTokenScaler(names).fit(tokens, mask)
        raster_scaler = RasterValueScaler().fit_tokens(tokens, mask, names)
        transformer = LazyNeuralInputs(
            "set_transformer", tokens, mask, frame_mask, context, np.arange(2), names,
            token_scaler=token_scaler,
        )
        cnn = LazyNeuralInputs(
            "cnn", tokens, mask, frame_mask, context, np.arange(2), names,
            raster_scaler=raster_scaler,
        )
        self.assertEqual(transformer.input_shapes["player_tokens"][0], None)
        self.assertEqual(transformer.input_shapes["time_to_event"], (None, 1))
        self.assertEqual(cnn.input_shapes["raster"][0], None)
        # Example zero observes only four frames, but every batch from this
        # source retains the seven-frame union needed by example one. This
        # prevents Keras from freezing a short TensorSpec from its first two
        # sampled batches.
        self.assertEqual(transformer.batch([0])["player_tokens"].shape[1], 7)
        self.assertEqual(transformer.batch([0])["time_to_event"].shape, (1, 7, 1))
        self.assertEqual(transformer.batch([0, 1])["player_tokens"].shape[1], 7)
        self.assertEqual(cnn.batch([0])["raster"].shape[1], 7)
        self.assertEqual(cnn.batch([0, 1])["raster"].shape[1], 7)
        self.assertTrue(np.all(~transformer.batch([0])["frame_mask"][:, :3]))

    def test_identity_policy_and_task_typed_edges(self) -> None:
        context = pd.DataFrame(
            {"possessionTeam": ["PHI"], "defensiveTeam": ["DAL"], "down": [3]}
        )
        self.assertEqual(primary_context_frame(context).columns.tolist(), ["down"])
        self.assertEqual(len(primary_context_frame(context, include_team_identity=True).columns), 3)

        names = (
            "x_rel", "y_rel", "offense", "defense", "football", "focal",
            "carrier",
        )
        tokens = np.zeros((1, 1, 5, len(names)), dtype=np.float32)
        mask = np.ones(tokens.shape[:3], dtype=bool)
        # candidate, near-ball blocker, official carrier, help defender, football
        tokens[0, 0, 0, names.index("defense")] = 1
        tokens[0, 0, 0, names.index("focal")] = 1
        tokens[0, 0, 1, names.index("offense")] = 1
        tokens[0, 0, 1, names.index("x_rel")] = 0.5
        tokens[0, 0, 2, names.index("offense")] = 1
        tokens[0, 0, 2, names.index("x_rel")] = 4.0
        tokens[0, 0, 2, names.index("carrier")] = 1
        tokens[0, 0, 3, names.index("defense")] = 1
        tokens[0, 0, 4, names.index("football")] = 1
        edges = relational_edge_types(tokens, mask, names, task_id="bdb2024_tackle")
        # Carrier structure follows the official role channel, not whichever
        # offensive player happens to be closest to the football.
        self.assertEqual(
            edges[0, 0, 0, 1], EDGE_TYPE_NAMES.index("blocker_candidate")
        )
        self.assertEqual(
            edges[0, 0, 0, 2], EDGE_TYPE_NAMES.index("candidate_carrier")
        )
        self.assertEqual(
            edges[0, 0, 0, 3], EDGE_TYPE_NAMES.index("same_team_influence")
        )
        self.assertEqual(
            edges[0, 0, 1, 2], EDGE_TYPE_NAMES.index("blocker_carrier")
        )
        self.assertEqual(
            edges[0, 0, 3, 2], EDGE_TYPE_NAMES.index("help_carrier")
        )
        ablated = structural_ablation_player_mask(
            tokens, mask, names, task_id="bdb2024_tackle",
            ablation_id="candidate_carrier_pair_only",
        )
        self.assertEqual(np.flatnonzero(ablated[0, 0]).tolist(), [0, 2])

        frames = np.ones((1, 1), dtype=bool)
        scaler = MaskedTokenScaler(names).fit_indexed(
            tokens,
            mask,
            np.array([0]),
            task_id="bdb2024_tackle",
            ablation_id="candidate_carrier_pair_only",
        )
        removed_changed = tokens.copy()
        removed_changed[0, 0, [1, 3], names.index("x_rel")] = [1e4, -1e4]
        changed_scaler = MaskedTokenScaler(names).fit_indexed(
            removed_changed,
            mask,
            np.array([0]),
            task_id="bdb2024_tackle",
            ablation_id="candidate_carrier_pair_only",
        )
        np.testing.assert_array_equal(scaler.means, changed_scaler.means)
        np.testing.assert_array_equal(scaler.scales, changed_scaler.scales)
        first = LazyNeuralInputs(
            "relnet", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2024_tackle", ablation_id="candidate_carrier_pair_only",
        ).batch([0])
        second = LazyNeuralInputs(
            "relnet", removed_changed, mask, frames,
            np.zeros((1, 1), np.float32), np.array([0]), names,
            token_scaler=changed_scaler, task_id="bdb2024_tackle",
            ablation_id="candidate_carrier_pair_only",
        ).batch([0])
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_each_task_graph_emits_exact_declared_edge_vocabulary(self) -> None:
        names = (
            "x_rel", "y_rel", "offense", "defense", "football", "focal",
            "position_qb", "position_wr", "position_te", "position_rb",
            "position_ol", "position_dl", "position_lb", "position_db",
            "carrier",
        )

        def graph(task_id, players):
            tokens = np.zeros((1, 1, len(players), len(names)), dtype=np.float32)
            for slot, roles in enumerate(players):
                for role in roles:
                    tokens[0, 0, slot, names.index(role)] = 1.0
            mask = np.ones(tokens.shape[:3], dtype=bool)
            encoded = relational_edge_types(
                tokens, mask, names, task_id=task_id
            )
            return {
                EDGE_TYPE_NAMES[int(value)]
                for value in np.unique(encoded)
                if int(value) != 0
            }

        football = ("football",)
        self.assertEqual(
            graph(
                "bdb2021_completion",
                [
                    ("offense", "focal", "position_qb"),
                    ("offense", "position_wr"),
                    ("offense", "position_te"),
                    ("defense", "position_db"),
                    football,
                ],
            ),
            {"quarterback_receiver", "receiver_defender", "receiver_receiver"},
        )
        self.assertEqual(
            graph(
                "bdb2022_punt_returns",
                [
                    ("offense", "focal"),
                    ("offense",),
                    ("offense",),
                    ("defense",),
                    ("defense",),
                    football,
                ],
            ),
            {"returner_blocker", "returner_coverer", "blocker_coverer", "same_team_lane"},
        )
        self.assertEqual(
            graph(
                "bdb2023_sack",
                [
                    ("offense", "focal", "position_qb"),
                    ("offense", "position_ol"),
                    ("defense", "position_dl"),
                    ("offense", "position_wr"),
                    ("defense", "position_db"),
                    football,
                ],
            ),
            {"protector_rusher", "quarterback_protector", "quarterback_rusher", "receiver_coverage"},
        )
        self.assertEqual(
            graph(
                "bdb2024_tackle",
                [
                    ("defense", "focal"),
                    ("offense", "carrier"),
                    ("offense",),
                    ("defense",),
                    football,
                ],
            ),
            {"candidate_carrier", "blocker_candidate", "blocker_carrier", "help_carrier", "same_team_influence"},
        )
        self.assertEqual(
            graph(
                "bdb2025_man_zone",
                [
                    ("offense", "focal", "position_qb"),
                    ("offense", "position_wr"),
                    ("offense", "position_te"),
                    ("defense", "position_db"),
                    ("defense", "position_db"),
                    football,
                ],
            ),
            {"receiver_defender", "defender_defender", "receiver_receiver", "quarterback_receiver"},
        )
        self.assertEqual(
            graph(
                "bdb2026_trajectory",
                [
                    ("offense", "focal"),
                    ("offense",),
                    ("offense",),
                    ("defense",),
                    ("defense",),
                ],
            ),
            {"focal_teammate", "focal_opponent", "same_side_social", "opposing_side_social"},
        )

    def test_node_and_snapshot_ablations_are_input_invariant(self) -> None:
        names = ("x_rel", "y_rel", "vx", "vy", "speed", "acceleration", "offense", "defense", "focal")
        tokens = np.zeros((1, 3, 3, len(names)), dtype=np.float32)
        mask = np.ones(tokens.shape[:3], dtype=bool)
        frames = np.ones((1, 3), dtype=bool)
        tokens[:, :, 0, names.index("offense")] = 1
        tokens[:, :, 0, names.index("focal")] = 1
        tokens[:, :, 1:, names.index("defense")] = 1
        earlier_changed = tokens.copy()
        earlier_changed[:, :2, :, :6] = 10_000
        snapshot_scaler = MaskedTokenScaler(names).fit_indexed(
            tokens, mask, np.array([0]), frame_mask=frames, final_frame_only=True
        )
        changed_snapshot_scaler = MaskedTokenScaler(names).fit_indexed(
            earlier_changed, mask, np.array([0]), frame_mask=frames, final_frame_only=True
        )
        np.testing.assert_array_equal(snapshot_scaler.means, changed_snapshot_scaler.means)
        snapshot_first = LazyNeuralInputs(
            "relnet", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=snapshot_scaler,
            task_id="bdb2025_man_zone", ablation_id="final_snapshot_only",
        ).batch([0])
        snapshot_second = LazyNeuralInputs(
            "relnet", earlier_changed, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=changed_snapshot_scaler,
            task_id="bdb2025_man_zone", ablation_id="final_snapshot_only",
        ).batch([0])
        np.testing.assert_allclose(snapshot_first["player_tokens"], snapshot_second["player_tokens"])
        self.assertEqual(snapshot_first["player_tokens"].shape[1], 1)

        scaler = MaskedTokenScaler(names).fit_indexed(
            tokens,
            mask,
            np.array([0]),
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        first = LazyNeuralInputs(
            "relnet", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2026_trajectory", ablation_id="remove_social_player_messages",
        ).batch([0])
        changed = tokens.copy()
        changed[:, -1, 1:, :6] = -10_000
        changed_scaler = MaskedTokenScaler(names).fit_indexed(
            changed,
            mask,
            np.array([0]),
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        np.testing.assert_array_equal(scaler.means, changed_scaler.means)
        np.testing.assert_array_equal(scaler.scales, changed_scaler.scales)
        second = LazyNeuralInputs(
            "relnet", changed, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=changed_scaler,
            task_id="bdb2026_trajectory", ablation_id="remove_social_player_messages",
        ).batch([0])
        np.testing.assert_array_equal(first["player_mask"], second["player_mask"])
        np.testing.assert_allclose(
            first["player_tokens"][:, -1, 1:], second["player_tokens"][:, -1, 1:]
        )
        self.assertTrue(np.all(first["player_tokens"][:, -1, 1:] == 0.0))
        np.testing.assert_array_equal(first["edge_type"], second["edge_type"])

        relnet_source = LazyNeuralInputs(
            "relnet", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        attention_source = LazyNeuralInputs(
            "attn_relnet", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        set_source = LazyNeuralInputs(
            "set_transformer", tokens, mask, frames, np.zeros((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        relnet_signature = representative_neural_input_signature(relnet_source)
        attention_signature = representative_neural_input_signature(attention_source)
        self.assertEqual(relnet_signature, attention_signature)
        self.assertNotEqual(
            relnet_signature["signature_sha256"],
            representative_neural_input_signature(set_source)["signature_sha256"],
        )
        shared_signatures = [
            representative_shared_neural_input_signature(source)
            for source in (relnet_source, attention_source, set_source)
        ]
        self.assertEqual(shared_signatures[0], shared_signatures[1])
        self.assertEqual(shared_signatures[0], shared_signatures[2])
        changed_context = LazyNeuralInputs(
            "attn_relnet", tokens, mask, frames, np.ones((1, 1), np.float32),
            np.array([0]), names, token_scaler=scaler,
            task_id="bdb2026_trajectory",
            ablation_id="remove_social_player_messages",
        )
        self.assertNotEqual(
            relnet_signature["signature_sha256"],
            representative_neural_input_signature(changed_context)["signature_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
