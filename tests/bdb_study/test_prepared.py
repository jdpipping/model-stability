from __future__ import annotations

from copy import deepcopy
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bdb_study.adapters.common import PreparedTask
from bdb_study.prepared import (
    PreparedArtifactError,
    build_prepared_semantic_receipt,
    load_prepared_task,
    validate_prepared_task_contract,
    write_prepared_task,
)
from bdb_study.contracts import sha256_json
from bdb_study.manifest import ManifestError, validate_prepared_binding


def synthetic_task(offset: float = 0.0) -> PreparedTask:
    examples = pd.DataFrame(
        {
            "example_id": ["1-1", "2-2"],
            "game_id": [1, 2],
            "stratum": ["2020", "2021"],
            "target": [0, 1],
        }
    )
    tokens = np.zeros((2, 1, 2, 4), dtype=np.float32)
    tokens[..., 0] = offset
    return PreparedTask(
        task_id="synthetic_binary",
        outcome_type="binary",
        primary_metric="brier",
        examples=examples,
        tabular=pd.DataFrame({"down": [1, 2], "formation": ["a", "b"]}),
        player_tokens=tokens,
        player_mask=np.ones((2, 1, 2), dtype=bool),
        frame_mask=np.ones((2, 1), dtype=bool),
        channel_names=("x_rel", "y_rel", "offense", "focal"),
        y=np.array([0, 1], dtype=np.int8),
        audit={"retained_examples": 2},
    )


class PreparedArtifactTests(unittest.TestCase):
    def test_round_trip_uses_memory_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared"
            receipt = write_prepared_task(synthetic_task(), target)
            loaded = load_prepared_task(target)
            self.assertEqual(receipt["examples"], 2)
            self.assertIsInstance(loaded.player_tokens, np.memmap)
            np.testing.assert_array_equal(loaded.y, [0, 1])
            self.assertEqual(list(loaded.tabular.columns), ["down", "formation"])
            self.assertEqual(
                receipt["semantic_receipt"], build_prepared_semantic_receipt(loaded)
            )

    def test_idempotent_and_refuses_different_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared"
            first = write_prepared_task(synthetic_task(), target)
            second = write_prepared_task(synthetic_task(), target)
            self.assertEqual(first["prepared_hash"], second["prepared_hash"])
            with self.assertRaises(PreparedArtifactError):
                write_prepared_task(synthetic_task(1.0), target)

    def test_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared"
            write_prepared_task(synthetic_task(), target)
            with (target / "examples.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            with self.assertRaises(PreparedArtifactError):
                load_prepared_task(target)

    def test_semantic_contract_rejects_reordered_features_and_metadata(self) -> None:
        task = synthetic_task()
        expected = build_prepared_semantic_receipt(task)
        validate_prepared_task_contract(task, expected)

        reordered = deepcopy(expected)
        reordered["tabular_feature_order"].reverse()
        reordered["semantic_hash"] = sha256_json(
            {key: value for key, value in reordered.items() if key != "semantic_hash"}
        )
        with self.assertRaisesRegex(PreparedArtifactError, "tabular_feature_order"):
            validate_prepared_task_contract(task, reordered)

        changed_metadata = deepcopy(expected)
        changed_metadata["adapter_metadata"] = {"cutoff": "future_frame"}
        changed_metadata["semantic_hash"] = sha256_json(
            {
                key: value
                for key, value in changed_metadata.items()
                if key != "semantic_hash"
            }
        )
        with self.assertRaisesRegex(PreparedArtifactError, "adapter_metadata"):
            validate_prepared_task_contract(task, changed_metadata)

    def test_embedded_semantic_receipt_is_independently_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared"
            write_prepared_task(synthetic_task(), target)
            path = target / "receipt.json"
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt["semantic_receipt"]["semantic_hash"] = "0" * 64
            unsigned = {
                key: value for key, value in receipt.items() if key != "prepared_hash"
            }
            receipt["prepared_hash"] = sha256_json(unsigned)
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(
                PreparedArtifactError, "semantic receipt hash"
            ):
                load_prepared_task(target, verify_files=True)

    def test_planning_binding_rechecks_semantics_and_frozen_identity(self) -> None:
        task = synthetic_task()
        semantic = build_prepared_semantic_receipt(task)
        receipt = {
            "task_id": task.task_id,
            "outcome_type": task.outcome_type,
            "primary_metric": task.primary_metric,
            "examples": 2,
            "games": 2,
            "prepared_hash": "a" * 64,
            "audit": dict(task.audit),
            "semantic_receipt": semantic,
        }
        spec = SimpleNamespace(
            task_id=task.task_id,
            outcome_type=task.outcome_type,
            primary_metric=task.primary_metric,
            prepared_contract=semantic,
            cohort={
                "prepared_hash": receipt["prepared_hash"],
                "prepared_examples": 2,
                "prepared_games": 2,
                "prepared_audit": dict(task.audit),
            },
        )
        validate_prepared_binding(
            spec, receipt, task, require_frozen_identity=True
        )

        changed = deepcopy(semantic)
        changed["arrays"]["frame_mask"]["shape"] = [2, 2]
        changed["semantic_hash"] = sha256_json(
            {key: value for key, value in changed.items() if key != "semantic_hash"}
        )
        bad_spec = SimpleNamespace(**{**vars(spec), "prepared_contract": changed})
        with self.assertRaisesRegex(ManifestError, "semantic binding"):
            validate_prepared_binding(
                bad_spec, receipt, task, require_frozen_identity=False
            )

        wrong_hash = SimpleNamespace(
            **{**vars(spec), "cohort": {**spec.cohort, "prepared_hash": "b" * 64}}
        )
        with self.assertRaisesRegex(ManifestError, "exact artifact frozen"):
            validate_prepared_binding(
                wrong_hash, receipt, task, require_frozen_identity=True
            )


if __name__ == "__main__":
    unittest.main()
