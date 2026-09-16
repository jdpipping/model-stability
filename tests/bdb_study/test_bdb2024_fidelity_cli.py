from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bdb_study.cli import build_parser, command_fidelity_bdb2024


class BDB2024FidelityCliTests(unittest.TestCase):
    def test_parser_and_command_wire_complete_artifact_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            output = root / "results" / "fidelity.json"
            arguments = build_parser().parse_args(
                [
                    "--repo-root",
                    str(root),
                    "fidelity-bdb2024",
                    "--prepared-dir",
                    str(prepared),
                    "--output",
                    str(output),
                    "--artifact-format",
                    "csv",
                    "--chunk-rows",
                    "17",
                ]
            )
            task = SimpleNamespace(
                task_id="bdb2024_tackle",
                examples=pd.DataFrame(
                    {
                        "example_id": ["1:1:1", "1:2:2", "2:1:3", "2:2:4"],
                        "game_id": [1, 1, 2, 2],
                        "play_id": [1, 2, 1, 2],
                        "nfl_id": [1, 2, 3, 4],
                        "week": [1, 8, 9, 9],
                    }
                ),
                tabular=pd.DataFrame(np.zeros((4, 9))),
                y=np.array([0, 1, 0, 1], dtype=np.int8),
            )
            model = object()
            fitted = SimpleNamespace(model=model, parameters={"n_estimators": 150})
            reference = {
                "receipt_hash": "e" * 64,
                "artifacts": {"model": {"sha256": "f" * 64}},
            }
            framewise = {
                "receipt_hash": "1" * 64,
                "complete_weeks": list(range(1, 10)),
                "week_receipts": [
                    {
                        "scored_defender_frames": 10,
                        "play_defender_summaries": 2,
                    }
                    for _ in range(9)
                ],
            }
            with (
                patch("bdb_study.prepared.load_prepared_task", return_value=task),
                patch(
                    "bdb_study.prepared.load_prepared_receipt",
                    return_value={"prepared_hash": "d" * 64},
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.fit_executed_xgboost",
                    return_value=fitted,
                ),
                patch(
                    "bdb_study.fidelity.bdb2024.predict_fidelity_score",
                    return_value=np.array([0.2, 0.8]),
                ),
                patch("bdb_study.fidelity.bdb2024.validate_fidelity_counts"),
                patch(
                    "bdb_study.fidelity.bdb2024.persist_reference_artifacts",
                    return_value=reference,
                ) as persist,
                patch(
                    "bdb_study.fidelity.bdb2024.run_framewise_fidelity_from_raw",
                    return_value=framewise,
                ) as run_framewise,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    command_fidelity_bdb2024(arguments)

            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["schema_version"], "bdb2024-fidelity-result-v2")
            self.assertEqual(value["artifacts"]["reference_receipt_hash"], "e" * 64)
            self.assertEqual(value["artifacts"]["framewise_receipt_hash"], "1" * 64)
            self.assertEqual(value["framewise"]["scored_defender_frames"], 90)
            persist.assert_called_once()
            self.assertEqual(persist.call_args.args[0], fitted)
            self.assertEqual(persist.call_args.kwargs["prepared_hash"], "d" * 64)
            run_framewise.assert_called_once()
            self.assertIs(run_framewise.call_args.args[0], model)
            self.assertEqual(run_framewise.call_args.kwargs["artifact_format"], "csv")
            self.assertEqual(run_framewise.call_args.kwargs["chunk_rows"], 17)
            self.assertEqual(
                run_framewise.call_args.args[1],
                (root / "data" / "bdb2024" / "raw").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
