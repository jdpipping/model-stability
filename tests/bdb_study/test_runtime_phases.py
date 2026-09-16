from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bdb_study.runtime_phases import (
    BETTY_ACCOUNT_CPU_LIMIT,
    BETTY_MIG90_LANES,
    RuntimePhasePlanError,
    betty_runtime_v4_resource_contract,
    betty_runtime_v3_resource_contract,
    canonical_neural_phase_shards,
    concrete_chain_units,
    lpt_lane_assignment,
    validate_lpt_lane_assignment,
    validate_betty_capacity_evidence,
    validate_neural_phase_shards,
)


ROOT = Path(__file__).resolve().parents[2]


class RuntimePhaseShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.anchors = [10, 20, 40, 60, 100, 130]
        self.models = ["relnet", "attn_relnet", "set_transformer"]
        self.bounds = {
            "relnet": {"selector": 4_900.0, "refit": 4_500.0},
            "attn_relnet": {"selector": 8_555.0, "refit": 8_556.0},
            "set_transformer": {"selector": 1_200.0, "refit": 1_000.0},
        }

    def shards(self):
        return canonical_neural_phase_shards(
            experimental_phase="primary",
            anchors=self.anchors,
            models=self.models,
            model_phase_seconds_bound=self.bounds,
        )

    def test_resource_contract_uses_fastest_measured_capacity(self) -> None:
        contract = betty_runtime_v3_resource_contract()
        self.assertEqual(contract["gpu"]["resource_class"], "b200-mig90")
        self.assertEqual(contract["gpu"]["lanes"], 4)
        self.assertEqual(contract["cpu"]["lanes"], 6)
        self.assertEqual(contract["account_gpu_limit"], 4)
        self.assertEqual(contract["qos_gpu_group_limit"], 16)
        self.assertEqual(contract["peak_scheduled_cpus"], 128)
        self.assertEqual(contract["spare_account_cpus"], 0)
        self.assertLessEqual(
            contract["peak_scheduled_cpus"], BETTY_ACCOUNT_CPU_LIMIT
        )
    def test_migrated_wrappers_do_not_reuse_historical_source_admission(self) -> None:
        with self.assertRaisesRegex(RuntimePhasePlanError, "source binding drifted"):
            validate_betty_capacity_evidence(ROOT)

    def test_capacity_receipt_and_source_tamper_fail_closed(self) -> None:
        relatives = (
            "scripts/betty/capacity/2026-08-22_fastest_lane.json",
            "scripts/betty/capacity/2026-08-22_fastest_lane.json.sha256",
            "scripts/betty/benchmark_models.py",
            "scripts/betty/mig45.sbatch",
            "scripts/betty/mig90.sbatch",
            "scripts/betty/b200.sbatch",
            "scripts/betty/mig90_14cpu.sbatch",
            "scripts/betty/cpu.sbatch",
        )
        with tempfile.TemporaryDirectory() as temporary:
            target_root = Path(temporary)
            for relative in relatives:
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            capacity = target_root / relatives[0]
            capacity.write_bytes(capacity.read_bytes() + b"\n")
            digest = hashlib.sha256(capacity.read_bytes()).hexdigest()
            capacity.with_name(f"{capacity.name}.sha256").write_text(
                f"{digest}  {capacity.name}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(RuntimePhasePlanError, "identity"):
                validate_betty_capacity_evidence(target_root)

            shutil.copyfile(ROOT / relatives[0], capacity)
            shutil.copyfile(ROOT / relatives[1], capacity.with_name(
                f"{capacity.name}.sha256"
            ))
            source = target_root / "scripts/betty/mig90_14cpu.sbatch"
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimePhasePlanError, "source binding"):
                validate_betty_capacity_evidence(target_root)

    def test_bdb2020_archive_only_probe_is_exactly_task_scoped(self) -> None:
        probe_relative = Path(
            "scripts/betty/probes/attempt9_mig45_current_pair"
        )
        amendment_relative = Path(
            "configs/bdb_suite/protocol_amendments/"
            "20260906_bdb2020_harmonized_five_role.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            target_root = Path(temporary)
            shutil.copytree(
                ROOT / probe_relative, target_root / probe_relative
            )
            amendment = target_root / amendment_relative
            amendment.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / amendment_relative, amendment)
            binding = json.loads(
                (target_root / probe_relative / "binding.json").read_text()
            )

            contract = betty_runtime_v4_resource_contract(
                binding,
                repo_root=target_root,
                task_id="bdb2020_rushing_harmonized",
            )
            self.assertTrue(contract["determinism_probe"]["terminal_green"])
            self.assertEqual(contract["gpu"]["lanes"], 4)
            self.assertEqual(contract["cpu"]["lanes"], 8)

            with self.assertRaisesRegex(
                RuntimePhasePlanError, "source binding"
            ):
                betty_runtime_v4_resource_contract(
                    binding,
                    repo_root=target_root,
                    task_id="bdb2021_completion",
                )

            amendment.write_bytes(amendment.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                RuntimePhasePlanError, "runtime amendment drifted"
            ):
                betty_runtime_v4_resource_contract(
                    binding,
                    repo_root=target_root,
                    task_id="bdb2020_rushing_harmonized",
                )

    def test_atomic_cover_is_exact_and_guarded(self) -> None:
        shards = self.shards()
        self.assertEqual(len(shards), 18)
        cover = {
            (shard["anchors"][0], shard["models"][0]) for shard in shards
        }
        self.assertEqual(
            cover,
            {(anchor, model) for anchor in self.anchors for model in self.models},
        )
        self.assertTrue(all(shard["cells_per_repeat"] == 1 for shard in shards))
        self.assertTrue(
            all(shard["selector_predicted_job_seconds"] <= 12_000 for shard in shards)
        )
        self.assertTrue(
            all(shard["refit_predicted_job_seconds"] <= 12_000 for shard in shards)
        )
        self.assertEqual(
            validate_neural_phase_shards(
                shards,
                experimental_phase="primary",
                anchors=self.anchors,
                models=self.models,
                model_phase_seconds_bound=self.bounds,
            ),
            shards,
        )

    def test_atomic_phase_over_guard_fails_closed(self) -> None:
        bad = copy.deepcopy(self.bounds)
        bad["attn_relnet"]["selector"] = 11_641.0
        with self.assertRaisesRegex(RuntimePhasePlanError, "atomic neural phase"):
            canonical_neural_phase_shards(
                experimental_phase="primary",
                anchors=self.anchors,
                models=self.models,
                model_phase_seconds_bound=bad,
            )

    def test_missing_overlap_and_self_rehashed_drift_rejected(self) -> None:
        shards = self.shards()
        for observed in (shards[:-1], [*shards, copy.deepcopy(shards[0])]):
            with self.assertRaisesRegex(RuntimePhasePlanError, "canonical exact cover"):
                validate_neural_phase_shards(
                    observed,
                    experimental_phase="primary",
                    anchors=self.anchors,
                    models=self.models,
                    model_phase_seconds_bound=self.bounds,
                )
        drift = copy.deepcopy(shards)
        drift[0]["selector_seconds_per_repeat_bound"] += 1.0
        with self.assertRaisesRegex(RuntimePhasePlanError, "canonical exact cover"):
            validate_neural_phase_shards(
                drift,
                experimental_phase="primary",
                anchors=self.anchors,
                models=self.models,
                model_phase_seconds_bound=self.bounds,
            )

    def test_concrete_union_once_and_deterministic_lpt(self) -> None:
        shards = self.shards()
        units = concrete_chain_units(shards, repeats=10)
        # The slow roles have one repeat per chain while faster roles may be
        # grouped, so validate the actual cell union rather than a job count.
        cells = []
        for unit in units:
            for repeat in unit["repeats"]:
                cells.append(
                    (repeat, unit["anchors"][0], unit["models"][0])
                )
        self.assertEqual(len(cells), 180)
        self.assertEqual(len(set(cells)), 180)
        lanes = lpt_lane_assignment(units)
        self.assertEqual(len(lanes), BETTY_MIG90_LANES)
        self.assertEqual(sum(map(len, lanes)), len(units))
        self.assertEqual(
            validate_lpt_lane_assignment(
                lanes, shards=shards, repeats=10
            ),
            lanes,
        )
        tampered = copy.deepcopy(lanes)
        tampered[0], tampered[1] = tampered[1], tampered[0]
        with self.assertRaisesRegex(RuntimePhasePlanError, "canonical LPT"):
            validate_lpt_lane_assignment(
                tampered, shards=shards, repeats=10
            )


if __name__ == "__main__":
    unittest.main()
