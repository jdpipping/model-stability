from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from bdb_study.execution import (
    ENVIRONMENT_POLICY_GPU_EXACT,
    ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY,
    ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
    ExecutionError,
    _effective_task_spec,
    _execution_design,
    _resolve_environment_policy,
    _validate_frozen_environment,
    _verify_pilot_execution_contract,
    _run_worker_subprocess,
    _verify_environment,
    _verify_installed_versions,
    _worker_environment,
    queue_status,
    run_missing_cells,
    run_repeat_queue,
    verify_run_manifest,
)
from bdb_study.contracts import sha256_file
from bdb_study.manifest import bind_run_identity, build_run_manifest
from bdb_study.manifest import (
    ManifestError,
    parse_tensorflow_probe_output,
)
from bdb_study.storage import initialize_run_dir, manifest_sha256


ROOT = Path(__file__).resolve().parents[2]


class ExecutionProjectionTests(unittest.TestCase):
    @staticmethod
    def _environment(**overrides):
        value = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": "Betty B200 compute image",
            "machine": "x86_64-b200",
            "processor": "NVIDIA B200 host",
            "cpu_count": 96,
            "packages": {},
            "tensorflow_runtime": {
                "available": True,
                "version": "2.17.0",
                "devices": [
                    {"name": "/physical_device:GPU:0", "device_type": "GPU"}
                ],
                "build_info": {"cuda_version": "12.5"},
                "intra_op_threads": 0,
                "inter_op_threads": 0,
            },
            "deterministic_environment": {
                "PYTHONHASHSEED": "20260817",
                "TF_DETERMINISTIC_OPS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
            },
        }
        value.update(overrides)
        return value

    def _task_mapping(self):
        value = json.loads(
            (ROOT / "configs/bdb_suite/tasks/bdb2023_sack.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in value["models"].values():
            if entry["family"] in {
                "relnet", "attn_relnet", "set_transformer"
            }:
                entry["selected_config"] = {
                    "learning_rate": 0.001,
                    "dropout": 0.3,
                    "max_epochs": 50,
                    "patience": 10,
                    "batch_size": 64,
                }
            else:
                entry["selected_config"] = {"alpha": 1 / 3} if entry["family"] == "glm" else {
                    "n_estimators": 200
                }
        return value

    def test_smoke_uses_five_largest_anchor_cells_and_one_epoch_only_for_neural(self):
        task = self._task_mapping()
        records = []
        for repeat in (1, 2):
            for anchor in (10, 20):
                for model, entry in task["models"].items():
                    records.append(
                        {
                            "branch": "fixed_main",
                            "repeat": repeat,
                            "n_train": anchor,
                            "model": model,
                            "queue": (
                                "cpu_tabular"
                                if entry["family"] in {"glm", "lightgbm"}
                                else "gpu_neural"
                            ),
                        }
                    )
        selected = [
            {key: row[key] for key in ("branch", "repeat", "n_train", "model", "queue")}
            for row in records
            if row["repeat"] == 1 and row["n_train"] == 20
        ]
        manifest = {
            "task_spec": task,
            "task_design": {"required_cells": records},
            "execution": {
                "smoke": {
                    "enabled": True,
                    "cells": selected,
                    "neural_max_epochs": 1,
                    "neural_patience": 0,
                }
            },
        }
        projection = _execution_design(manifest)
        self.assertEqual(len(projection["required_cells"]), 5)
        self.assertEqual({row["n_train"] for row in projection["required_cells"]}, {20})
        effective = _effective_task_spec(manifest)
        for model, entry in effective.models.items():
            selected_config = entry["selected_config"]
            if entry["family"] in {
                "relnet", "attn_relnet", "set_transformer"
            }:
                self.assertEqual(selected_config["max_epochs"], 1)
                self.assertEqual(selected_config["patience"], 0)
            else:
                self.assertNotIn("max_epochs", selected_config)
        # The frozen manifest payload is never rewritten by a smoke run.
        self.assertEqual(task["models"]["relnet"]["selected_config"]["max_epochs"], 50)

    def test_malformed_smoke_projection_fails_closed(self):
        manifest = {
            "task_design": {"required_cells": []},
            "execution": {"smoke": {"enabled": True, "cells": []}},
        }
        with self.assertRaisesRegex(ExecutionError, "projection"):
            _execution_design(manifest)

    def test_sensitivity_benchmark_projection_executes_intervention(self):
        records = [
            {
                "branch": "fixed_main",
                "ablation_id": None,
                "sensitivity_id": None,
                "repeat": 1,
                "n_train": anchor,
                "model": model,
                "queue": "cpu_tabular",
            }
            for anchor in (20, 60)
            for model in (
                "linear_structure", "boosted_structure", "relnet",
                "attn_relnet", "set_transformer",
            )
        ]
        intervention = {
            "branch": "frozen_sensitivity",
            "sensitivity_id": "include_team_identity",
            "prepared_variant": "primary",
        }
        benchmark_cells = [
            {
                **record,
                "branch": "frozen_sensitivity",
                "source_branch": "fixed_main",
                "sensitivity_id": "include_team_identity",
            }
            for record in records
        ]
        manifest = {
            "task_design": {"required_cells": records},
            "execution": {
                "smoke": {"enabled": False},
                "benchmark": {
                    "enabled": True,
                    "cells": benchmark_cells,
                    "intervention": intervention,
                },
            },
        }
        projection = _execution_design(manifest)
        self.assertEqual(len(projection["required_cells"]), 10)
        self.assertEqual(
            {record["branch"] for record in projection["required_cells"]},
            {"frozen_sensitivity"},
        )
        self.assertEqual(
            {record["sensitivity_id"] for record in projection["required_cells"]},
            {"include_team_identity"},
        )

    def test_sensitivity_smoke_projection_executes_intervention(self):
        records = [
            {
                "branch": "fixed_main",
                "ablation_id": None,
                "sensitivity_id": None,
                "repeat": 1,
                "n_train": 60,
                "model": model,
                "queue": (
                    "cpu_tabular"
                    if model in {"linear_structure", "boosted_structure"}
                    else "gpu_neural"
                ),
            }
            for model in (
                "linear_structure", "boosted_structure", "relnet",
                "attn_relnet", "set_transformer",
            )
        ]
        intervention = {
            "branch": "frozen_sensitivity",
            "sensitivity_id": "include_team_identity",
            "prepared_variant": "primary",
        }
        manifest = {
            "task_design": {"required_cells": records},
            "execution": {
                "smoke": {
                    "enabled": True,
                    "cells": [
                        {
                            **record,
                            "branch": "frozen_sensitivity",
                            "source_branch": "fixed_main",
                            "sensitivity_id": "include_team_identity",
                        }
                        for record in records
                    ],
                    "intervention": intervention,
                },
                "benchmark": {"enabled": False},
            },
        }
        projected = _execution_design(manifest)["required_cells"]
        self.assertEqual(len(projected), 5)
        self.assertEqual(
            {(row["branch"], row["sensitivity_id"]) for row in projected},
            {("frozen_sensitivity", "include_team_identity")},
        )

    def test_worker_environment_is_deterministic_and_cpu_hides_gpu(self):
        cpu = _worker_environment(ROOT, "cpu_tabular")
        gpu = _worker_environment(ROOT, "gpu_neural")
        for name in (
            "PYTHONHASHSEED",
            "TF_DETERMINISTIC_OPS",
            "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        ):
            expected = {
                "PYTHONHASHSEED": "20260817",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }.get(name, "1")
            self.assertEqual(cpu[name], expected)
            self.assertEqual(cpu[name], gpu[name])
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(cpu["PYTHONPATH"].split(":")[0], str(ROOT))

    def test_worker_subprocess_places_global_repo_root_before_subcommand(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"completed": 1}\n',
            stderr="",
        )
        with patch("bdb_study.execution.subprocess.run", return_value=completed) as run:
            result = _run_worker_subprocess(
                ROOT / "data" / "bdb_suite_runs" / "test",
                7,
                "cpu_tabular",
                ROOT,
            )

        command = run.call_args.args[0]
        self.assertEqual(
            command[:7],
            [
                command[0],
                "-m",
                "bdb_study",
                "--repo-root",
                str(ROOT),
                "_worker",
                "--run-dir",
            ],
        )
        self.assertLess(command.index("--repo-root"), command.index("_worker"))
        self.assertEqual(
            run.call_args.kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"],
            ":4096:8",
        )
        self.assertEqual(result, {"completed": 1})

    def test_cpu_queue_and_status_do_not_require_visible_gpu_replay(self):
        record = {
            "branch": "fixed_main",
            "ablation_id": None,
            "sensitivity_id": None,
            "repeat": 1,
            "n_train": 10,
            "model": "linear_structure",
            "queue": "cpu_tabular",
        }
        storage = SimpleNamespace(
            manifest={
                "task_design": {"required_cells": [record]},
                "execution": {
                    "smoke": {"enabled": False},
                    "benchmark": {"enabled": False},
                    "queues": {
                        "cpu_tabular": {"workers": 12},
                        "gpu_neural": {"workers": 1},
                    },
                },
            },
            manifest_hash="a" * 64,
            validate_cell=lambda key: SimpleNamespace(is_complete=False),
        )
        with (
            patch("bdb_study.execution.open_run", return_value=storage) as opened,
            patch(
                "bdb_study.execution.pending_cells_by_queue",
                return_value={"cpu_tabular": [], "gpu_neural": []},
            ),
            patch("bdb_study.execution.queue_status", return_value={}),
        ):
            run_missing_cells(ROOT, repo_root=ROOT, queues=["cpu_tabular"])
        self.assertEqual(
            opened.call_args.kwargs["environment_policy"],
            ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
        )

        status_storage = SimpleNamespace(
            manifest=storage.manifest,
            manifest_hash="a" * 64,
            validate_cell=lambda key: SimpleNamespace(is_complete=False),
        )
        with (
            patch(
                "bdb_study.execution.open_run", return_value=status_storage
            ) as opened,
            patch(
                "bdb_study.execution.status_by_queue",
                return_value={"cpu_tabular": SimpleNamespace(as_dict=lambda: {})},
            ),
        ):
            queue_status(ROOT, repo_root=ROOT, queues=["cpu_tabular"])
        self.assertEqual(
            opened.call_args.kwargs["environment_policy"],
            ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
        )

    def test_repeat_workers_select_queue_specific_runtime_policy(self):
        for queue, expected in (
            ("cpu_tabular", ENVIRONMENT_POLICY_QUEUE_NEUTRAL),
            ("gpu_neural", ENVIRONMENT_POLICY_GPU_EXACT),
        ):
            with self.subTest(queue=queue):
                with patch(
                    "bdb_study.execution.open_run",
                    side_effect=ExecutionError("stop after admission"),
                ) as opened:
                    with self.assertRaisesRegex(ExecutionError, "after admission"):
                        run_repeat_queue(ROOT, repeat=1, queue=queue, repo_root=ROOT)
                self.assertEqual(
                    opened.call_args.kwargs["environment_policy"], expected
                )
                self.assertTrue(
                    opened.call_args.kwargs["require_deterministic_environment"]
                )

    def test_legacy_tensorflow_flag_maps_to_nonconflicting_runtime_policy(self):
        self.assertEqual(
            _resolve_environment_policy(None, False),
            ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
        )
        self.assertEqual(
            _resolve_environment_policy(None, True),
            ENVIRONMENT_POLICY_GPU_EXACT,
        )
        with self.assertRaisesRegex(ExecutionError, "conflicts"):
            _resolve_environment_policy(
                ENVIRONMENT_POLICY_GPU_EXACT, False
            )

    def test_queue_neutral_cpu_accepts_genoa_host_for_b200_planned_manifest(self):
        planned = self._environment()
        deterministic = planned["deterministic_environment"]
        with (
            patch("bdb_study.execution.platform.platform", return_value="Genoa Linux"),
            patch("bdb_study.execution.platform.machine", return_value="x86_64"),
            patch("bdb_study.execution.platform.processor", return_value="AMD Genoa"),
            patch("bdb_study.execution.os.cpu_count", return_value=192),
            patch("bdb_study.execution.subprocess.run") as tensorflow_probe,
            patch.dict("bdb_study.execution.os.environ", deterministic, clear=False),
        ):
            _verify_environment(
                planned,
                policy=ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
                require_deterministic_environment=True,
            )
        tensorflow_probe.assert_not_called()

    def test_queue_neutral_still_rejects_dependency_package_and_determinism_drift(self):
        with patch(
            "bdb_study.execution.importlib.metadata.version",
            return_value="2.0",
        ):
            with self.assertRaisesRegex(ExecutionError, "dependency drift"):
                _verify_installed_versions(
                    {"numpy": "1.0"}, category="dependency"
                )

        planned = self._environment(packages={"numpy": "1.0"})
        with patch(
            "bdb_study.execution.importlib.metadata.version",
            return_value="2.0",
        ):
            with self.assertRaisesRegex(ExecutionError, "package drift"):
                _verify_environment(
                    planned,
                    policy=ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
                    require_deterministic_environment=False,
                )

        planned = self._environment()
        with patch.dict(
            "bdb_study.execution.os.environ", {"OMP_NUM_THREADS": "8"}, clear=False
        ):
            with self.assertRaisesRegex(
                ExecutionError, "deterministic worker environment differs"
            ):
                _verify_environment(
                    planned,
                    policy=ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
                    require_deterministic_environment=True,
                )

        incomplete = self._environment(
            deterministic_environment={
                name: value
                for name, value in self._environment()[
                    "deterministic_environment"
                ].items()
                if name != "CUBLAS_WORKSPACE_CONFIG"
            }
        )
        with patch.dict(
            "bdb_study.execution.os.environ",
            incomplete["deterministic_environment"],
            clear=False,
        ):
            with self.assertRaisesRegex(
                ExecutionError, "canonical deterministic worker settings"
            ):
                _verify_environment(
                    incomplete,
                    policy=ENVIRONMENT_POLICY_QUEUE_NEUTRAL,
                    require_deterministic_environment=True,
                )

    def test_host_artifact_only_replays_artifacts_and_skips_live_host_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "code.py"
            frozen = root / "frozen.json"
            prepared = root / "prepared"
            preflight = root / "preflight.json"
            runtime = root / "runtime.json"
            lock = root / "requirements-lock.txt"
            prepared.mkdir()
            payload = prepared / "payload.bin"
            original_bytes = {
                code: b"SOURCE = 'frozen'\n",
                frozen: b"{}\n",
                preflight: b'{"receipt": "preflight"}\n',
                runtime: b'{"receipt": "runtime"}\n',
                lock: b"numpy==1.0\n",
                payload: b"prepared payload\n",
            }
            for path, content in original_bytes.items():
                path.write_bytes(content)

            prepared_unsigned = {
                "schema_version": "bdb-prepared-task-v2",
                "task_id": "bdb_host_admission_test",
                "files": {
                    "payload": {
                        "path": payload.name,
                        "size_bytes": payload.stat().st_size,
                        "sha256": sha256_file(payload),
                    }
                },
            }
            prepared_receipt = {
                **prepared_unsigned,
                "prepared_hash": hashlib.sha256(
                    json.dumps(
                        prepared_unsigned,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            prepared_receipt_path = prepared / "receipt.json"
            prepared_receipt_path.write_text(
                json.dumps(prepared_receipt), encoding="utf-8"
            )

            task_id = "bdb_host_admission_test"
            spec = SimpleNamespace(
                task_id=task_id,
                spec_hash="s" * 64,
                models={},
                sensitivities=(),
                as_dict=lambda: {},
            )
            design = {
                "task_id": task_id,
                "design_hash": "d" * 64,
                "profile": "custom",
                "required_cells": [],
                "primary_cells": [],
                "ablation_cells": [],
                "sensitivity_cells": [],
            }
            manifest = bind_run_identity(
                {
                    "schema_version": "bdb-task-run-v1",
                    "task_id": task_id,
                    "task_spec": {},
                    "task_spec_hash": spec.spec_hash,
                    "task_spec_receipt": {
                        "path": frozen.relative_to(root).as_posix(),
                        "sha256": sha256_file(frozen),
                    },
                    "task_design": design,
                    "required_cells": [],
                    "primary_required_cells": [],
                    "ablation_required_cells": [],
                    "sensitivity_required_cells": [],
                    "storage_backend": {},
                    "shared_registry_peer": None,
                    "prepared": {
                        "path": prepared.relative_to(root).as_posix(),
                        "receipt_sha256": sha256_file(prepared_receipt_path),
                        "prepared_hash": prepared_receipt["prepared_hash"],
                        "semantic_hash": "q" * 64,
                    },
                    "preflight": {
                        "path": preflight.relative_to(root).as_posix(),
                        "sha256": sha256_file(preflight),
                    },
                    "runtime_plan": {
                        "path": runtime.relative_to(root).as_posix(),
                        "sha256": sha256_file(runtime),
                    },
                    "provenance": {
                        "code": [
                            {
                                "path": code.relative_to(root).as_posix(),
                                "size_bytes": code.stat().st_size,
                                "sha256": sha256_file(code),
                            }
                        ],
                        "dependency_lock": {
                            "path": lock.relative_to(root).as_posix(),
                            "sha256": sha256_file(lock),
                            "expected": {"numpy": "1.0"},
                            "installed": {"numpy": "1.0"},
                            "mismatches": {},
                        },
                        # Deliberately invalid on this host.  This provenance is
                        # still bound by the manifest hash, but live comparison
                        # belongs only to CPU/GPU container workers.
                        "environment": {
                            "python": "0.0-host-mismatch",
                            "implementation": "HostMismatchPython",
                            "packages": {"numpy": "0.0-host-mismatch"},
                            "deterministic_environment": {
                                "PYTHONHASHSEED": "different-host-value"
                            },
                            "tensorflow_runtime": {
                                "available": False,
                                "probe_error_type": "deliberate-host-mismatch",
                            },
                        },
                    },
                }
            )

            with (
                patch("bdb_study.execution.verify_storage_backend_receipt"),
                patch("bdb_study.execution.validate_task_spec", return_value=spec),
                patch("bdb_study.execution.load_task_spec", return_value=spec),
                patch("bdb_study.manifest.validate_task_scientific_receipts"),
                patch(
                    "bdb_study.execution.build_shared_frozen_registry_binding",
                    return_value=None,
                ),
                patch("bdb_study.execution.validate_task_design"),
                patch("bdb_study.execution.load_prepared_task", return_value=object()),
                patch(
                    "bdb_study.execution.prepared_contract_for_variant",
                    return_value={},
                ),
                patch(
                    "bdb_study.execution.validate_prepared_binding",
                    return_value={"semantic_hash": "q" * 64},
                ),
                patch(
                    "bdb_study.execution._verify_installed_versions",
                    side_effect=AssertionError("host package inventory was consulted"),
                ),
                patch(
                    "bdb_study.execution._verify_environment",
                    side_effect=AssertionError("host environment was consulted"),
                ),
            ):
                verify_run_manifest(
                    manifest,
                    repo_root=root,
                    environment_policy=ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY,
                )

                tamper_cases = (
                    (code, "code/config provenance drift"),
                    (frozen, "frozen TaskSpec receipt"),
                    (payload, "prepared file failed checksum"),
                    (preflight, "preflight receipt changed"),
                    (runtime, "runtime plan changed"),
                    (lock, "dependency lock provenance is invalid"),
                )
                for path, message in tamper_cases:
                    with self.subTest(path=path.name):
                        path.write_bytes(original_bytes[path] + b"tamper\n")
                        try:
                            with self.assertRaisesRegex(Exception, message):
                                verify_run_manifest(
                                    manifest,
                                    repo_root=root,
                                    environment_policy=(
                                        ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY
                                    ),
                                )
                        finally:
                            path.write_bytes(original_bytes[path])

                malformed_environment_cases = (
                    ("python", None, "invalid python"),
                    ("implementation", [], "invalid implementation"),
                    ("packages", [], "package registry is invalid"),
                    (
                        "deterministic_environment",
                        {},
                        "deterministic settings are invalid",
                    ),
                    ("tensorflow_runtime", [], "TensorFlow receipt is invalid"),
                )
                for field, replacement, message in malformed_environment_cases:
                    with self.subTest(environment_field=field):
                        unsigned = {
                            key: value
                            for key, value in json.loads(
                                json.dumps(manifest)
                            ).items()
                            if key not in {"manifest_hash", "run_hash"}
                        }
                        unsigned["provenance"]["environment"][field] = replacement
                        malformed = bind_run_identity(unsigned)
                        with self.assertRaisesRegex(ExecutionError, message):
                            verify_run_manifest(
                                malformed,
                                repo_root=root,
                                environment_policy=(
                                    ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY
                                ),
                            )

    def test_frozen_environment_static_validation_never_reads_live_state(self):
        frozen = self._environment(
            python="0.0-frozen",
            implementation="FrozenPython",
            packages={"not-installed-here": None},
            deterministic_environment={"FROZEN_ONLY": "1"},
            tensorflow_runtime={"available": False},
        )
        with (
            patch("bdb_study.execution.platform.python_version") as python_version,
            patch("bdb_study.execution.importlib.metadata.version") as packages,
            patch("bdb_study.execution.subprocess.run") as tensorflow_probe,
        ):
            _validate_frozen_environment(frozen)
        python_version.assert_not_called()
        packages.assert_not_called()
        tensorflow_probe.assert_not_called()

    def test_host_artifact_only_cannot_authorize_a_worker(self):
        with self.assertRaisesRegex(
            ExecutionError, "cannot authorize a deterministic worker"
        ):
            verify_run_manifest(
                {},
                environment_policy=ENVIRONMENT_POLICY_HOST_ARTIFACT_ONLY,
                require_deterministic_environment=True,
            )

    def test_gpu_exact_rejects_host_and_tensorflow_device_drift(self):
        planned = self._environment()
        with (
            patch("bdb_study.execution.platform.platform", return_value=planned["os"]),
            patch("bdb_study.execution.platform.machine", return_value=planned["machine"]),
            patch("bdb_study.execution.platform.processor", return_value="AMD Genoa"),
            patch("bdb_study.execution.os.cpu_count", return_value=planned["cpu_count"]),
            patch("bdb_study.execution.subprocess.run") as tensorflow_probe,
        ):
            with self.assertRaisesRegex(ExecutionError, "processor"):
                _verify_environment(
                    planned,
                    policy=ENVIRONMENT_POLICY_GPU_EXACT,
                    require_deterministic_environment=False,
                )
        tensorflow_probe.assert_not_called()

        observed_tf = {
            **planned["tensorflow_runtime"],
            "devices": [
                {"name": "/physical_device:GPU:0", "device_type": "CPU"}
            ],
        }
        completed = SimpleNamespace(
            stdout="BDB_TF_PROBE_JSON=" + json.dumps(observed_tf) + "\n"
        )
        with (
            patch("bdb_study.execution.platform.platform", return_value=planned["os"]),
            patch("bdb_study.execution.platform.machine", return_value=planned["machine"]),
            patch("bdb_study.execution.platform.processor", return_value=planned["processor"]),
            patch("bdb_study.execution.os.cpu_count", return_value=planned["cpu_count"]),
            patch("bdb_study.execution.subprocess.run", return_value=completed),
        ):
            with self.assertRaisesRegex(ExecutionError, "TensorFlow"):
                _verify_environment(
                    planned,
                    policy=ENVIRONMENT_POLICY_GPU_EXACT,
                    require_deterministic_environment=False,
                )

    def test_tensorflow_probe_ignores_trailing_plugin_stdout(self):
        value = {"available": True, "devices": [{"device_type": "CPU"}]}
        stdout = (
            "plugin setup noise\n"
            "BDB_TF_PROBE_JSON=" + json.dumps(value, sort_keys=True) + "\n"
            "No supported GPU was found.\n"
        )
        self.assertEqual(parse_tensorflow_probe_output(stdout), value)
        with self.assertRaisesRegex(ManifestError, "exactly one"):
            parse_tensorflow_probe_output(stdout + "BDB_TF_PROBE_JSON={}\n")

    def test_build_run_manifest_identity_is_accepted_by_storage_canonical_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.json"
            prepared = root / "prepared"
            preflight = root / "preflight.json"
            prepared.mkdir()
            frozen.write_text("{}\n", encoding="utf-8")
            preflight.write_text("{}\n", encoding="utf-8")
            (prepared / "receipt.json").write_text("{}\n", encoding="utf-8")
            task_id = "bdb2020_rushing_harmonized"
            spec = SimpleNamespace(
                task_id=task_id,
                anchors=(10, 20),
                cohort={"game_registry": {"registry_hash": "g" * 64}},
                spec_hash="s" * 64,
                as_dict=lambda: {
                    "task_id": task_id,
                    "description": "Venn\N{EN DASH}Abers \N{GREEK SMALL LETTER DELTA}",
                },
            )
            analysis_seed_key = json.dumps(
                ["task", task_id, "analysis", "bootstrap"], separators=(",", ":")
            )
            required_cells = [
                {
                    "branch": "fixed_main",
                    "repeat": 1,
                    "n_train": anchor,
                    "model": model,
                    "queue": queue,
                }
                for anchor in (10, 20)
                for model, queue in (
                    ("linear_structure", "cpu_tabular"),
                    ("boosted_structure", "cpu_tabular"),
                    ("relnet", "gpu_neural"),
                    ("attn_relnet", "gpu_neural"),
                    ("set_transformer", "gpu_neural"),
                )
            ]
            design = {
                "task_id": task_id,
                "design_hash": "d" * 64,
                "required_cells": required_cells,
                "primary_cells": required_cells,
                "ablation_cells": [],
                "sensitivity_cells": [],
                "seed_registry": {analysis_seed_key: 123},
            }
            shared_peer = {
                "schema_version": "bdb-shared-frozen-registry-v1",
                "task_id": "peer",
                "file_sha256": "f" * 64,
            }
            with (
                patch("bdb_study.manifest.load_task_spec", return_value=spec),
                patch(
                    "bdb_study.manifest.build_shared_frozen_registry_binding",
                    return_value=shared_peer,
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={"task_id": task_id, "prepared_hash": "p" * 64},
                ),
                patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch("bdb_study.manifest.build_task_design", return_value=design),
                patch("bdb_study.manifest.validate_task_design"),
                patch("bdb_study.manifest.validate_task_scientific_receipts"),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "q" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest._bdb2020_harmonized_amendment_binding",
                    return_value={
                        "path": "test-amendment.json",
                        "sha256": "a" * 64,
                        "amendment_id": "test-harmonized-amendment",
                        "evidence_status": "retrospective",
                    },
                ),
                patch(
                    "bdb_study.manifest._environment_provenance",
                    return_value={
                        "tensorflow_runtime": {
                            "available": True,
                            "devices": [{"device_type": "GPU"}],
                        }
                    },
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64, "installed": {}},
                ) as dependency_lock,
                patch(
                    "bdb_study.manifest.bind_storage_backend",
                    side_effect=lambda value, repo_root: {
                        **value,
                        "storage_backend": {"label": "backend \N{GREEK SMALL LETTER DELTA}"},
                    },
                ),
                patch(
                    "bdb_study.preflight.validate_preflight_receipt",
                    return_value={"receipt_hash": "r" * 64},
                ),
            ):
                manifest = build_run_manifest(
                    frozen,
                    prepared,
                    repo_root=root,
                    repeats=1,
                    smoke=True,
                )
                benchmark_manifest = build_run_manifest(
                    frozen,
                    prepared,
                    repo_root=root,
                    repeats=1,
                    profile="full50",
                    cpu_workers=1,
                    benchmark=True,
                    preflight_receipt=preflight,
                )
            self.assertEqual(manifest_sha256(manifest), manifest["manifest_hash"])
            self.assertEqual(
                len(manifest["execution"]["smoke"]["cells"]), 4
            )
            self.assertNotIn(
                "set_transformer",
                {
                    cell["model"]
                    for cell in manifest["execution"]["smoke"]["cells"]
                },
            )
            self.assertEqual(
                manifest["execution"]["queues"]["gpu_neural"]["model_families"],
                ["relnet", "attn_relnet"],
            )
            self.assertEqual(
                len(benchmark_manifest["execution"]["benchmark"]["cells"]), 8
            )
            self.assertNotIn(
                "set_transformer",
                {
                    cell["model"]
                    for cell in benchmark_manifest["execution"]["benchmark"]["cells"]
                },
            )
            self.assertEqual(manifest["shared_registry_peer"], shared_peer)
            self.assertEqual(
                manifest["provenance"]["bdb2020_harmonized_amendment"][
                    "amendment_id"
                ],
                "test-harmonized-amendment",
            )
            self.assertNotIn("git", manifest["provenance"])
            self.assertEqual(dependency_lock.call_count, 2)
            dependency_lock.assert_called_with(root.resolve(), require_complete=True)
            storage = initialize_run_dir(
                root / "run", manifest, require_backend_pin=False
            )
            self.assertEqual(storage.manifest_hash, manifest["manifest_hash"])

    def test_pilot_manifest_has_distinct_mode_seed_and_provisional_runtime_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.json"
            prepared = root / "prepared"
            preflight = root / "preflight.json"
            runtime = root / "runtime.json"
            prepared.mkdir()
            for path in (frozen, preflight):
                path.write_text("{}\n", encoding="utf-8")
            runtime.write_text(
                json.dumps({"schema_version": "bdb-runtime-plan-v4"}) + "\n",
                encoding="utf-8",
            )
            (prepared / "receipt.json").write_text("{}\n", encoding="utf-8")
            task_id = "bdb2023_sack"
            spec = SimpleNamespace(
                task_id=task_id,
                anchors=(10, 20, 30, 40, 50, 60),
                cohort={"game_registry": {"registry_hash": "g" * 64}},
                models={},
                spec_hash="s" * 64,
                as_dict=lambda: {"task_id": task_id, "models": {}},
            )
            analysis_key = json.dumps(
                ["task", task_id, "pilot10", "analysis", "bootstrap"],
                separators=(",", ":"),
            )
            primary = [
                {
                    "branch": "fixed_main",
                    "repeat": 1 + index // 30,
                    "n_train": (10, 20, 30, 40, 50, 60)[(index // 5) % 6],
                    "model": (
                        "linear_structure",
                        "boosted_structure",
                        "relnet",
                        "attn_relnet",
                        "set_transformer",
                    )[index % 5],
                    "queue": (
                        "cpu_tabular" if index % 5 < 2 else "gpu_neural"
                    ),
                }
                for index in range(300)
            ]
            design = {
                "task_id": task_id,
                "design_hash": "d" * 64,
                "profile": "pilot10",
                "seed_profile": "pilot10",
                "repeats": 10,
                "required_cells": primary,
                "primary_cells": primary,
                "ablation_cells": [],
                "sensitivity_cells": [],
                "cell_counts": {
                    "primary": 300,
                    "structural_ablation": 0,
                    "frozen_sensitivity": 0,
                    "required": 300,
                },
                "seed_registry": {analysis_key: 707},
            }
            with (
                patch("bdb_study.manifest.load_task_spec", return_value=spec),
                patch(
                    "bdb_study.manifest.build_shared_frozen_registry_binding",
                    return_value=None,
                ),
                patch(
                    "bdb_study.manifest.load_prepared_receipt",
                    return_value={"task_id": task_id, "prepared_hash": "p" * 64},
                ),
                patch("bdb_study.manifest.load_prepared_task", return_value=object()),
                patch("bdb_study.manifest.game_records_from_prepared", return_value=[]),
                patch("bdb_study.manifest.build_task_design", return_value=design),
                patch("bdb_study.manifest.validate_task_design"),
                patch("bdb_study.manifest.validate_task_scientific_receipts"),
                patch(
                    "bdb_study.manifest.validate_prepared_binding",
                    return_value={"semantic_hash": "q" * 64},
                ),
                patch("bdb_study.manifest._code_receipts", return_value=[]),
                patch(
                    "bdb_study.manifest._protocol_amendment_binding",
                    return_value={"amendment_id": "test-amendment"},
                ),
                patch(
                    "bdb_study.manifest._environment_provenance",
                    return_value={
                        "tensorflow_runtime": {
                            "available": True,
                            "devices": [{"device_type": "GPU"}],
                        }
                    },
                ),
                patch(
                    "bdb_study.manifest.verify_dependency_lock",
                    return_value={"sha256": "l" * 64, "installed": {}},
                ),
                patch(
                    "bdb_study.manifest.bind_storage_backend",
                    side_effect=lambda value, repo_root: value,
                ),
                patch(
                    "bdb_study.preflight.validate_preflight_receipt",
                    return_value={"receipt_hash": "r" * 64},
                ),
                patch(
                    "bdb_study.preflight.validate_runtime_plan",
                    return_value={
                        "schema_version": "bdb-runtime-plan-v4",
                        "runtime_plan_hash": "t" * 64,
                        "gpu_lanes": 4,
                        "resource_contract": {"fixture": "v4"},
                    },
                ),
            ):
                manifest = build_run_manifest(
                    frozen,
                    prepared,
                    repo_root=root,
                    profile="pilot10",
                    preflight_receipt=preflight,
                    runtime_plan=runtime,
                    cpu_workers=12,
                )
                benchmark_manifest = build_run_manifest(
                    frozen,
                    prepared,
                    repo_root=root,
                    repeats=1,
                    profile="pilot10",
                    benchmark=True,
                    preflight_receipt=preflight,
                    cpu_workers=12,
                )
            execution = manifest["execution"]
            self.assertEqual(execution["mode"], "pilot")
            self.assertEqual(execution["profile"], "pilot10")
            self.assertEqual(execution["external_gpu_lanes"], 4)
            self.assertEqual(execution["evidence_status"], "exploratory_provisional")
            self.assertEqual(
                execution["inference_scope"],
                "descriptive_only_no_confirmatory_inference",
            )
            self.assertEqual(manifest["analysis"]["bootstrap_seed"], 707)
            self.assertEqual(manifest["analysis"]["bootstrap_draws"], 0)
            self.assertEqual(manifest["analysis"]["max_t_family"], "none_pilot_descriptive_only")
            self.assertEqual(manifest["runtime_plan"]["runtime_plan_hash"], "t" * 64)
            self.assertEqual(
                manifest["runtime_plan"]["schema_version"],
                "bdb-runtime-plan-v4",
            )
            self.assertEqual(
                len(benchmark_manifest["execution"]["benchmark"]["cells"]), 10
            )
            self.assertEqual(
                {
                    cell["model"]
                    for cell in benchmark_manifest["execution"]["benchmark"]["cells"]
                },
                {
                    "linear_structure",
                    "boosted_structure",
                    "relnet",
                    "attn_relnet",
                    "set_transformer",
                },
            )

    def test_pilot_execution_contract_rejects_definitive_and_secondary_branches(self):
        design = {
            "profile": "pilot10",
            "cell_counts": {
                "primary": 300,
                "structural_ablation": 0,
                "frozen_sensitivity": 0,
                "required": 300,
            },
        }
        manifest = {
            "execution": {
                "mode": "pilot",
                "profile": "pilot10",
                "external_gpu_lanes": 4,
                "evidence_status": "exploratory_provisional",
                "inference_scope": "descriptive_only_no_confirmatory_inference",
            }
        }
        _verify_pilot_execution_contract(manifest, design)
        mislabeled = json.loads(json.dumps(manifest))
        mislabeled["execution"]["mode"] = "definitive"
        with self.assertRaisesRegex(ExecutionError, "exact pilot"):
            _verify_pilot_execution_contract(mislabeled, design)
        for field in ("structural_ablation", "frozen_sensitivity"):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(design))
                changed["cell_counts"][field] = 1
                changed["cell_counts"]["required"] = 301
                with self.assertRaisesRegex(ExecutionError, "300 primary"):
                    _verify_pilot_execution_contract(manifest, changed)

    def test_smoke_and_main_planning_require_shared_frozen_peer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "configs/bdb_suite/frozen/bdb2024_tackle.json"
            prepared = root / "data/processed/bdb_suite/bdb2024_tackle"
            preflight = root / "configs/bdb_suite/preflight/bdb2024_tackle.json"
            frozen.parent.mkdir(parents=True)
            prepared.mkdir(parents=True)
            preflight.parent.mkdir(parents=True)
            frozen.write_text("{}\n", encoding="utf-8")
            preflight.write_text("{}\n", encoding="utf-8")
            spec = SimpleNamespace(
                task_id="bdb2024_tackle",
                cohort={"game_registry": {"registry_hash": "r" * 64}},
            )
            for smoke in (True, False):
                with self.subTest(smoke=smoke):
                    with (
                        patch("bdb_study.manifest.load_task_spec", return_value=spec),
                        patch("bdb_study.manifest.validate_task_scientific_receipts"),
                        patch(
                            "bdb_study.manifest.build_shared_frozen_registry_binding",
                            side_effect=ManifestError(
                                "shared-registry frozen peer is required"
                            ),
                        ),
                        patch("bdb_study.manifest.load_prepared_receipt") as prepared_load,
                    ):
                        with self.assertRaisesRegex(
                            ManifestError, "frozen peer is required"
                        ):
                            build_run_manifest(
                                frozen,
                                prepared,
                                repo_root=root,
                                repeats=1 if smoke else 50,
                                cpu_workers=1 if smoke else 12,
                                smoke=smoke,
                                preflight_receipt=None if smoke else preflight,
                            )
                    prepared_load.assert_not_called()

    def test_runtime_rejects_shared_peer_receipt_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "configs/bdb_suite/frozen/bdb2024_tackle.json"
            frozen.parent.mkdir(parents=True)
            frozen.write_text("{}\n", encoding="utf-8")
            spec = SimpleNamespace(
                task_id="bdb2024_tackle",
                spec_hash="s" * 64,
                as_dict=lambda: {},
                models={},
            )
            base = {
                "schema_version": "bdb-task-run-v1",
                "task_id": "bdb2024_tackle",
                "task_spec": {},
                "task_spec_hash": "s" * 64,
                "task_spec_receipt": {
                    "path": frozen.relative_to(root).as_posix(),
                    "sha256": sha256_file(frozen),
                },
                "task_design": {
                    "task_id": "bdb2024_tackle",
                    "design_hash": "d" * 64,
                    "required_cells": [],
                },
                "required_cells": [],
                "storage_backend": {},
                "provenance": {"code": []},
                "shared_registry_peer": {"file_sha256": "planned"},
            }
            manifest = bind_run_identity(base)
            with (
                patch("bdb_study.execution.verify_storage_backend_receipt"),
                patch("bdb_study.execution.validate_task_spec", return_value=spec),
                patch("bdb_study.execution.load_task_spec", return_value=spec),
                patch("bdb_study.manifest.validate_task_scientific_receipts"),
                patch(
                    "bdb_study.execution.build_shared_frozen_registry_binding",
                    return_value={"file_sha256": "changed"},
                ),
            ):
                with self.assertRaisesRegex(
                    ExecutionError, "differs from run planning"
                ):
                    verify_run_manifest(
                        manifest, repo_root=root, verify_tensorflow_runtime=False
                    )


if __name__ == "__main__":
    unittest.main()
