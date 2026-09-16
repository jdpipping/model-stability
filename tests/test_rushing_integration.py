"""Fast two-anchor integration test with deterministic stub model cells."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from rushing_study.data import StudyData
from rushing_study.cli import (
    _attach_preflight_attestation,
    build_parser,
    command_verify_determinism,
)
from rushing_study.design import (
    HYBRID_EXECUTION_PROFILE,
    HYBRID_EXECUTION_QUEUES,
    SMOKE_DESCRIPTION_PREFIX,
    SeedRegistry,
    load_config,
    manifest_seed,
)
from rushing_study.execution import (
    SENSITIVITY_STAGE1_RECEIPT,
    classify_scientific_cells,
    configure_worker_device,
    execute_repeat,
    expected_cell_keys,
    load_metrics_frame,
    schedule_subprocesses,
    validate_sensitivity_stage1_receipt,
    validate_scientific_cell,
)
from rushing_study.metrics import evaluate_cell
from rushing_study.models import MODEL_IDS, frozen_model_config, sensitivity_candidates
from rushing_study.runner import CellPayload
from rushing_study.storage import (
    CellKey,
    CellStatus,
    atomic_write_json,
    initialize_run_dir,
    register_existing_artifact,
)


def _scientific_manifest() -> dict:
    config = {
        "models": {
            model: {
                "stub": True,
                **(
                    {"expected_parameters": 80}
                    if model in {"zoo_cnn", "set_transformer"}
                    else {}
                ),
            }
            for model in MODEL_IDS
        },
        "splits": {"nested_train_anchors": [20, 40, 80, 160, 240, 360]},
        "execution": {"confirmatory_repeats": 50},
        "sensitivity": {
            "stage1": {"anchors": [20, 160, 360], "repeats": 20},
            "extension": {"target_total_repeats": 50},
        },
        "uncertainty": {"alpha": 0.1, "local_k": 5},
    }
    registry = SeedRegistry()
    for model in MODEL_IDS:
        for stage in ("fit", "epoch_selection", "refit", "prediction_rebuild"):
            registry.get("cell", 1, stage, model, 20)
    return {"config": config, "seed_registry": registry.snapshot(), "study": "validator-test"}


def _valid_payload(manifest: dict, key: CellKey) -> CellPayload:
    rng = np.random.default_rng(100 + MODEL_IDS.index(key.model))
    calibration_proba = rng.dirichlet(np.ones(80), size=12).astype(np.float32)
    test_proba = rng.dirichlet(np.ones(80), size=8).astype(np.float32)
    calibration_metadata = pd.DataFrame(
        {
            "game_id": [f"cal-{index // 3}" for index in range(12)],
            "play_id": [f"cal-play-{index}" for index in range(12)],
            "season": [2017] * 12,
        }
    )
    test_metadata = pd.DataFrame(
        {
            "game_id": ["test-a"] * 3 + ["test-b"] * 5,
            "play_id": [f"test-play-{index}" for index in range(8)],
            "season": [2018] * 8,
        }
    )
    evaluated = evaluate_cell(
        calibration_proba,
        test_proba,
        rng.integers(0, 80, size=12),
        rng.integers(0, 80, size=8),
        calibration_metadata,
        test_metadata,
        alpha=0.1,
        local_k=5,
    )
    seed = lambda stage: manifest_seed(
        manifest,
        "cell",
        key.repeat,
        stage,
        key.model,
        key.n_train,
    )
    if key.branch == "main":
        model_config = frozen_model_config(manifest["config"], key.model)
        selected_index = None
    else:
        candidates = sensitivity_candidates(manifest["config"], key.model)
        selected_index = 0
        model_config = candidates[selected_index]
    config_hash = hashlib.sha256(
        json.dumps(
            model_config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if key.model in {"ridge_sgd_l2", "lightgbm_multiclass"}:
        history = {"fit_seed": seed("fit")}
    else:
        history = {
            "selection_seed": seed("epoch_selection"),
            "refit_seed": seed("refit"),
        }
        if key.branch == "sensitivity":
            history["prediction_rebuild_seed"] = seed("prediction_rebuild")
    history.update({"model_config": model_config, "model_config_hash": config_hash})
    candidate_scores = []
    if key.branch == "sensitivity":
        history.update(
            {
                "selected_candidate_index": selected_index,
                "selected_tune_crps": 0.1,
                "candidate_count": len(candidates),
            }
        )
        candidate_scores = []
        for index, candidate in enumerate(candidates):
            candidate_hash = hashlib.sha256(
                json.dumps(
                    candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
            candidate_scores.append(
                {
                    "candidate_index": index,
                    "config": candidate,
                    "config_hash": candidate_hash,
                    "fit_seed": seed("fit")
                    if key.model in {"ridge_sgd_l2", "lightgbm_multiclass"}
                    else seed("epoch_selection"),
                    "refit_seed": None
                    if key.model in {"ridge_sgd_l2", "lightgbm_multiclass"}
                    else seed("refit"),
                    "tune_crps": 0.1 + index / 100,
                }
            )
    parameter_count = int(model_config.get("expected_parameters", 80))
    metrics = {
        **evaluated.metrics,
        "branch": key.branch,
        "repeat": key.repeat,
        "n_train": key.n_train,
        "model": key.model,
        "model_config_hash": config_hash,
        "n_train_plays": 100,
        "parameter_count": parameter_count,
        "elapsed_seconds": 0.25,
    }
    if key.branch == "sensitivity":
        metrics["selected_config"] = json.dumps(
            model_config, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    return CellPayload(
        metrics=metrics,
        calibration_predictions=evaluated.calibration,
        test_predictions=evaluated.test,
        arrays=evaluated.arrays,
        history=history,
        candidate_scores=candidate_scores,
    )


def _write_payload(
    storage,
    key: CellKey,
    payload: CellPayload,
    *,
    metric_overrides: dict | None = None,
) -> None:
    predictions = pd.concat(
        [
            payload.calibration_predictions.assign(partition="calibration"),
            payload.test_predictions.assign(partition="test"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics = dict(payload.metrics)
    metrics["manifest_hash"] = storage.manifest_hash
    if metric_overrides:
        metrics.update(metric_overrides)
    history = dict(payload.history)
    if payload.candidate_scores:
        history["candidate_scores"] = payload.candidate_scores
    storage.write_cell_artifacts(
        key,
        metrics=metrics,
        predictions=predictions,
        history=history,
        arrays=payload.arrays,
        overwrite=True,
    )


class RushingStudyIntegrationTests(unittest.TestCase):
    def test_plan_cli_defaults_to_50_and_exposes_locked_full_profile(self):
        parser = build_parser()
        default = parser.parse_args(["plan", "--smoke"])
        full = parser.parse_args(["plan", "--smoke", "--full"])
        hybrid = parser.parse_args(["plan", "--smoke", "--hybrid"])
        self.assertFalse(default.full)
        self.assertFalse(default.hybrid)
        self.assertTrue(full.full)
        self.assertTrue(hybrid.hybrid)

    def test_preflight_attestation_rejects_a_different_repeat_profile(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        config["execution"]["plan"] = "smoke"
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(
                Path(directory) / "run",
                {"config": config},
            )
            final_manifest = {
                "config": {"execution": {"confirmatory_repeats": 100}}
            }
            with self.assertRaisesRegex(RuntimeError, "same 50- or 100-repeat"):
                _attach_preflight_attestation(final_manifest, storage.run_dir)

    def test_preflight_attestation_uses_hashes_without_git_metadata(self):
        final_config = load_config("configs/rushing_confirmatory_v1.json")
        smoke_config = deepcopy(final_config)
        smoke_config["execution"]["plan"] = "smoke"
        smoke_config["neural_training"]["max_epochs"] = 1
        smoke_config["neural_training"]["early_stopping_patience"] = 0
        smoke_config["description"] = (
            SMOKE_DESCRIPTION_PREFIX + smoke_config["description"]
        )
        shared_files = {
            "study": {
                "path": "study.py",
                "size_bytes": 7,
                "sha256": "a" * 64,
            }
        }
        shared_data = {"files": {}}
        shared_environment = {"runtime": "same"}
        smoke_provenance = {
            "code": {"files": shared_files},
            "data": shared_data,
            "environment": shared_environment,
        }
        final_provenance = deepcopy(smoke_provenance)
        keys = [CellKey("main", 1, 20, model) for model in MODEL_IDS]

        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(
                Path(directory) / "smoke",
                {"config": smoke_config, "provenance": smoke_provenance},
            )
            for key in keys:
                marker = storage.cell_dir(key) / "_SUCCESS"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_bytes(b"complete\n")
            receipts = []
            for model in ("zoo_cnn", "set_transformer"):
                path = storage.run_dir / "verification" / f"{model}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                receipts.append((path, {"model": model, "passed": True}))
            classified = {
                key: SimpleNamespace(is_complete=True, reason=None) for key in keys
            }
            with mock.patch(
                "rushing_study.cli.load_study_data", return_value=object()
            ), mock.patch(
                "rushing_study.cli.expected_cell_keys", return_value=keys
            ), mock.patch(
                "rushing_study.cli.classify_scientific_cells",
                return_value=classified,
            ), mock.patch(
                "rushing_study.cli._determinism_receipts", return_value=receipts
            ):
                updated = _attach_preflight_attestation(
                    {"config": final_config, "provenance": final_provenance},
                    storage.run_dir,
                )

        self.assertNotIn("git", json.dumps(updated).lower())

    def test_determinism_subprocess_uses_frozen_repo_root(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            frozen_root = Path(directory).resolve()
            storage = initialize_run_dir(
                Path(directory) / "run",
                {
                    "config": config,
                    "provenance": {"code": {"repo_root": str(frozen_root)}},
                },
            )
            args = Namespace(
                run_dir=storage.run_dir,
                runtime_ready=False,
                branch="main",
                repeat_id=1,
                n_train=20,
                model="zoo_cnn",
                rtol=0.0,
                atol=0.0,
            )
            with mock.patch(
                "rushing_study.cli._verify_runtime_if_frozen"
            ), mock.patch("rushing_study.cli.subprocess.run") as launched:
                launched.return_value.returncode = 0
                self.assertEqual(command_verify_determinism(args), 0)

        environment = launched.call_args.kwargs["env"]
        self.assertEqual(launched.call_args.kwargs["cwd"], str(frozen_root))
        self.assertEqual(environment["PYTHONPATH"].split(":", 1)[0], str(frozen_root))

    def test_scheduler_uses_manifest_bound_deterministic_environment(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            frozen_root = Path(directory).resolve()
            storage = initialize_run_dir(
                Path(directory) / "run",
                {
                    "config": config,
                    "provenance": {"code": {"repo_root": str(frozen_root)}},
                },
            )
            with mock.patch(
                "rushing_study.execution._verify_runtime_if_frozen"
            ), mock.patch("rushing_study.execution.subprocess.run") as launched:
                schedule_subprocesses(
                    storage.run_dir,
                    "main",
                    repeat_ids=[1],
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    workers=1,
                )

        environment = launched.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONHASHSEED"], "20260817")
        self.assertEqual(environment["TF_DETERMINISTIC_OPS"], "1")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        self.assertEqual(environment["TF_NUM_INTEROP_THREADS"], "1")
        self.assertEqual(environment["TF_NUM_INTRAOP_THREADS"], "1")
        self.assertEqual(launched.call_args.kwargs["cwd"], str(frozen_root))
        self.assertEqual(environment["PYTHONPATH"].split(":", 1)[0], str(frozen_root))

    def test_hybrid_scheduler_routes_two_concurrent_manifest_queues(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        config["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        config["execution"]["queues"] = HYBRID_EXECUTION_QUEUES
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(
                Path(directory) / "run",
                {"config": config},
            )
            with mock.patch(
                "rushing_study.execution._verify_runtime_if_frozen"
            ), mock.patch("rushing_study.execution.subprocess.run") as launched:
                schedule_subprocesses(
                    storage.run_dir,
                    "main",
                    repeat_ids=[1, 2],
                    anchors=[20],
                    workers=None,
                )

        self.assertEqual(launched.call_count, 4)
        routed = []
        for call in launched.call_args_list:
            command = call.args[0]
            queue = command[command.index("--queue") + 1]
            models = set(command[command.index("--models") + 1].split(","))
            routed.append((queue, models, call.kwargs["env"]["RUSHING_STUDY_QUEUE"]))
        self.assertEqual(
            sum(queue == "cpu_tabular" for queue, _, _ in routed), 2
        )
        self.assertEqual(sum(queue == "gpu_neural" for queue, _, _ in routed), 2)
        for queue, models, environment_queue in routed:
            self.assertEqual(environment_queue, queue)
            self.assertEqual(models, set(HYBRID_EXECUTION_QUEUES[queue]["models"]))

    def test_hybrid_scheduler_enforces_locked_queue_worker_counts(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        config["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        config["execution"]["queues"] = HYBRID_EXECUTION_QUEUES
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", {"config": config})
            with self.assertRaisesRegex(ValueError, "complete hybrid run"):
                schedule_subprocesses(
                    storage.run_dir,
                    "main",
                    repeat_ids=[1],
                    anchors=[20],
                    workers=1,
                )
            with mock.patch("rushing_study.execution.subprocess.run") as launched:
                schedule_subprocesses(
                    storage.run_dir,
                    "main",
                    repeat_ids=[1],
                    anchors=[20],
                    queue="cpu_tabular",
                    workers=12,
                )
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[command.index("--queue") + 1], "cpu_tabular")
        self.assertEqual(
            set(command[command.index("--models") + 1].split(",")),
            set(HYBRID_EXECUTION_QUEUES["cpu_tabular"]["models"]),
        )

    def test_hybrid_cpu_worker_uses_cpu_only_routing_without_tensorflow(self):
        config = {
            "execution": {
                "profile": HYBRID_EXECUTION_PROFILE,
                "queues": HYBRID_EXECUTION_QUEUES,
            }
        }
        manifest = {"config": config}
        with mock.patch.dict("sys.modules", {"tensorflow": None}):
            result = configure_worker_device(manifest, "cpu_tabular")
        self.assertEqual(result["device"], "cpu")
        self.assertIsNone(result["visible_gpu_count"])
        self.assertEqual(result["isolation"], "cpu_only_model_routing")

    def test_hybrid_gpu_worker_requires_a_visible_tensorflow_gpu(self):
        manifest = {
            "config": {
                "execution": {
                    "profile": HYBRID_EXECUTION_PROFILE,
                    "queues": HYBRID_EXECUTION_QUEUES,
                }
            }
        }
        tensorflow = SimpleNamespace(
            config=SimpleNamespace(get_visible_devices=lambda device_type: [])
        )
        with mock.patch.dict("sys.modules", {"tensorflow": tensorflow}):
            with self.assertRaisesRegex(RuntimeError, "TensorFlow-visible GPU"):
                configure_worker_device(manifest, "gpu_neural")

        gpu = SimpleNamespace(name="GPU:0", device_type="GPU")
        tensorflow = SimpleNamespace(
            config=SimpleNamespace(get_visible_devices=lambda device_type: [gpu])
        )
        with mock.patch.dict("sys.modules", {"tensorflow": tensorflow}):
            result = configure_worker_device(manifest, "gpu_neural")
        self.assertEqual(result["device"], "gpu")
        self.assertEqual(result["visible_gpu_count"], 1)

    def test_repeat_verifies_host_before_applying_queue_device_policy(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        config["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        config["execution"]["queues"] = HYBRID_EXECUTION_QUEUES
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", {"config": config})
            events: list[str] = []
            with mock.patch(
                "rushing_study.execution._verify_runtime_if_frozen",
                side_effect=lambda *args, **kwargs: events.append(
                    f"verify_tf={kwargs['verify_tensorflow']}"
                ),
            ), mock.patch(
                "rushing_study.execution.configure_worker_device",
                side_effect=lambda *args, **kwargs: events.append("device"),
            ), mock.patch(
                "rushing_study.execution.classify_scientific_cells",
                return_value={
                    CellKey("main", 1, 20, "ridge_sgd_l2"): SimpleNamespace(
                        is_complete=True,
                        status=CellStatus.COMPLETE,
                    )
                },
            ):
                result = execute_repeat(
                    storage,
                    "main",
                    1,
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    queue="cpu_tabular",
                )
        self.assertEqual(events, ["verify_tf=False", "device"])
        self.assertEqual(result, {"completed": 1, "executed": 0, "skipped": 1})

    def test_synthetic_two_size_stub_study_is_complete_and_resumable(self):
        config = {
            "models": {model: {"stub": True} for model in MODEL_IDS},
            "splits": {"nested_train_anchors": [20, 40, 80, 160, 240, 360]},
            "execution": {"confirmatory_repeats": 50},
            "uncertainty": {"alpha": 0.1, "local_k": 5},
            "sensitivity": {
                "stage1": {"anchors": [20, 160, 360], "repeats": 20},
                "extension": {"target_total_repeats": 50},
            },
        }
        registry = SeedRegistry()
        for n_train in (20, 40):
            for model in MODEL_IDS:
                stages = (
                    ("fit",)
                    if model in {"ridge_sgd_l2", "lightgbm_multiclass"}
                    else ("epoch_selection", "refit")
                )
                for stage in stages:
                    registry.get("cell", 1, stage, model, n_train)
        manifest = {
            "config": config,
            "study": "synthetic-two-anchor",
            "seed_registry": registry.snapshot(),
        }
        running_states = []

        def fake_cell(_data, _manifest, branch, repeat, n_train, model):
            running_states.append(
                storage.validate_cell(CellKey(branch, repeat, n_train, model)).status
            )
            probability_cal = np.full((6, 80), 1 / 80, dtype=np.float32)
            probability_test = np.full((4, 80), 1 / 80, dtype=np.float32)
            calibration_metadata = pd.DataFrame(
                {
                    "game_id": ["cal-a"] * 3 + ["cal-b"] * 3,
                    "play_id": [f"cal-{index}" for index in range(6)],
                    "season": [2017] * 6,
                }
            )
            test_metadata = pd.DataFrame(
                {
                    "game_id": ["test-a"] * 2 + ["test-b"] * 2,
                    "play_id": [f"test-{index}" for index in range(4)],
                    "season": [2018] * 4,
                }
            )
            evaluated = evaluate_cell(
                probability_cal,
                probability_test,
                np.array([4, 12, 24, 40, 58, 72]),
                np.array([8, 30, 55, 70]),
                calibration_metadata,
                test_metadata,
                alpha=0.1,
                local_k=5,
            )
            if model in {"ridge_sgd_l2", "lightgbm_multiclass"}:
                history = {
                    "fit_seed": manifest_seed(
                        manifest, "cell", repeat, "fit", model, n_train
                    )
                }
            else:
                history = {
                    "selection_seed": manifest_seed(
                        manifest,
                        "cell",
                        repeat,
                        "epoch_selection",
                        model,
                        n_train,
                    ),
                    "refit_seed": manifest_seed(
                        manifest, "cell", repeat, "refit", model, n_train
                    ),
                }
            model_config = frozen_model_config(config, model)
            config_hash = hashlib.sha256(
                json.dumps(
                    model_config, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
            history.update(
                {"model_config": model_config, "model_config_hash": config_hash}
            )
            return CellPayload(
                metrics={
                    **evaluated.metrics,
                    "branch": branch,
                    "repeat": repeat,
                    "n_train": n_train,
                    "model": model,
                    "model_config_hash": config_hash,
                    "parameter_count": 80,
                    "elapsed_seconds": 0.01,
                },
                calibration_predictions=evaluated.calibration,
                test_predictions=evaluated.test,
                arrays=evaluated.arrays,
                history=history,
                candidate_scores=[],
            )

        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            keys = expected_cell_keys(
                manifest,
                "main",
                repeat_ids=[1],
                anchors=[20, 40],
            )
            with mock.patch("rushing_study.execution.load_study_data", return_value=object()), mock.patch(
                "rushing_study.execution.run_cell", side_effect=fake_cell
            ):
                first = execute_repeat(storage, "main", 1, anchors=[20, 40], resume=True)
                marker_bytes = {
                    key: (storage.cell_dir(key) / "_SUCCESS").read_bytes() for key in keys
                }
                second = execute_repeat(storage, "main", 1, anchors=[20, 40], resume=True)

            self.assertEqual(first, {"completed": 8, "executed": 8, "skipped": 0})
            self.assertEqual(second, {"completed": 8, "executed": 0, "skipped": 8})
            self.assertEqual(running_states, [CellStatus.RUNNING] * 8)
            self.assertTrue(all(validate_scientific_cell(storage, key).is_complete for key in keys))
            self.assertEqual(
                marker_bytes,
                {key: (storage.cell_dir(key) / "_SUCCESS").read_bytes() for key in keys},
            )

    def test_scientific_validator_accepts_all_seed_contract_shapes(self):
        manifest = _scientific_manifest()
        keys = (
            CellKey("main", 1, 20, "ridge_sgd_l2"),
            CellKey("main", 1, 20, "zoo_cnn"),
            CellKey("sensitivity", 1, 20, "ridge_sgd_l2"),
            CellKey("sensitivity", 1, 20, "zoo_cnn"),
        )
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            for key in keys:
                with self.subTest(key=key):
                    _write_payload(storage, key, _valid_payload(manifest, key))
                    self.assertTrue(validate_scientific_cell(storage, key).is_complete)

    def test_scientific_validator_binds_split_games_seasons_plays_and_outcomes(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        payload = _valid_payload(manifest, key)
        payload.metrics["n_train_plays"] = 20
        train_games = [f"train-{index}" for index in range(20)]
        calibration_games = sorted(
            set(payload.calibration_predictions["game_id"].astype(str))
        )
        test_games = sorted(set(payload.test_predictions["game_id"].astype(str)))
        manifest["split_manifests"] = [
            {
                "repeat_id": 1,
                "calibration_games": calibration_games,
                "test_games": test_games,
                "anchors": {"20": {"train_games": train_games}},
                "by_season": {
                    "2017": {
                        "calibration_games": calibration_games,
                        "test_games": [],
                    },
                    "2018": {
                        "calibration_games": [],
                        "test_games": test_games,
                    },
                },
            }
        ]
        train_metadata = pd.DataFrame(
            {
                "game_id": train_games,
                "play_id": [f"train-play-{index}" for index in range(20)],
                "season": [2019] * 20,
            }
        )
        metadata = pd.concat(
            [
                train_metadata,
                payload.calibration_predictions[["game_id", "play_id", "season"]],
                payload.test_predictions[["game_id", "play_id", "season"]],
            ],
            ignore_index=True,
        )
        y = np.concatenate(
            [
                np.zeros(20, dtype=int),
                payload.calibration_predictions["y_class"].to_numpy(dtype=int),
                payload.test_predictions["y_class"].to_numpy(dtype=int),
            ]
        )
        n_rows = len(metadata)
        data = StudyData(
            spatial=np.zeros((n_rows, 11, 10, 10), dtype=np.float32),
            player_set=np.zeros((n_rows, 22, 10), dtype=np.float32),
            tabular=np.zeros((n_rows, 40), dtype=np.float32),
            y=y,
            metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            _write_payload(storage, key, payload)
            self.assertTrue(
                validate_scientific_cell(storage, key, study_data=data).is_complete
            )

            wrong_game = _valid_payload(manifest, key)
            wrong_game.metrics["n_train_plays"] = 20
            wrong_game.calibration_predictions.loc[0, "game_id"] = "not-in-split"
            _write_payload(storage, key, wrong_game)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("frozen repeat split", result.reason)

            wrong_play = _valid_payload(manifest, key)
            wrong_play.metrics["n_train_plays"] = 20
            wrong_play.test_predictions.loc[0, "play_id"] = "wrong-play"
            _write_payload(storage, key, wrong_play)
            result = validate_scientific_cell(storage, key, study_data=data)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("play IDs/order", result.reason)

    def test_sensitivity_extension_requires_triggered_immutable_stage1_receipt(self):
        manifest = _scientific_manifest()
        key = CellKey("sensitivity", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            storage.write_cell_artifacts(
                key,
                metrics={"ok": 1},
                predictions={"prediction": np.array([1.0])},
            )
            analysis_path = storage.run_dir / "stage1.csv"
            analysis_path.write_text("ok\n1\n", encoding="utf-8")
            register_existing_artifact(analysis_path)
            marker = storage.cell_dir(key) / "_SUCCESS"
            receipt = {
                "schema_version": 1,
                "manifest_hash": storage.manifest_hash,
                "branch": "sensitivity",
                "analysis_stage": "stage1",
                "repeat_ids": list(range(1, 21)),
                "anchors": [20, 160, 360],
                "models": list(MODEL_IDS),
                "extension_trigger": False,
                "cell_markers": [
                    {
                        "path": str(key.relative_dir),
                        "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
                    }
                ],
                "analysis_artifacts": [
                    {
                        "file": analysis_path.name,
                        "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                    }
                ],
            }
            receipt_path = storage.run_dir / SENSITIVITY_STAGE1_RECEIPT
            atomic_write_json(receipt_path, receipt)
            with mock.patch(
                "rushing_study.execution.expected_cell_keys", return_value=[key]
            ):
                with self.assertRaisesRegex(RuntimeError, "not authorized"):
                    validate_sensitivity_stage1_receipt(storage)
                receipt["extension_trigger"] = True
                atomic_write_json(receipt_path, receipt)
                validated_path, saved = validate_sensitivity_stage1_receipt(storage)
            self.assertEqual(validated_path, receipt_path)
            self.assertTrue(saved["extension_trigger"])

    def test_scientific_validator_rejects_identity_manifest_and_actual_seeds(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        sensitivity_neural = CellKey("sensitivity", 1, 20, "zoo_cnn")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)

            _write_payload(
                storage,
                key,
                _valid_payload(manifest, key),
                metric_overrides={"repeat": 2},
            )
            self.assertTrue(storage.validate_cell(key).is_complete)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("metric identity", result.reason)

            _write_payload(
                storage,
                key,
                _valid_payload(manifest, key),
                metric_overrides={"manifest_hash": "0" * 64},
            )
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("manifest hash", result.reason)

            wrong_main_seed = _valid_payload(manifest, key)
            wrong_main_seed.history["fit_seed"] += 1
            _write_payload(storage, key, wrong_main_seed)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("fit_seed", result.reason)

            wrong_sensitivity_seed = _valid_payload(manifest, sensitivity_neural)
            wrong_sensitivity_seed.history["prediction_rebuild_seed"] += 1
            _write_payload(storage, sensitivity_neural, wrong_sensitivity_seed)
            result = validate_scientific_cell(storage, sensitivity_neural)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("prediction_rebuild_seed", result.reason)

    def test_scientific_validator_rejects_wrong_model_configuration_provenance(self):
        manifest = _scientific_manifest()
        main_key = CellKey("main", 1, 20, "ridge_sgd_l2")
        sensitivity_key = CellKey("sensitivity", 1, 20, "ridge_sgd_l2")
        neural_key = CellKey("main", 1, 20, "zoo_cnn")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)

            wrong_hash = _valid_payload(manifest, main_key)
            wrong_hash.history["model_config_hash"] = "0" * 64
            _write_payload(storage, main_key, wrong_hash)
            result = validate_scientific_cell(storage, main_key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("history model_config_hash", result.reason)

            wrong_main = _valid_payload(manifest, main_key)
            wrong_main.history["model_config"] = {
                **wrong_main.history["model_config"],
                "unlocked": True,
            }
            changed_hash = hashlib.sha256(
                json.dumps(
                    wrong_main.history["model_config"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            wrong_main.history["model_config_hash"] = changed_hash
            wrong_main.metrics["model_config_hash"] = changed_hash
            _write_payload(storage, main_key, wrong_main)
            result = validate_scientific_cell(storage, main_key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("frozen manifest", result.reason)

            wrong_selected = _valid_payload(manifest, sensitivity_key)
            wrong_selected.metrics["selected_config"] = json.dumps({"wrong": True})
            _write_payload(storage, sensitivity_key, wrong_selected)
            result = validate_scientific_cell(storage, sensitivity_key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("selected_config differs", result.reason)

            wrong_parameter_count = _valid_payload(manifest, neural_key)
            wrong_parameter_count.metrics["parameter_count"] = 79
            _write_payload(storage, neural_key, wrong_parameter_count)
            result = validate_scientific_cell(storage, neural_key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("parameter_count", result.reason)

    def test_scientific_validator_binds_hybrid_cells_to_their_queue(self):
        manifest = _scientific_manifest()
        manifest["config"]["execution"].update(
            {
                "profile": HYBRID_EXECUTION_PROFILE,
                "queues": HYBRID_EXECUTION_QUEUES,
            }
        )
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            payload = _valid_payload(manifest, key)
            _write_payload(storage, key, payload)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("execution queue", result.reason)

            _write_payload(
                storage,
                key,
                payload,
                metric_overrides={"execution_queue": "cpu_tabular"},
            )
            self.assertTrue(validate_scientific_cell(storage, key).is_complete)

    def test_scientific_validator_rejects_probability_shape_normalization_and_finiteness(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            mutations = {
                "shape": lambda arrays: arrays.__setitem__(
                    "test_proba", arrays["test_proba"][:, :79]
                ),
                "normalization": lambda arrays: arrays.__setitem__(
                    "test_proba", arrays["test_proba"] * 0.5
                ),
                "nonfinite": lambda arrays: arrays["test_proba"].__setitem__(
                    (0, 0), np.nan
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    payload = _valid_payload(manifest, key)
                    mutate(payload.arrays)
                    _write_payload(storage, key, payload)
                    self.assertTrue(storage.validate_cell(key).is_complete)
                    result = validate_scientific_cell(storage, key)
                    self.assertEqual(result.status, CellStatus.CORRUPT)
                    self.assertIn("test_proba", result.reason)

    def test_scientific_validator_recomputes_intervals_predictions_and_metrics(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)

            bad_interval = _valid_payload(manifest, key)
            bad_interval.test_predictions.loc[0, "conformal_lo"] = 79
            bad_interval.test_predictions.loc[0, "conformal_hi"] = 0
            _write_payload(storage, key, bad_interval)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("interval ordering", result.reason)

            bad_prediction = _valid_payload(manifest, key)
            bad_prediction.test_predictions.loc[0, "predictive_mean_class"] += 2.0
            _write_payload(storage, key, bad_prediction)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("predictive_mean_class", result.reason)

            valid = _valid_payload(manifest, key)
            _write_payload(
                storage,
                key,
                valid,
                metric_overrides={"crps": float(valid.metrics["crps"]) + 0.1},
            )
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("metric 'crps'", result.reason)
            with self.assertRaisesRegex(RuntimeError, "scientific validation failed"):
                load_metrics_frame(storage, [key])

    def test_scientific_validator_rejects_checksummed_nonfinite_metric(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            _write_payload(storage, key, _valid_payload(manifest, key))
            metrics_path = storage.cell_dir(key) / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["crps"] = float("nan")
            metrics_path.write_text(
                json.dumps(metrics, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
            )
            record = register_existing_artifact(metrics_path, "json")
            marker_path = storage.cell_dir(key) / "_SUCCESS"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["artifacts"]["metrics"]["sha256"] = record.sha256
            marker_path.write_text(
                json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )

            self.assertTrue(storage.validate_cell(key).is_complete)
            result = validate_scientific_cell(storage, key)
            self.assertEqual(result.status, CellStatus.CORRUPT)
            self.assertIn("nonfinite", result.reason)

    def test_resume_overwrites_a_checksummed_scientifically_invalid_cell(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            valid_payload = _valid_payload(manifest, key)
            _write_payload(
                storage,
                key,
                valid_payload,
                metric_overrides={"crps": float(valid_payload.metrics["crps"]) + 0.1},
            )
            old_marker = (storage.cell_dir(key) / "_SUCCESS").read_bytes()
            self.assertEqual(
                classify_scientific_cells(storage, [key])[key].status,
                CellStatus.CORRUPT,
            )

            observed_running = []

            def rerun(*_args, **_kwargs):
                observed_running.append(storage.validate_cell(key).status)
                return _valid_payload(manifest, key)

            with mock.patch("rushing_study.execution.load_study_data", return_value=object()), mock.patch(
                "rushing_study.execution.run_cell", side_effect=rerun
            ):
                result = execute_repeat(
                    storage,
                    "main",
                    1,
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    resume=True,
                )

            self.assertEqual(result, {"completed": 1, "executed": 1, "skipped": 0})
            self.assertEqual(observed_running, [CellStatus.RUNNING])
            self.assertTrue(validate_scientific_cell(storage, key).is_complete)
            self.assertNotEqual(old_marker, (storage.cell_dir(key) / "_SUCCESS").read_bytes())
            self.assertFalse((storage.cell_dir(key) / "_RUNNING").exists())

    def test_resume_does_not_duplicate_a_live_running_cell(self):
        manifest = _scientific_manifest()
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            storage.mark_cell_running(key)
            with mock.patch("rushing_study.execution.run_cell") as run_cell_mock, mock.patch(
                "rushing_study.execution.load_study_data"
            ) as load_mock:
                result = execute_repeat(
                    storage,
                    "main",
                    1,
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    resume=True,
                )

            self.assertEqual(result, {"completed": 1, "executed": 0, "skipped": 1})
            self.assertEqual(storage.validate_cell(key).status, CellStatus.RUNNING)
            run_cell_mock.assert_not_called()
            load_mock.assert_not_called()

    def test_finalized_run_noops_only_for_scientifically_complete_requested_cells(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        registry = SeedRegistry(namespace=config["study_id"])
        for stage in ("fit", "epoch_selection", "refit", "prediction_rebuild"):
            registry.get("cell", 1, stage, "ridge_sgd_l2", 20)
        manifest = {"config": config, "seed_registry": registry.snapshot()}
        key = CellKey("main", 1, 20, "ridge_sgd_l2")
        with tempfile.TemporaryDirectory() as directory:
            storage = initialize_run_dir(Path(directory) / "run", manifest)
            _write_payload(storage, key, _valid_payload(manifest, key))
            storage.finalize_run([key])
            root_marker = (storage.run_dir / "_SUCCESS").read_bytes()

            with mock.patch("rushing_study.execution.subprocess.run") as launched:
                schedule_subprocesses(
                    storage.run_dir,
                    "main",
                    repeat_ids=[1],
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    workers=1,
                )
            launched.assert_not_called()

            metrics_path = storage.cell_dir(key) / "metrics.json"
            metrics_path.write_text("{}\n", encoding="utf-8")
            with mock.patch("rushing_study.execution.subprocess.run") as launched:
                with self.assertRaisesRegex(RuntimeError, "Finalized run"):
                    schedule_subprocesses(
                        storage.run_dir,
                        "main",
                        repeat_ids=[1],
                        anchors=[20],
                        models=["ridge_sgd_l2"],
                        workers=1,
                    )
            launched.assert_not_called()
            self.assertEqual((storage.run_dir / "_SUCCESS").read_bytes(), root_marker)

    def test_scheduler_and_worker_enforce_frozen_runtime_provenance(self):
        config = load_config("configs/rushing_confirmatory_v1.json")
        scheduler_manifest = {
            "config": config,
            "provenance": {"code": {"repo_root": "/frozen/repo"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            scheduler_storage = initialize_run_dir(
                Path(directory) / "scheduler", scheduler_manifest
            )
            with mock.patch(
                "rushing_study.execution.verify_runtime_provenance"
            ) as verify, mock.patch("rushing_study.execution.subprocess.run"):
                schedule_subprocesses(
                    scheduler_storage.run_dir,
                    "main",
                    repeat_ids=[1],
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    workers=1,
                )
            verify.assert_called_once_with(
                scheduler_storage.manifest,
                "/frozen/repo",
                require_environment=False,
            )

            worker_manifest = _scientific_manifest()
            worker_manifest["provenance"] = {"code": {"repo_root": "/frozen/repo"}}
            worker_storage = initialize_run_dir(Path(directory) / "worker", worker_manifest)
            key = CellKey("main", 1, 20, "ridge_sgd_l2")
            worker_storage.mark_cell_running(key)
            with mock.patch(
                "rushing_study.execution.verify_runtime_provenance"
            ) as verify, mock.patch("rushing_study.execution.load_study_data"):
                execute_repeat(
                    worker_storage,
                    "main",
                    1,
                    anchors=[20],
                    models=["ridge_sgd_l2"],
                    resume=True,
                )
            verify.assert_called_once_with(
                worker_storage.manifest,
                "/frozen/repo",
                require_environment=True,
            )


if __name__ == "__main__":
    unittest.main()
