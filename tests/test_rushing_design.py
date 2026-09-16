"""Contract tests for the immutable rushing confirmatory-study design."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rushing_study.design import (
    CANONICAL_MODEL_IDS,
    ConfigValidationError,
    DETERMINISTIC_ENVIRONMENT,
    FULL_CONFIRMATORY_REPEATS,
    HYBRID_EXECUTION_PROFILE,
    HYBRID_EXECUTION_QUEUES,
    DesignError,
    IdentifierValidationError,
    LOCKED_FINAL_CONFIG_SHA256,
    ManifestCollisionError,
    ProvenanceError,
    REQUIRED_CODE_PROVENANCE,
    SMOKE_DESCRIPTION_PREFIX,
    SeedCollisionError,
    SeedRegistry,
    build_split_manifests,
    build_study_manifest,
    canonical_json,
    collect_code_provenance,
    collect_data_provenance,
    collect_environment_provenance,
    declared_execution_settings,
    largest_remainder_allocation,
    load_config,
    manifest_seed,
    parse_requirements_lock,
    proportional_nested_allocations,
    sha256_file,
    sha256_json,
    stable_seed,
    validate_config,
    verify_runtime_provenance,
    write_manifest_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rushing_confirmatory_v1.json"
SEASON_SIZES = {2017: 256, 2018: 256, 2019: 176}


def synthetic_games() -> dict[int, list[str]]:
    """Return the exact season-complete one-row-per-game universe."""

    return {
        season: [f"{season}-{number:03d}" for number in range(size)]
        for season, size in SEASON_SIZES.items()
    }


def explicit_smoke_config(config: dict, max_epochs: int = 1) -> dict:
    """Apply exactly the smoke transformation performed by cli.command_plan."""

    smoke = deepcopy(config)
    smoke["execution"]["plan"] = "smoke"
    smoke["neural_training"]["max_epochs"] = max_epochs
    smoke["neural_training"]["early_stopping_patience"] = 0
    smoke["description"] = SMOKE_DESCRIPTION_PREFIX + smoke["description"]
    return smoke


class ConfigAndSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_config_freezes_requested_design_and_models(self) -> None:
        self.assertEqual(sha256_json(self.config), LOCKED_FINAL_CONFIG_SHA256)
        self.assertEqual(self.config["execution"]["confirmatory_repeats"], 50)
        self.assertFalse(self.config["execution"]["retune_hyperparameters"])
        self.assertEqual(self.config["splits"]["nested_train_anchors"], [20, 40, 80, 160, 240, 360])
        self.assertEqual(self.config["sensitivity"]["stage1"]["repeat_ids"], list(range(1, 21)))
        self.assertEqual(
            self.config["sensitivity"]["extension"]["additional_repeat_ids"],
            list(range(21, 51)),
        )
        self.assertEqual(self.config["analysis"]["bootstrap_draws"], 10_000)
        self.assertEqual(self.config["analysis"]["coverage_target"], 0.90)
        self.assertEqual(self.config["execution"]["workers"], 1)
        self.assertEqual(self.config["execution"]["lightgbm_n_jobs"], 1)
        self.assertEqual(self.config["execution"]["environment"], DETERMINISTIC_ENVIRONMENT)
        self.assertNotIn("require_clean_repo_for_final", self.config["provenance"])
        self.assertNotIn("allow_dirty_repo_for_smoke", self.config["execution"])
        self.assertEqual(
            declared_execution_settings(self.config),
            {
                "worker_count": 1,
                "lightgbm_n_jobs": 1,
                "environment_variables": dict(sorted(DETERMINISTIC_ENVIRONMENT.items())),
            },
        )
        self.assertEqual(set(self.config["models"]), set(CANONICAL_MODEL_IDS))
        self.assertEqual(self.config["models"]["ridge_sgd_l2"]["eta0"], 0.01)
        self.assertEqual(
            {
                name: self.config["provenance"]["code_files"][name]
                for name in REQUIRED_CODE_PROVENANCE
            },
            REQUIRED_CODE_PROVENANCE,
        )
        self.assertEqual(
            self.config["sensitivity"]["grids"]["ridge_sgd_l2"]["values"],
            [1 / 3, 10.0, 10 / 3, 1.0, 0.1],
        )
        expected_neural_grid = [
            {"learning_rate": 1e-3, "dropout": 0.3},
            {"learning_rate": 3e-4, "dropout": 0.1},
            {"learning_rate": 3e-4, "dropout": 0.3},
            {"learning_rate": 1e-3, "dropout": 0.1},
        ]
        self.assertEqual(self.config["sensitivity"]["grids"]["zoo_cnn"], expected_neural_grid)
        self.assertEqual(
            self.config["sensitivity"]["grids"]["set_transformer"], expected_neural_grid
        )

    def test_entire_final_config_is_locked_and_only_explicit_smoke_is_allowed(self) -> None:
        mutations = {
            "CNN dropout": lambda value: value["models"]["zoo_cnn"].update(dropout=0.9),
            "Ridge epochs": lambda value: value["models"]["ridge_sgd_l2"].update(epochs=1),
            "tree estimators": lambda value: value["models"]["lightgbm_multiclass"].update(
                n_estimators=1
            ),
            "neural learning rate": lambda value: value["neural_training"].update(
                learning_rate=0.5
            ),
            "later sensitivity candidate": lambda value: value["sensitivity"]["grids"][
                "zoo_cnn"
            ][1].update(dropout=0.99),
            "unrecognized field": lambda value: value.update(extra_contract_field=True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = deepcopy(self.config)
                mutate(changed)
                with self.assertRaises(ConfigValidationError):
                    validate_config(changed)

        smoke = explicit_smoke_config(self.config, max_epochs=2)
        validate_config(smoke)
        smoke["features"]["use_mirror_augmentation"] = True
        with self.assertRaisesRegex(ConfigValidationError, "Smoke configuration"):
            validate_config(smoke)

    def test_full_profile_is_the_exact_100_repeat_projection(self) -> None:
        full = deepcopy(self.config)
        full["execution"]["confirmatory_repeats"] = FULL_CONFIRMATORY_REPEATS
        validate_config(full)
        validate_config(explicit_smoke_config(full, max_epochs=1))

        invalid_count = deepcopy(self.config)
        invalid_count["execution"]["confirmatory_repeats"] = 75
        with self.assertRaisesRegex(ConfigValidationError, "50-repeat default or 100-repeat"):
            validate_config(invalid_count)

        changed_full = deepcopy(full)
        changed_full["models"]["zoo_cnn"]["dropout"] = 0.1
        with self.assertRaises(ConfigValidationError):
            validate_config(changed_full)

    def test_hybrid_profile_is_an_exact_locked_projection(self) -> None:
        hybrid = deepcopy(self.config)
        hybrid["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        hybrid["execution"]["queues"] = deepcopy(HYBRID_EXECUTION_QUEUES)
        validate_config(hybrid)
        settings = declared_execution_settings(hybrid)
        self.assertEqual(settings["profile"], HYBRID_EXECUTION_PROFILE)
        self.assertEqual(settings["queues"], HYBRID_EXECUTION_QUEUES)
        validate_config(explicit_smoke_config(hybrid, max_epochs=1))

        full_hybrid = deepcopy(hybrid)
        full_hybrid["execution"]["confirmatory_repeats"] = FULL_CONFIRMATORY_REPEATS
        validate_config(full_hybrid)
        validate_config(explicit_smoke_config(full_hybrid, max_epochs=1))

        changed = deepcopy(hybrid)
        changed["execution"]["queues"]["cpu_tabular"]["workers"] = 11
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)

    def test_ridge_eta0_matches_pinned_sklearn_default(self) -> None:
        import inspect

        from sklearn.linear_model import SGDClassifier

        pinned_default = inspect.signature(SGDClassifier).parameters["eta0"].default
        self.assertEqual(pinned_default, 0.01)
        self.assertEqual(self.config["models"]["ridge_sgd_l2"]["eta0"], pinned_default)

    def test_canonical_json_and_seed_are_stable(self) -> None:
        left = {"z": 1, "a": [True, {"b": 2, "a": 1}]}
        right = {"a": [True, {"a": 1, "b": 2}], "z": 1}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(sha256_json(left), sha256_json(right))
        self.assertEqual(stable_seed(20260817, "analysis", "bootstrap"), 2_127_296_457)
        self.assertEqual(
            stable_seed(
                20260817,
                "rushing_confirmatory_v1",
                "cell",
                1,
                "fit",
                "ridge_sgd_l2",
                20,
            ),
            534_632_946,
        )
        self.assertNotEqual(
            stable_seed(20260817, "study-a", "cell", 1),
            stable_seed(20260817, "study-b", "cell", 1),
        )

    def test_config_rejects_confirmatory_retuning(self) -> None:
        changed = deepcopy(self.config)
        changed["execution"]["retune_hyperparameters"] = True
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)

    def test_config_rejects_previous_source_layout(self) -> None:
        legacy = deepcopy(self.config)
        legacy["provenance"]["code_files"]["neural_models"] = "neural_models.py"
        legacy["provenance"]["code_files"]["prepare_data"] = "prepare_data.py"
        legacy["provenance"]["code_files"]["training_size_sweep"] = "training_size_sweep.py"
        del legacy["provenance"]["code_files"]["rushing_study_intervals"]
        legacy["models"]["zoo_cnn"]["implementation"] = "neural_models.get_conv_net"
        legacy["models"]["set_transformer"]["implementation"] = "neural_models.get_set_transformer"
        with self.assertRaises(ConfigValidationError):
            validate_config(legacy)

    def test_config_rejects_missing_local_code_dependency(self) -> None:
        changed = deepcopy(self.config)
        del changed["provenance"]["code_files"]["neural_models"]
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)

    def test_config_rejects_nondeterministic_worker_or_thread_settings(self) -> None:
        changed = deepcopy(self.config)
        changed["execution"]["workers"] = 2
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)
        changed = deepcopy(self.config)
        changed["execution"]["environment"]["TF_NUM_INTRAOP_THREADS"] = "2"
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)
        changed = deepcopy(self.config)
        changed["provenance"]["environment_variables"].remove("OMP_NUM_THREADS")
        with self.assertRaises(ConfigValidationError):
            validate_config(changed)

    def test_seed_registry_rejects_integer_collisions(self) -> None:
        registry = SeedRegistry(seed_function=lambda _base, *_parts: 7)
        registry.get("first")
        with self.assertRaises(SeedCollisionError):
            registry.get("second")


class AllocationTests(unittest.TestCase):
    def test_largest_remainder_has_exact_deterministic_ties(self) -> None:
        capacities = {"2017": 139, "2018": 139, "2019": 96}
        self.assertEqual(
            largest_remainder_allocation(20, capacities, ["2017", "2018", "2019"]),
            {"2017": 8, "2018": 7, "2019": 5},
        )
        self.assertEqual(
            proportional_nested_allocations(
                capacities,
                [20, 40, 80, 160, 240, 360],
                ["2017", "2018", "2019"],
            ),
            {
                "20": {"2017": 8, "2018": 7, "2019": 5},
                "40": {"2017": 15, "2018": 15, "2019": 10},
                "80": {"2017": 30, "2018": 30, "2019": 20},
                "160": {"2017": 60, "2018": 59, "2019": 41},
                "240": {"2017": 89, "2018": 89, "2019": 62},
                "360": {"2017": 134, "2018": 134, "2019": 92},
            },
        )


class SplitManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.games = synthetic_games()
        cls.splits, cls.registry = build_split_manifests(cls.config, cls.games)

    def test_outer_counts_partitions_and_season_stratification(self) -> None:
        self.assertEqual(len(self.splits), 50)
        self.assertEqual(len({split["split_hash"] for split in self.splits}), 50)
        expected_by_season = {
            "2017": (139, 15, 51, 51),
            "2018": (139, 15, 51, 51),
            "2019": (96, 10, 35, 35),
        }
        for split in self.splits:
            self.assertEqual(len(split["train_games"]), 374)
            self.assertEqual(len(split["tune_games"]), 40)
            self.assertEqual(len(split["calibration_games"]), 137)
            self.assertEqual(len(split["test_games"]), 137)
            partitions = [
                split["train_games"],
                split["tune_games"],
                split["calibration_games"],
                split["test_games"],
            ]
            self.assertEqual(sum(map(len, partitions)), 688)
            self.assertEqual(len(set().union(*map(set, partitions))), 688)
            for season, expected in expected_by_season.items():
                record = split["by_season"][season]
                actual = tuple(
                    len(record[field])
                    for field in (
                        "train_games",
                        "tune_games",
                        "calibration_games",
                        "test_games",
                    )
                )
                self.assertEqual(actual, expected)

    def test_anchors_are_nested_and_inner_splits_are_exact(self) -> None:
        split = self.splits[0]
        prior: set[str] = set()
        for anchor in self.config["splits"]["nested_train_anchors"]:
            record = split["anchors"][str(anchor)]
            train = set(record["train_games"])
            fit = set(record["neural_fit_games"])
            validation = set(record["neural_validation_games"])
            self.assertEqual(len(train), anchor)
            self.assertTrue(prior.issubset(train))
            self.assertFalse(fit & validation)
            self.assertEqual(fit | validation, train)
            self.assertEqual(len(fit), anchor * 4 // 5)
            self.assertEqual(len(validation), anchor // 5)
            prior = train
        unhashed = deepcopy(split)
        split_hash = unhashed.pop("split_hash")
        self.assertEqual(split_hash, sha256_json(unhashed))

    def test_all_required_runner_seeds_are_materialized(self) -> None:
        self.assertEqual(len(self.registry), 6_201)
        manifest = {"seed_registry": self.registry}
        self.assertEqual(
            manifest_seed(manifest, "analysis", "bootstrap"),
            stable_seed(
                self.config["base_seed"],
                self.config["study_id"],
                "analysis",
                "bootstrap",
            ),
        )
        for parts in (
            ("cell", 50, "prediction_rebuild", "set_transformer", 360),
            ("cell", 1, "fit", "zoo_cnn", 20),
            ("cell", 50, "refit", "lightgbm_multiclass", 360),
        ):
            self.assertIsInstance(manifest_seed(manifest, *parts), int)
        with self.assertRaises(KeyError):
            manifest_seed(manifest, "cell", 51, "fit", "ridge_sgd_l2", 20)

    def test_anchor_tie_order_is_repeat_seeded_and_materialized(self) -> None:
        observed_orders = set()
        manifest = {"seed_registry": self.registry}
        capacities = {"2017": 139, "2018": 139, "2019": 96}
        for split in self.splits:
            repeat_id = split["repeat_id"]
            seed = manifest_seed(manifest, "split", "anchor_allocation_tie_order", repeat_id)
            self.assertEqual(split["anchor_allocation_tie_seed"], seed)
            tie_order = split["anchor_allocation_tie_order"]
            self.assertEqual(set(tie_order), set(capacities))
            observed_orders.add(tuple(tie_order))
            expected_allocations = proportional_nested_allocations(
                capacities,
                self.config["splits"]["nested_train_anchors"],
                tie_order,
            )
            for anchor, allocation in expected_allocations.items():
                self.assertEqual(split["anchors"][anchor]["season_counts"], allocation)
        self.assertGreater(len(observed_orders), 1)

    def test_same_inputs_rebuild_byte_identical_splits_and_seeds(self) -> None:
        splits, registry = build_split_manifests(self.config, self.games)
        self.assertEqual(canonical_json(splits), canonical_json(self.splits))
        self.assertEqual(registry, self.registry)

    def test_full_profile_materializes_100_splits_and_complete_seed_registry(self) -> None:
        full = deepcopy(self.config)
        full["execution"]["confirmatory_repeats"] = FULL_CONFIRMATORY_REPEATS
        splits, registry = build_split_manifests(full, self.games)
        self.assertEqual(len(splits), 100)
        self.assertEqual(len({split["split_hash"] for split in splits}), 100)
        self.assertEqual(len(registry), 12_401)
        manifest = {"seed_registry": registry}
        self.assertIsInstance(
            manifest_seed(
                manifest,
                "cell",
                100,
                "prediction_rebuild",
                "set_transformer",
                360,
            ),
            int,
        )
        with self.assertRaises(KeyError):
            manifest_seed(manifest, "cell", 101, "fit", "ridge_sgd_l2", 20)

    def test_augmented_and_colliding_ids_are_rejected(self) -> None:
        augmented = deepcopy(self.games)
        augmented[2017][0] += "_aug"
        with self.assertRaises(IdentifierValidationError):
            build_split_manifests(self.config, augmented)

        colliding = deepcopy(self.games)
        colliding[2018][0] = colliding[2017][0]
        with self.assertRaises(IdentifierValidationError):
            build_split_manifests(self.config, colliding)

    def test_augmented_play_id_is_rejected_when_supplied(self) -> None:
        records = [
            {"Season": season, "GameId": game_id, "PlayId": f"play-{game_id}"}
            for season, ids in self.games.items()
            for game_id in ids
        ]
        records[0]["PlayId"] += "_aug"
        with self.assertRaises(IdentifierValidationError):
            build_split_manifests(self.config, records)


class ProvenanceAndBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_code_provenance_is_hash_only_and_does_not_require_git(self) -> None:
        config = deepcopy(self.config)
        config["provenance"]["code_files"] = {
            "z_file": "study/z.py",
            "a_file": "study/a.py",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "study").mkdir()
            (root / "study" / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "study" / "z.py").write_text("z = 1\n", encoding="utf-8")
            provenance = collect_code_provenance(config, root)

        self.assertEqual(set(provenance), {"repo_root", "files"})
        self.assertEqual(set(provenance["files"]), {"a_file", "z_file"})
        self.assertNotIn("git", canonical_json(provenance).lower())

    def test_declared_data_file_provenance_records_hash_and_size(self) -> None:
        smoke = deepcopy(self.config)
        smoke["execution"]["plan"] = "smoke"
        smoke["provenance"]["data_files"] = {"sample": "inputs/sample.bin"}
        smoke["provenance"]["code_files"] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "inputs" / "sample.bin"
            target.parent.mkdir()
            target.write_bytes(b"immutable-input\n")
            provenance = collect_data_provenance(smoke, root)
            self.assertEqual(provenance["files"]["sample"]["size_bytes"], len(b"immutable-input\n"))
            self.assertEqual(provenance["files"]["sample"]["sha256"], sha256_file(target))

    def test_requirements_lock_honors_platform_markers_without_relaxing_pins(self) -> None:
        config = deepcopy(self.config)
        config["provenance"]["code_files"]["requirements_lock"] = "requirements-lock.txt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements-lock.txt").write_text(
                "always-on==1.2.3\n"
                "never-on==9.9.9; python_version < '0'\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_requirements_lock(config, root),
                {"always-on": "1.2.3"},
            )

            (root / "requirements-lock.txt").write_text(
                "not-exact>=1.2.3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ProvenanceError, "exact name==version"):
                parse_requirements_lock(config, root)

    def test_environment_provenance_records_settings_os_hardware_and_tensorflow(self) -> None:
        devices = [
            SimpleNamespace(name="/physical_device:GPU:0", device_type="GPU"),
            SimpleNamespace(name="/physical_device:CPU:0", device_type="CPU"),
        ]
        tensorflow = SimpleNamespace(
            __version__="9.9.0-test",
            config=SimpleNamespace(
                list_physical_devices=lambda: devices,
                threading=SimpleNamespace(
                    get_inter_op_parallelism_threads=lambda: 1,
                    get_intra_op_parallelism_threads=lambda: 1,
                ),
            ),
            sysconfig=SimpleNamespace(
                get_build_info=lambda: {"is_cuda_build": True, "cpu_compiler": "test-cc"}
            ),
        )
        with (
            patch("rushing_study.design.os.cpu_count", return_value=12),
            patch("rushing_study.design.platform.processor", return_value="test-cpu"),
            patch.dict(os.environ, DETERMINISTIC_ENVIRONMENT, clear=False),
            patch("rushing_study.design.importlib.import_module", return_value=tensorflow),
        ):
            provenance = collect_environment_provenance(self.config, ROOT)
        expected_packages = parse_requirements_lock(self.config, ROOT)
        self.assertEqual(provenance["declared_execution_settings"]["worker_count"], 1)
        self.assertEqual(provenance["declared_execution_settings"]["lightgbm_n_jobs"], 1)
        self.assertEqual(
            provenance["declared_execution_settings"]["environment_variables"],
            DETERMINISTIC_ENVIRONMENT,
        )
        self.assertEqual(
            {name: provenance["environment_variables"][name] for name in DETERMINISTIC_ENVIRONMENT},
            DETERMINISTIC_ENVIRONMENT,
        )
        self.assertIn("system", provenance["os"])
        self.assertEqual(provenance["hardware"]["cpu_count"], 12)
        self.assertEqual(provenance["hardware"]["processor"], "test-cpu")
        self.assertEqual(provenance["packages"], expected_packages)
        self.assertEqual(provenance["requirements_lock"]["package_count"], 53)
        self.assertEqual(
            provenance["requirements_lock"]["expected_packages"], expected_packages
        )
        self.assertEqual(
            provenance["requirements_lock"]["sha256"],
            sha256_file(ROOT / "requirements-lock.txt"),
        )
        self.assertEqual(provenance["tensorflow"]["version"], "9.9.0-test")
        self.assertEqual(
            provenance["tensorflow"]["physical_devices"],
            [
                {"name": "/physical_device:CPU:0", "device_type": "CPU"},
                {"name": "/physical_device:GPU:0", "device_type": "GPU"},
            ],
        )
        self.assertTrue(provenance["tensorflow"]["build_info"]["is_cuda_build"])
        self.assertEqual(
            provenance["tensorflow"]["runtime_threading"],
            {"inter_op_threads": 1, "intra_op_threads": 1},
        )

    def test_hybrid_environment_requires_a_tensorflow_visible_gpu(self) -> None:
        hybrid = deepcopy(self.config)
        hybrid["execution"]["profile"] = HYBRID_EXECUTION_PROFILE
        hybrid["execution"]["queues"] = deepcopy(HYBRID_EXECUTION_QUEUES)
        tensorflow = {
            "available": True,
            "version": "test",
            "physical_devices": [
                {"name": "/physical_device:CPU:0", "device_type": "CPU"}
            ],
            "build_info": {},
            "runtime_threading": {"inter_op_threads": 1, "intra_op_threads": 1},
        }
        with patch(
            "rushing_study.design._tensorflow_environment_provenance",
            return_value=tensorflow,
        ):
            with self.assertRaisesRegex(ProvenanceError, "requires a TensorFlow-visible GPU"):
                collect_environment_provenance(hybrid, ROOT)

    def test_cpu_child_environment_check_can_defer_tensorflow_import(self) -> None:
        with patch(
            "rushing_study.design._tensorflow_environment_provenance",
            side_effect=AssertionError("TensorFlow must not be imported"),
        ):
            provenance = collect_environment_provenance(
                self.config,
                ROOT,
                include_tensorflow=False,
            )
        self.assertEqual(
            provenance["tensorflow"],
            {"verification_skipped": True},
        )

    def test_environment_provenance_tolerates_unavailable_tensorflow(self) -> None:
        with patch(
            "rushing_study.design.importlib.import_module",
            side_effect=ImportError("tensorflow unavailable"),
        ):
            provenance = collect_environment_provenance(self.config, ROOT)
        self.assertFalse(provenance["tensorflow"]["available"])
        self.assertEqual(provenance["tensorflow"]["import_error"]["type"], "ImportError")

    def test_environment_provenance_rejects_any_requirements_lock_mismatch(self) -> None:
        expected = parse_requirements_lock(self.config, ROOT)

        def installed_version(name: str) -> str:
            if name == "numpy":
                return "0.0.invalid"
            return expected[name]

        with patch(
            "rushing_study.design.importlib.metadata.version",
            side_effect=installed_version,
        ):
            with self.assertRaisesRegex(ProvenanceError, "numpy==2.0.2"):
                collect_environment_provenance(self.config, ROOT)

    def _frozen_runtime_manifest(self, root: Path) -> tuple[dict, dict, Path, Path]:
        code_path = root / "frozen_code.py"
        data_path = root / "frozen_data.bin"
        code_path.write_bytes(b"code-v1\n")
        data_path.write_bytes(b"data-v1\n")
        frozen_environment = {
            "declared_execution_settings": declared_execution_settings(self.config),
            "python": {
                "version": "3.12.test",
                "implementation": "CPython",
                "executable": "/frozen/python",
            },
            "os": {
                "system": "TestOS",
                "release": "1",
                "version": "1.0",
                "platform": "TestOS-1",
            },
            "hardware": {
                "machine": "test-machine",
                "processor": "test-processor",
                "cpu_count": 8,
                "byteorder": "little",
            },
            "packages": {"numpy": "2.test", "tensorflow": "9.test"},
            "requirements_lock": {
                "path": "requirements-lock.txt",
                "size_bytes": 123,
                "sha256": "d" * 64,
                "expected_packages": {"numpy": "2.test", "tensorflow": "9.test"},
                "package_count": 2,
            },
            "environment_variables": {name: None for name in DETERMINISTIC_ENVIRONMENT},
            "tensorflow": {
                "available": True,
                "version": "9.test",
                "physical_devices": [
                    {"name": "/physical_device:CPU:0", "device_type": "CPU"}
                ],
                "build_info": {"test_build": True},
                "runtime_threading": {"inter_op_threads": 1, "intra_op_threads": 1},
            },
        }
        payload = {
            "schema_version": 1,
            "study_id": self.config["study_id"],
            "config": deepcopy(self.config),
            "config_hash": sha256_json(self.config),
            "seed_registry": {},
            "split_manifests": [],
            "provenance": {
                "code": {
                    "files": {
                        "code": {
                            "path": code_path.name,
                            "size_bytes": code_path.stat().st_size,
                            "sha256": sha256_file(code_path),
                        }
                    },
                },
                "data": {
                    "files": {
                        "data": {
                            "path": data_path.name,
                            "size_bytes": data_path.stat().st_size,
                            "sha256": sha256_file(data_path),
                        }
                    }
                },
                "environment": frozen_environment,
            },
        }
        manifest_hash = sha256_json(payload)
        payload["manifest_hash"] = manifest_hash
        payload["run_hash"] = manifest_hash[:24]
        return payload, frozen_environment, code_path, data_path

    def test_runtime_verifier_can_defer_worker_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, current_environment, _code_path, _data_path = self._frozen_runtime_manifest(root)

            with (
                patch(
                    "rushing_study.design.collect_environment_provenance",
                    return_value=current_environment,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                report = verify_runtime_provenance(manifest, root, require_environment=False)
                self.assertTrue(report["verified"])
                self.assertEqual(report["code_files_verified"], 1)
                self.assertEqual(report["data_files_verified"], 1)
                with self.assertRaisesRegex(ProvenanceError, "environment variable"):
                    verify_runtime_provenance(manifest, root, require_environment=True)

            with (
                patch(
                    "rushing_study.design.collect_environment_provenance",
                    return_value=current_environment,
                ),
                patch.dict(os.environ, DETERMINISTIC_ENVIRONMENT, clear=True),
            ):
                report = verify_runtime_provenance(manifest, root, require_environment=True)
                self.assertTrue(report["environment_required"])

    def test_runtime_verifier_rejects_declared_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, current_environment, code_path, data_path = self._frozen_runtime_manifest(root)
            with (
                patch(
                    "rushing_study.design.collect_environment_provenance",
                    return_value=current_environment,
                ),
            ):
                report = verify_runtime_provenance(manifest, root)
                self.assertTrue(report["verified"])

            code_path.write_bytes(b"code-v2\n")  # Same size, different hash.
            with patch(
                "rushing_study.design.collect_environment_provenance",
                return_value=current_environment,
            ):
                with self.assertRaisesRegex(ProvenanceError, "code.*SHA-256"):
                    verify_runtime_provenance(manifest, root)
            code_path.write_bytes(b"code-v1\n")
            data_path.write_bytes(b"data-v1-expanded\n")
            with patch(
                "rushing_study.design.collect_environment_provenance",
                return_value=current_environment,
            ):
                with self.assertRaisesRegex(ProvenanceError, "data.*size changed"):
                    verify_runtime_provenance(manifest, root)

    def test_runtime_verifier_rejects_version_os_hardware_and_device_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, frozen_environment, _code_path, _data_path = self._frozen_runtime_manifest(root)
            mutations = {
                "Python version": lambda value: value["python"].update(version="3.13.changed"),
                "package versions": lambda value: value["packages"].update(numpy="changed"),
                "requirements lock details": lambda value: value["requirements_lock"].update(
                    sha256="changed"
                ),
                "OS": lambda value: value["os"].update(release="changed"),
                "hardware machine": lambda value: value["hardware"].update(machine="changed"),
                "hardware CPU count": lambda value: value["hardware"].update(cpu_count=99),
                "TensorFlow physical device inventory": lambda value: value["tensorflow"].update(
                    physical_devices=[
                        {"name": "/physical_device:GPU:0", "device_type": "GPU"}
                    ]
                ),
                "TensorFlow version": lambda value: value["tensorflow"].update(
                    version="changed"
                ),
                "TensorFlow build info": lambda value: value["tensorflow"].update(
                    build_info={"changed": True}
                ),
                "TensorFlow runtime threading": lambda value: value["tensorflow"].update(
                    runtime_threading={"inter_op_threads": 2, "intra_op_threads": 1}
                ),
            }
            for expected_message, mutate in mutations.items():
                with self.subTest(expected_message=expected_message):
                    current = deepcopy(frozen_environment)
                    mutate(current)
                    with (
                        patch(
                            "rushing_study.design.collect_environment_provenance",
                            return_value=current,
                        ),
                    ):
                        with self.assertRaisesRegex(ProvenanceError, expected_message):
                            verify_runtime_provenance(manifest, root)

    def test_runtime_verifier_can_validate_cross_queue_cpu_software(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, frozen_environment, _code_path, _data_path = self._frozen_runtime_manifest(root)
            cpu_environment = deepcopy(frozen_environment)
            cpu_environment["hardware"]["cpu_count"] = 64
            cpu_environment["tensorflow"]["physical_devices"] = []
            with patch(
                "rushing_study.design.collect_environment_provenance",
                return_value=cpu_environment,
            ):
                report = verify_runtime_provenance(
                    manifest,
                    root,
                    verify_tensorflow=False,
                    verify_hardware=False,
                )
            self.assertTrue(report["verified"])
            self.assertFalse(report["tensorflow_verified"])
            self.assertFalse(report["hardware_verified"])

            cpu_environment["packages"]["numpy"] = "changed"
            with patch(
                "rushing_study.design.collect_environment_provenance",
                return_value=cpu_environment,
            ):
                with self.assertRaisesRegex(ProvenanceError, "package versions"):
                    verify_runtime_provenance(
                        manifest,
                        root,
                        verify_tensorflow=False,
                        verify_hardware=False,
                    )

    def test_runtime_verifier_rejects_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _environment, _code_path, _data_path = self._frozen_runtime_manifest(root)
            manifest["config"]["description"] = "tampered"
            with self.assertRaisesRegex(ProvenanceError, "manifest_hash"):
                verify_runtime_provenance(manifest, root)

    def _mock_manifest(self) -> dict:
        smoke = explicit_smoke_config(self.config)
        code = {
            "files": {},
        }
        with (
            patch("rushing_study.design.collect_code_provenance", return_value=code),
            patch("rushing_study.design.collect_data_provenance", return_value={"files": {}}),
            patch(
                "rushing_study.design.collect_environment_provenance",
                return_value={"python": {"version": "test"}, "platform": {}, "packages": {}},
            ),
        ):
            return build_study_manifest(smoke, synthetic_games(), ROOT)

    def test_runner_manifest_contract_and_immutable_bundle(self) -> None:
        manifest = self._mock_manifest()
        self.assertEqual(manifest["run_hash"], manifest["manifest_hash"][:24])
        self.assertEqual(manifest["config"]["execution"]["workers"], 1)
        self.assertEqual(manifest["config"]["execution"]["environment"], DETERMINISTIC_ENVIRONMENT)
        self.assertEqual(manifest["seed_derivation"]["namespace"], self.config["study_id"])
        self.assertEqual(len(manifest["split_manifests"]), 50)
        self.assertEqual(len(manifest["seed_registry"]), 6_201)
        self.assertEqual(
            manifest_seed(manifest, "cell", 1, "epoch_selection", "zoo_cnn", 20),
            manifest["seed_registry"][
                canonical_json(["cell", 1, "epoch_selection", "zoo_cnn", 20])
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_manifest_bundle(directory, manifest)
            same_paths = write_manifest_bundle(directory, manifest)
            self.assertEqual(paths, same_paths)
            loaded = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(loaded, manifest)
            self.assertEqual(paths["hash"].read_text(encoding="ascii").strip(), manifest["manifest_hash"])

            tampered = deepcopy(manifest)
            tampered["config"]["description"] = "changed without rehashing"
            with self.assertRaises(DesignError):
                write_manifest_bundle(directory, tampered)

            replacement = deepcopy(manifest)
            replacement["provenance"]["environment_marker"] = "different valid payload"
            identity = {
                key: value
                for key, value in replacement.items()
                if key not in {"manifest_hash", "run_hash"}
            }
            replacement["manifest_hash"] = sha256_json(identity)
            replacement["run_hash"] = replacement["manifest_hash"][:24]
            with self.assertRaises(ManifestCollisionError):
                write_manifest_bundle(directory, replacement)


if __name__ == "__main__":
    unittest.main()
