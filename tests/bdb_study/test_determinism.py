from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bdb_study.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    configure_tensorflow_determinism,
    deterministic_environment,
)


ROOT = Path(__file__).resolve().parents[2]


def _fake_tensorflow(*, cuda: bool = True, gpu: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        keras=SimpleNamespace(
            utils=SimpleNamespace(set_random_seed=Mock()),
        ),
        sysconfig=SimpleNamespace(
            get_build_info=Mock(
                return_value={
                    "is_cuda_build": cuda,
                    "cuda_version": "12.8" if cuda else None,
                }
            )
        ),
        config=SimpleNamespace(
            list_physical_devices=Mock(
                return_value=["GPU:0"] if gpu else []
            ),
            experimental=SimpleNamespace(enable_op_determinism=Mock()),
        ),
    )


class DeterminismContractTests(unittest.TestCase):
    def test_canonical_environment_binds_the_documented_cublas_workspace(self) -> None:
        first = deterministic_environment()
        second = deterministic_environment()
        self.assertEqual(first["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(first["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG)
        first["CUBLAS_WORKSPACE_CONFIG"] = "tampered"
        self.assertEqual(second["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG)

        common = (ROOT / "scripts/betty/common.sh").read_text(encoding="utf-8")
        self.assertIn(
            'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
            common,
        )

    def test_cuda_training_fails_before_ops_without_process_start_workspace(self) -> None:
        tensorflow = _fake_tensorflow()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "process-start"):
                configure_tensorflow_determinism(
                    tensorflow, 73, deterministic=True
                )
        tensorflow.keras.utils.set_random_seed.assert_called_once_with(73)
        tensorflow.config.experimental.enable_op_determinism.assert_not_called()

    def test_cuda_training_enables_op_determinism_with_canonical_workspace(self) -> None:
        tensorflow = _fake_tensorflow()
        with patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG},
            clear=True,
        ):
            configure_tensorflow_determinism(
                tensorflow, 91, deterministic=True
            )
        tensorflow.keras.utils.set_random_seed.assert_called_once_with(91)
        tensorflow.config.experimental.enable_op_determinism.assert_called_once_with()

    def test_non_cuda_and_explicit_nondeterministic_paths_do_not_claim_cublas(self) -> None:
        cpu_tensorflow = _fake_tensorflow(cuda=False, gpu=False)
        with patch.dict(os.environ, {}, clear=True):
            configure_tensorflow_determinism(
                cpu_tensorflow, 19, deterministic=True
            )
        cpu_tensorflow.config.experimental.enable_op_determinism.assert_called_once_with()

        disabled = _fake_tensorflow()
        with patch.dict(os.environ, {}, clear=True):
            configure_tensorflow_determinism(
                disabled, 23, deterministic=False
            )
        disabled.keras.utils.set_random_seed.assert_called_once_with(23)
        disabled.config.experimental.enable_op_determinism.assert_not_called()


if __name__ == "__main__":
    unittest.main()
