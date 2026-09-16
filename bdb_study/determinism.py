"""Process-start and TensorFlow runtime guarantees for bit-exact BDB fits."""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any, Mapping


# CUDA documents these two workspace layouts as the reproducible cuBLAS
# configurations.  The larger layout is the faster of the two and costs only a
# small, fixed amount of device memory.  It must be present before the process
# creates its first cuBLAS handle, so callers may validate it here but must set
# it in the worker environment before Python starts.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

DETERMINISTIC_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "PYTHONHASHSEED": "20260817",
        "TF_DETERMINISTIC_OPS": "1",
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "OMP_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }
)


def deterministic_environment() -> dict[str, str]:
    """Return a mutable copy of the canonical process-start environment."""

    return dict(DETERMINISTIC_ENVIRONMENT)


def _uses_cuda_gpu(tensorflow: Any) -> bool:
    """Return whether this TensorFlow runtime can execute work through CUDA."""

    try:
        build = tensorflow.sysconfig.get_build_info()
    except (AttributeError, TypeError):
        build = {}
    if not isinstance(build, Mapping) or not (
        build.get("is_cuda_build") is True or build.get("cuda_version")
    ):
        return False
    try:
        return bool(tensorflow.config.list_physical_devices("GPU"))
    except (AttributeError, TypeError):
        return False


def configure_tensorflow_determinism(
    tensorflow: Any,
    seed: int,
    *,
    deterministic: bool,
) -> None:
    """Reset all Keras PRNGs and fail closed on an unsafe CUDA workspace.

    ``CUBLAS_WORKSPACE_CONFIG`` cannot be repaired here: importing TensorFlow
    may already have initialized CUDA state.  The Betty launcher and worker
    subprocess set it before interpreter startup; this guard prevents direct
    neural callers from silently bypassing that process boundary.
    """

    tensorflow.keras.utils.set_random_seed(int(seed))
    if not deterministic:
        return
    if _uses_cuda_gpu(tensorflow):
        observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if observed != CUBLAS_WORKSPACE_CONFIG:
            raise RuntimeError(
                "deterministic CUDA neural training requires process-start "
                f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG!r}; "
                f"observed {observed!r}"
            )
    tensorflow.config.experimental.enable_op_determinism()
