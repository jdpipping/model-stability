"""Immutable, memory-mappable prepared-task artifacts.

Adapters produce :class:`~bdb_study.adapters.common.PreparedTask` objects from
raw competition files.  This module persists that deterministic boundary
without fitting an imputer, scaler, vocabulary, or model.  Those learned
objects are intentionally created inside each development fold or main cell.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .adapters.common import PreparedTask
from .contracts import canonical_json, sha256_json


PREPARED_SCHEMA_VERSION = "bdb-prepared-task-v2"
PREPARED_SEMANTIC_SCHEMA_VERSION = "bdb-prepared-semantics-v2"
_ARRAY_NAMES = (
    "player_tokens",
    "player_mask",
    "frame_mask",
    "y",
    "target_values",
    "target_mask",
    "target_baseline",
)


class PreparedArtifactError(RuntimeError):
    """A prepared artifact is incomplete, corrupt, or incompatible."""


def _canonical_json(value: Any) -> str:
    # Prepared artifacts and TaskSpecs must use one canonicalization rule.
    # Keeping this small wrapper avoids changing the established on-disk API
    # while making every new semantic hash agree with contracts.sha256_json.
    return canonical_json(value)


_ARRAY_AXES = {
    "player_tokens": ("example", "time", "tracked_object", "channel"),
    "player_mask": ("example", "time", "tracked_object"),
    "frame_mask": ("example", "time"),
    "y": ("example",),
    "target_values": ("example", "horizon", "coordinate"),
    "target_mask": ("example", "horizon"),
    "target_baseline": ("example", "horizon", "coordinate"),
}


def _normalized(value: Any) -> Any:
    """Return the canonical JSON value, not merely canonical bytes."""

    return json.loads(canonical_json(value))


def _array_semantics(task: PreparedTask, name: str) -> dict[str, Any] | None:
    value = getattr(task, name)
    if value is None:
        return None
    array = np.asarray(value)
    return {
        "axes": list(_ARRAY_AXES[name]),
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
    }


def _temporal_mask_layout(mask: np.ndarray, *, label: str) -> str:
    """Describe and validate the only temporal padding layouts we support."""

    observed = np.asarray(mask, dtype=bool)
    if observed.ndim != 2 or observed.shape[1] < 1:
        raise PreparedArtifactError(f"{label} must have shape [N,T] with T >= 1")
    if np.any(observed.sum(axis=1) == 0):
        raise PreparedArtifactError(f"{label} cannot contain an entirely padded example")
    if np.all(observed):
        return "all_observed"
    changes = np.diff(observed.astype(np.int8), axis=1)
    left_padded = bool(np.all(changes >= 0))
    right_padded = bool(np.all(changes <= 0))
    if left_padded and not right_padded:
        return "left_padded"
    if right_padded and not left_padded:
        return "right_padded"
    raise PreparedArtifactError(
        f"{label} must use one contiguous left- or right-padding convention"
    )


def _validate_player_mask_contract(task: PreparedTask) -> None:
    tokens = np.asarray(task.player_tokens)
    player_mask = np.asarray(task.player_mask, dtype=bool)
    frame_mask = np.asarray(task.frame_mask, dtype=bool)
    # Do not materialize a whole-dataset boolean selection from the 2026
    # 8.6-GiB token mmap.  Chunking keeps semantic validation bounded while
    # still reading every masked slot.
    for start in range(0, len(tokens), 256):
        stop = min(start + 256, len(tokens))
        chunk_mask = player_mask[start:stop]
        chunk_frames = frame_mask[start:stop]
        counts = chunk_mask.sum(axis=2)
        if np.any(counts[chunk_frames] == 0):
            raise PreparedArtifactError(
                "every observed frame must contain a tracked object"
            )
        chunk_tokens = tokens[start:stop]
        if np.any(chunk_tokens[~chunk_mask] != 0):
            raise PreparedArtifactError(
                "masked player-token slots must contain exact zeros"
            )


def _support_semantics(support: tuple[float, ...] | None) -> dict[str, Any] | None:
    if support is None:
        return None
    values = [float(value) for value in support]
    if not values or not np.all(np.isfinite(values)):
        raise PreparedArtifactError("outcome support must be finite and nonempty")
    if len(values) > 1 and np.any(np.diff(np.asarray(values, dtype=float)) <= 0):
        raise PreparedArtifactError("outcome support must be strictly increasing")
    return {
        "count": len(values),
        "minimum": values[0],
        "maximum": values[-1],
        "values_sha256": sha256_json(values),
    }


def build_prepared_semantic_receipt(task: PreparedTask) -> dict[str, Any]:
    """Describe the exact adapter-to-model boundary in canonical form.

    This receipt deliberately includes scientific meaning that file checksums
    alone cannot express: ordered feature/channel names, target encoding,
    tensor axes, padding masks, and the adapter's semantic metadata.  Freeze
    and planning compare it byte-for-byte (after canonical normalization) with
    the prospective TaskSpec contract.
    """

    _validate_player_mask_contract(task)
    frame_layout = _temporal_mask_layout(task.frame_mask, label="frame_mask")
    arrays = {name: _array_semantics(task, name) for name in _ARRAY_NAMES}
    n_examples = int(len(task.examples))
    targets = task.examples["target"]
    modeled_outcome = "binary" if task.outcome_type == "frame_event" else task.outcome_type

    if modeled_outcome == "binary":
        if task.y is None or any(
            value is not None
            for value in (task.target_values, task.target_mask, task.target_baseline)
        ):
            raise PreparedArtifactError("binary tasks require y and forbid trajectory arrays")
        labels = np.asarray(task.y).reshape(-1)
        observed_targets = pd.to_numeric(targets, errors="coerce").to_numpy()
        if np.any(~np.isfinite(observed_targets)) or not np.array_equal(
            labels.astype(float), observed_targets.astype(float)
        ):
            raise PreparedArtifactError("binary examples.target must equal y")
        target_contract = {
            "encoding": "zero_one",
            "examples_target": "equals_y",
            "model_output": "positive_class_probability",
        }
        target_mask_contract = None
    elif modeled_outcome == "distribution":
        if task.y is None or task.support is None or any(
            value is not None
            for value in (task.target_values, task.target_mask, task.target_baseline)
        ):
            raise PreparedArtifactError(
                "distribution tasks require y/support and forbid trajectory arrays"
            )
        labels = np.asarray(task.y).reshape(-1).astype(np.int64)
        support_values = np.asarray(task.support, dtype=float)
        observed_targets = pd.to_numeric(targets, errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(observed_targets)) or not np.array_equal(
            observed_targets, support_values[labels]
        ):
            raise PreparedArtifactError(
                "distribution examples.target must equal support[y]"
            )
        target_contract = {
            "encoding": "zero_based_support_index",
            "examples_target": "equals_support_at_y",
            "model_output": "ordered_support_probability_vector",
        }
        target_mask_contract = None
    elif modeled_outcome == "trajectory":
        if task.y is not None or any(
            value is None
            for value in (task.target_values, task.target_mask, task.target_baseline)
        ):
            raise PreparedArtifactError(
                "trajectory tasks forbid y and require values/mask/baseline"
            )
        if not bool(pd.isna(targets).all()):
            raise PreparedArtifactError(
                "trajectory examples.target must be a NaN placeholder"
            )
        target_mask = np.asarray(task.target_mask, dtype=bool)
        target_layout = _temporal_mask_layout(target_mask, label="target_mask")
        if target_layout not in {"all_observed", "right_padded"}:
            raise PreparedArtifactError(
                "trajectory target_mask must be an observed prefix followed by padding"
            )
        values = np.asarray(task.target_values)
        if not np.all(np.isnan(values[~target_mask])):
            raise PreparedArtifactError(
                "masked trajectory target coordinates must contain NaN"
            )
        target_contract = {
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
        }
        target_mask_contract = {
            "meaning": "true_iff_official_future_coordinate_is_observed_and_scored",
            "layout": target_layout,
            "masked_target_value": "nan",
        }
    else:
        raise PreparedArtifactError(f"unsupported outcome_type {task.outcome_type!r}")

    unsigned = {
        "schema_version": PREPARED_SEMANTIC_SCHEMA_VERSION,
        "task_id": str(task.task_id),
        "outcome_type": str(task.outcome_type),
        "primary_metric": str(task.primary_metric),
        "examples": n_examples,
        "games": int(task.examples["game_id"].nunique()),
        "tabular_feature_order": [str(value) for value in task.tabular.columns],
        "channel_names": [str(value) for value in task.channel_names],
        "arrays": arrays,
        "support": _support_semantics(task.support),
        "mask_contract": {
            "player_mask": {
                "meaning": "true_iff_tracked_object_slot_is_observed",
                "observed_frame_slot_layout": "stable_per_play_slots_with_sparse_observation",
                "slot_identity": "alignment_only_never_a_model_feature",
                "masked_token_value": 0.0,
                "requires_observed_frame": True,
            },
            "frame_mask": {
                "meaning": "true_iff_time_slot_is_observed_not_padding",
                "layout": frame_layout,
            },
            "target_mask": target_mask_contract,
        },
        "target_contract": target_contract,
        "adapter_metadata": _normalized(dict(task.metadata)),
    }
    normalized = _normalized(unsigned)
    return {**normalized, "semantic_hash": sha256_json(normalized)}


def validate_prepared_task_contract(
    task: PreparedTask,
    expected_contract: Mapping[str, Any],
    *,
    embedded_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a prepared task exactly matches its TaskSpec contract."""

    actual = build_prepared_semantic_receipt(task)
    expected = _normalized(expected_contract)
    if canonical_json(actual) != canonical_json(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if canonical_json(actual.get(key)) != canonical_json(expected.get(key))
        )
        raise PreparedArtifactError(
            "prepared task differs from TaskSpec semantic contract in: "
            f"{differing}"
        )
    if embedded_receipt is not None and canonical_json(actual) != canonical_json(
        embedded_receipt
    ):
        raise PreparedArtifactError(
            "prepared task differs from its embedded semantic receipt"
        )
    return actual


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_record(path: Path, array: np.ndarray, root: Path) -> dict[str, Any]:
    record = _file_record(path, root)
    record.update(
        {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
        }
    )
    return record


def write_prepared_task(
    task: PreparedTask,
    output_dir: str | Path,
    *,
    source_receipt_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a complete prepared task once, refusing incompatible overwrite."""

    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        examples_path = temporary / "examples.csv"
        tabular_path = temporary / "tabular.csv"
        task.examples.to_csv(examples_path, index=False)
        task.tabular.to_csv(tabular_path, index=False)

        files: dict[str, dict[str, Any]] = {
            "examples": _file_record(examples_path, temporary),
            "tabular": _file_record(tabular_path, temporary),
        }
        for name in _ARRAY_NAMES:
            value = getattr(task, name)
            if value is None:
                continue
            array = np.asarray(value)
            array_path = temporary / f"{name}.npy"
            np.save(array_path, array, allow_pickle=False)
            files[name] = _array_record(array_path, array, temporary)

        receipt: dict[str, Any] = {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "task_id": task.task_id,
            "outcome_type": task.outcome_type,
            "primary_metric": task.primary_metric,
            "examples": int(len(task.examples)),
            "games": int(task.examples["game_id"].nunique()),
            "channel_names": list(task.channel_names),
            "support": None if task.support is None else [float(value) for value in task.support],
            "audit": dict(task.audit),
            "metadata": dict(task.metadata),
            "semantic_receipt": build_prepared_semantic_receipt(task),
            "source_receipt_hashes": dict(sorted((source_receipt_hashes or {}).items())),
            "files": files,
        }
        receipt["prepared_hash"] = hashlib.sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()
        _atomic_json(temporary / "receipt.json", receipt)

        if destination.exists():
            existing = load_prepared_receipt(destination, verify_files=True)
            if existing.get("prepared_hash") != receipt["prepared_hash"]:
                raise PreparedArtifactError(
                    f"refusing to replace {destination}: prepared hashes differ"
                )
            return existing
        os.replace(temporary, destination)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_prepared_receipt(
    prepared_dir: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    root = Path(prepared_dir).resolve()
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        raise PreparedArtifactError(f"missing prepared receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise PreparedArtifactError("unsupported prepared-task schema")
    claimed_hash = receipt.get("prepared_hash")
    unsigned = {key: value for key, value in receipt.items() if key != "prepared_hash"}
    actual_hash = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if claimed_hash != actual_hash:
        raise PreparedArtifactError("prepared receipt hash is invalid")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise PreparedArtifactError("prepared receipt has no file registry")
    semantic = receipt.get("semantic_receipt")
    if semantic is not None:
        if not isinstance(semantic, Mapping):
            raise PreparedArtifactError("prepared semantic receipt must be an object")
        unsigned_semantic = {
            key: value for key, value in semantic.items() if key != "semantic_hash"
        }
        if (
            semantic.get("schema_version") != PREPARED_SEMANTIC_SCHEMA_VERSION
            or semantic.get("semantic_hash") != sha256_json(unsigned_semantic)
        ):
            raise PreparedArtifactError("prepared semantic receipt hash is invalid")
    if verify_files:
        for name, record in files.items():
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise PreparedArtifactError(f"unsafe prepared path for {name}")
            path = root / relative
            if not path.is_file():
                raise PreparedArtifactError(f"missing prepared file: {path}")
            if path.stat().st_size != int(record.get("size_bytes", -1)) or _sha256(path) != record.get("sha256"):
                raise PreparedArtifactError(f"prepared file failed checksum: {path}")
    return receipt


def load_prepared_task(
    prepared_dir: str | Path,
    *,
    mmap_mode: str | None = "r",
    verify_files: bool = True,
) -> PreparedTask:
    """Load a persisted task; large numeric arrays remain memory mapped."""

    root = Path(prepared_dir).resolve()
    receipt = load_prepared_receipt(root, verify_files=verify_files)
    files = receipt["files"]

    def array(name: str) -> np.ndarray | None:
        if name not in files:
            return None
        value = np.load(root / files[name]["path"], mmap_mode=mmap_mode, allow_pickle=False)
        expected_shape = tuple(int(item) for item in files[name].get("shape", ()))
        if expected_shape and value.shape != expected_shape:
            raise PreparedArtifactError(f"{name} shape differs from receipt")
        if str(value.dtype) != files[name].get("dtype"):
            raise PreparedArtifactError(f"{name} dtype differs from receipt")
        return value

    examples = pd.read_csv(root / files["examples"]["path"], low_memory=False)
    tabular = pd.read_csv(root / files["tabular"]["path"], low_memory=False)
    task = PreparedTask(
        task_id=str(receipt["task_id"]),
        outcome_type=str(receipt["outcome_type"]),
        primary_metric=str(receipt["primary_metric"]),
        examples=examples,
        tabular=tabular,
        player_tokens=array("player_tokens"),
        player_mask=array("player_mask"),
        frame_mask=array("frame_mask"),
        channel_names=tuple(str(value) for value in receipt["channel_names"]),
        y=array("y"),
        support=None if receipt.get("support") is None else tuple(float(value) for value in receipt["support"]),
        audit=dict(receipt.get("audit", {})),
        metadata=dict(receipt.get("metadata", {})),
        target_values=array("target_values"),
        target_mask=array("target_mask"),
        target_baseline=array("target_baseline"),
    )
    embedded = receipt.get("semantic_receipt")
    if embedded is not None:
        # Recompute from the checksum-verified CSV/NPY payload, rather than
        # trusting the duplicated human-readable semantic fields.
        actual = build_prepared_semantic_receipt(task)
        if canonical_json(actual) != canonical_json(embedded):
            raise PreparedArtifactError(
                "prepared payload differs from its embedded semantic receipt"
            )
    return task
