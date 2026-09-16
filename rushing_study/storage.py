"""Immutable, resumable artifact storage for rushing-study sweeps.

The storage protocol uses a small commit marker at both the cell and run level.
Payloads are written atomically and carry SHA-256 sidecars; a cell is complete
only after its ``_SUCCESS`` marker has been written and every referenced payload
still matches both the marker and its sidecar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MANIFEST_FILE = "manifest.json"
SUCCESS_FILE = "_SUCCESS"
RUNNING_FILE = "_RUNNING"
SCHEMA_VERSION = 1

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StorageError(RuntimeError):
    """Base class for study-storage failures."""


class RunDirectoryCollisionError(StorageError):
    """Raised when initialization would reuse an unrelated nonempty directory."""


class ManifestMismatchError(StorageError):
    """Raised when a run directory is bound to another manifest."""


class CorruptArtifactError(StorageError):
    """Raised when a committed artifact no longer passes validation."""


class IncompleteRunError(StorageError):
    """Raised when finalization is attempted with incomplete required cells."""


class RunFinalizedError(StorageError):
    """Raised when code attempts to mutate a finalized run."""


class CellStatus(str, Enum):
    COMPLETE = "complete"
    RUNNING = "running"
    CORRUPT = "corrupt"
    MISSING = "missing"


@dataclass(frozen=True, order=True)
class CellKey:
    """Coordinates for one model fit/evaluation cell."""

    branch: str
    repeat: int
    n_train: int
    model: str

    def __post_init__(self) -> None:
        _validate_component("branch", self.branch)
        _validate_component("model", self.model)
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int) or self.repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if isinstance(self.n_train, bool) or not isinstance(self.n_train, int) or self.n_train <= 0:
            raise ValueError("n_train must be a positive integer")

    @property
    def relative_dir(self) -> Path:
        return (
            Path("cells")
            / self.branch
            / f"repeat_{self.repeat:03d}"
            / f"n_{self.n_train:04d}"
            / self.model
        )


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata recorded for one checksummed payload."""

    file: str
    sha256: str
    format: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    status: CellStatus
    reason: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status is CellStatus.COMPLETE


@dataclass(frozen=True)
class StatusCounts:
    complete: int = 0
    running: int = 0
    corrupt: int = 0
    missing: int = 0

    @property
    def total(self) -> int:
        return self.complete + self.running + self.corrupt + self.missing

    def as_dict(self) -> dict[str, int]:
        return {
            CellStatus.COMPLETE.value: self.complete,
            CellStatus.RUNNING.value: self.running,
            CellStatus.CORRUPT.value: self.corrupt,
            CellStatus.MISSING.value: self.missing,
            "total": self.total,
        }


def _validate_component(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must be a nonempty path-safe slug containing only letters, "
            "numbers, dots, underscores, or hyphens"
        )


def _normalize_hash(value: str) -> str:
    normalized = str(value).lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError("manifest_hash must be a 64-character SHA-256 hex digest")
    return normalized


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes suitable for manifest hashing."""

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    # Design manifests carry their own identity fields.  Their hash is defined
    # over the scientific payload, excluding those two derived fields; generic
    # storage manifests retain the original hash-everything behavior.
    if "manifest_hash" in manifest or "run_hash" in manifest:
        if "manifest_hash" not in manifest or "run_hash" not in manifest:
            raise ManifestMismatchError("design manifest must contain both manifest_hash and run_hash")
        payload = {
            key: value
            for key, value in manifest.items()
            if key not in {"manifest_hash", "run_hash"}
        }
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if str(manifest["manifest_hash"]).lower() != digest:
            raise ManifestMismatchError("embedded manifest_hash does not match the scientific payload")
        if str(manifest["run_hash"]).lower() != digest[:24]:
            raise ManifestMismatchError("embedded run_hash does not match manifest_hash")
        return digest
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    """Create ``path`` exactly once, failing if another worker claimed it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise StorageError(f"artifact already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_writer(path: Path, writer: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f".tmp{path.suffix}",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise StorageError(f"writer did not create temporary artifact for {path}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.sha256")


def _write_checksum(path: Path, digest: str) -> None:
    _atomic_write_bytes(checksum_path(path), f"{digest}  {path.name}\n".encode("ascii"))


def _read_checksum(path: Path) -> tuple[str, str]:
    sidecar = checksum_path(path)
    try:
        line = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise CorruptArtifactError(f"missing or unreadable checksum sidecar: {sidecar}") from exc
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or not _SHA256.fullmatch(parts[0]) or parts[1] != path.name:
        raise CorruptArtifactError(f"invalid checksum sidecar: {sidecar}")
    return parts[0], parts[1]


def validate_checksum(path: str | Path, expected_sha256: str | None = None) -> bool:
    """Return whether an artifact exists and agrees with its sidecar/expected hash."""

    artifact = Path(path)
    if not artifact.is_file():
        return False
    try:
        sidecar_digest, _ = _read_checksum(artifact)
    except CorruptArtifactError:
        return False
    actual = _sha256_file(artifact)
    expected = sidecar_digest if expected_sha256 is None else str(expected_sha256).lower()
    return actual == sidecar_digest == expected


def _finish_artifact(path: Path, artifact_format: str) -> ArtifactRecord:
    digest = _sha256_file(path)
    _write_checksum(path, digest)
    return ArtifactRecord(file=path.name, sha256=digest, format=artifact_format)


def register_existing_artifact(path: str | Path, artifact_format: str | None = None) -> ArtifactRecord:
    """Checksum an already-created final artifact without rewriting its bytes."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    inferred = artifact_format or target.suffix.lower().lstrip(".") or "binary"
    return _finish_artifact(target, inferred)


def atomic_write_json(path: str | Path, value: Any) -> ArtifactRecord:
    path = Path(path)
    _atomic_write_bytes(path, _pretty_json_bytes(value))
    return _finish_artifact(path, "json")


def atomic_write_csv(
    path: str | Path,
    value: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
    *,
    index: bool = False,
) -> ArtifactRecord:
    path = Path(path)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    _atomic_writer(path, lambda temporary: frame.to_csv(temporary, index=index))
    return _finish_artifact(path, "csv")


def atomic_write_npz(path: str | Path, **arrays: Any) -> ArtifactRecord:
    path = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    _atomic_writer(path, write)
    return _finish_artifact(path, "npz")


def _parquet_engine() -> str | None:
    if importlib.util.find_spec("pyarrow") is not None:
        return "pyarrow"
    if importlib.util.find_spec("fastparquet") is not None:
        return "fastparquet"
    return None


def atomic_write_dataframe(
    directory: str | Path,
    frame: pd.DataFrame,
    *,
    basename: str = "predictions",
    preferred: str = "parquet",
    index: bool = False,
) -> ArtifactRecord:
    """Atomically write a frame, explicitly falling back from Parquet to CSV."""

    _validate_component("basename", basename)
    destination = Path(directory)
    if preferred not in {"parquet", "csv"}:
        raise ValueError("preferred must be 'parquet' or 'csv'")

    engine = _parquet_engine() if preferred == "parquet" else None
    if engine is not None:
        parquet_path = destination / f"{basename}.parquet"
        try:
            _atomic_writer(
                parquet_path,
                lambda temporary: frame.to_parquet(temporary, engine=engine, index=index),
            )
        except (ImportError, ModuleNotFoundError):
            # Dependency discovery can race with an environment change. Missing
            # optional engines are the only Parquet errors that become CSV.
            pass
        else:
            return _finish_artifact(parquet_path, "parquet")

    return atomic_write_csv(destination / f"{basename}.csv", frame, index=index)


def _coerce_key(
    key: CellKey | None = None,
    *,
    branch: str | None = None,
    repeat: int | None = None,
    n_train: int | None = None,
    model: str | None = None,
) -> CellKey:
    if key is not None:
        if any(value is not None for value in (branch, repeat, n_train, model)):
            raise TypeError("pass either key or branch/repeat/n_train/model, not both")
        if not isinstance(key, CellKey):
            raise TypeError("key must be a CellKey")
        return key
    if branch is None or repeat is None or n_train is None or model is None:
        raise TypeError("branch, repeat, n_train, and model are required when key is omitted")
    return CellKey(branch=branch, repeat=repeat, n_train=n_train, model=model)


def cell_dir(
    run_dir: str | Path,
    key: CellKey | str,
    repeat: int | None = None,
    n_train: int | None = None,
    model: str | None = None,
) -> Path:
    """Return the deterministic directory for a cell without creating it."""

    resolved = (
        key
        if isinstance(key, CellKey)
        else _coerce_key(branch=key, repeat=repeat, n_train=n_train, model=model)
    )
    return Path(run_dir) / resolved.relative_dir


class RunStorage:
    """A run directory bound to one canonical manifest hash."""

    def __init__(self, run_dir: Path, manifest: Mapping[str, Any], manifest_hash: str):
        self.run_dir = Path(run_dir)
        self.manifest = dict(manifest)
        self.manifest_hash = _normalize_hash(manifest_hash)

    @classmethod
    def open(cls, run_dir: str | Path, *, manifest_hash: str | None = None) -> "RunStorage":
        directory = Path(run_dir)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.is_file():
            raise RunDirectoryCollisionError(f"run directory has no {MANIFEST_FILE}: {directory}")
        if not validate_checksum(manifest_path):
            raise CorruptArtifactError(f"manifest checksum failed: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptArtifactError(f"manifest is unreadable: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise CorruptArtifactError("manifest root must be a JSON object")
        stored_hash = manifest_sha256(manifest)
        if manifest_hash is not None and stored_hash != _normalize_hash(manifest_hash):
            raise ManifestMismatchError(
                f"run manifest hash {stored_hash} does not match requested {manifest_hash}"
            )
        return cls(directory, manifest, stored_hash)

    def cell_dir(self, key: CellKey) -> Path:
        return cell_dir(self.run_dir, key)

    def _validate_running_marker(self, key: CellKey) -> ValidationResult:
        running_path = self.cell_dir(key) / RUNNING_FILE
        if not running_path.is_file():
            return ValidationResult(CellStatus.CORRUPT, "cell commit marker is missing")
        try:
            running = json.loads(running_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return ValidationResult(CellStatus.CORRUPT, f"running marker is unreadable: {exc}")
        if not isinstance(running, dict):
            return ValidationResult(CellStatus.CORRUPT, "running marker is not an object")
        if running.get("schema_version") != SCHEMA_VERSION:
            return ValidationResult(CellStatus.CORRUPT, "unsupported running-marker schema")
        if running.get("manifest_hash") != self.manifest_hash:
            return ValidationResult(CellStatus.CORRUPT, "running cell belongs to another manifest")
        if running.get("key") != asdict(key):
            return ValidationResult(CellStatus.CORRUPT, "running cell key does not match its path")
        pid = running.get("pid")
        hostname = running.get("hostname")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(hostname, str)
            or not hostname
        ):
            return ValidationResult(CellStatus.CORRUPT, "running marker has invalid process identity")
        if hostname == socket.gethostname():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return ValidationResult(
                    CellStatus.CORRUPT, f"stale running marker for dead local pid {pid}"
                )
            except PermissionError:
                pass
            except OSError as exc:
                return ValidationResult(
                    CellStatus.CORRUPT, f"could not validate local running pid {pid}: {exc}"
                )
        return ValidationResult(CellStatus.RUNNING, "cell execution is in progress")

    def validate_cell(self, key: CellKey) -> ValidationResult:
        directory = self.cell_dir(key)
        if not directory.exists():
            return ValidationResult(CellStatus.MISSING, "cell directory does not exist")
        if not directory.is_dir():
            return ValidationResult(CellStatus.CORRUPT, "cell path is not a directory")
        marker_path = directory / SUCCESS_FILE
        if not marker_path.is_file():
            return self._validate_running_marker(key)

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return ValidationResult(CellStatus.CORRUPT, f"cell commit marker is unreadable: {exc}")
        if not isinstance(marker, dict):
            return ValidationResult(CellStatus.CORRUPT, "cell commit marker is not an object")
        if marker.get("schema_version") != SCHEMA_VERSION:
            return ValidationResult(CellStatus.CORRUPT, "unsupported cell schema version")
        if marker.get("manifest_hash") != self.manifest_hash:
            return ValidationResult(CellStatus.CORRUPT, "cell belongs to another manifest")
        if marker.get("key") != asdict(key):
            return ValidationResult(CellStatus.CORRUPT, "cell key does not match its path")

        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, dict):
            return ValidationResult(CellStatus.CORRUPT, "cell artifact index is missing")
        if not {"metrics", "predictions"}.issubset(artifacts):
            return ValidationResult(CellStatus.CORRUPT, "required cell artifacts are missing")
        for logical_name, entry in artifacts.items():
            if not isinstance(logical_name, str) or not isinstance(entry, dict):
                return ValidationResult(CellStatus.CORRUPT, "invalid cell artifact entry")
            filename = entry.get("file")
            digest = entry.get("sha256")
            artifact_format = entry.get("format")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                or artifact_format not in {"json", "csv", "parquet", "npz"}
            ):
                return ValidationResult(
                    CellStatus.CORRUPT, f"invalid metadata for artifact {logical_name!r}"
                )
            artifact_path = directory / filename
            if not validate_checksum(artifact_path, digest):
                return ValidationResult(
                    CellStatus.CORRUPT,
                    f"artifact {logical_name!r} is missing or failed its checksum",
                )
        return ValidationResult(CellStatus.COMPLETE)

    def mark_cell_running(self, key: CellKey, *, overwrite: bool = False) -> None:
        """Atomically mark a cell active before fitting starts.

        ``overwrite`` is required to invalidate an existing generic commit, as
        happens when the scientific validator rejects an otherwise checksummed
        cell during resume.
        """

        if (self.run_dir / SUCCESS_FILE).exists():
            raise RunFinalizedError(f"run is already finalized: {self.run_dir}")
        directory = self.cell_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        running_path = directory / RUNNING_FILE
        if running_path.exists():
            running = self._validate_running_marker(key)
            if running.status is CellStatus.RUNNING:
                raise StorageError(f"cell is already running: {key}")
            running_path.unlink(missing_ok=True)
        success = directory / SUCCESS_FILE
        if success.exists() and not overwrite:
            raise StorageError(f"refusing to replace committed cell without overwrite=True: {key}")
        if overwrite:
            success.unlink(missing_ok=True)
        marker = {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "key": asdict(key),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_create_bytes(running_path, _pretty_json_bytes(marker))

    def clear_cell_running(self, key: CellKey) -> None:
        """Clear this process-independent activity marker after success/failure."""

        (self.cell_dir(key) / RUNNING_FILE).unlink(missing_ok=True)

    def classify_cells(self, keys: Iterable[CellKey]) -> dict[CellKey, ValidationResult]:
        return {key: self.validate_cell(key) for key in keys}

    def status_counts(self, keys: Iterable[CellKey]) -> StatusCounts:
        counts = {status: 0 for status in CellStatus}
        for result in self.classify_cells(keys).values():
            counts[result.status] += 1
        return StatusCounts(
            complete=counts[CellStatus.COMPLETE],
            running=counts[CellStatus.RUNNING],
            corrupt=counts[CellStatus.CORRUPT],
            missing=counts[CellStatus.MISSING],
        )

    def pending_cells(self, keys: Iterable[CellKey]) -> list[CellKey]:
        """Return cells that resume logic must run; only valid cells are skipped."""

        return [key for key in keys if not self.validate_cell(key).is_complete]

    def write_cell_artifacts(
        self,
        key: CellKey,
        *,
        metrics: Mapping[str, Any],
        predictions: pd.DataFrame | Mapping[str, Any],
        history: Mapping[str, Any] | Sequence[Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
        preferred_predictions: str = "parquet",
        overwrite: bool = False,
    ) -> bool:
        """Write and commit one cell; return False when a valid cell is skipped."""

        if (self.run_dir / SUCCESS_FILE).exists():
            raise RunFinalizedError(f"run is already finalized: {self.run_dir}")
        existing = self.validate_cell(key)
        if existing.is_complete and not overwrite:
            return False

        directory = self.cell_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        # Removing the old marker first makes any subsequent interruption an
        # explicit corrupt/partial cell that resume logic will recompute.
        (directory / SUCCESS_FILE).unlink(missing_ok=True)

        artifact_index: dict[str, ArtifactRecord] = {}
        artifact_index["metrics"] = atomic_write_json(directory / "metrics.json", metrics)
        if isinstance(predictions, pd.DataFrame):
            artifact_index["predictions"] = atomic_write_dataframe(
                directory,
                predictions,
                basename="predictions",
                preferred=preferred_predictions,
                index=False,
            )
        elif isinstance(predictions, Mapping):
            artifact_index["predictions"] = atomic_write_npz(
                directory / "predictions.npz", **dict(predictions)
            )
        else:
            raise TypeError("predictions must be a pandas DataFrame or a mapping of arrays")
        if history is not None:
            artifact_index["history"] = atomic_write_json(directory / "history.json", history)
        if arrays is not None:
            artifact_index["arrays"] = atomic_write_npz(directory / "arrays.npz", **dict(arrays))

        marker = {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "key": asdict(key),
            "artifacts": {
                name: record.as_dict() for name, record in sorted(artifact_index.items())
            },
        }
        _atomic_write_bytes(directory / SUCCESS_FILE, _pretty_json_bytes(marker))
        return True

    def _load_marker(self, key: CellKey) -> dict[str, Any]:
        result = self.validate_cell(key)
        if not result.is_complete:
            raise CorruptArtifactError(
                f"cannot load {key}: cell is {result.status.value} ({result.reason})"
            )
        with (self.cell_dir(key) / SUCCESS_FILE).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_metrics(self, key: CellKey) -> dict[str, Any]:
        marker = self._load_marker(key)
        path = self.cell_dir(key) / marker["artifacts"]["metrics"]["file"]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptArtifactError(f"metrics payload is unreadable: {path}") from exc
        if not isinstance(value, dict):
            raise CorruptArtifactError("metrics payload must be a JSON object")
        return value

    def load_predictions(self, key: CellKey) -> pd.DataFrame | dict[str, np.ndarray]:
        marker = self._load_marker(key)
        entry = marker["artifacts"]["predictions"]
        path = self.cell_dir(key) / entry["file"]
        if entry["format"] == "csv":
            return pd.read_csv(path)
        if entry["format"] == "parquet":
            try:
                return pd.read_parquet(path)
            except (ImportError, ModuleNotFoundError) as exc:
                raise CorruptArtifactError(
                    "predictions are Parquet but no Parquet engine is installed"
                ) from exc
        if entry["format"] == "npz":
            with np.load(path, allow_pickle=False) as payload:
                return {name: payload[name] for name in payload.files}
        raise CorruptArtifactError(f"unsupported predictions format: {entry['format']}")

    def load_history(self, key: CellKey) -> Any:
        marker = self._load_marker(key)
        entry = marker["artifacts"].get("history")
        if entry is None:
            return None
        path = self.cell_dir(key) / entry["file"]
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptArtifactError(f"history payload is unreadable: {path}") from exc

    def load_arrays(self, key: CellKey) -> dict[str, np.ndarray] | None:
        marker = self._load_marker(key)
        entry = marker["artifacts"].get("arrays")
        if entry is None:
            return None
        path = self.cell_dir(key) / entry["file"]
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name] for name in payload.files}

    def finalize_run(
        self,
        required_cells: Iterable[CellKey],
        *,
        final_artifacts: Iterable[str | Path] = (),
    ) -> bool:
        """Write the run-level commit marker after every requirement validates."""

        keys = list(required_cells)
        if len(set(keys)) != len(keys):
            raise ValueError("required_cells contains duplicate cell keys")
        expected_directories = {self.cell_dir(key).resolve() for key in keys}
        cells_root = self.run_dir / "cells"
        unexpected_markers = []
        if cells_root.is_dir():
            unexpected_markers = sorted(
                marker
                for marker in cells_root.rglob(SUCCESS_FILE)
                if marker.parent.resolve() not in expected_directories
            )
        if unexpected_markers:
            relative = [str(path.relative_to(self.run_dir)) for path in unexpected_markers]
            raise IncompleteRunError(
                "cannot finalize with committed cells outside the required grid: "
                + ", ".join(relative[:10])
            )
        existing_marker = self.run_dir / SUCCESS_FILE
        if existing_marker.exists():
            if self.validate_final(keys):
                return False
            raise CorruptArtifactError(f"existing final marker is invalid: {existing_marker}")

        classifications = self.classify_cells(keys)
        incomplete = {
            key: result
            for key, result in classifications.items()
            if result.status is not CellStatus.COMPLETE
        }
        if incomplete:
            detail = ", ".join(
                f"{key.relative_dir}={result.status.value}" for key, result in incomplete.items()
            )
            raise IncompleteRunError(f"cannot finalize with incomplete cells: {detail}")

        indexed_final_artifacts: list[dict[str, str]] = []
        root = self.run_dir.resolve()
        for supplied_path in final_artifacts:
            path = Path(supplied_path)
            if not path.is_absolute():
                path = self.run_dir / path
            try:
                relative = path.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(f"final artifact is outside run directory: {path}") from exc
            if not validate_checksum(path):
                raise CorruptArtifactError(f"final artifact failed checksum validation: {path}")
            indexed_final_artifacts.append(
                {"file": str(relative), "sha256": _sha256_file(path)}
            )

        cells = []
        for key in sorted(keys):
            marker_path = self.cell_dir(key) / SUCCESS_FILE
            cells.append(
                {
                    "key": asdict(key),
                    "path": str(key.relative_dir),
                    "marker_sha256": _sha256_file(marker_path),
                }
            )
        marker = {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "cell_count": len(cells),
            "cells": cells,
            "final_artifacts": sorted(indexed_final_artifacts, key=lambda item: item["file"]),
        }
        _atomic_write_bytes(existing_marker, _pretty_json_bytes(marker))
        return True

    def validate_final(self, required_cells: Iterable[CellKey] | None = None) -> bool:
        marker_path = self.run_dir / SUCCESS_FILE
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if (
            not isinstance(marker, dict)
            or marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("manifest_hash") != self.manifest_hash
            or not isinstance(marker.get("cells"), list)
            or marker.get("cell_count") != len(marker["cells"])
        ):
            return False

        recorded: dict[CellKey, str] = {}
        try:
            for entry in marker["cells"]:
                key = CellKey(**entry["key"])
                if entry["path"] != str(key.relative_dir):
                    return False
                recorded[key] = entry["marker_sha256"]
        except (KeyError, TypeError, ValueError):
            return False
        if len(recorded) != len(marker["cells"]):
            return False

        expected = set(recorded) if required_cells is None else set(required_cells)
        if set(recorded) != expected:
            return False
        for key in expected:
            marker_digest = recorded.get(key)
            cell_marker = self.cell_dir(key) / SUCCESS_FILE
            if (
                self.validate_cell(key).status is not CellStatus.COMPLETE
                or not isinstance(marker_digest, str)
                or not cell_marker.is_file()
                or _sha256_file(cell_marker) != marker_digest
            ):
                return False

        final_artifacts = marker.get("final_artifacts", [])
        if not isinstance(final_artifacts, list):
            return False
        for entry in final_artifacts:
            if not isinstance(entry, dict):
                return False
            filename = entry.get("file")
            digest = entry.get("sha256")
            if not isinstance(filename, str) or not isinstance(digest, str):
                return False
            path = self.run_dir / filename
            try:
                path.resolve().relative_to(self.run_dir.resolve())
            except ValueError:
                return False
            if not validate_checksum(path, digest):
                return False
        return True


def initialize_run_dir(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_hash: str | None = None,
) -> RunStorage:
    """Create or idempotently reopen a run directory bound to ``manifest``."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    manifest_object = dict(manifest)
    calculated_hash = manifest_sha256(manifest_object)
    if manifest_hash is not None and calculated_hash != _normalize_hash(manifest_hash):
        raise ManifestMismatchError(
            f"provided manifest hash {manifest_hash} does not match manifest {calculated_hash}"
        )

    directory = Path(run_dir)
    if directory.exists() and not directory.is_dir():
        raise RunDirectoryCollisionError(f"run path exists and is not a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILE
    entries = list(directory.iterdir())
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise RunDirectoryCollisionError(f"manifest path is not a file: {manifest_path}")
        # A process can be interrupted after the atomically replaced manifest
        # but before its sidecar is installed. The requested manifest gives us
        # enough information to recover that one safe partial-initialization
        # case; an existing but incorrect sidecar remains a hard corruption.
        if not checksum_path(manifest_path).exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CorruptArtifactError(f"manifest is unreadable: {manifest_path}") from exc
            if not isinstance(existing_manifest, dict):
                raise CorruptArtifactError("manifest root must be a JSON object")
            if manifest_sha256(existing_manifest) != calculated_hash:
                raise ManifestMismatchError(
                    "partial run initialization contains a different manifest"
                )
            _write_checksum(manifest_path, _sha256_file(manifest_path))
        storage = RunStorage.open(directory)
        if storage.manifest_hash != calculated_hash:
            raise ManifestMismatchError(
                f"run directory is bound to {storage.manifest_hash}, not {calculated_hash}"
            )
        return storage
    if entries:
        raise RunDirectoryCollisionError(
            f"refusing to initialize nonempty directory without {MANIFEST_FILE}: {directory}"
        )

    atomic_write_json(manifest_path, manifest_object)
    return RunStorage(directory, manifest_object, calculated_hash)


def _as_storage(
    storage_or_run_dir: RunStorage | str | Path,
    manifest_hash: str | None,
) -> RunStorage:
    if isinstance(storage_or_run_dir, RunStorage):
        if manifest_hash is not None and storage_or_run_dir.manifest_hash != _normalize_hash(
            manifest_hash
        ):
            raise ManifestMismatchError("RunStorage is bound to another manifest hash")
        return storage_or_run_dir
    return RunStorage.open(storage_or_run_dir, manifest_hash=manifest_hash)


def write_cell_artifacts(
    storage_or_run_dir: RunStorage | str | Path,
    key: CellKey | None = None,
    *,
    branch: str | None = None,
    repeat: int | None = None,
    n_train: int | None = None,
    model: str | None = None,
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame | Mapping[str, Any],
    history: Mapping[str, Any] | Sequence[Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
    manifest_hash: str | None = None,
    preferred_predictions: str = "parquet",
    overwrite: bool = False,
) -> bool:
    storage = _as_storage(storage_or_run_dir, manifest_hash)
    resolved = _coerce_key(
        key, branch=branch, repeat=repeat, n_train=n_train, model=model
    )
    return storage.write_cell_artifacts(
        resolved,
        metrics=metrics,
        predictions=predictions,
        history=history,
        arrays=arrays,
        preferred_predictions=preferred_predictions,
        overwrite=overwrite,
    )


def validate_cell(
    storage_or_run_dir: RunStorage | str | Path,
    key: CellKey | None = None,
    *,
    branch: str | None = None,
    repeat: int | None = None,
    n_train: int | None = None,
    model: str | None = None,
    manifest_hash: str | None = None,
) -> ValidationResult:
    storage = _as_storage(storage_or_run_dir, manifest_hash)
    resolved = _coerce_key(
        key, branch=branch, repeat=repeat, n_train=n_train, model=model
    )
    return storage.validate_cell(resolved)


def classify_cells(
    storage_or_run_dir: RunStorage | str | Path,
    keys: Iterable[CellKey],
    *,
    manifest_hash: str | None = None,
) -> dict[CellKey, ValidationResult]:
    return _as_storage(storage_or_run_dir, manifest_hash).classify_cells(keys)


def status_counts(
    storage_or_run_dir: RunStorage | str | Path,
    keys: Iterable[CellKey],
    *,
    manifest_hash: str | None = None,
) -> StatusCounts:
    return _as_storage(storage_or_run_dir, manifest_hash).status_counts(keys)


def load_cell_metrics(
    storage_or_run_dir: RunStorage | str | Path,
    key: CellKey,
    *,
    manifest_hash: str | None = None,
) -> dict[str, Any]:
    return _as_storage(storage_or_run_dir, manifest_hash).load_metrics(key)


def load_cell_predictions(
    storage_or_run_dir: RunStorage | str | Path,
    key: CellKey,
    *,
    manifest_hash: str | None = None,
) -> pd.DataFrame | dict[str, np.ndarray]:
    return _as_storage(storage_or_run_dir, manifest_hash).load_predictions(key)


def finalize_run(
    storage_or_run_dir: RunStorage | str | Path,
    required_cells: Iterable[CellKey],
    *,
    manifest_hash: str | None = None,
    final_artifacts: Iterable[str | Path] = (),
) -> bool:
    return _as_storage(storage_or_run_dir, manifest_hash).finalize_run(
        required_cells, final_artifacts=final_artifacts
    )


__all__ = [
    "ArtifactRecord",
    "CellKey",
    "CellStatus",
    "CorruptArtifactError",
    "IncompleteRunError",
    "ManifestMismatchError",
    "RunDirectoryCollisionError",
    "RunFinalizedError",
    "RunStorage",
    "StatusCounts",
    "StorageError",
    "ValidationResult",
    "atomic_write_csv",
    "atomic_write_dataframe",
    "atomic_write_json",
    "atomic_write_npz",
    "canonical_json_bytes",
    "cell_dir",
    "checksum_path",
    "classify_cells",
    "finalize_run",
    "initialize_run_dir",
    "load_cell_metrics",
    "load_cell_predictions",
    "manifest_sha256",
    "register_existing_artifact",
    "status_counts",
    "validate_cell",
    "validate_checksum",
    "write_cell_artifacts",
]
