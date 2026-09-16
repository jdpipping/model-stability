"""Task-neutral, hash-pinned storage and queue helpers for the BDB suite.

The byte-level implementation is intentionally imported from the already
tested rushing-study storage protocol instead of being forked.  Each suite run
must pin the exact backend file hash in its manifest, so later changes cannot
silently alter resume or checksum semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from rushing_study import storage as _backend


# Stable public re-exports.  Importing this module does not import TensorFlow.
ArtifactRecord = _backend.ArtifactRecord
CellKey = _backend.CellKey
CellStatus = _backend.CellStatus
CorruptArtifactError = _backend.CorruptArtifactError
IncompleteRunError = _backend.IncompleteRunError
ManifestMismatchError = _backend.ManifestMismatchError
RunDirectoryCollisionError = _backend.RunDirectoryCollisionError
RunFinalizedError = _backend.RunFinalizedError
RunStorage = _backend.RunStorage
GenericRunStorage = RunStorage
StatusCounts = _backend.StatusCounts
StorageError = _backend.StorageError
ValidationResult = _backend.ValidationResult

atomic_write_csv = _backend.atomic_write_csv
atomic_write_dataframe = _backend.atomic_write_dataframe
atomic_write_json = _backend.atomic_write_json
atomic_write_npz = _backend.atomic_write_npz
checksum_path = _backend.checksum_path
classify_cells = _backend.classify_cells
finalize_run = _backend.finalize_run
load_cell_metrics = _backend.load_cell_metrics
load_cell_predictions = _backend.load_cell_predictions
manifest_sha256 = _backend.manifest_sha256
register_existing_artifact = _backend.register_existing_artifact
status_counts = _backend.status_counts
validate_cell = _backend.validate_cell
validate_checksum = _backend.validate_checksum
write_cell_artifacts = _backend.write_cell_artifacts


PHASE_LEASE_SCHEMA_VERSION = "bdb-neural-phase-lease-v1"
PHASE_LEASE_RUNNING_FILE = "_RUNNING"
_PHASE_LEASES = frozenset({"selector", "refit"})
_SLURM_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_SLURM_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
_SLURM_LIVE_STATES = frozenset(
    {
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
)


class PhaseLeaseError(StorageError):
    """A neural phase marker is unsafe, foreign, or cannot be proven stale."""


@dataclass(frozen=True)
class PhaseLeaseClaim:
    """Result of an atomic selector/refit claim attempt.

    ``status`` is one of ``claimed``, ``running``, or ``complete``.  Only a
    claimed lease may be released, and release verifies the exact marker bytes
    so it can never delete another worker's evidence.
    """

    status: str
    marker_path: Path
    marker_sha256: str | None = None
    reclaimed_marker_sha256: str | None = None
    replaced_completion_sha256: str | None = None


def _phase_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _phase_canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _immutable_create_bytes(path: Path, payload: bytes) -> None:
    """Create immutable evidence or accept the exact bytes already present."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PhaseLeaseError(f"phase evidence cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise PhaseLeaseError(
                    f"refusing to replace different phase evidence: {path}"
                ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_phase_marker(path: Path, payload: bytes) -> None:
    """Atomically create a live marker; an existing marker always wins."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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
            raise PhaseLeaseError(f"phase marker appeared during claim: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_slurm_state(value: str) -> str:
    state = str(value).strip().split(maxsplit=1)[0].rstrip("+").upper()
    if state not in _SLURM_TERMINAL_STATES | _SLURM_LIVE_STATES:
        raise PhaseLeaseError(f"unrecognized Slurm job state: {value!r}")
    return state


def _query_slurm_job_state(job_id: str) -> str:
    """Return one exact root-job state, failing closed when Slurm is ambiguous."""

    if not _SLURM_JOB_ID.fullmatch(str(job_id)):
        raise PhaseLeaseError(f"invalid Slurm job id in phase marker: {job_id!r}")
    try:
        queued = subprocess.run(
            ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        queued = None
        queue_error = exc
    else:
        queue_error = None
    if queued is not None and queued.returncode == 0:
        states = [line.strip() for line in queued.stdout.splitlines() if line.strip()]
        if states:
            normalized = {_normalize_slurm_state(state) for state in states}
            if len(normalized) != 1:
                raise PhaseLeaseError(
                    f"Slurm returned ambiguous live states for job {job_id}: "
                    f"{sorted(normalized)}"
                )
            return next(iter(normalized))
    try:
        accounting = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                str(job_id),
                "--format=JobIDRaw,State",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = queue_error or exc
        raise PhaseLeaseError(
            f"could not query Slurm state for prior job {job_id}: {detail}"
        ) from exc
    if accounting.returncode != 0:
        detail = accounting.stderr.strip() or (
            "squeue failed" if queued is None else queued.stderr.strip()
        )
        raise PhaseLeaseError(
            f"could not query Slurm state for prior job {job_id}: {detail}"
        )
    root_states = []
    for line in accounting.stdout.splitlines():
        fields = line.strip().split("|", maxsplit=2)
        if len(fields) >= 2 and fields[0] == str(job_id):
            root_states.append(_normalize_slurm_state(fields[1]))
    if len(root_states) != 1:
        raise PhaseLeaseError(
            f"Slurm accounting has no unique root state for prior job {job_id}"
        )
    return root_states[0]


def _phase_owner(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    environ = os.environ if environment is None else environment
    job_id = str(environ.get("SLURM_JOB_ID", "")).strip()
    slurm = None
    if job_id:
        if not _SLURM_JOB_ID.fullmatch(job_id):
            raise PhaseLeaseError(f"current Slurm job id is invalid: {job_id!r}")
        restart_raw = str(
            environ.get(
                "SLURM_RESTART_COUNT",
                environ.get("SLURM_JOB_RESTART_COUNT", "0"),
            )
        ).strip()
        if not restart_raw.isdigit():
            raise PhaseLeaseError("current Slurm restart count is invalid")
        slurm = {
            "job_id": job_id,
            "restart_count": int(restart_raw),
            "cluster_name": str(environ.get("SLURM_CLUSTER_NAME", "")).strip() or None,
            "array_job_id": str(environ.get("SLURM_ARRAY_JOB_ID", "")).strip() or None,
            "array_task_id": str(environ.get("SLURM_ARRAY_TASK_ID", "")).strip() or None,
            "dependency": str(environ.get("SLURM_JOB_DEPENDENCY", "")).strip()
            or None,
        }
    owner = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm": slurm,
    }
    owner["owner_token"] = hashlib.sha256(
        _phase_canonical_json_bytes(owner)
    ).hexdigest()
    return owner


def _build_phase_marker(
    manifest_hash: str,
    key: CellKey,
    phase: str,
    owner: Mapping[str, Any],
    *,
    takeover: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        # Keep the imported backend's schema and required identity fields so a
        # final-cell refit lease remains visible to its normal validator.
        "schema_version": 1,
        "manifest_hash": manifest_hash,
        "key": asdict(key),
        "pid": int(owner["pid"]),
        "hostname": str(owner["hostname"]),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "phase_lease": {
            "schema_version": PHASE_LEASE_SCHEMA_VERSION,
            "phase": phase,
            "owner": dict(owner),
            "takeover": None if takeover is None else dict(takeover),
        },
    }


def _parse_phase_marker(
    payload: bytes,
    *,
    manifest_hash: str,
    key: CellKey,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        marker = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseLeaseError("phase marker is unreadable") from exc
    if not isinstance(marker, dict) or marker.get("schema_version") != 1:
        raise PhaseLeaseError("phase marker schema is invalid")
    if payload != _phase_json_bytes(marker):
        raise PhaseLeaseError("phase marker JSON is not canonical")
    if marker.get("manifest_hash") != manifest_hash:
        raise PhaseLeaseError("phase marker belongs to a different run")
    if marker.get("key") != asdict(key):
        raise PhaseLeaseError("phase marker belongs to a different cell")
    pid = marker.get("pid")
    hostname = marker.get("hostname")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(hostname, str)
        or not hostname
    ):
        raise PhaseLeaseError("phase marker process identity is invalid")
    started = marker.get("started_utc")
    try:
        parsed_started = datetime.fromisoformat(str(started))
    except ValueError as exc:
        raise PhaseLeaseError("phase marker start time is invalid") from exc
    if parsed_started.tzinfo is None:
        raise PhaseLeaseError("phase marker start time has no timezone")
    lease = marker.get("phase_lease")
    if lease is None:
        expected_legacy = {
            "schema_version",
            "manifest_hash",
            "key",
            "pid",
            "hostname",
            "started_utc",
        }
        if set(marker) != expected_legacy:
            raise PhaseLeaseError("legacy phase marker has unexpected fields")
        return marker, None
    if set(marker) != {
        "schema_version",
        "manifest_hash",
        "key",
        "pid",
        "hostname",
        "started_utc",
        "phase_lease",
    }:
        raise PhaseLeaseError("phase marker has unexpected fields")
    if not isinstance(lease, dict) or set(lease) != {
        "schema_version",
        "phase",
        "owner",
        "takeover",
    }:
        raise PhaseLeaseError("phase lease fields are invalid")
    if lease.get("schema_version") != PHASE_LEASE_SCHEMA_VERSION:
        raise PhaseLeaseError("phase lease schema is invalid")
    if lease.get("phase") != phase:
        raise PhaseLeaseError("phase marker belongs to a different phase")
    owner = lease.get("owner")
    if not isinstance(owner, dict) or set(owner) != {
        "hostname",
        "pid",
        "slurm",
        "owner_token",
    }:
        raise PhaseLeaseError("phase lease owner fields are invalid")
    if owner.get("hostname") != hostname or owner.get("pid") != pid:
        raise PhaseLeaseError("phase lease owner conflicts with its marker")
    claimed_token = owner.pop("owner_token")
    actual_token = hashlib.sha256(_phase_canonical_json_bytes(owner)).hexdigest()
    owner["owner_token"] = claimed_token
    if claimed_token != actual_token:
        raise PhaseLeaseError("phase lease owner token is invalid")
    slurm = owner.get("slurm")
    if slurm is not None:
        if not isinstance(slurm, dict) or set(slurm) != {
            "job_id",
            "restart_count",
            "cluster_name",
            "array_job_id",
            "array_task_id",
            "dependency",
        }:
            raise PhaseLeaseError("phase lease Slurm identity is invalid")
        if (
            not _SLURM_JOB_ID.fullmatch(str(slurm.get("job_id", "")))
            or isinstance(slurm.get("restart_count"), bool)
            or not isinstance(slurm.get("restart_count"), int)
            or slurm["restart_count"] < 0
        ):
            raise PhaseLeaseError("phase lease Slurm identity is invalid")
        for optional in (
            "cluster_name",
            "array_job_id",
            "array_task_id",
            "dependency",
        ):
            if slurm.get(optional) is not None and not isinstance(slurm[optional], str):
                raise PhaseLeaseError("phase lease Slurm identity is invalid")
    takeover = lease.get("takeover")
    if takeover is not None and not isinstance(takeover, dict):
        raise PhaseLeaseError("phase lease takeover evidence is invalid")
    return marker, owner


def _stale_phase_evidence(
    prior_owner: Mapping[str, Any] | None,
    legacy_marker: Mapping[str, Any],
    current_owner: Mapping[str, Any],
    scheduler_state_lookup: Any,
) -> dict[str, Any] | None:
    """Return immutable-safe stale evidence, or ``None`` for a live owner."""

    prior_host = str(legacy_marker["hostname"])
    prior_pid = int(legacy_marker["pid"])
    prior_slurm = None if prior_owner is None else prior_owner.get("slurm")
    current_slurm = current_owner.get("slurm")
    if isinstance(prior_slurm, Mapping) and isinstance(current_slurm, Mapping):
        prior_cluster = prior_slurm.get("cluster_name")
        current_cluster = current_slurm.get("cluster_name")
        if prior_cluster and current_cluster and prior_cluster != current_cluster:
            raise PhaseLeaseError("phase marker belongs to a different Slurm cluster")
        prior_job = str(prior_slurm["job_id"])
        current_job = str(current_slurm["job_id"])
        if prior_job == current_job:
            prior_restart = int(prior_slurm["restart_count"])
            current_restart = int(current_slurm["restart_count"])
            if current_restart > prior_restart:
                return {
                    "kind": "slurm_restart_count_advanced",
                    "prior_job_id": prior_job,
                    "prior_restart_count": prior_restart,
                    "current_restart_count": current_restart,
                }
            if current_restart < prior_restart:
                raise PhaseLeaseError("phase marker has a future Slurm restart count")
            return None
        dependency = current_slurm.get("dependency")
        if isinstance(dependency, str):
            for clause in re.split(r"[?,]", dependency):
                fields = clause.strip().split(":")
                if (
                    len(fields) >= 2
                    and fields[0]
                    in {"afterany", "afterok", "afternotok", "aftercorr"}
                    and prior_job in fields[1:]
                ):
                    return {
                        "kind": "slurm_satisfied_dependency",
                        "prior_job_id": prior_job,
                        "dependency_type": fields[0],
                        "current_job_id": current_job,
                    }
        state = _normalize_slurm_state(scheduler_state_lookup(prior_job))
        if state in _SLURM_LIVE_STATES:
            return None
        return {
            "kind": "slurm_terminal_job",
            "prior_job_id": prior_job,
            "terminal_state": state,
        }
    if prior_host != str(current_owner["hostname"]):
        raise PhaseLeaseError(
            "cannot safely reclaim a foreign-host phase marker without "
            "both prior and current Slurm identities"
        )
    if prior_pid == int(current_owner["pid"]):
        return None
    try:
        os.kill(prior_pid, 0)
    except ProcessLookupError:
        return {
            "kind": "dead_local_process",
            "hostname": prior_host,
            "pid": prior_pid,
        }
    except PermissionError:
        return None
    except OSError as exc:
        raise PhaseLeaseError(
            f"could not validate local phase-marker process {prior_pid}: {exc}"
        ) from exc
    return None


def claim_phase_lease(
    marker_path: str | Path,
    *,
    manifest_hash: str,
    key: CellKey,
    phase: str,
    completion_path: str | Path | None = None,
    completion_validator: Any = None,
    environment: Mapping[str, str] | None = None,
    scheduler_state_lookup: Any = None,
) -> PhaseLeaseClaim:
    """Claim selector/refit work, reclaiming only a provably stale marker.

    A foreign-node marker is never aged out heuristically.  A new allocation
    may take it over only when Slurm reports the exact prior root job in a
    terminal state (or a restart count proves a later incarnation of the same
    job).  The prior marker bytes are create-only archived before replacement.
    """

    marker = Path(marker_path)
    if phase not in _PHASE_LEASES:
        raise PhaseLeaseError(f"unsupported neural phase lease: {phase}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest_hash)):
        raise PhaseLeaseError("phase lease manifest hash is invalid")
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.is_symlink() or marker.parent.is_symlink():
        raise PhaseLeaseError(f"phase marker path is unsafe: {marker}")
    completion = None if completion_path is None else Path(completion_path)
    if completion is not None and not callable(completion_validator):
        raise PhaseLeaseError(
            "phase completion paths require an explicit validity callback"
        )
    owner = _phase_owner(environment)
    lookup = _query_slurm_job_state if scheduler_state_lookup is None else scheduler_state_lookup
    lock_path = marker.with_name(f".{marker.name}.lease.lock")
    if lock_path.is_symlink():
        raise PhaseLeaseError(f"phase marker lock is unsafe: {lock_path}")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise PhaseLeaseError(f"could not lock phase marker: {marker}") from exc
        invalid_completion: tuple[bytes, str, Path] | None = None
        if completion is not None and completion.exists():
            if completion.is_symlink() or not completion.is_file():
                raise PhaseLeaseError(
                    f"phase completion marker path is unsafe: {completion}"
                )
            try:
                completion_is_valid = completion_validator()
            except Exception as exc:
                raise PhaseLeaseError(
                    f"phase completion validity check failed: {completion}: {exc}"
                ) from exc
            if not isinstance(completion_is_valid, bool):
                raise PhaseLeaseError(
                    "phase completion validity callback must return boolean"
                )
            if completion_is_valid:
                return PhaseLeaseClaim("complete", marker)
            completion_payload = completion.read_bytes()
            completion_sha256 = hashlib.sha256(completion_payload).hexdigest()
            completion_archive = (
                completion.parent
                / f".{completion.name}.takeovers"
                / f"{completion_sha256}.json"
            )
            invalid_completion = (
                completion_payload,
                completion_sha256,
                completion_archive,
            )
        if not marker.exists():
            takeover = None
            replaced_completion_sha256 = None
            if invalid_completion is not None:
                completion_payload, completion_sha256, completion_archive = (
                    invalid_completion
                )
                _immutable_create_bytes(completion_archive, completion_payload)
                completion.unlink()
                takeover = {
                    "prior_completion_sha256": completion_sha256,
                    "archive": str(completion_archive.relative_to(marker.parent)),
                    "evidence": {"kind": "explicit_invalid_completion"},
                }
                replaced_completion_sha256 = completion_sha256
            value = _build_phase_marker(
                str(manifest_hash), key, phase, owner, takeover=takeover
            )
            payload = _phase_json_bytes(value)
            _atomic_create_phase_marker(marker, payload)
            return PhaseLeaseClaim(
                "claimed",
                marker,
                hashlib.sha256(payload).hexdigest(),
                replaced_completion_sha256=replaced_completion_sha256,
            )
        if marker.is_symlink() or not marker.is_file():
            raise PhaseLeaseError(f"phase marker path is unsafe: {marker}")
        prior_payload = marker.read_bytes()
        prior_marker, prior_owner = _parse_phase_marker(
            prior_payload,
            manifest_hash=str(manifest_hash),
            key=key,
            phase=phase,
        )
        if (
            prior_owner is not None
            and prior_owner.get("owner_token") == owner.get("owner_token")
        ):
            return PhaseLeaseClaim("running", marker)
        evidence = _stale_phase_evidence(
            prior_owner, prior_marker, owner, lookup
        )
        if evidence is None:
            return PhaseLeaseClaim("running", marker)
        prior_sha256 = hashlib.sha256(prior_payload).hexdigest()
        archive = (
            marker.parent
            / f".{marker.name}.takeovers"
            / f"{prior_sha256}.json"
        )
        _immutable_create_bytes(archive, prior_payload)
        marker.unlink()
        takeover = {
            "prior_marker_sha256": prior_sha256,
            "archive": str(archive.relative_to(marker.parent)),
            "evidence": evidence,
        }
        replaced_completion_sha256 = None
        if invalid_completion is not None:
            completion_payload, completion_sha256, completion_archive = invalid_completion
            _immutable_create_bytes(completion_archive, completion_payload)
            completion.unlink()
            takeover["replaced_completion"] = {
                "prior_completion_sha256": completion_sha256,
                "archive": str(completion_archive.relative_to(marker.parent)),
                "evidence": {"kind": "explicit_invalid_completion"},
            }
            replaced_completion_sha256 = completion_sha256
        value = _build_phase_marker(
            str(manifest_hash), key, phase, owner, takeover=takeover
        )
        payload = _phase_json_bytes(value)
        _atomic_create_phase_marker(marker, payload)
        return PhaseLeaseClaim(
            "claimed",
            marker,
            hashlib.sha256(payload).hexdigest(),
            reclaimed_marker_sha256=prior_sha256,
            replaced_completion_sha256=replaced_completion_sha256,
        )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def release_phase_lease(claim: PhaseLeaseClaim) -> None:
    """Release only the exact marker created by ``claim_phase_lease``."""

    if claim.status != "claimed" or claim.marker_sha256 is None:
        raise PhaseLeaseError("only a claimed phase lease can be released")
    marker = Path(claim.marker_path)
    lock_path = marker.with_name(f".{marker.name}.lease.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise PhaseLeaseError(f"could not lock phase marker: {marker}") from exc
        if not marker.is_file() or marker.is_symlink():
            raise PhaseLeaseError(f"claimed phase marker disappeared: {marker}")
        actual = hashlib.sha256(marker.read_bytes()).hexdigest()
        if actual != claim.marker_sha256:
            raise PhaseLeaseError(
                f"refusing to clear a different phase marker: {marker}"
            )
        marker.unlink()
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def immutable_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically create one canonical JSON receipt or accept an exact match.

    Unlike :func:`atomic_write_json`, this helper never replaces an existing
    scientific receipt.  The hard-link commit makes concurrent creators safe:
    exactly one payload wins, and every other writer must prove byte equality.
    """

    if not isinstance(value, Mapping):
        raise TypeError("immutable JSON receipt must be a mapping")
    target = Path(path)
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace a different immutable JSON receipt: {target}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return target


STORAGE_BACKEND_SCHEMA_VERSION = "bdb-storage-backend-v1"
DEFAULT_QUEUE_SCHEMA_VERSION = "bdb-execution-queues-v1"
BACKEND_RELATIVE_PATH = "rushing_study/storage.py"


class BackendPinError(StorageError):
    """The run manifest does not pin the exact imported storage backend."""


class QueuePlanError(StorageError):
    """A queue plan is ambiguous, incomplete, or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_backend_receipt(repo_root: str | Path = ".") -> dict[str, Any]:
    """Return the content pin required in every suite run manifest."""

    root = Path(repo_root).resolve()
    path = (root / BACKEND_RELATIVE_PATH).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BackendPinError("storage backend escapes repository root") from exc
    if not path.is_file():
        raise BackendPinError(f"storage backend is missing: {path}")
    return {
        "schema_version": STORAGE_BACKEND_SCHEMA_VERSION,
        "module": "rushing_study.storage",
        "path": BACKEND_RELATIVE_PATH,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_storage_backend_receipt(
    receipt: Mapping[str, Any], *, repo_root: str | Path = "."
) -> None:
    if not isinstance(receipt, Mapping):
        raise BackendPinError("storage_backend must be an object")
    actual = storage_backend_receipt(repo_root)
    if dict(receipt) != actual:
        raise BackendPinError(
            "run storage-backend pin does not match the current implementation"
        )


def bind_storage_backend(
    manifest: Mapping[str, Any], *, repo_root: str | Path = "."
) -> dict[str, Any]:
    """Return an independent run manifest bound to the current backend bytes."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    output = dict(manifest)
    current = storage_backend_receipt(repo_root)
    existing = output.get("storage_backend")
    if existing is not None and existing != current:
        raise BackendPinError("manifest is already bound to a different storage backend")
    output["storage_backend"] = current
    return output


def initialize_run_dir(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_hash: str | None = None,
    repo_root: str | Path = ".",
    require_backend_pin: bool = True,
) -> RunStorage:
    """Initialize generic storage after enforcing its implementation pin."""

    if require_backend_pin:
        receipt = manifest.get("storage_backend") if isinstance(manifest, Mapping) else None
        verify_storage_backend_receipt(receipt, repo_root=repo_root)
    return _backend.initialize_run_dir(
        run_dir, manifest, manifest_hash=manifest_hash
    )


@dataclass(frozen=True)
class QueueSpec:
    name: str
    device: str
    workers: int
    model_families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.name not in {"cpu_tabular", "gpu_neural"}:
            raise QueuePlanError(f"unsupported queue name: {self.name}")
        if self.device not in {"cpu", "gpu"}:
            raise QueuePlanError(f"unsupported device: {self.device}")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers <= 0:
            raise QueuePlanError("queue workers must be a positive integer")
        if not self.model_families or len(self.model_families) != len(set(self.model_families)):
            raise QueuePlanError("queue model_families must be unique and nonempty")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_families"] = list(self.model_families)
        return value


DEFAULT_QUEUE_SPECS = (
    QueueSpec("cpu_tabular", "cpu", 12, ("glm", "lightgbm")),
    QueueSpec(
        "gpu_neural",
        "gpu",
        1,
        ("relnet", "attn_relnet", "set_transformer"),
    ),
)


def recommended_cpu_workers(
    total_ram_bytes: int,
    worker_peak_bytes: int,
    *,
    cap: int = 12,
    usable_fraction: float = 0.75,
) -> int:
    """Apply the frozen RAM-aware CPU concurrency rule, with a safe floor of 1."""

    for value, name in ((total_ram_bytes, "total_ram_bytes"), (worker_peak_bytes, "worker_peak_bytes")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise QueuePlanError(f"{name} must be a positive integer")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise QueuePlanError("cap must be a positive integer")
    if not isinstance(usable_fraction, (int, float)) or isinstance(usable_fraction, bool):
        raise QueuePlanError("usable_fraction must be numeric")
    if not 0 < float(usable_fraction) <= 1:
        raise QueuePlanError("usable_fraction must be in (0, 1]")
    memory_limit = int(float(usable_fraction) * total_ram_bytes) // worker_peak_bytes
    return max(1, min(cap, memory_limit))


def queue_manifest(
    *,
    cpu_workers: int = 12,
    gpu_workers: int = 1,
    include_set_transformer: bool = True,
) -> dict[str, Any]:
    if not isinstance(include_set_transformer, bool):
        raise QueuePlanError("include_set_transformer must be boolean")
    neural_families = (
        ("relnet", "attn_relnet", "set_transformer")
        if include_set_transformer
        else ("relnet", "attn_relnet")
    )
    queues = (
        QueueSpec("cpu_tabular", "cpu", cpu_workers, ("glm", "lightgbm")),
        QueueSpec(
            "gpu_neural",
            "gpu",
            gpu_workers,
            neural_families,
        ),
    )
    return {
        "schema_version": DEFAULT_QUEUE_SCHEMA_VERSION,
        "queues": {queue.name: queue.as_dict() for queue in queues},
    }


def cell_keys_from_design(design: Mapping[str, Any]) -> list[CellKey]:
    records = design.get("required_cells") if isinstance(design, Mapping) else None
    if not isinstance(records, list):
        raise QueuePlanError("design.required_cells must be an array")
    keys: list[CellKey] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise QueuePlanError("required cell records must be objects")
        try:
            keys.append(
                CellKey(
                    str(record["branch"]),
                    int(record["repeat"]),
                    int(record["n_train"]),
                    str(record["model"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QueuePlanError(f"invalid required cell: {record}") from exc
    if len(keys) != len(set(keys)):
        raise QueuePlanError("design contains duplicate cell keys")
    return keys


def _queue_records(design: Mapping[str, Any]) -> dict[str, list[tuple[CellKey, Mapping[str, Any]]]]:
    records = design.get("required_cells") if isinstance(design, Mapping) else None
    if not isinstance(records, list):
        raise QueuePlanError("design.required_cells must be an array")
    output = {"cpu_tabular": [], "gpu_neural": []}
    for record in records:
        if not isinstance(record, Mapping):
            raise QueuePlanError("required cell records must be objects")
        queue = record.get("queue")
        if queue not in output:
            raise QueuePlanError(f"cell has unknown queue {queue!r}")
        try:
            key = CellKey(
                str(record["branch"]),
                int(record["repeat"]),
                int(record["n_train"]),
                str(record["model"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QueuePlanError(f"invalid required cell: {record}") from exc
        output[queue].append((key, record))
    return output


def status_by_queue(
    storage: RunStorage,
    design: Mapping[str, Any],
) -> dict[str, StatusCounts]:
    """Classify resume state independently for the CPU and GPU queues."""

    return {
        queue: storage.status_counts(key for key, _record in records)
        for queue, records in _queue_records(design).items()
    }


def pending_cells_by_queue(
    storage: RunStorage,
    design: Mapping[str, Any],
) -> dict[str, list[CellKey]]:
    """Return only incomplete/corrupt cells; valid commits remain untouched."""

    return {
        queue: storage.pending_cells(key for key, _record in records)
        for queue, records in _queue_records(design).items()
    }


def estimate_queue_eta_seconds(
    storage: RunStorage,
    design: Mapping[str, Any],
    seconds_per_model: Mapping[str, float],
    *,
    cpu_workers: int = 12,
    gpu_workers: int = 1,
) -> dict[str, Any]:
    """Estimate concurrent queue completion; run ETA is the slower queue."""

    workers = {"cpu_tabular": cpu_workers, "gpu_neural": gpu_workers}
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in workers.values()):
        raise QueuePlanError("queue worker counts must be positive integers")
    queue_seconds: dict[str, float] = {}
    pending_counts: dict[str, int] = {}
    for queue, records in _queue_records(design).items():
        duration = 0.0
        count = 0
        for key, record in records:
            if storage.validate_cell(key).is_complete:
                continue
            model = str(record["model"])
            if model not in seconds_per_model:
                raise QueuePlanError(f"missing runtime estimate for model {model!r}")
            estimate = seconds_per_model[model]
            if isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or estimate < 0:
                raise QueuePlanError(f"invalid runtime estimate for model {model!r}")
            duration += float(estimate)
            count += 1
        pending_counts[queue] = count
        queue_seconds[queue] = duration / workers[queue]
    return {
        "queue_seconds": queue_seconds,
        "pending_cells": pending_counts,
        "eta_seconds": max(queue_seconds.values(), default=0.0),
    }
