"""Verified local imports for the missing BDB competition releases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceDefinition:
    release: int
    source_id: str
    source_path: str
    destination: str
    catalog_receipts: tuple[str, ...]


SOURCE_DEFINITIONS: dict[int, SourceDefinition] = {
    2021: SourceDefinition(
        2021,
        "nfl_big_data_bowl_2021",
        "/Users/Jonathan/wsabi/data/lake/imports/nfl/bdb_2021",
        "data/bdb2021/raw",
        (
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062851Z-3589656c8ba7.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062632Z-7ba00b44d356.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062632Z-02e69f8c17f0.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062632Z-75b24ba1de15.json",
        ),
    ),
    2022: SourceDefinition(
        2022,
        "nfl_big_data_bowl_2022",
        "/Users/Jonathan/wsabi/data/lake/imports/nfl/bdb_2022",
        "data/bdb2022/raw",
        (
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062953Z-2e4890a5a16e.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062632Z-86a22d475d7c.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062633Z-752849f110bb.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062633Z-86aae424385c.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-19/20260719T062633Z-a2495c6a068a.json",
        ),
    ),
    2024: SourceDefinition(
        2024,
        "nfl_big_data_bowl_2024",
        "/Users/Jonathan/wsabi/data/lake/raw/source_id=nfl_big_data_bowl_2024",
        "data/bdb2024/raw",
        (
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-18/20260719T024319Z-f4974f29c639.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-18/20260719T024255Z-350d82446b36.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-18/20260719T024046Z-59cff21460ab.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-18/20260719T024256Z-c37a136c96de.json",
            "/Users/Jonathan/wsabi/data/lake/manifests/snapshot_date=2026-07-18/20260719T024257Z-97a9e2602008.json",
        ),
    ),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def tree_inventory(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in base.rglob("*") if item.is_file()), key=lambda item: item.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"source tree {base} contains no files")
    return rows


def _catalog_receipts(paths: Iterable[str]) -> list[dict[str, Any]]:
    receipts = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("status") != "success":
            raise ValueError(f"catalog receipt is not successful: {path}")
        receipts.append(
            {
                "dataset_id": payload.get("dataset_id"),
                "source_id": payload.get("source_id"),
                "run_id": payload.get("run_id"),
                "row_count": payload.get("row_count"),
                "byte_count": payload.get("byte_count"),
                "manifest_sha256": sha256_file(path),
            }
        )
    return receipts


def verify_copy(source: str | Path, destination: str | Path) -> tuple[list[dict[str, Any]], str]:
    source_rows = tree_inventory(source)
    destination_rows = tree_inventory(destination)
    if source_rows != destination_rows:
        source_by_path = {row["path"]: row for row in source_rows}
        destination_by_path = {row["path"]: row for row in destination_rows}
        differences = sorted(
            path
            for path in set(source_by_path) | set(destination_by_path)
            if source_by_path.get(path) != destination_by_path.get(path)
        )
        raise ValueError(f"source/destination copy differs at {differences[:10]}")
    return destination_rows, canonical_json_hash(destination_rows)


def build_import_receipt(definition: SourceDefinition, repo_root: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    source = Path(definition.source_path).resolve()
    destination = (repo / definition.destination).resolve()
    files, tree_hash = verify_copy(source, destination)
    catalog = _catalog_receipts(definition.catalog_receipts)
    scientific_identity = {
        "schema_version": 1,
        "release": definition.release,
        "source_id": definition.source_id,
        "destination": definition.destination,
        "tree_sha256": tree_hash,
        "file_count": len(files),
        "byte_count": sum(int(row["bytes"]) for row in files),
        "catalog_receipts": catalog,
        "files": files,
    }
    return {
        **scientific_identity,
        "receipt_sha256": canonical_json_hash(scientific_identity),
        "source_path_observed": str(source),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_local_tree_receipt(
    *,
    source_id: str,
    destination: str,
    repo_root: str | Path = ".",
    include: Iterable[str] = ("**/*",),
) -> dict[str, Any]:
    """Receipt an already-local release without copying or host-path identity.

    ``include`` is a frozen list of relative glob patterns.  It is useful for
    existing releases where derived convenience files (for example
    ``tracking_all.csv``) must not silently enter the scientific source tree.
    """

    repo = Path(repo_root).resolve()
    root = (repo / destination).resolve()
    try:
        root.relative_to(repo)
    except ValueError as exc:
        raise ValueError("local source destination must be inside the repository") from exc
    selected: set[Path] = set()
    patterns = tuple(str(pattern) for pattern in include)
    if not patterns:
        raise ValueError("local source include patterns cannot be empty")
    for pattern in patterns:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    files = []
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"no local files matched {patterns!r} under {root}")
    scientific_identity = {
        "schema_version": 1,
        "source_id": source_id,
        "destination": destination,
        "tree_sha256": canonical_json_hash(files),
        "file_count": len(files),
        "byte_count": sum(int(row["bytes"]) for row in files),
        "catalog_receipts": [],
        "files": files,
        "include_patterns": list(patterns),
    }
    return {
        **scientific_identity,
        "receipt_sha256": canonical_json_hash(scientific_identity),
        "source_path_observed": str(root),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_import_receipt(receipt: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text())
        # Wall-clock and observed host path are audit metadata, not identity.
        if existing.get("receipt_sha256") != receipt.get("receipt_sha256"):
            raise FileExistsError(f"different immutable receipt already exists at {path}")
        return path
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def import_release(
    release: int,
    *,
    repo_root: str | Path = ".",
    copy_if_missing: bool = True,
) -> dict[str, Any]:
    if release not in SOURCE_DEFINITIONS:
        raise KeyError(f"no import definition for BDB{release}")
    definition = SOURCE_DEFINITIONS[release]
    repo = Path(repo_root).resolve()
    source = Path(definition.source_path).resolve()
    destination = (repo / definition.destination).resolve()
    if not destination.exists():
        if not copy_if_missing:
            raise FileNotFoundError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    receipt = build_import_receipt(definition, repo)
    write_import_receipt(
        receipt,
        repo / "configs" / "bdb_suite" / "sources" / f"bdb{release}_import.json",
    )
    return receipt


def inventory(repo_root: str | Path = ".") -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    releases = {}
    for release, definition in SOURCE_DEFINITIONS.items():
        destination = repo / definition.destination
        releases[str(release)] = {
            **asdict(definition),
            "source_exists": Path(definition.source_path).is_dir(),
            "destination_exists": destination.is_dir(),
            "destination_bytes": sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) if destination.is_dir() else 0,
        }
    return {"repo_root": str(repo), "releases": releases}
