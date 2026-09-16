"""Resolve recorded source-layout paths without changing receipt identities.

This module relocates files only. Callers must still validate the original
recorded checksum against the bytes they consume. Changed executable source
must fail its historical admission checks.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def resolve_record_path(root: str | Path, relative: str | Path) -> Path:
    """Find an original-layout record or its explicitly relocated Betty file.

    Original paths remain supported for complete historical checkouts and test
    fixtures. A relocated file is considered only when the original path does
    not exist; this cannot conceal a corrupt original file.
    """
    project = Path(root).resolve()
    recorded = PurePosixPath(str(relative))
    if recorded.is_absolute() or not recorded.parts or ".." in recorded.parts:
        raise ValueError("recorded path is unsafe")
    def checked_target(parts: tuple[str, ...]) -> Path:
        candidate = project
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("recorded path cannot contain a symlink")
        return candidate

    target = checked_target(recorded.parts)
    if recorded.parts[:2] == ("hpc", "betty") and not target.exists():
        target = checked_target(("scripts", "betty", *recorded.parts[2:]))
    resolved = target.resolve()
    try:
        resolved.relative_to(project)
    except ValueError as exc:
        raise ValueError("recorded path escapes project root") from exc
    return resolved
