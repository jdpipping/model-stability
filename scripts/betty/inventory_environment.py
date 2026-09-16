"""Print exact installed versions for every distribution in a project lock."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from packaging.requirements import Requirement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    arguments = parser.parse_args()
    packages: list[dict[str, str | bool | None]] = []
    for raw in arguments.lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        active = requirement.marker is None or requirement.marker.evaluate()
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        packages.append(
            {
                "name": requirement.name,
                "active": active,
                "installed": installed,
            }
        )
    print(
        "ZOO_ENVIRONMENT="
        + json.dumps({"lock": str(arguments.lock), "packages": packages}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
