"""Versioned data adapters for the multi-release BDB stability suite."""

from importlib import import_module

from .common import CANONICAL_PLAYER_CHANNELS, CohortAudit, PreparedTask


ADAPTER_MODULES = {
    "bdb2020_rushing_harmonized": "bdb_study.adapters.bdb2020",
    "bdb2021_completion": "bdb_study.adapters.bdb2021",
    "bdb2022_punt_returns": "bdb_study.adapters.bdb2022",
    "bdb2023_sack": "bdb_study.adapters.bdb2023",
    "bdb2024_tackle": "bdb_study.adapters.bdb2024",
    "bdb2025_man_zone": "bdb_study.adapters.bdb2025",
    "bdb2026_trajectory": "bdb_study.adapters.bdb2026",
}


def get_adapter(task_id: str):
    """Resolve an adapter lazily so CPU inventory commands stay lightweight."""

    try:
        module_name = ADAPTER_MODULES[str(task_id)]
    except KeyError as exc:
        raise KeyError(f"unknown BDB task adapter {task_id!r}") from exc
    return import_module(module_name)


__all__ = [
    "ADAPTER_MODULES",
    "CANONICAL_PLAYER_CHANNELS",
    "CohortAudit",
    "PreparedTask",
    "get_adapter",
]
