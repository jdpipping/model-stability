"""Data loading and row selection for the prospective rushing study."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .models import make_tabular_features


def _read_numpy_portable_pickle(path: Path) -> Any:
    """Read NumPy-2 pickles with the NumPy 1.26 ABI used by NGC TensorFlow.

    NumPy 2 records private module names below ``numpy._core`` in pickles.
    NumPy 1.26 exposes the same reconstruction functions below ``numpy.core``.
    Registering those two legacy aliases avoids changing the frozen data file or
    upgrading NumPy beyond the version supported by the container's TensorFlow.
    """
    try:
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        for name in ("multiarray", "numeric"):
            module = importlib.import_module(f"numpy.core.{name}")
            sys.modules.setdefault(f"numpy._core.{name}", module)
    return pd.read_pickle(path)


@dataclass(frozen=True)
class StudyData:
    spatial: np.ndarray
    player_set: np.ndarray
    tabular: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame

    def indices_for_games(self, game_ids: Iterable[Any]) -> np.ndarray:
        wanted = {str(value) for value in game_ids}
        mask = self.metadata["game_id"].astype(str).isin(wanted).to_numpy()
        return np.flatnonzero(mask)

    def game_table(self) -> pd.DataFrame:
        return (
            self.metadata[["game_id", "season"]]
            .drop_duplicates()
            .sort_values(["season", "game_id"], kind="stable")
            .reset_index(drop=True)
        )


def load_study_data(config: dict[str, Any], mmap_mode: str | None = "r") -> StudyData:
    """Load aligned, non-augmented representations and attach GameId/Season."""
    data_cfg = config["data"]
    processed = Path(data_cfg["processed_dir"])
    raw_path = Path(data_cfg["raw_train_csv"])
    spatial = np.load(processed / "train_x.npy", mmap_mode=mmap_mode)
    player_set = np.load(processed / "train_x_set.npy", mmap_mode=mmap_mode)
    labels = _read_numpy_portable_pickle(processed / "train_y.pkl").reset_index(drop=True)
    if not (len(spatial) == len(player_set) == len(labels)):
        raise ValueError("Processed feature arrays and labels are not row-aligned.")

    raw = pd.read_csv(raw_path, usecols=["GameId", "PlayId", "Season"])
    raw["PlayId"] = raw["PlayId"].astype(str)
    mapping_counts = raw.groupby("PlayId").agg(
        game_count=("GameId", "nunique"),
        season_count=("Season", "nunique"),
    )
    if ((mapping_counts["game_count"] != 1) | (mapping_counts["season_count"] != 1)).any():
        raise ValueError("Every base PlayId must map to exactly one GameId and Season.")
    raw = raw[["PlayId", "GameId", "Season"]].drop_duplicates("PlayId")

    play_ids = labels["PlayId"].astype(str)
    is_augmented = play_ids.str.endswith("_aug")
    if data_cfg.get("augmentation", "none") != "none":
        raise ValueError("The confirmatory study requires augmentation='none'.")
    keep = np.flatnonzero(~is_augmented.to_numpy())
    labels = labels.iloc[keep].reset_index(drop=True)
    base_ids = labels["PlayId"].astype(str).str.replace("_aug", "", regex=False)
    metadata = pd.DataFrame({"play_id": base_ids, "source_row": keep})
    metadata = metadata.merge(
        raw.rename(columns={"PlayId": "play_id", "GameId": "game_id", "Season": "season"}),
        on="play_id",
        how="left",
        validate="one_to_one",
    )
    if metadata[["game_id", "season"]].isna().any().any():
        raise ValueError("Some processed plays do not map to a raw GameId and Season.")
    metadata["season"] = metadata["season"].astype(int)
    metadata["game_id"] = metadata["game_id"].astype(str)
    metadata["play_id"] = metadata["play_id"].astype(str)
    if metadata["play_id"].str.endswith("_aug").any():
        raise AssertionError("Augmented play survived the confirmatory data filter.")

    y = labels["YardIndexClipped"].to_numpy(dtype=np.int64) - 71
    if np.any((y < 0) | (y >= 80)):
        raise ValueError("YardIndexClipped must map to the frozen 80-class support.")
    spatial_kept = np.asarray(spatial[keep], dtype=np.float32)
    set_kept = np.asarray(player_set[keep], dtype=np.float32)
    return StudyData(
        spatial=spatial_kept,
        player_set=set_kept,
        tabular=make_tabular_features(spatial_kept),
        y=y,
        metadata=metadata,
    )
