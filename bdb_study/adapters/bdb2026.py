"""BDB 2026 variable-horizon trajectory adapter.

The competition predicts only rows whose input record has
``player_to_predict == True``.  Every input-time player record is retained as
context.  Absolute official future coordinates and a deterministic
constant-velocity path are stored separately with explicit padding masks. The
development-only decoder selection freezes whether a model learns residual or
absolute coordinates; every candidate is reconstructed in absolute field
coordinates for scoring.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_PLAYER_CHANNELS,
    CohortAudit,
    PreparedTask,
    load_adapter_task_spec,
)


TASK_ID = "bdb2026_trajectory"
INPUT_REQUIRED = {
    "game_id",
    "play_id",
    "player_to_predict",
    "nfl_id",
    "frame_id",
    "play_direction",
    "absolute_yardline_number",
    "player_height",
    "player_weight",
    "player_position",
    "player_side",
    "player_role",
    "x",
    "y",
    "s",
    "a",
    "dir",
    "o",
    "num_frames_output",
    "ball_land_x",
    "ball_land_y",
}
OUTPUT_REQUIRED = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"}
KEY_COLUMNS = ["game_id", "play_id", "nfl_id"]
FIELD_WIDTH = 160.0 / 3.0
TOKEN_MEMMAP_THRESHOLD_BYTES = 512 * 1024 * 1024
_POSITION_MEMBERS = {
    "qb": {"QB"},
    "rb": {"RB", "FB", "HB"},
    "wr": {"WR"},
    "te": {"TE"},
    "ol": {"C", "G", "OG", "OT", "T", "OL"},
    "dl": {"DE", "DT", "NT", "DL"},
    "lb": {"ILB", "MLB", "OLB", "LB"},
    "db": {"CB", "DB", "FS", "SS", "S"},
    "special": {"K", "P", "LS"},
}
TABULAR_FEATURE_NAMES = (
    "x",
    "y",
    "speed",
    "acceleration",
    "vx",
    "vy",
    "dir_sin",
    "dir_cos",
    "orientation_sin",
    "orientation_cos",
    "absolute_yardline_number",
    "player_weight",
    "player_height_inches",
    "ball_land_x_rel",
    "ball_land_y_rel",
    "num_frames_output",
    "player_position",
    "player_side",
    "player_role",
)


def task_spec():
    return load_adapter_task_spec(TASK_ID)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _prediction_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _week_from_path(path: str | Path) -> int:
    match = re.search(r"[_-]w(?:eek)?[_-]?(\d{1,2})(?:\D|$)", Path(path).stem, re.I)
    if match is None:
        raise ValueError(f"cannot infer BDB2026 week from {path}")
    return int(match.group(1))


def _read_official_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    """Read only fields that cross the frozen adapter boundary.

    The official input contains two high-cardinality identity columns
    (``player_name`` and ``player_birth_date``) that are neither allowed model
    features nor needed for alignment.  Avoiding them materially lowers the
    peak memory of the full 4.9-million-row load without changing any prepared
    value.
    """

    source = Path(path)
    available = set(pd.read_csv(source, nrows=0).columns)
    missing = required.difference(available)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    return pd.read_csv(source, usecols=lambda column: column in required)


def load_week_files(
    input_files: Sequence[str | Path], output_files: Sequence[str | Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and week-annotate paired official train input/output files."""

    inputs_by_week = {_week_from_path(path): Path(path) for path in input_files}
    outputs_by_week = {_week_from_path(path): Path(path) for path in output_files}
    if inputs_by_week.keys() != outputs_by_week.keys():
        raise ValueError(
            "BDB2026 input/output week sets differ: "
            f"input={sorted(inputs_by_week)}, output={sorted(outputs_by_week)}"
        )
    input_parts: list[pd.DataFrame] = []
    output_parts: list[pd.DataFrame] = []
    for week in sorted(inputs_by_week):
        input_part = _read_official_csv(inputs_by_week[week], INPUT_REQUIRED)
        output_part = _read_official_csv(outputs_by_week[week], OUTPUT_REQUIRED)
        input_part["week"] = week
        output_part["week"] = week
        input_parts.append(input_part)
        output_parts.append(output_part)
    return (
        pd.concat(input_parts, ignore_index=True),
        pd.concat(output_parts, ignore_index=True),
    )


def _train_root(raw_dir: str | Path) -> Path:
    root = Path(raw_dir)
    candidates = [root / "train", root]
    train_root = next(
        (
            candidate
            for candidate in candidates
            if list(candidate.glob("input_2023_w*.csv"))
        ),
        None,
    )
    if train_root is None:
        raise FileNotFoundError(f"no input_2023_w*.csv files below {root}")
    return train_root


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all official BDB2026 training weeks below ``raw_dir``."""

    train_root = _train_root(raw_dir)
    return load_week_files(
        sorted(train_root.glob("input_2023_w*.csv")),
        sorted(train_root.glob("output_2023_w*.csv")),
    )


def load_selected_raw(
    raw_dir: str | Path,
    *,
    game_ids: Iterable[int] | None,
    max_examples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream only selected games/plays for smoke and split-scoped preparation."""

    if game_ids is None and max_examples is None:
        return load_raw(raw_dir)
    if max_examples is not None and int(max_examples) <= 0:
        raise ValueError("max_examples must be positive")
    selected_games = None if game_ids is None else {int(value) for value in game_ids}
    remaining = None if max_examples is None else int(max_examples)
    root = _train_root(raw_dir)
    input_parts: list[pd.DataFrame] = []
    output_parts: list[pd.DataFrame] = []
    for input_path in sorted(root.glob("input_2023_w*.csv"), key=_week_from_path):
        week = _week_from_path(input_path)
        inputs = _read_official_csv(input_path, INPUT_REQUIRED)
        if selected_games is not None:
            inputs = inputs.loc[inputs["game_id"].astype(int).isin(selected_games)]
        if inputs.empty:
            continue
        requested = (
            inputs.loc[_prediction_mask(inputs["player_to_predict"]), KEY_COLUMNS]
            .drop_duplicates()
            .sort_values(KEY_COLUMNS, kind="stable")
        )
        if remaining is not None:
            requested = requested.head(remaining)
        if requested.empty:
            continue
        selected_plays = requested[["game_id", "play_id"]].drop_duplicates()
        inputs = inputs.merge(selected_plays, on=["game_id", "play_id"], how="inner")
        outputs = _read_official_csv(
            root / f"output_2023_w{week:02d}.csv", OUTPUT_REQUIRED
        )
        outputs = outputs.merge(selected_plays, on=["game_id", "play_id"], how="inner")
        inputs["week"] = week
        outputs["week"] = week
        input_parts.append(inputs)
        output_parts.append(outputs)
        if remaining is not None:
            retained_keys = inputs.loc[
                _prediction_mask(inputs["player_to_predict"]), KEY_COLUMNS
            ].drop_duplicates()
            remaining -= min(remaining, len(retained_keys))
            if remaining <= 0:
                break
    if not input_parts or not output_parts:
        raise ValueError("BDB2026 selection contains no aligned input/output rows")
    return pd.concat(input_parts, ignore_index=True), pd.concat(output_parts, ignore_index=True)


def _standard_geometry(
    frame: pd.DataFrame, *, copy: bool = True
) -> pd.DataFrame:
    """Normalize plays left-to-right using an x reflection.

    ``dir``/``o`` in NGS data use 0 degrees toward the top sideline and grow
    clockwise.  The clean angles use 0 along +x and grow counter-clockwise.
    """

    result = frame.copy() if copy else frame
    direction = (90.0 - pd.to_numeric(result["dir"], errors="coerce")) % 360.0
    orientation = (90.0 - pd.to_numeric(result["o"], errors="coerce")) % 360.0
    left = result["play_direction"].astype(str).str.lower().eq("left")
    result["x_clean"] = np.where(left, 120.0 - result["x"], result["x"])
    result["y_clean"] = pd.to_numeric(result["y"], errors="coerce")
    result["dir_clean"] = np.where(left, (180.0 - direction) % 360.0, direction)
    result["o_clean"] = np.where(left, (180.0 - orientation) % 360.0, orientation)
    result["vx"] = result["s"] * np.cos(np.radians(result["dir_clean"]))
    result["vy"] = result["s"] * np.sin(np.radians(result["dir_clean"]))
    return result


def _canonical_output(output: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    merged = output.merge(directions, on=["game_id", "play_id"], how="left", validate="many_to_one")
    if merged["play_direction"].isna().any():
        raise ValueError("output contains a play absent from the input data")
    left = merged["play_direction"].astype(str).str.lower().eq("left")
    merged["x_clean"] = np.where(left, 120.0 - merged["x"], merged["x"])
    merged["y_clean"] = merged["y"].astype(float)
    return merged


def _position_channels(position: object) -> tuple[float, ...]:
    value = str(position).upper()
    flags = [float(value in _POSITION_MEMBERS[name]) for name in _POSITION_MEMBERS]
    flags.append(float(not any(flags)))
    return tuple(flags)


def _token_row(row: pd.Series, focal: pd.Series, focal_id: int) -> np.ndarray:
    angle = np.radians(float(row["dir_clean"]))
    orientation = np.radians(float(row["o_clean"]))
    side = str(row["player_side"]).lower()
    return np.asarray(
        [
            float(row["x_clean"] - focal["x_clean"]),
            float(row["y_clean"] - focal["y_clean"]),
            float(row["vx"]),
            float(row["vy"]),
            float(row["s"]),
            float(row["a"]),
            float(np.sin(angle)),
            float(np.cos(angle)),
            float(np.sin(orientation)),
            float(np.cos(orientation)),
            float(side == "offense"),
            float(side == "defense"),
            0.0,
            float(int(row["nfl_id"]) == int(focal_id)),
            *_position_channels(row["player_position"]),
        ],
        dtype=np.float32,
    )


def _height_inches(value: object) -> float:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(value))
    if match is None:
        return np.nan
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def _constant_velocity_path(final_state: pd.Series, max_horizon: int) -> np.ndarray:
    steps = np.arange(1, max_horizon + 1, dtype=np.float32)[:, None] / 10.0
    origin = np.asarray([final_state["x_clean"], final_state["y_clean"]], dtype=np.float32)
    velocity = np.asarray([final_state["vx"], final_state["vy"]], dtype=np.float32)
    return origin[None, :] + steps * velocity[None, :]


def add_residual_predictions(baseline: np.ndarray, residual: np.ndarray) -> np.ndarray:
    base = np.asarray(baseline, dtype=np.float64)
    offset = np.asarray(residual, dtype=np.float64)
    if base.shape != offset.shape or base.ndim != 3 or base.shape[-1] != 2:
        raise ValueError("baseline and residual must share shape [examples, horizon, 2]")
    return base + offset


def pooled_coordinate_rmse(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    """Official RMSE pooled across every required x/y output coordinate."""

    actual = np.asarray(truth, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if actual.shape != forecast.shape or actual.ndim != 3 or actual.shape[-1] != 2:
        raise ValueError("trajectory arrays must share shape [examples, horizon, 2]")
    if valid.shape != actual.shape[:2] or not np.any(valid):
        raise ValueError("trajectory mask is empty or has the wrong shape")
    if not np.all(np.isfinite(actual[valid])) or not np.all(np.isfinite(forecast[valid])):
        raise ValueError("valid trajectory coordinates must be finite")
    return float(np.sqrt(np.square(actual[valid] - forecast[valid]).mean()))


def game_equal_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    game_ids: Iterable[int],
) -> float:
    """Average the independently pooled RMSE of each game."""

    games = np.asarray(list(game_ids))
    if len(games) != len(truth):
        raise ValueError("game IDs and trajectory examples are misaligned")
    unique = np.unique(games)
    if len(unique) == 0:
        raise ValueError("no games supplied")
    return float(
        np.mean(
            [
                pooled_coordinate_rmse(
                    np.asarray(truth)[games == game],
                    np.asarray(prediction)[games == game],
                    np.asarray(mask)[games == game],
                )
                for game in unique
            ]
        )
    )


def horizon_rmse(
    truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return pooled coordinate RMSE for each forecast horizon."""

    actual = np.asarray(truth, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if actual.shape != forecast.shape or valid.shape != actual.shape[:2]:
        raise ValueError("trajectory arrays and mask are incompatible")
    result = np.full(actual.shape[1], np.nan, dtype=np.float64)
    for horizon in range(actual.shape[1]):
        keep = valid[:, horizon]
        if np.any(keep):
            result[horizon] = float(
                np.sqrt(np.square(actual[keep, horizon] - forecast[keep, horizon]).mean())
            )
    return result


def bdb2026_prepare_resource_estimate(
    *,
    examples: int,
    input_frames: int,
    players: int,
    output_horizon: int,
    channels: int = len(CANONICAL_PLAYER_CHANNELS),
) -> dict[str, int]:
    """Return exact dense artifact bytes and the resident non-token payload.

    The token artifact is intentionally dense because that shape is frozen in
    protocol v2.  Full preparation maps that one array to anonymous scratch
    storage, so it does not also need to reside in RAM.
    """

    dimensions = (examples, input_frames, players, output_horizon, channels)
    if any(int(value) <= 0 for value in dimensions):
        raise ValueError("BDB2026 resource dimensions must be positive")
    n = int(examples)
    time = int(input_frames)
    objects = int(players)
    horizon = int(output_horizon)
    feature_count = int(channels)
    token_bytes = n * time * objects * feature_count * np.dtype(np.float32).itemsize
    player_mask_bytes = n * time * objects * np.dtype(bool).itemsize
    frame_mask_bytes = n * time * np.dtype(bool).itemsize
    target_value_bytes = n * horizon * 2 * np.dtype(np.float32).itemsize
    target_mask_bytes = n * horizon * np.dtype(bool).itemsize
    baseline_bytes = target_value_bytes
    resident_array_bytes = (
        player_mask_bytes
        + frame_mask_bytes
        + target_value_bytes
        + target_mask_bytes
        + baseline_bytes
    )
    return {
        "player_tokens_bytes": int(token_bytes),
        "player_mask_bytes": int(player_mask_bytes),
        "frame_mask_bytes": int(frame_mask_bytes),
        "target_values_bytes": int(target_value_bytes),
        "target_mask_bytes": int(target_mask_bytes),
        "target_baseline_bytes": int(baseline_bytes),
        "resident_array_bytes_excluding_mapped_tokens": int(resident_array_bytes),
        "prepared_array_bytes": int(token_bytes + resident_array_bytes),
    }


def _anonymous_token_memmap(
    shape: tuple[int, ...], *, scratch_dir: str | Path | None
) -> np.memmap:
    """Create a zeroed, process-lifetime mmap with no stale path to clean up."""

    if scratch_dir is None:
        configured = os.environ.get("BDB2026_PREPARE_SCRATCH_DIR")
        scratch = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "bdb_suite"
            / ".scratch"
        )
    else:
        scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    required = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    available = int(shutil.disk_usage(scratch).free)
    if available < required:
        raise OSError(
            "BDB2026 token scratch lacks capacity: "
            f"requires {required} bytes, has {available} bytes at {scratch}"
        )
    # TemporaryFile is unlinked immediately on Betty/Linux and remains valid
    # through the mmap after the descriptor is closed. SIGKILL therefore
    # cannot strand a multi-gigabyte scratch pathname.
    backing = tempfile.TemporaryFile(prefix="bdb2026-player-tokens-", dir=scratch)
    try:
        backing.truncate(required)
        result = np.memmap(
            backing,
            mode="r+",
            dtype=np.float32,
            shape=shape,
            order="C",
        )
    finally:
        backing.close()
    return result


def _allocate_token_tensor(
    shape: tuple[int, ...],
    *,
    scratch_dir: str | Path | None,
    memmap_threshold_bytes: int,
) -> np.ndarray:
    if int(memmap_threshold_bytes) < 0:
        raise ValueError("memmap_threshold_bytes cannot be negative")
    required = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if required > int(memmap_threshold_bytes):
        return _anonymous_token_memmap(shape, scratch_dir=scratch_dir)
    return np.zeros(shape, dtype=np.float32)


def _position_channel_matrix(position: pd.Series) -> np.ndarray:
    values = position.astype(str).str.upper().to_numpy(dtype=object)
    result = np.zeros((len(values), len(_POSITION_MEMBERS) + 1), dtype=np.float32)
    known = np.zeros(len(values), dtype=bool)
    for index, members in enumerate(_POSITION_MEMBERS.values()):
        selected = np.isin(values, tuple(members))
        result[:, index] = selected
        known |= selected
    result[:, -1] = ~known
    return result


def _prepare_bdb2026_frames_reference(
    input_tracking: pd.DataFrame,
    output_tracking: pd.DataFrame,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
    max_input_frames: int | None = None,
) -> PreparedTask:
    """Slow row-wise oracle retained only for exact adapter equivalence tests."""

    _require_columns(input_tracking, INPUT_REQUIRED, "BDB2026 input")
    _require_columns(output_tracking, OUTPUT_REQUIRED, "BDB2026 output")
    if input_tracking.empty or output_tracking.empty:
        raise ValueError("BDB2026 input and output must be non-empty")
    if max_input_frames is not None and max_input_frames < 1:
        raise ValueError("max_input_frames must be positive when supplied")

    inputs = input_tracking.copy()
    outputs = output_tracking.copy()
    if game_ids is not None:
        selected_games = {int(value) for value in game_ids}
        inputs = inputs.loc[inputs["game_id"].astype(int).isin(selected_games)].copy()
        outputs = outputs.loc[outputs["game_id"].astype(int).isin(selected_games)].copy()
    if inputs.empty or outputs.empty:
        raise ValueError("BDB2026 selection contains no aligned input/output rows")
    if "week" not in inputs:
        inputs["week"] = 0
    if "week" not in outputs:
        outputs["week"] = 0
    if inputs.duplicated([*KEY_COLUMNS, "frame_id"]).any():
        raise ValueError("BDB2026 input has duplicate player-frame rows")
    if outputs.duplicated([*KEY_COLUMNS, "frame_id"]).any():
        raise ValueError("BDB2026 output has duplicate player-frame rows")

    directions = inputs[["game_id", "play_id", "play_direction"]].drop_duplicates()
    if directions.duplicated(["game_id", "play_id"]).any():
        raise ValueError("play_direction changes within a BDB2026 play")
    inputs = _standard_geometry(inputs)
    outputs = _canonical_output(outputs, directions)

    examples = (
        inputs.loc[_prediction_mask(inputs["player_to_predict"]), KEY_COLUMNS]
        .drop_duplicates()
        .sort_values(KEY_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )
    output_keys = outputs[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not examples.equals(output_keys):
        missing_output = examples.merge(output_keys, on=KEY_COLUMNS, how="left", indicator=True)
        missing_input = output_keys.merge(examples, on=KEY_COLUMNS, how="left", indicator=True)
        raise ValueError(
            "player_to_predict keys do not exactly match output keys "
            f"(missing output={(missing_output._merge == 'left_only').sum()}, "
            f"unexpected output={(missing_input._merge == 'left_only').sum()})"
        )
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError("max_examples must be positive")
        examples = examples.head(int(max_examples)).copy()
        selected_targets = examples.assign(_selected=True)
        outputs = outputs.merge(selected_targets, on=KEY_COLUMNS, how="inner").drop(columns="_selected")
        selected_plays = examples[["game_id", "play_id"]].drop_duplicates().assign(_selected=True)
        inputs = inputs.merge(selected_plays, on=["game_id", "play_id"], how="inner").drop(columns="_selected")

    target_rows: list[dict[str, object]] = []
    tabular_rows: list[dict[str, float]] = []
    play_frames: list[pd.DataFrame] = []
    focal_ids: list[int] = []
    final_states: list[pd.Series] = []
    output_groups: list[pd.DataFrame] = []
    max_frames_observed = 0
    max_players_observed = 0
    max_horizon = int(outputs["frame_id"].max())

    input_play_groups = {
        (int(game), int(play)): group.sort_values(["frame_id", "nfl_id"], kind="stable")
        for (game, play), group in inputs.groupby(["game_id", "play_id"], sort=False)
    }
    output_groups_by_key = {
        (int(game), int(play), int(player)): group.sort_values("frame_id", kind="stable")
        for (game, play, player), group in outputs.groupby(KEY_COLUMNS, sort=False)
    }
    for key in examples.itertuples(index=False, name=None):
        game_id, play_id, nfl_id = (int(key[0]), int(key[1]), int(key[2]))
        play = input_play_groups[(game_id, play_id)]
        frames = np.sort(play["frame_id"].unique())
        if max_input_frames is not None:
            frames = frames[-max_input_frames:]
        play = play.loc[play["frame_id"].isin(frames)].copy()
        focal_history = play.loc[play["nfl_id"].eq(nfl_id)].sort_values("frame_id")
        if len(focal_history) != len(frames):
            raise ValueError(f"focal player {key} is absent from one or more input frames")
        final = focal_history.iloc[-1]
        expected_horizon = int(final["num_frames_output"])
        target = output_groups_by_key[(game_id, play_id, nfl_id)]
        target_frames = target["frame_id"].astype(int).to_numpy()
        if (
            len(target_frames) != expected_horizon
            or not np.array_equal(target_frames, np.arange(1, expected_horizon + 1))
        ):
            raise ValueError(
                f"output frames for {key} do not match num_frames_output={expected_horizon}"
            )
        week_values = play["week"].dropna().astype(int).unique()
        if len(week_values) != 1:
            raise ValueError(f"week is not unique for play {(game_id, play_id)}")
        target_rows.append(
            {
                "example_id": f"{game_id}:{play_id}:{nfl_id}",
                "game_id": game_id,
                "stratum": f"2023-w{int(week_values[0]):02d}",
                "target": np.nan,
                "play_id": play_id,
                "nfl_id": nfl_id,
                "horizon": expected_horizon,
            }
        )
        angle = np.radians(float(final["dir_clean"]))
        orientation = np.radians(float(final["o_clean"]))
        ball_x = 120.0 - float(final["ball_land_x"]) if str(final["play_direction"]).lower() == "left" else float(final["ball_land_x"])
        tabular_rows.append(
            {
                "x": float(final["x_clean"]),
                "y": float(final["y_clean"]),
                "speed": float(final["s"]),
                "acceleration": float(final["a"]),
                "vx": float(final["vx"]),
                "vy": float(final["vy"]),
                "dir_sin": float(np.sin(angle)),
                "dir_cos": float(np.cos(angle)),
                "orientation_sin": float(np.sin(orientation)),
                "orientation_cos": float(np.cos(orientation)),
                "absolute_yardline_number": float(final["absolute_yardline_number"]),
                "player_weight": float(final["player_weight"]),
                "player_height_inches": _height_inches(final["player_height"]),
                "ball_land_x_rel": ball_x - float(final["x_clean"]),
                "ball_land_y_rel": float(final["ball_land_y"]) - float(final["y_clean"]),
                "num_frames_output": float(expected_horizon),
                "player_position": str(final["player_position"]),
                "player_side": str(final["player_side"]),
                "player_role": str(final["player_role"]),
            }
        )
        play_frames.append(play)
        focal_ids.append(nfl_id)
        final_states.append(final)
        output_groups.append(target)
        max_frames_observed = max(max_frames_observed, len(frames))
        max_players_observed = max(
            max_players_observed, int(play["nfl_id"].nunique())
        )

    n_examples = len(target_rows)
    tokens = np.zeros(
        (n_examples, max_frames_observed, max_players_observed, len(CANONICAL_PLAYER_CHANNELS)),
        dtype=np.float32,
    )
    player_mask = np.zeros(tokens.shape[:3], dtype=bool)
    frame_mask = np.zeros(tokens.shape[:2], dtype=bool)
    targets = np.full((n_examples, max_horizon, 2), np.nan, dtype=np.float32)
    target_mask = np.zeros((n_examples, max_horizon), dtype=bool)
    baselines = np.zeros((n_examples, max_horizon, 2), dtype=np.float32)

    slot_by_play: dict[tuple[int, int], dict[int, int]] = {}
    for play_key, play in input_play_groups.items():
        alignment = (
            play[["nfl_id", "player_side"]]
            .drop_duplicates("nfl_id")
            .assign(
                _offense=lambda value: value["player_side"]
                .astype(str)
                .str.lower()
                .eq("offense")
            )
            .sort_values(
                ["_offense", "nfl_id"],
                ascending=[False, True],
                kind="stable",
            )
        )
        slot_by_play[play_key] = {
            int(nfl_id): slot
            for slot, nfl_id in enumerate(alignment["nfl_id"].astype(int))
        }

    for example_index, (example, play, focal_id, final, target) in enumerate(
        zip(target_rows, play_frames, focal_ids, final_states, output_groups)
    ):
        frames = np.sort(play["frame_id"].unique())
        frame_offset = max_frames_observed - len(frames)
        play_key = (int(example["game_id"]), int(example["play_id"]))
        slot_by_id = slot_by_play[play_key]
        for local_frame, frame_id in enumerate(frames):
            output_frame = frame_offset + local_frame
            state = play.loc[play["frame_id"].eq(frame_id)].sort_values(
                "nfl_id", kind="stable"
            ).copy()
            focal = state.loc[state["nfl_id"].eq(focal_id)]
            if len(focal) != 1:
                raise ValueError("focal player is not unique in an input frame")
            focal_row = focal.iloc[0]
            for _, row in state.iterrows():
                player_index = slot_by_id[int(row["nfl_id"])]
                tokens[example_index, output_frame, player_index] = _token_row(
                    row, focal_row, focal_id
                )
                player_mask[example_index, output_frame, player_index] = True
            frame_mask[example_index, output_frame] = True
        baseline = _constant_velocity_path(final, max_horizon)
        baselines[example_index] = baseline
        indices = target["frame_id"].astype(int).to_numpy() - 1
        targets[example_index, indices, 0] = target["x_clean"].to_numpy(dtype=np.float32)
        targets[example_index, indices, 1] = target["y_clean"].to_numpy(dtype=np.float32)
        target_mask[example_index, indices] = True

    examples_frame = pd.DataFrame(target_rows)
    audit = {
        "input_rows": int(len(inputs)),
        "output_rows": int(len(outputs)),
        "eligible_examples": int(n_examples),
        "eligible_games": int(examples_frame["game_id"].nunique()),
        "eligible_plays": int(examples_frame[["game_id", "play_id"]].drop_duplicates().shape[0]),
        "max_input_frames": int(max_frames_observed),
        "max_players": int(max_players_observed),
        "max_output_horizon": int(max_horizon),
        "target_definition": "official player_to_predict rows",
        "coordinate_normalization": "left plays reflected in x; Euclidean error invariant",
        "stored_target": "absolute official future x/y",
        "training_target": (
            "development_selected_residual_or_absolute_from_stored_"
            "absolute_xy_and_baseline"
        ),
    }
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="trajectory",
        primary_metric="rmse",
        support=None,
        examples=examples_frame,
        tabular=pd.DataFrame(tabular_rows, columns=TABULAR_FEATURE_NAMES),
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CANONICAL_PLAYER_CHANNELS,
        audit=audit,
        target_values=targets,
        target_mask=target_mask,
        target_baseline=baselines,
        metadata={
            "tabular_feature_order": list(TABULAR_FEATURE_NAMES),
            "input_scope": "all official input frames and tracked players",
            "global_context_scope": "official focal-player and play fields in tabular",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nfl_id alignment only and never encoded"
            ),
            "graph": (
                "focal-player social graph with horizon-conditioned "
                "development-selected decoder"
            ),
            "output_head": "shared_horizon_conditioned_development_selected_v1",
            "structural_ablation": "remove_social_player_messages",
            "trajectory_tube_domain": {
                "field_rectangle_x": [0.0, 120.0],
                "field_rectangle_y": [0.0, FIELD_WIDTH],
                "reported_region": "radial_conformal_tube_intersected_with_field_rectangle",
            },
        },
    )


def prepare_bdb2026_frames(
    input_tracking: pd.DataFrame,
    output_tracking: pd.DataFrame,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
    max_input_frames: int | None = None,
    scratch_dir: str | Path | None = None,
    memmap_threshold_bytes: int = TOKEN_MEMMAP_THRESHOLD_BYTES,
    _reuse_input_memory: bool = False,
) -> PreparedTask:
    """Build BDB2026 once per play with vectorized focal-relative expansion.

    Slot allocation and common player features are computed once for a play.
    Only the two focal-relative coordinates and focal indicator vary across
    requested-player examples.  The frozen dense output is written directly
    into anonymous mapped storage when large, bounding resident memory without
    altering its shape, dtype, ordering, masks, or scientific meaning.
    """

    _require_columns(input_tracking, INPUT_REQUIRED, "BDB2026 input")
    _require_columns(output_tracking, OUTPUT_REQUIRED, "BDB2026 output")
    if input_tracking.empty or output_tracking.empty:
        raise ValueError("BDB2026 input and output must be non-empty")
    if max_input_frames is not None and max_input_frames < 1:
        raise ValueError("max_input_frames must be positive when supplied")

    inputs = input_tracking if _reuse_input_memory else input_tracking.copy()
    outputs = output_tracking.copy()
    if game_ids is not None:
        selected_games = {int(value) for value in game_ids}
        inputs = inputs.loc[inputs["game_id"].astype(int).isin(selected_games)].copy()
        outputs = outputs.loc[outputs["game_id"].astype(int).isin(selected_games)].copy()
    if inputs.empty or outputs.empty:
        raise ValueError("BDB2026 selection contains no aligned input/output rows")
    if "week" not in inputs:
        inputs["week"] = 0
    if "week" not in outputs:
        outputs["week"] = 0
    if inputs.duplicated([*KEY_COLUMNS, "frame_id"]).any():
        raise ValueError("BDB2026 input has duplicate player-frame rows")
    if outputs.duplicated([*KEY_COLUMNS, "frame_id"]).any():
        raise ValueError("BDB2026 output has duplicate player-frame rows")

    directions = inputs[["game_id", "play_id", "play_direction"]].drop_duplicates()
    if directions.duplicated(["game_id", "play_id"]).any():
        raise ValueError("play_direction changes within a BDB2026 play")
    inputs = _standard_geometry(inputs, copy=False)
    outputs = _canonical_output(outputs, directions)

    examples = (
        inputs.loc[_prediction_mask(inputs["player_to_predict"]), KEY_COLUMNS]
        .drop_duplicates()
        .sort_values(KEY_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )
    output_keys = (
        outputs[KEY_COLUMNS]
        .drop_duplicates()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    if not examples.equals(output_keys):
        missing_output = examples.merge(
            output_keys, on=KEY_COLUMNS, how="left", indicator=True
        )
        missing_input = output_keys.merge(
            examples, on=KEY_COLUMNS, how="left", indicator=True
        )
        raise ValueError(
            "player_to_predict keys do not exactly match output keys "
            f"(missing output={(missing_output._merge == 'left_only').sum()}, "
            f"unexpected output={(missing_input._merge == 'left_only').sum()})"
        )
    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError("max_examples must be positive")
        examples = examples.head(int(max_examples)).copy()
        selected_targets = examples.assign(_selected=True)
        outputs = outputs.merge(
            selected_targets, on=KEY_COLUMNS, how="inner"
        ).drop(columns="_selected")
        selected_plays = examples[["game_id", "play_id"]].drop_duplicates().assign(
            _selected=True
        )
        inputs = inputs.merge(
            selected_plays, on=["game_id", "play_id"], how="inner"
        ).drop(columns="_selected")
    if examples.empty:
        raise ValueError("BDB2026 selection contains no player_to_predict examples")

    inputs.sort_values(
        ["game_id", "play_id", "frame_id", "nfl_id"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    frame_counts = inputs.groupby(["game_id", "play_id"], sort=False)[
        "frame_id"
    ].nunique()
    if max_input_frames is None:
        max_frames_observed = int(frame_counts.max())
        max_players_observed = int(
            inputs.groupby(["game_id", "play_id"], sort=False)["nfl_id"]
            .nunique()
            .max()
        )
    else:
        max_frames_observed = int(
            np.minimum(frame_counts.to_numpy(dtype=int), int(max_input_frames)).max()
        )
        retained_frame_keys = (
            inputs[["game_id", "play_id", "frame_id"]]
            .drop_duplicates()
            .groupby(["game_id", "play_id"], sort=False, group_keys=False)
            .tail(int(max_input_frames))
        )
        retained_for_size = inputs.merge(
            retained_frame_keys,
            on=["game_id", "play_id", "frame_id"],
            how="inner",
        )
        max_players_observed = int(
            retained_for_size.groupby(["game_id", "play_id"], sort=False)["nfl_id"]
            .nunique()
            .max()
        )
        del retained_for_size, retained_frame_keys
    max_horizon = int(outputs["frame_id"].max())
    n_examples = int(len(examples))
    token_shape = (
        n_examples,
        max_frames_observed,
        max_players_observed,
        len(CANONICAL_PLAYER_CHANNELS),
    )
    tokens = _allocate_token_tensor(
        token_shape,
        scratch_dir=scratch_dir,
        memmap_threshold_bytes=memmap_threshold_bytes,
    )
    player_mask = np.zeros(token_shape[:3], dtype=bool)
    frame_mask = np.zeros(token_shape[:2], dtype=bool)
    targets = np.full((n_examples, max_horizon, 2), np.nan, dtype=np.float32)
    target_mask = np.zeros((n_examples, max_horizon), dtype=bool)
    baselines = np.zeros((n_examples, max_horizon, 2), dtype=np.float32)

    example_play_indices = {
        (int(game), int(play)): np.asarray(indices, dtype=np.int64)
        for (game, play), indices in examples.groupby(
            ["game_id", "play_id"], sort=False
        ).indices.items()
    }
    target_rows: list[dict[str, object] | None] = [None] * n_examples
    tabular_rows: list[dict[str, object] | None] = [None] * n_examples
    expected_horizons = np.zeros(n_examples, dtype=np.int64)
    steps = (
        np.arange(1, max_horizon + 1, dtype=np.float32)[None, :, None]
        / np.float32(10.0)
    )
    channel_count = len(CANONICAL_PLAYER_CHANNELS)

    for (game_value, play_value), full_play in inputs.groupby(
        ["game_id", "play_id"], sort=False
    ):
        play_key = (int(game_value), int(play_value))
        example_indices = example_play_indices.get(play_key)
        if example_indices is None:
            continue
        frames = np.sort(full_play["frame_id"].unique())
        if max_input_frames is not None:
            frames = frames[-int(max_input_frames) :]
        play = full_play.loc[full_play["frame_id"].isin(frames)]
        frame_offset = max_frames_observed - len(frames)

        alignment = (
            full_play[["nfl_id", "player_side"]]
            .drop_duplicates("nfl_id")
            .assign(
                _offense=lambda value: value["player_side"]
                .astype(str)
                .str.lower()
                .eq("offense")
            )
            .sort_values(
                ["_offense", "nfl_id"],
                ascending=[False, True],
                kind="stable",
            )
        )
        slot_by_id = {
            int(nfl_id): slot
            for slot, nfl_id in enumerate(alignment["nfl_id"].astype(int))
        }
        row_nfl_ids = play["nfl_id"].to_numpy(dtype=np.int64)
        row_frame_ids = play["frame_id"].to_numpy()
        frame_codes = np.searchsorted(frames, row_frame_ids)
        output_frame_codes = frame_offset + frame_codes
        slot_codes = np.fromiter(
            (slot_by_id[int(value)] for value in row_nfl_ids),
            dtype=np.int64,
            count=len(row_nfl_ids),
        )
        if len(slot_codes) and int(slot_codes.max()) >= max_players_observed:
            raise ValueError(
                "retained input players exceed the allocated play-global slot range"
            )

        row_x = play["x_clean"].to_numpy(dtype=np.float64)
        row_y = play["y_clean"].to_numpy(dtype=np.float64)
        common = np.zeros((len(play), channel_count), dtype=np.float32)
        common[:, 2] = play["vx"].to_numpy(dtype=np.float64)
        common[:, 3] = play["vy"].to_numpy(dtype=np.float64)
        common[:, 4] = play["s"].to_numpy(dtype=np.float64)
        common[:, 5] = play["a"].to_numpy(dtype=np.float64)
        direction_radians = np.radians(
            play["dir_clean"].to_numpy(dtype=np.float64)
        )
        orientation_radians = np.radians(
            play["o_clean"].to_numpy(dtype=np.float64)
        )
        common[:, 6] = np.sin(direction_radians)
        common[:, 7] = np.cos(direction_radians)
        common[:, 8] = np.sin(orientation_radians)
        common[:, 9] = np.cos(orientation_radians)
        sides = play["player_side"].astype(str).str.lower()
        common[:, 10] = sides.eq("offense").to_numpy()
        common[:, 11] = sides.eq("defense").to_numpy()
        common[:, 14:] = _position_channel_matrix(play["player_position"])

        focal_ids = examples.loc[example_indices, "nfl_id"].to_numpy(dtype=np.int64)
        focal_x = np.empty((len(focal_ids), len(frames)), dtype=np.float64)
        focal_y = np.empty_like(focal_x)
        final_states: list[pd.Series] = []
        for local_index, focal_id in enumerate(focal_ids):
            focal_history = play.loc[play["nfl_id"].eq(int(focal_id))].sort_values(
                "frame_id", kind="stable"
            )
            if len(focal_history) != len(frames):
                key = (play_key[0], play_key[1], int(focal_id))
                raise ValueError(
                    f"focal player {key} is absent from one or more input frames"
                )
            focal_x[local_index] = focal_history["x_clean"].to_numpy(
                dtype=np.float64
            )
            focal_y[local_index] = focal_history["y_clean"].to_numpy(
                dtype=np.float64
            )
            final_states.append(focal_history.iloc[-1])

        expanded = np.broadcast_to(
            common[None, :, :], (len(focal_ids), len(play), channel_count)
        ).copy()
        expanded[:, :, 0] = row_x[None, :] - focal_x[:, frame_codes]
        expanded[:, :, 1] = row_y[None, :] - focal_y[:, frame_codes]
        expanded[:, :, 13] = row_nfl_ids[None, :] == focal_ids[:, None]
        tokens[
            example_indices[:, None],
            output_frame_codes[None, :],
            slot_codes[None, :],
            :,
        ] = expanded
        player_mask[
            example_indices[:, None],
            output_frame_codes[None, :],
            slot_codes[None, :],
        ] = True
        frame_mask[
            example_indices[:, None],
            (frame_offset + np.arange(len(frames), dtype=np.int64))[None, :],
        ] = True

        week_values = play["week"].dropna().astype(int).unique()
        if len(week_values) != 1:
            raise ValueError(f"week is not unique for play {play_key}")
        origins = np.asarray(
            [[state["x_clean"], state["y_clean"]] for state in final_states],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [[state["vx"], state["vy"]] for state in final_states],
            dtype=np.float32,
        )
        baselines[example_indices] = (
            origins[:, None, :] + steps * velocities[:, None, :]
        )

        for example_index, focal_id, final in zip(
            example_indices, focal_ids, final_states
        ):
            expected_horizon = int(final["num_frames_output"])
            expected_horizons[example_index] = expected_horizon
            target_rows[example_index] = {
                "example_id": f"{play_key[0]}:{play_key[1]}:{int(focal_id)}",
                "game_id": play_key[0],
                "stratum": f"2023-w{int(week_values[0]):02d}",
                "target": np.nan,
                "play_id": play_key[1],
                "nfl_id": int(focal_id),
                "horizon": expected_horizon,
            }
            angle = np.radians(float(final["dir_clean"]))
            orientation = np.radians(float(final["o_clean"]))
            ball_x = (
                120.0 - float(final["ball_land_x"])
                if str(final["play_direction"]).lower() == "left"
                else float(final["ball_land_x"])
            )
            tabular_rows[example_index] = {
                "x": float(final["x_clean"]),
                "y": float(final["y_clean"]),
                "speed": float(final["s"]),
                "acceleration": float(final["a"]),
                "vx": float(final["vx"]),
                "vy": float(final["vy"]),
                "dir_sin": float(np.sin(angle)),
                "dir_cos": float(np.cos(angle)),
                "orientation_sin": float(np.sin(orientation)),
                "orientation_cos": float(np.cos(orientation)),
                "absolute_yardline_number": float(final["absolute_yardline_number"]),
                "player_weight": float(final["player_weight"]),
                "player_height_inches": _height_inches(final["player_height"]),
                "ball_land_x_rel": ball_x - float(final["x_clean"]),
                "ball_land_y_rel": float(final["ball_land_y"])
                - float(final["y_clean"]),
                "num_frames_output": float(expected_horizon),
                "player_position": str(final["player_position"]),
                "player_side": str(final["player_side"]),
                "player_role": str(final["player_role"]),
            }

    if any(value is None for value in target_rows) or any(
        value is None for value in tabular_rows
    ):
        raise ValueError("one or more BDB2026 requested players lack an input play")

    indexed_examples = examples.copy()
    indexed_examples["_example_index"] = np.arange(n_examples, dtype=np.int64)
    aligned_output = outputs.merge(
        indexed_examples,
        on=KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    output_frame_indices = aligned_output["frame_id"].astype(int).to_numpy()
    aligned_output = aligned_output.assign(_frame_index=output_frame_indices)
    summary = (
        aligned_output.groupby("_example_index", sort=False)["_frame_index"]
        .agg(["count", "min", "max", "nunique"])
        .reindex(np.arange(n_examples))
    )
    valid_output = (
        summary["count"].to_numpy() == expected_horizons
    ) & (summary["min"].to_numpy() == 1) & (
        summary["max"].to_numpy() == expected_horizons
    ) & (
        summary["nunique"].to_numpy() == expected_horizons
    )
    if not np.all(valid_output):
        example_index = int(np.flatnonzero(~valid_output)[0])
        key = tuple(
            int(value) for value in examples.loc[example_index, KEY_COLUMNS].tolist()
        )
        raise ValueError(
            f"output frames for {key} do not match "
            f"num_frames_output={expected_horizons[example_index]}"
        )
    output_example_indices = aligned_output["_example_index"].to_numpy(dtype=np.int64)
    output_horizon_indices = aligned_output["_frame_index"].to_numpy(dtype=np.int64) - 1
    targets[output_example_indices, output_horizon_indices, 0] = aligned_output[
        "x_clean"
    ].to_numpy(dtype=np.float32)
    targets[output_example_indices, output_horizon_indices, 1] = aligned_output[
        "y_clean"
    ].to_numpy(dtype=np.float32)
    target_mask[output_example_indices, output_horizon_indices] = True
    if isinstance(tokens, np.memmap):
        tokens.flush()

    examples_frame = pd.DataFrame(target_rows)
    audit = {
        "input_rows": int(len(inputs)),
        "output_rows": int(len(outputs)),
        "eligible_examples": int(n_examples),
        "eligible_games": int(examples_frame["game_id"].nunique()),
        "eligible_plays": int(
            examples_frame[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
        "max_input_frames": int(max_frames_observed),
        "max_players": int(max_players_observed),
        "max_output_horizon": int(max_horizon),
        "target_definition": "official player_to_predict rows",
        "coordinate_normalization": (
            "left plays reflected in x; Euclidean error invariant"
        ),
        "stored_target": "absolute official future x/y",
        "training_target": (
            "development_selected_residual_or_absolute_from_stored_"
            "absolute_xy_and_baseline"
        ),
    }
    return PreparedTask(
        task_id=TASK_ID,
        outcome_type="trajectory",
        primary_metric="rmse",
        support=None,
        examples=examples_frame,
        tabular=pd.DataFrame(tabular_rows, columns=TABULAR_FEATURE_NAMES),
        player_tokens=tokens,
        player_mask=player_mask,
        frame_mask=frame_mask,
        channel_names=CANONICAL_PLAYER_CHANNELS,
        audit=audit,
        target_values=targets,
        target_mask=target_mask,
        target_baseline=baselines,
        metadata={
            "tabular_feature_order": list(TABULAR_FEATURE_NAMES),
            "input_scope": "all official input frames and tracked players",
            "global_context_scope": "official focal-player and play fields in tabular",
            "stable_player_slots": (
                "play_global_focal_independent_v1; nfl_id alignment only and never encoded"
            ),
            "graph": (
                "focal-player social graph with horizon-conditioned "
                "development-selected decoder"
            ),
            "output_head": "shared_horizon_conditioned_development_selected_v1",
            "structural_ablation": "remove_social_player_messages",
            "trajectory_tube_domain": {
                "field_rectangle_x": [0.0, 120.0],
                "field_rectangle_y": [0.0, FIELD_WIDTH],
                "reported_region": (
                    "radial_conformal_tube_intersected_with_field_rectangle"
                ),
            },
        },
    )


def audit_cohort(raw_dir: Path) -> CohortAudit:
    # Audit streams the small key projection rather than materializing all
    # ~4.9M rich input records in memory.
    root = _train_root(raw_dir)
    play_keys: list[pd.DataFrame] = []
    eligible_parts: list[pd.DataFrame] = []
    context_player_keys = 0
    output_rows = 0
    invalid_output_trajectories = 0
    for input_path in sorted(root.glob("input_2023_w*.csv")):
        week = _week_from_path(input_path)
        output_path = root / f"output_2023_w{week:02d}.csv"
        inputs = pd.read_csv(
            input_path,
            usecols=[*KEY_COLUMNS, "player_to_predict", "num_frames_output"],
        )
        outputs = pd.read_csv(output_path, usecols=[*KEY_COLUMNS, "frame_id"])
        play_keys.append(inputs[["game_id", "play_id"]].drop_duplicates())
        all_players = inputs[KEY_COLUMNS].drop_duplicates()
        eligible = inputs.loc[
            _prediction_mask(inputs["player_to_predict"]),
            [*KEY_COLUMNS, "num_frames_output"],
        ].drop_duplicates()
        eligible_parts.append(eligible[KEY_COLUMNS])
        context_player_keys += int(len(all_players) - len(eligible))
        summary = (
            outputs.groupby(KEY_COLUMNS)["frame_id"]
            .agg(["count", "min", "max", "nunique"])
            .reset_index()
        )
        aligned = eligible.merge(summary, on=KEY_COLUMNS, how="outer", indicator=True)
        valid = (
            aligned["_merge"].eq("both")
            & aligned["count"].eq(aligned["num_frames_output"])
            & aligned["min"].eq(1)
            & aligned["max"].eq(aligned["num_frames_output"])
            & aligned["nunique"].eq(aligned["num_frames_output"])
        )
        invalid_output_trajectories += int((~valid).sum())
        output_rows += int(len(outputs))
    plays = pd.concat(play_keys, ignore_index=True).drop_duplicates()
    eligible = pd.concat(eligible_parts, ignore_index=True).drop_duplicates()
    retained = len(eligible) - invalid_output_trajectories
    return CohortAudit(
        task_id=TASK_ID,
        total_play_rows=int(len(plays)),
        outcome_eligible=int(len(eligible)),
        cutoff_eligible=int(retained),
        retained_examples=int(retained),
        retained_games=int(plays["game_id"].nunique()),
        exclusions={
            "context_player_not_requested": int(context_player_keys),
            "invalid_or_missing_output_trajectory": int(invalid_output_trajectories),
        },
        details={
            "target": "official player_to_predict",
            "variable_horizon": True,
            "output_rows": int(output_rows),
            "expected_plays": 14_108,
            "expected_examples": 46_045,
            "expected_output_rows": 562_936,
        },
    )


def prepare_bdb2026(
    raw_dir: Path,
    *,
    game_ids: Iterable[int] | None = None,
    max_examples: int | None = None,
    max_input_frames: int | None = None,
) -> PreparedTask:
    inputs, outputs = load_selected_raw(
        raw_dir, game_ids=game_ids, max_examples=max_examples
    )
    return prepare_bdb2026_frames(
        inputs,
        outputs,
        game_ids=game_ids,
        max_examples=max_examples,
        max_input_frames=max_input_frames,
        _reuse_input_memory=True,
    )


prepare = prepare_bdb2026
