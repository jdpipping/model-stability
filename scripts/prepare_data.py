"""
Prepare data once: read from data/raw/, write to data/processed/.
Input: data/raw/train.csv. Outputs: data/processed/train_x.npy, train_x_set.npy, train_y.pkl, df_season.pkl.
"""

import argparse
import math
import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TRAIN_CSV = f"{RAW_DIR}/train.csv"
TRAIN_X_PATH = f"{OUT_DIR}/train_x.npy"
TRAIN_X_SET_PATH = f"{OUT_DIR}/train_x_set.npy"
TRAIN_Y_PATH = f"{OUT_DIR}/train_y.pkl"
DF_SEASON_PATH = f"{OUT_DIR}/df_season.pkl"
AUGMENT_SEED = 42

# CNN input: [n_plays, 11 def, 10 off, 10 channels]. Ch 0-3: off minus def (Sx,Sy,X,Y). Ch 4-9: def–rusher + def.


def split_play_and_player_cols(df, predicting=False):
    df["IsRusher"] = df["NflId"] == df["NflIdRusher"]
    df["PlayId"] = df["PlayId"].astype(str)
    player_cols = ["PlayId", "Season", "Team", "X", "Y", "S", "Dis", "Dir", "NflId", "IsRusher"]
    df_players = df[player_cols]
    play_cols = ["PlayId", "Season", "PossessionTeam", "HomeTeamAbbr", "PlayDirection"]
    if not predicting:
        play_cols.append("Yards")
    df_play = df[play_cols].copy().groupby("PlayId").first().reset_index()
    assert df_play.PlayId.nunique() == df.PlayId.nunique()
    return df_play, df_players


def process_team_abbr(df, train):
    map_abbr = {"ARI": "ARZ", "BAL": "BLT", "CLE": "CLV", "HOU": "HST"}
    for abb in train["PossessionTeam"].unique():
        map_abbr[abb] = abb
    df["PossessionTeam"] = df["PossessionTeam"].map(map_abbr)
    df["HomeTeamAbbr"] = df["HomeTeamAbbr"].map(map_abbr)
    df["HomePossession"] = df["PossessionTeam"] == df["HomeTeamAbbr"]


def standardize_direction(df):
    df["Dir_rad"] = np.mod(90 - df.Dir, 360) * math.pi / 180.0
    df["ToLeft"] = df.PlayDirection == "left"
    df["TeamOnOffense"] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, "TeamOnOffense"] = "away"
    df["IsOnOffense"] = df.Team == df.TeamOnOffense
    df["X_std"] = df.X
    df.loc[df.ToLeft, "X_std"] = 120 - df.loc[df.ToLeft, "X"]
    df["Y_std"] = df.Y
    df.loc[df.ToLeft, "Y_std"] = 160 / 3 - df.loc[df.ToLeft, "Y"]
    df["Dir_std"] = df.Dir_rad
    df.loc[df.ToLeft, "Dir_std"] = np.mod(np.pi + df.loc[df.ToLeft, "Dir_rad"], 2 * np.pi)
    df.loc[(df.IsOnOffense) & df["Dir_std"].isna(), "Dir_std"] = 0.0
    df.loc[~(df.IsOnOffense) & df["Dir_std"].isna(), "Dir_std"] = np.pi


def data_augmentation(df, sample_ids):
    df_sample = df.loc[df.PlayId.isin(sample_ids)].copy()
    df_sample["Y_std"] = 160 / 3 - df_sample["Y_std"]
    df_sample["Dir_std"] = df_sample["Dir_std"].apply(lambda x: 2 * np.pi - x)
    df_sample["PlayId"] = df_sample["PlayId"].apply(lambda x: x + "_aug")
    return df_sample


def process_tracking_data(df):
    df["Sx"] = df["S"] * df["Dir_std"].apply(math.cos)
    df["Sy"] = df["S"] * df["Dir_std"].apply(math.sin)
    rushers = df[df["IsRusher"]].copy().set_index("PlayId", drop=True)
    pmap = rushers[["X_std", "Y_std", "Sx", "Sy"]].to_dict(orient="index")
    df["player_minus_rusher_x"] = df["PlayId"].map(lambda v: pmap[v]["X_std"]) - df["X_std"]
    df["player_minus_rusher_y"] = df["PlayId"].map(lambda v: pmap[v]["Y_std"]) - df["Y_std"]
    df["player_minus_rusher_Sx"] = df["PlayId"].map(lambda v: pmap[v]["Sx"]) - df["Sx"]
    df["player_minus_rusher_Sy"] = df["PlayId"].map(lambda v: pmap[v]["Sy"]) - df["Sy"]


def build_cnn_tensor(df_all, play_ids_ordered):
    """Return [n_plays, 11, 10, 10] Zoo tensor ordered to match df_play."""
    grouped = df_all.groupby("PlayId", sort=False)
    train_x = np.zeros([len(play_ids_ordered), 11, 10, 10], dtype=np.float32)
    for i, play_id in enumerate(play_ids_ordered):
        group = grouped.get_group(play_id)
        offense_ids = group[group.IsOnOffense & ~group.IsRusher].index
        defense_ids = group[~group.IsOnOffense].index
        if len(offense_ids) != 10 or len(defense_ids) != 11:
            raise ValueError(
                f"Expected 10 offense non-rusher and 11 defense rows for PlayId={play_id}; "
                f"got {len(offense_ids)} offense and {len(defense_ids)} defense."
            )
        for j, def_id in enumerate(defense_ids):
            def_x, def_y, def_Sx, def_Sy = group.loc[def_id, ["X_std", "Y_std", "Sx", "Sy"]].values
            dr_x, dr_y = group.loc[def_id, ["player_minus_rusher_x", "player_minus_rusher_y"]].values
            dr_Sx, dr_Sy = group.loc[def_id, ["player_minus_rusher_Sx", "player_minus_rusher_Sy"]].values
            off_vals = group.loc[offense_ids, ["Sx", "Sy", "X_std", "Y_std"]].values
            train_x[i, j, :, :4] = off_vals - np.array([def_Sx, def_Sy, def_x, def_y])
            train_x[i, j, :, -6:] = [dr_Sx, dr_Sy, dr_x, dr_y, def_Sx, def_Sy]
    return train_x


def build_set_tensor(df_all, play_ids_ordered):
    """Return [n_plays, 22, 10] player-token tensor ordered to match df_play."""
    token_cols = [
        "X_std",
        "Y_std",
        "Sx",
        "Sy",
        "player_minus_rusher_x",
        "player_minus_rusher_y",
        "player_minus_rusher_Sx",
        "player_minus_rusher_Sy",
        "IsOnOffense",
        "IsRusher",
    ]
    grouped = df_all.groupby("PlayId", sort=False)
    train_x_set = np.zeros([len(play_ids_ordered), 22, len(token_cols)], dtype=np.float32)
    for i, play_id in enumerate(play_ids_ordered):
        group = grouped.get_group(play_id).copy()
        group["token_group"] = np.where(group["IsRusher"], 0, np.where(group["IsOnOffense"], 1, 2))
        group = group.sort_values(["token_group", "Y_std", "X_std"], kind="mergesort")
        if len(group) != 22:
            raise ValueError(f"Expected exactly 22 player rows for PlayId={play_id}; got {len(group)}.")
        train_x_set[i] = group[token_cols].to_numpy(dtype=np.float32)
    return train_x_set


def main():
    """Build CNN input tensor and labels from raw CSV; add ~50% mirror-augmented plays (_aug)."""
    train = pd.read_csv(TRAIN_CSV, dtype={"WindSpeed": "object"})
    df_play, df_players = split_play_and_player_cols(train)
    process_team_abbr(df_play, train)
    df_players = df_players.merge(
        df_play[["PlayId", "PossessionTeam", "HomeTeamAbbr", "PlayDirection"]], how="left", on="PlayId"
    )
    df_players.loc[df_players.Season == 2017, "S"] = 10 * df_players.loc[df_players.Season == 2017, "Dis"]
    standardize_direction(df_players)

    np.random.seed(AUGMENT_SEED)
    n_plays = len(df_play.PlayId.unique())
    sample_ids = np.random.choice(df_play.PlayId.unique(), int(0.5 * n_plays))  # Mirror ~50% of plays.
    df_players_aug = data_augmentation(df_players, sample_ids)
    df_players = pd.concat([df_players, df_players_aug], ignore_index=True)
    df_play_aug = df_play.loc[df_play.PlayId.isin(sample_ids)].copy()
    df_play_aug["PlayId"] = df_play_aug["PlayId"] + "_aug"
    df_play = pd.concat([df_play, df_play_aug], ignore_index=True)
    df_players.sort_values(by=["PlayId"], inplace=True)
    df_play.sort_values(by=["PlayId"], inplace=True)
    process_tracking_data(df_players)

    feats = [
        "PlayId", "IsOnOffense", "X_std", "Y_std", "Sx", "Sy",
        "player_minus_rusher_x", "player_minus_rusher_y", "player_minus_rusher_Sx", "player_minus_rusher_Sy", "IsRusher",
    ]
    df_all = df_players[feats]
    play_ids_ordered = df_play.PlayId.values
    train_x = build_cnn_tensor(df_all, play_ids_ordered)
    train_x_set = build_set_tensor(df_all, play_ids_ordered)

    min_idx_y, max_idx_y = 71, 150
    train_y = df_play[["PlayId", "Yards"]].copy()
    train_y["YardIndex"] = train_y["Yards"] + 99
    train_y["YardIndexClipped"] = train_y["YardIndex"].clip(min_idx_y, max_idx_y).astype(int)
    df_season = df_play[["PlayId", "Season"]].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(TRAIN_X_PATH, train_x)
    np.save(TRAIN_X_SET_PATH, train_x_set)
    train_y.to_pickle(TRAIN_Y_PATH)
    df_season.to_pickle(DF_SEASON_PATH)
    print("Saved:", TRAIN_X_PATH, TRAIN_X_SET_PATH, TRAIN_Y_PATH, DF_SEASON_PATH)
    print("train_x shape:", train_x.shape, "train_x_set shape:", train_x_set.shape, "plays:", len(train_y))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
