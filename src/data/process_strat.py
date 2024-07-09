import pandas as pd
import numpy as np
from typing import Dict, Tuple


def first_pass_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "workerId" in df.columns:
        df["sona_id"] = df["workerId"].astype(str)
    elif "PROLIFIC_PID" in df.columns:
        df["sona_id"] = df["PROLIFIC_PID"].astype(str)
    else:
        df["sona_id"] = df["sona_id"].astype(str)
    sub_counts = df.groupby("sona_id")[["word"]].count()

    good_subs = sub_counts[sub_counts["word"] == 224].index.values
    df = df[df["sona_id"].isin(good_subs)]
    if "rule" in df.columns:
        df = df.rename(columns={"rule": "item_rule", "current_rule": "correct_rule"})
    df = df.reset_index(drop=True)
    return df


def get_strat_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: dataframe of all subjects
    returns a dataframe of the rule guessing task for all subjects
    """
    strat_indices = np.where(~df["correct_rule"].isnull())[0]
    strat_df = df.loc[strat_indices].reset_index()
    strat_df["corr_rule_numeric"] = strat_df["correct_rule"].map(
        {"small": 0, "large": 1, "natural": 2, "manmade": 3}
    )
    return strat_df


def get_discs(sub_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sub_df = sub_df.reset_index(drop=True)
    disc_rules = {
        11.0: [1.0, 10.0],
        0.0: [1.0, 10.0],
        1.0: [11.0, 0.0],
        10.0: [11.0, 0.0],
    }
    sub_df["bin_rule"] = sub_df["item_rule"].map(disc_rules)
    sub_df["item_shift"] = sub_df["item_rule"].shift(1)
    sub_df["is_disc"] = sub_df[["item_shift", "bin_rule"]].apply(
        lambda x: x["item_shift"] in x["bin_rule"], axis=1
    )
    disc_words = sub_df[sub_df["is_disc"]]["word"].values
    non_disc_words = sub_df[~sub_df["is_disc"]]["word"].values
    return {"disc": disc_words, "non_disc": non_disc_words}


def make_disc_dict(strat_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    return {sub: get_discs(sub_df) for sub, sub_df in strat_df.groupby("sona_id")}


def first_missed_after_boundary(sub_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    sub_df: dataframe of a single subject
    returns two lists, one of which are the objective boundary indices and the other are the subjective boundary indices
    """
    # sub_df = sub_df.reset_index()
    boundary_indices = sub_df[
        sub_df["corr_rule_numeric"] != sub_df["corr_rule_numeric"].shift(1)
    ]
    boundary_indices = boundary_indices.index.values
    missed_indices = sub_df[sub_df["points"] == 0].index.values
    missed_after_boundary = []
    for boundary in boundary_indices:
        if missed_indices[missed_indices >= boundary].shape[0] == 0:
            continue
        missed_after_boundary.append(missed_indices[missed_indices >= boundary][0])
    missed_after_boundary = np.array(missed_after_boundary)
    return boundary_indices, missed_after_boundary


def label_strat_blocks(sub_strat_df: pd.DataFrame) -> pd.DataFrame:
    sub_strat_df["block"] = (
        sub_strat_df.groupby("run")["corr_rule_numeric"].diff().ne(0).cumsum()
    )  # I think this is now correct to account for runs
    sub_strat_df["trial_within_block"] = sub_strat_df.groupby("block").cumcount()
    # get max trial_in_block for each block
    sub_strat_df["max_trial_within_block"] = sub_strat_df.groupby("block")[
        "trial_within_block"
    ].transform("max")
    # get avg accuracy for last 3 trials in each block
    sub_strat_df["avg_acc_last_2"] = sub_strat_df.groupby("block")["points"].apply(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    sub_strat_df["good_block"] = 0
    sub_strat_df.loc[
        (sub_strat_df["trial_within_block"] == sub_strat_df["max_trial_within_block"])
        & (sub_strat_df["avg_acc_last_2"] == 1),
        "good_block",
    ] = 1
    # label blocks as good or bad depending on the last trial in the block
    sub_strat_df["good_block"] = sub_strat_df.groupby("block")["good_block"].transform(
        "max"
    )
    # label whether the previous block was good
    sub_strat_df["prev_good_block"] = sub_strat_df["good_block"].shift(1)
    # find objective and subjective event boundaries
    boundary_indices, missed_after_boundary = first_missed_after_boundary(sub_strat_df)
    sub_strat_df["obj_boundary"] = 0
    sub_strat_df["subj_boundary"] = 0
    sub_strat_df.loc[boundary_indices, "obj_boundary"] = 1
    sub_strat_df.loc[missed_after_boundary, "subj_boundary"] = 1
    return sub_strat_df


def label_item_relative_to_index(sub_strat_df: pd.DataFrame) -> pd.DataFrame:
    sub_strat_df = sub_strat_df.reset_index(drop=True)

    # Find indices of all rows with obj_boundary or subj_boundary flag set to 1
    obj_boundary_indices = sub_strat_df.index[sub_strat_df["obj_boundary"] == 1].values
    subj_boundary_indices = sub_strat_df.index[
        sub_strat_df["subj_boundary"] == 1
    ].values

    if len(obj_boundary_indices) > 0:
        # Calculate relative distances between each row and the nearest row with obj_boundary flag set to 1
        obj_dists = (
            sub_strat_df.index[:, np.newaxis] - obj_boundary_indices[np.newaxis, :]
        )
        obj_dists = obj_dists[
            np.arange(len(obj_dists)), np.abs(obj_dists).argmin(axis=1)
        ]

        sub_strat_df["rel_obj_boundary"] = obj_dists
    else:
        sub_strat_df["rel_obj_boundary"] = np.nan

    if len(subj_boundary_indices) > 0:
        # Calculate relative distances between each row and the nearest row with subj_boundary flag set to 1
        subj_dists = (
            sub_strat_df.index[:, np.newaxis] - subj_boundary_indices[np.newaxis, :]
        )
        subj_dists = subj_dists[
            np.arange(len(subj_dists)), np.abs(subj_dists).argmin(axis=1)
        ]

        sub_strat_df["rel_subj_boundary"] = subj_dists
    else:
        sub_strat_df["rel_subj_boundary"] = np.nan

    return sub_strat_df


def compute_mean_corr_rule(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["sona_id", "block"])["corr_rule_numeric"]
        .mean()
        .rename("corr_rule_mean")
        .reset_index()
    )


def is_within_or_across(
    x: pd.DataFrame,
    prev_corr_rule_mean: pd.DataFrame,
    corr_rule_shifts: Dict[str, np.ndarray],
):
    if pd.isna(prev_corr_rule_mean):
        return np.nan
    elif (x["corr_rule_mean"], prev_corr_rule_mean) in corr_rule_shifts["within"]:
        return "within"
    else:
        return "across"  # TODO fix this with respect to previous block rule


def get_within_across(
    df: pd.DataFrame, corr_rule_shifts: Dict[str, np.ndarray]
) -> pd.DataFrame:
    df["within_across"] = df.apply(
        lambda x: is_within_or_across(
            x,
            (
                df.loc[x.name - 1, "corr_rule_mean"]
                if x.name > 0 and x["sona_id"] == df.loc[x.name - 1, "sona_id"]
                else np.nan
            ),
            corr_rule_shifts,
        ),
        axis=1,
    )
    return df


def is_discriminating(x, disc_dict):
    return (
        x["word"] in disc_dict[x["sona_id"]]["disc"]
    )  # TODO make sure that this is discriminative with respect to previous block rule


def get_block_size_sequences(df: pd.DataFrame) -> pd.DataFrame:
    block_size_sequences = (
        df.groupby(["sona_id", "block"])["max_trial_within_block"]
        .mean()
        .rename("block_size_mean")
        .reset_index()
    )
    block_size_sequences["prev_block_size"] = block_size_sequences.groupby("sona_id")[
        "block_size_mean"
    ].shift(1)
    return block_size_sequences


def process_dataframe(df: pd.DataFrame, disc_dict: Dict[str, Dict[str, np.ndarray]]):
    corr_rule_shifts = {
        "within": [(2, 3), (3, 2), (0, 1), (1, 0)],
        "across": [(2, 0), (0, 2), (0, 3), (3, 0), (1, 3), (3, 1), (1, 2), (2, 1)],
    }
    corr_rule_sequences = compute_mean_corr_rule(df)
    corr_rule_sequences = get_within_across(corr_rule_sequences, corr_rule_shifts)
    df = df.merge(
        corr_rule_sequences[["sona_id", "block", "within_across"]],
        on=["sona_id", "block"],
        how="left",
    )  # TODO shift corr_rule_sequences by 1 so I can see if the current block overall is within or across

    block_size_sequences = get_block_size_sequences(df)
    df = df.merge(
        block_size_sequences[["sona_id", "block", "prev_block_size"]],
        on=["sona_id", "block"],
        how="left",
    )

    df["disc"] = df.apply(lambda x: is_discriminating(x, disc_dict), axis=1)
    df = add_cols(df)

    return df


def get_only_blocksize_3(strat_df: pd.DataFrame) -> pd.DataFrame:
    num_blocks = (
        strat_df.groupby(["sona_id"])["max_trial_within_block"].nunique().reset_index()
    )
    curr_subs = num_blocks[num_blocks["max_trial_within_block"] == 3]["sona_id"].values
    strat_df = strat_df[strat_df["sona_id"].isin(curr_subs)].reset_index(drop=True)
    return strat_df


def add_inv_item_rule(strat_df: pd.DataFrame) -> pd.DataFrame:
    strat_df["inv_item_rule_idx0"] = 1 - strat_df["item_rule_idx0"]
    strat_df["inv_item_rule_idx1"] = 1 - strat_df["item_rule_idx1"]
    return strat_df


def add_cols(strat_df: pd.DataFrame) -> pd.DataFrame:
    strat_df["resp_numeric"] = strat_df["response"].apply(
        lambda x: 1 if x == "f" else (0 if x == "j" else -1)
    )
    strat_df["item_rule_idx0"] = strat_df["item_rule"].apply(
        lambda x: 0 if x in [0, 1] else 1
    )
    strat_df["item_rule_idx1"] = strat_df["item_rule"].apply(
        lambda x: 0 if x in [10, 0] else 1
    )
    strat_df = add_inv_item_rule(strat_df)
    return strat_df


def add_trial_nums_and_runs(strat_df: pd.DataFrame) -> pd.DataFrame:
    strat_df["trial_number"] = strat_df.groupby("sona_id").cumcount() + 1
    # label the runs for strat_df for each subject, (run 1 = 0:96, run 2 = 97:192, run 3 = 193:288)

    strat_df["run"] = strat_df["trial_number"].apply(
        lambda x: (1 if x <= 56 else (2 if x <= 56 * 2 else (3 if x <= 56 * 3 else 4)))
    )
    # label the trial_number_per_run for each subject in the strat_df that just ticks up from 1 to 96 for each run
    strat_df["trial_number_per_run"] = (
        strat_df.groupby(["sona_id", "run"]).cumcount() + 1
    )
    return strat_df
