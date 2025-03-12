import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set


def standardize_subid_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the subid column in the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: The DataFrame with the subid column standardized.
    """
    if "workerId" in df.columns:
        df["subid"] = df["workerId"].astype(str)
    elif "PROLIFIC_PID" in df.columns:
        df["subid"] = df["PROLIFIC_PID"].astype(str)
    elif "sona_id" in df.columns:
        df["subid"] = df["sona_id"].astype(str)
    return df


def check_run_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the first valid run for each participant.
    A valid run is defined as having more than 10 points.
    """
    filtered_df = pd.DataFrame()
    for subid in df["subid"].unique():
        # check if subid has more than one run_id
        if len(df[df["subid"] == subid]["run_id"].unique()) > 1:
            # if they do, grab the lowest run_id belonging to them
            lowest_run_id = df[df["subid"] == subid]["run_id"].min()
            filtered_df = pd.concat(
                [
                    filtered_df,
                    df[(df["subid"] == subid) & (df["run_id"] == lowest_run_id)],
                ]
            )
        else:
            filtered_df = pd.concat([filtered_df, df[df["subid"] == subid]])
    return filtered_df.reset_index(drop=True)


def first_pass_counts(df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """
    Filters the given DataFrame based on the number of occurrences of each word for each subid.

    Args:
        df (pd.DataFrame): The input DataFrame containing columns "subid", "word", "item", and optionally "rule" and "current_rule".
        n_trials (int): The desired number of occurrences for each word.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only the rows with subids that have the desired number of occurrences for each word.
    """
    sub_counts = df.groupby("subid")[["word"]].count()
    good_subs = sub_counts[sub_counts["word"] == n_trials].index.values
    df = df[df["subid"].isin(good_subs)]
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


def get_prop_list(strat_df: pd.DataFrame) -> Set[str]:
    hist_df = (
        strat_df[
            (strat_df["good_block"] == 1)
            & (strat_df["trial_within_block"] == strat_df["max_trial_within_block"])
        ]
        .groupby("subid")["good_block"]
        .count()
        .reset_index()
    )
    hist_df["good_block"] = hist_df["good_block"] / strat_df["block"].max()
    num_block_sizes = strat_df["max_trial_within_block"].nunique()
    proportion = 0.375
    # filter out who have less proportion than the proportion
    return set(hist_df.loc[(hist_df["good_block"] < proportion), "subid"].values)


def get_discs(sub_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get the discriminative and non-discriminative words from a DataFrame.

    Args:
        sub_df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        Dict[str, List[str]]: A dictionary containing two lists of words:
            - "disc": The discriminative words.
            - "non_disc": The non-discriminative words.
    """
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


def make_disc_dict(strat_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate a dictionary of discriminative and non-discriminative words for each subject.

    Parameters:
        strat_df (pd.DataFrame): The input DataFrame containing subject strategy data.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where the keys are subject IDs and the values are dictionaries
        containing disc and non_disc as keys and lists of strings as values corresponding to either discriminative words or non_discriminative words.
    """
    return {sub: get_discs(sub_df) for sub, sub_df in strat_df.groupby("subid")}


def first_missed_after_boundary(sub_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    sub_df: dataframe of a single subject
    returns two lists, one of which are the objective boundary indices and the other are the subjective boundary indices
    """
    # sub_df = sub_df.reset_index()
    boundary_indices = sub_df[
        sub_df["corr_rule_numeric"] != sub_df["corr_rule_numeric"].shift(1)
    ].index.values
    missed_indices = sub_df[sub_df["points"] == 0].index.values
    missed_after_boundary = []
    for boundary in boundary_indices:
        if missed_indices[missed_indices >= boundary].shape[0] == 0:
            continue
        missed_after_boundary.append(missed_indices[missed_indices >= boundary][0])
    missed_after_boundary = np.array(missed_after_boundary)
    return boundary_indices, missed_after_boundary


def label_strat_blocks(sub_strat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels the strategy blocks in the given DataFrame based on certain criteria.

    Args:
        sub_strat_df (pd.DataFrame): The DataFrame containing the strategy data.

    Returns:
        pd.DataFrame: The DataFrame with additional columns indicating the labeled blocks.

    Raises:
        None
    """
    sub_strat_df["block"] = (
        sub_strat_df.groupby("run")["corr_rule_numeric"].diff().ne(0).cumsum()
    )
    sub_strat_df["trial_within_block"] = sub_strat_df.groupby("block").cumcount()
    # get max trial_in_block for each block
    sub_strat_df["max_trial_within_block"] = sub_strat_df.groupby("block")[
        "trial_within_block"
    ].transform("max")
    # get avg accuracy for last 2 trials in each block
    sub_strat_df["avg_acc_last_2"] = sub_strat_df.groupby("block")["points"].transform(
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
    sub_strat_df.loc[boundary_indices[1:], "obj_boundary"] = 1
    sub_strat_df.loc[missed_after_boundary, "subj_boundary"] = 1
    return sub_strat_df


def label_item_relative_to_index(sub_strat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each item in the DataFrame relative to the indices of rows with obj_boundary or subj_boundary flags set to 1.

    Parameters:
        sub_strat_df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with additional columns "rel_obj_boundary" and "rel_subj_boundary" representing the relative distances between each row and the nearest row with obj_boundary or subj_boundary flag set to 1, respectively.
    """
    sub_strat_df = sub_strat_df.reset_index(drop=True)

    obj_boundary_indices = sub_strat_df.index[sub_strat_df["obj_boundary"] == 1].values
    subj_boundary_indices = sub_strat_df.index[
        sub_strat_df["subj_boundary"] == 1
    ].values

    if len(obj_boundary_indices) > 0:
        # Calculate relative distances between each row and the nearest row with obj_boundary flag set to 1
        obj_dists = (
            np.array(sub_strat_df.index)[:, np.newaxis]
            - np.array(obj_boundary_indices)[np.newaxis, :]
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
            np.array(sub_strat_df.index)[:, np.newaxis]
            - np.array(subj_boundary_indices)[np.newaxis, :]
        )
        subj_dists = subj_dists[
            np.arange(len(subj_dists)), np.abs(subj_dists).argmin(axis=1)
        ]

        sub_strat_df["rel_subj_boundary"] = subj_dists
    else:
        sub_strat_df["rel_subj_boundary"] = np.nan

    return sub_strat_df


def get_block_corr_rule(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subid", "block"])["corr_rule_numeric"]
        .mean()
        .rename("corr_rule_mean")
        .reset_index()
    )


def is_within_or_across(
    x: pd.DataFrame,
    prev_corr_rule_mean: int,
    corr_rule_shifts: Dict[str, List[Tuple[int]]],
) -> str:
    """
    Determines whether the given value of `x["corr_rule_mean"]` is within or across the range of previous correlation rule means.

    Args:
        x (pd.DataFrame): The input DataFrame.
        prev_corr_rule_mean (int): The previous correlation rule mean.
        corr_rule_shifts (Dict[str, List[Tuple[int]]]): A dictionary containing the shifts for within and across ranges.

    Returns:
        str: The result indicating whether the value is within or across the range.
    """
    if pd.isna(prev_corr_rule_mean):
        return np.nan
    elif (x["corr_rule_mean"], prev_corr_rule_mean) in corr_rule_shifts["within"]:
        return "within"
    else:
        return "across"


def get_within_across(
    df: pd.DataFrame, corr_rule_shifts: Dict[str, List[Tuple[int]]]
) -> pd.DataFrame:
    """
    Calculate the 'within_across' column for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        corr_rule_shifts (Dict[str, List[Tuple[int]]]): The dictionary containing correlation rule shifts.

    Returns:
        pd.DataFrame: The DataFrame with the 'within_across' column added.
    """
    df["within_across"] = df.apply(
        lambda x: is_within_or_across(
            x,
            (
                df.loc[x.name - 1, "corr_rule_mean"]
                if x.name > 0 and x["subid"] == df.loc[x.name - 1, "subid"]
                else np.nan
            ),
            corr_rule_shifts,
        ),
        axis=1,
    )
    return df


def is_discriminating(
    x: pd.DataFrame, disc_dict: Dict[str, Dict[str, List[str]]]
) -> bool:
    return x["word"] in disc_dict[x["subid"]]["disc"]


def get_block_size_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the mean block size for each subid and block in the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame with the calculated block size mean for each subid and block.
    """
    block_size_sequences = (
        df.groupby(["subid", "block"])["max_trial_within_block"]
        .mean()
        .rename("block_size_mean")
        .reset_index()
    )
    block_size_sequences["prev_block_size"] = block_size_sequences.groupby("subid")[
        "block_size_mean"
    ].shift(1)
    return block_size_sequences


def get_disc_and_within_across(
    df: pd.DataFrame, disc_dict: Dict[str, Dict[str, List[str]]]
) -> pd.DataFrame:
    """
    Calculates the 'disc' column and the 'within_across' column for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        disc_dict (Dict[str, Dict[str, List[str]]]): A dictionary containing discrimination rules.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'disc' and 'within_across' columns added.
    """
    corr_rule_shifts = {
        "within": [(2, 3), (3, 2), (0, 1), (1, 0)],
        "across": [(2, 0), (0, 2), (0, 3), (3, 0), (1, 3), (3, 1), (1, 2), (2, 1)],
    }
    corr_rule_sequences = get_block_corr_rule(df)
    corr_rule_sequences = get_within_across(corr_rule_sequences, corr_rule_shifts)
    df = df.merge(
        corr_rule_sequences[["subid", "block", "within_across"]],
        on=["subid", "block"],
        how="left",
    )

    block_size_sequences = get_block_size_sequences(df)
    df = df.merge(
        block_size_sequences[["subid", "block", "prev_block_size"]],
        on=["subid", "block"],
        how="left",
    )

    df["disc"] = df.apply(lambda x: is_discriminating(x, disc_dict), axis=1)

    return df


def get_only_blocksize_n(strat_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter the given DataFrame to include only rows where the 'max_trial_within_block' column
    has a unique value of 'n' for each 'subid' group.

    Parameters:
        strat_df (pd.DataFrame): The input DataFrame containing the data.
        n (int): The desired value of 'max_trial_within_block' for filtering.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only rows with the desired 'max_trial_within_block' value.
    """
    num_blocks = (
        strat_df.groupby(["subid"])["max_trial_within_block"].nunique().reset_index()
    )
    curr_subs = num_blocks[num_blocks["max_trial_within_block"] == n]["subid"].values
    strat_df = strat_df[strat_df["subid"].isin(curr_subs)].reset_index(drop=True)
    return strat_df


def add_inv_item_rule(strat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds inverse item rule columns to the given DataFrame.

    Parameters:
        strat_df (pd.DataFrame): The DataFrame to which inverse item rule columns will be added.

    Returns:
        pd.DataFrame: The DataFrame with inverse item rule columns added.
    """
    strat_df["inv_item_rule_idx0"] = 1 - strat_df["item_rule_idx0"]
    strat_df["inv_item_rule_idx1"] = 1 - strat_df["item_rule_idx1"]
    return strat_df


def add_cols(strat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional columns to the given DataFrame. Primarily those used by the RL model later to understand the stimulus.

    Parameters:
        strat_df (pd.DataFrame): The DataFrame to which the columns will be added.

    Returns:
        pd.DataFrame: The DataFrame with the added columns.
    """
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


def drop_unimportant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unimportant columns from the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to drop columns.

    Returns:
    - pd.DataFrame: The DataFrame with unimportant columns dropped.
    """
    cognition_cols = [
        "view_history",
        "ip",
        "recorded_at",
        "user_agent",
        "device",
        "browser",
        "browser_version",
        "platform",
        "platform_version",
        "referer",
        "accept_language",
        "fbclid",
    ]
    df = df.drop(columns=[col for col in cognition_cols if col in df.columns])
    return df


def add_trial_nums_and_runs(strat_df: pd.DataFrame, nruns: int) -> pd.DataFrame:
    """
    Adds trial numbers and runs to the given DataFrame.

    Parameters:
        strat_df (pd.DataFrame): The DataFrame to which trial numbers and runs will be added.

    Returns:
        pd.DataFrame: The DataFrame with trial numbers and runs added.
    """
    # make runs by applying to trial_number and splitting evenly
    strat_df["trial_number"] = strat_df.groupby("subid").cumcount() + 1
    # nruns should be 1, 4, or 8
    assert nruns in [1, 4, 8], "nruns should be 1, 4, or 8"
    # using nruns apply to trial_number and split evenly
    n_total_trials = strat_df["trial_number"].max()
    strat_df["run"] = strat_df["trial_number"].apply(
        lambda x: (
            x // (n_total_trials // nruns)
            if x % (n_total_trials // nruns) == 0
            else (
                (x // (n_total_trials // nruns)) + 1
                if x % (n_total_trials // nruns) != 0
                else 0
            )
        )
    )
    strat_df["trial_number_per_run"] = strat_df.groupby(["subid", "run"]).cumcount() + 1
    return strat_df


def preproc_df(df: pd.DataFrame, n_trials: int, check_run: bool = True) -> pd.DataFrame:
    df = standardize_subid_col(df)
    if check_run:
        df = check_run_id(df)
    df = drop_unimportant_cols(df)

    df = first_pass_counts(df, n_trials=n_trials)
    return df


def process_strat(df: pd.DataFrame, nruns: int) -> pd.DataFrame:
    strat_df = get_strat_df(df)
    strat_df = add_cols(strat_df)
    strat_df = add_trial_nums_and_runs(strat_df, nruns=nruns)
    labelled_strat_df = (
        strat_df.groupby("subid")
        .apply(
            lambda x: label_item_relative_to_index(label_strat_blocks(x))
        )  # this line stopped working for experiment 2 for reasons unknown
        .reset_index(drop=True)
    )
    disc_dict = make_disc_dict(labelled_strat_df)
    labelled_strat_df = get_disc_and_within_across(labelled_strat_df, disc_dict)

    return labelled_strat_df
