import numpy as np
import pandas as pd
from psifr import fr
from nltk.metrics.distance import edit_distance
from nltk.corpus import words
from typing import Tuple, Set

english_words = set(words.words())
english_words = {word.upper() for word in english_words}


def filter_mem_df(mem: pd.DataFrame, cleaned_exp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the given memory DataFrame based on the provided cleaned experiment DataFrame.

    Args:
        mem (pd.DataFrame): The memory DataFrame to be filtered.
        cleaned_exp_df (pd.DataFrame): The cleaned experiment DataFrame used for filtering.

    Returns:
        pd.DataFrame: The filtered memory DataFrame.

    """
    mem = mem[~mem["block"].isnull()].reset_index(drop=True).copy()
    mem = mem[mem["sona_id"].isin(cleaned_exp_df["sona_id"].unique())].reset_index(
        drop=True
    )
    return mem


def get_encoding_subset(strat_df: pd.DataFrame, sub: str) -> pd.DataFrame:
    """
    Retrieves a subset of encoding data for a specific subject.

    Args:
        strat_df (pd.DataFrame): The dataframe containing the encoding data.
        sub (str): The subject ID.

    Returns:
        pd.DataFrame: A subset of the encoding data for the specified subject, including columns for 'word',
        'corr_rule_numeric', 'rel_subj_boundary', 'block', 'run', and 'points'.
    """
    sub_strat_data = strat_df[strat_df["sona_id"] == sub].reset_index(drop=True)
    sub_strat_data["word"] = sub_strat_data["word"].str.lower()
    encoding_subset = sub_strat_data[
        ["word", "corr_rule_numeric", "rel_subj_boundary", "block", "run", "points"]
    ]
    return encoding_subset


def correct_response(response: str, word_list: Set[str]) -> str:
    """
    Corrects the given response by finding the closest match in the word list.

    Args:
        response (str): The response to be corrected.
        word_list (Set[str]): The set of words to compare the response against.

    Returns:
        str: The corrected response, if a close match is found. Otherwise, returns "No Match".
    """
    # Initialize minimum distance to a high value
    min_distance = float("inf")
    corrected_word = response  # Default to the original response
    candidate_words = []
    if response in word_list:
        return response
    elif response in english_words:
        return "No Match"

    # Calculate edit distance to each word in the list
    for word in word_list:
        distance = edit_distance(response, word, substitution_cost=2)

        # Check if this word is a closer match
        if distance < min_distance:
            min_distance = distance
            candidate_words = [
                word
            ]  # Start a new list with this word as the best match so far

        # If this word is as close as the best matches so far, add it to the list of candidates
        elif distance == min_distance:
            candidate_words.append(word)

    # If the closest word has an edit distance of 2 and is unique, correct the response
    if min_distance <= 2 and len(candidate_words) == 1:
        corrected_word = candidate_words[0]
        return corrected_word
    else:
        return "No Match"


def spellcheck_and_filter(
    recall_run: pd.DataFrame, encoding_subset: pd.DataFrame
) -> pd.DataFrame:
    """
    Spellchecks and filters the recall_run DataFrame based on a given encoding_subset.

    Args:
        recall_run (pd.DataFrame): The DataFrame containing the recall run data.
        encoding_subset (pd.DataFrame): The DataFrame containing the encoding subset data.

    Returns:
        pd.DataFrame: The filtered recall_run DataFrame with corrected responses.

    """
    word_list = set(encoding_subset["word"].str.lower().values)
    responses = list(set(recall_run["response"]))
    # filter out non-strings from responses
    responses = [response for response in responses if isinstance(response, str)]

    corrected_responses = [
        correct_response(response, word_list) for response in responses
    ]

    # make a dictionary of these to insert them back into the dataframe
    correction_dict = dict(zip(responses, corrected_responses))

    # Apply the correction to the recall_df
    recall_run["corrected_response"] = recall_run["response"].map(correction_dict)

    # drop the 'No Match' rows
    recall_run = recall_run[recall_run["corrected_response"] != "No Match"]

    return recall_run


def make_recall_df(
    strat_df: pd.DataFrame,
    full_df: pd.DataFrame,
    sub: str,
) -> pd.DataFrame:
    """
    Generate a recall dataframe for a specific subject.

    Args:
        strat_df (pd.DataFrame): The strategic dataframe.
        full_df (pd.DataFrame): The full dataframe.
        sub (str): The subject ID.

    Returns:
        pd.DataFrame: The recall dataframe.
    """
    sub_data = full_df[full_df["sona_id"] == sub].reset_index(drop=True)
    encoding_subset = get_encoding_subset(strat_df, sub)

    # grab recall trials
    recall_trials = sub_data[sub_data["trial_type"] == "html-free-recall"].reset_index(
        drop=True
    )

    recall_trials["response"] = recall_trials["response"].str.lower()
    recall_trials["response"] = recall_trials["response"].str.strip()
    recall_end_of_run_idxs = recall_trials[recall_trials["button"] == "pressed"].index
    recall_runs = []
    start_idx = 0
    for end_idx in recall_end_of_run_idxs:
        recall_run = recall_trials.loc[start_idx:end_idx]
        recall_runs.append(recall_run)
        start_idx = end_idx + 1
    num_runs = 4
    checked_runs = []
    for run in recall_runs:
        run = spellcheck_and_filter(run, encoding_subset)
        checked_runs.append(run)

    enc_lists = []
    rec_lists = []
    for i in range(1, num_runs + 1):
        enc_lists.append(
            encoding_subset[encoding_subset["run"] == i]["word"].str.lower().values
        )
        rec_lists.append(checked_runs[i - 1]["corrected_response"].values)

    list_subject = [sub] * num_runs

    fr_data = fr.table_from_lists(list_subject, enc_lists, rec_lists)
    return fr_data


def get_final_psifr_df(
    fr_data: pd.DataFrame, strat_df: pd.DataFrame, sub: str
) -> pd.DataFrame:
    """
    Returns a DataFrame containing the final dataframe style preferred by the psifr package.

    Parameters:
        fr_data (pd.DataFrame): The DataFrame containing the free recall data.
        strat_df (pd.DataFrame): The DataFrame containing the strategic data.
        sub (str): The subject identifier.

    Returns:
        pd.DataFrame: The DataFrame in psifr format.
    """
    encoding_subset = get_encoding_subset(strat_df, sub)
    merged = fr.merge_free_recall(fr_data)
    merged = merged.merge(
        encoding_subset, left_on=["item"], right_on=["word"], how="left"
    )
    return merged


def get_fr_df_for_all_subs(
    strat_df: pd.DataFrame, full_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves the free recall data and the merged data for all subjects.

    Args:
        strat_df (pd.DataFrame): The stratified dataframe.
        full_df (pd.DataFrame): The full dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the free recall data and the merged data.
    """
    full_fr_df = pd.DataFrame()
    full_merged_df = pd.DataFrame()
    for sub in strat_df.sona_id.unique():
        fr_data = make_recall_df(strat_df, full_df, sub)
        final_df = get_final_psifr_df(fr_data, strat_df, sub)
        full_fr_df = full_fr_df.append(fr_data)
        full_merged_df = full_merged_df.append(final_df)
    return full_fr_df, full_merged_df
