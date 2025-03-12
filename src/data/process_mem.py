import numpy as np
import pandas as pd
from psifr import fr
from nltk.metrics.distance import edit_distance
from nltk.corpus import words

english_words = set(words.words())
english_words = {word.upper() for word in english_words}


def filter_mem_df(mem, cleaned_exp_df):
    mem = mem[~mem["block"].isnull()].reset_index(drop=True)
    mem = mem[mem["subid"].isin(cleaned_exp_df["subid"].unique())].reset_index(
        drop=True
    )
    return mem


def get_encoding_subset(strat_df, sub):
    sub_strat_data = strat_df[strat_df["subid"] == sub].reset_index(drop=True)
    sub_strat_data["word"] = sub_strat_data["word"].str.lower()
    encoding_subset = sub_strat_data[
        ["word", "corr_rule_numeric", "rel_subj_boundary", "block", "run", "points"]
    ]
    return encoding_subset


def correct_response(response, word_list):
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


def spellcheck_and_filter(recall_run, encoding_subset):
    word_list = set(encoding_subset["word"].str.lower().values)
    responses = list(set(recall_run["response"]))
    responses = [response for response in responses if isinstance(response, str)]

    corrected_responses = [
        correct_response(response, word_list) for response in responses
    ]
    correction_dict = dict(zip(responses, corrected_responses))

    # Create a copy of recall_run to avoid SettingWithCopyWarning
    recall_run = recall_run.copy()

    # Apply the correction to the recall_df
    recall_run["corrected_response"] = recall_run["response"].map(correction_dict)

    # Filter out 'No Match' rows
    recall_run = recall_run[recall_run["corrected_response"] != "No Match"]

    return recall_run


def make_recall_df(strat_df, full_df, sub):
    sub_data = full_df[full_df["subid"] == sub].reset_index(drop=True)
    encoding_subset = get_encoding_subset(strat_df, sub)

    # Create a copy of sub_data to avoid SettingWithCopyWarning
    recall_trials = (
        sub_data[sub_data["trial_type"] == "html-free-recall"]
        .copy()
        .reset_index(drop=True)
    )

    recall_trials["response"] = recall_trials["response"].str.lower()
    recall_trials["response"] = recall_trials["response"].str.strip()
    recall_end_of_run_idxs = recall_trials[recall_trials["button"] == "pressed"].index
    recall_run_1 = recall_trials.loc[0 : recall_end_of_run_idxs[0]].copy()
    recall_run_2 = recall_trials.loc[
        recall_end_of_run_idxs[0] : recall_end_of_run_idxs[1]
    ].copy()

    recall_run_3 = recall_trials.loc[
        recall_end_of_run_idxs[1] : recall_end_of_run_idxs[2]
    ].copy()
    recall_run_4 = recall_trials.loc[recall_end_of_run_idxs[2] :].copy()
    recall_runs = [recall_run_1, recall_run_2, recall_run_3, recall_run_4]
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


def get_final_psifr_df(fr_data, strat_df, sub):
    encoding_subset = get_encoding_subset(strat_df, sub)
    merged = fr.merge_free_recall(fr_data)
    merged = merged.merge(
        encoding_subset, left_on=["item"], right_on=["word"], how="left"
    )
    return merged


def get_fr_df_for_all_subs(strat_df, full_df):
    full_fr_df_list = []
    full_merged_df_list = []
    for sub in strat_df.subid.unique():
        fr_data = make_recall_df(strat_df, full_df, sub)
        final_df = get_final_psifr_df(fr_data, strat_df, sub)
        full_fr_df_list.append(fr_data)
        full_merged_df_list.append(final_df)

    full_fr_df = pd.concat(full_fr_df_list, ignore_index=True)
    full_merged_df = pd.concat(full_merged_df_list, ignore_index=True)
    full_fr_df["subid"] = full_fr_df["subject"]
    full_merged_df["subid"] = full_merged_df["subject"]
    return full_fr_df, full_merged_df


# the following functions are not currently being used
def get_good_block_idxs(mem):
    bound_and_after_idx = np.where(
        (mem["rel_subj_boundary"] <= 0) & mem["prev_good_block"] == 1
    )
    bound_and_before_idx = np.where(
        (mem["rel_subj_boundary"] > 0) & mem["good_block"] == 1
    )
    lure_idx = np.where(mem["type"] == "lure")
    return np.concatenate(
        (bound_and_after_idx[0], bound_and_before_idx[0], lure_idx[0])
    )


def filter_by_good_block_idxs(mem):
    idxs = get_good_block_idxs(mem)
    return mem.iloc[idxs].reset_index(drop=True)
