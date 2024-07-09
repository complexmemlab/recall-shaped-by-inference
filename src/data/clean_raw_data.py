import os
import pandas as pd
import argparse
from typing import List
from src.data.process_strat import (
    first_pass_counts,
    get_strat_df,
    make_disc_dict,
    add_trial_nums_and_runs,
    label_strat_blocks,
    label_item_relative_to_index,
    process_dataframe,
)
from src.data.process_mem import get_fr_df_for_all_subs


def load_data(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, dtype={"sona_id": str})


def get_math_df(df: pd.DataFrame) -> pd.DataFrame:
    math_trials = df[df["stimulus"].str.contains("Does")]

    math_trials["eq"] = math_trials.stimulus.apply(lambda x: x[5:-1])

    def is_correct_equation(equation):
        left, right = equation.split("=")

        iseq = eval(left) == int(right)
        if iseq:
            return "f"
        else:
            return "j"

    math_trials["correct_math_ans"] = math_trials["eq"].apply(
        lambda x: is_correct_equation(x)
    )

    math_trials["correct_math"] = (
        math_trials["response"] == math_trials["correct_math_ans"]
    )
    math_trials = math_trials.dropna(subset="rt")

    math_df = math_trials.groupby("sona_id")["correct_math"].mean().reset_index()
    return math_df


def get_score_df(strat_df: pd.DataFrame, math_df: pd.DataFrame) -> pd.DataFrame:
    score_df = strat_df.groupby("sona_id")[["points", "rt"]].mean().reset_index()
    score_df = score_df.merge(math_df, on="sona_id")
    return score_df


def get_bad_sublist(score_df: pd.DataFrame) -> List[str]:
    thresholded = score_df[
        (score_df["points"] > 0.5)
        & (score_df["rt"] > 400)
        & (score_df["correct_math"] >= 0.5)
    ]  # include subject as long as average score is above 0.5 and average rt is above 400ms
    print(f"Sample size before removal: {score_df['sona_id'].nunique()}")
    good_people = set(thresholded["sona_id"].values)
    print(f"Sample size after removal: {len(good_people)}")
    bad_people = set(score_df["sona_id"].values) - good_people

    math_removal = set(score_df[score_df["correct_math"] < 0.5]["sona_id"].values)
    print("\n")
    print(
        f"Number of people removed for worse than chance performance on encoding: {len(bad_people) - len(math_removal)}"
    )
    print(
        f"Number of people removed for worse than chance performance on distractor: {len(math_removal)}"
    )
    return list(bad_people)


def remove_bad_sub(df: pd.DataFrame, bad_people: List[str]) -> pd.DataFrame:
    return df[~df["sona_id"].isin(bad_people)]


def save_clean_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path where the raw csv is stored",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the dataframes should be saved to",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    assert os.path.exists(input_path), "Input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"
    df = load_data(input_path)
    df = first_pass_counts(df)
    pre_math = df.dropna(subset=["stimulus"])
    math_df = get_math_df(pre_math)
    strat_df = get_strat_df(df)
    strat_df = add_trial_nums_and_runs(strat_df)
    labelled_strat_df = (
        strat_df.groupby("sona_id")
        .apply(
            lambda x: label_item_relative_to_index(label_strat_blocks(x))
        )  # this line stopped working for experiment 2 for reasons unknown
        .reset_index(drop=True)
    )
    disc_dict = make_disc_dict(labelled_strat_df)

    labelled_strat_df = process_dataframe(labelled_strat_df, disc_dict)
    fr_df, recall_df = get_fr_df_for_all_subs(labelled_strat_df, df)
    score_df = get_score_df(labelled_strat_df, math_df)
    bad_people = get_bad_sublist(score_df)

    labelled_strat_df = remove_bad_sub(labelled_strat_df, bad_people)
    fr_df["sona_id"] = fr_df["subject"]  # psifr requires the col to be named 'subject'
    recall_df["sona_id"] = recall_df["subject"]
    fr_df = remove_bad_sub(fr_df, bad_people)
    recall_df = remove_bad_sub(recall_df, bad_people)
    save_clean_data(labelled_strat_df, os.path.join(output_path, "strat_df.csv"))
    save_clean_data(recall_df, os.path.join(output_path, "mem_df.csv"))
    save_clean_data(fr_df, os.path.join(output_path, "fr_df.csv"))


if __name__ == "__main__":
    main()
