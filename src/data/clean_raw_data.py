import os
import pandas as pd
import argparse
from typing import List, Set
from src.data.process_strat import (
    preproc_df,
    process_strat,
    get_prop_list,
)
from src.data.process_mem import get_fr_df_for_all_subs


def load_data(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, dtype={"subid": str, "subject": str})


def get_math_df(df: pd.DataFrame) -> pd.DataFrame:
    math_trials = df[df["stimulus"].str.contains("Does")].copy()

    math_trials.loc[:, "eq"] = math_trials.loc[:, "stimulus"].apply(lambda x: x[5:-1])

    def is_correct_equation(equation):
        left, right = equation.split("=")

        iseq = eval(left) == int(right)
        return "f" if iseq else "j"

    math_trials.loc[:, "correct_math_ans"] = math_trials.loc[:, "eq"].apply(
        lambda x: is_correct_equation(x)
    )

    math_trials.loc[:, "correct_math"] = (
        math_trials.loc[:, "response"] == math_trials.loc[:, "correct_math_ans"]
    )
    math_trials = math_trials.dropna(subset=["rt"])

    math_df = math_trials.groupby("subid")["correct_math"].mean().reset_index()
    return math_df


def get_score_df(strat_df: pd.DataFrame, math_df: pd.DataFrame) -> pd.DataFrame:
    score_df = strat_df.groupby("subid")[["points", "rt"]].mean().reset_index()
    score_df = score_df.merge(math_df, on="subid")
    return score_df


def find_bad_sub_mem(recall_df: pd.DataFrame) -> Set[str]:
    recall_df["recalled"] = recall_df["recall"] & recall_df["study"]
    recalls = recall_df.groupby(["subject", "list"])["recalled"].sum()

    bad_subs = set(recalls[recalls < 1].reset_index()["subject"])
    return bad_subs


# the get_bad_sublist function will depend on the study
def get_bad_sublist(
    labelled_strat_df: pd.DataFrame,
    math_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    exp: int,
) -> List[str]:
    math_thresh = 0
    score_df = get_score_df(labelled_strat_df, math_df)
    threshold_for_points = score_df[score_df["points"] < 0.5]["subid"].values
    threshold_for_rt = score_df[score_df["rt"] < 0]["subid"].values
    threshold_for_math = score_df[score_df["correct_math"] < math_thresh][
        "subid"
    ].values
    print(f"Sample size before removal: {score_df['subid'].nunique()}")
    bad_people = (
        set(threshold_for_points) | set(threshold_for_rt) | set(threshold_for_math)
    )
    # print the number of people removed uniquely for each condition (points, math, rt, proportion) making sure there is no overlap between conditions using set operations
    bad_mem = find_bad_sub_mem(recall_df)
    bad_mem = bad_mem - bad_people
    bad_people.update(bad_mem)
    prop_bad = get_prop_list(labelled_strat_df)
    # get only the unique ones not in threshold_for_points, threshold_for_rt, threshold_for_math
    prop_bad = (
        prop_bad
        - set(threshold_for_points)
        - set(threshold_for_rt)
        - set(threshold_for_math)
        - set(bad_mem)
    )
    bad_people.update(prop_bad)

    print(f"Number of people removed for points: {len(threshold_for_points)}")
    print(f"Number of people removed for rt: {len(threshold_for_rt)}")
    print(f"Number of people removed for math: {len(threshold_for_math)}")
    print(f"Number of people removed for proportion: {len(prop_bad)}")
    print(f"Number of people removed for memory: {len(bad_mem)}")
    print(f"Sample size after removal: {score_df['subid'].nunique() - len(bad_people)}")
    return bad_people


def remove_bad_sub(df: pd.DataFrame, bad_people: List[str]) -> pd.DataFrame:
    return df[~df["subid"].isin(bad_people)].reset_index(drop=True)


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
    parser.add_argument("-e", "--exp", required=True, type=int)
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the dataframes should be saved to",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    exp = args.exp
    assert os.path.exists(input_path), "Input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"
    df = load_data(input_path)
    n_completed = sum(
        (df.groupby(["PROLIFIC_PID"]).points.count().reset_index())["points"] >= 224
    )
    print(f"Number of people completed the experiment: {n_completed}")
    df = preproc_df(df, n_trials=224)
    pre_math = df.dropna(subset=["stimulus"])
    math_df = get_math_df(pre_math)
    # getting the labelled strat df
    labelled_strat_df = process_strat(df, nruns=4)
    # getting the fr and recall df
    fr_df, recall_df = get_fr_df_for_all_subs(labelled_strat_df, df)
    # getting the bad sublist
    bad_people = get_bad_sublist(labelled_strat_df, math_df, recall_df, exp)

    # removing bad subjects
    labelled_strat_df = remove_bad_sub(labelled_strat_df, bad_people)
    fr_df = remove_bad_sub(fr_df, bad_people)
    recall_df = remove_bad_sub(recall_df, bad_people)
    # saving the data
    save_clean_data(labelled_strat_df, os.path.join(output_path, "strat_df.csv"))
    save_clean_data(recall_df, os.path.join(output_path, "mem_df.csv"))
    save_clean_data(fr_df, os.path.join(output_path, "fr_df.csv"))


if __name__ == "__main__":
    main()
