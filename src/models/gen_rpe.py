# this code will combine the fitted model values for a given participant with the trials they experienced during encoding to get an RPE estimate for each trial

import pandas as pd
import argparse
import os
import time
from src.models.RL_constants import MODELS
from src.models.feature_rl import FeatureRL, DecayFeatureRL
from typing import List, Tuple, Type, Union


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        input_path (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(input_path)
    return data


def get_single_subject_data(data: pd.DataFrame, sub: str) -> pd.DataFrame:
    """
    Get data for a single subject.

    Args:
        data (pd.DataFrame): The data containing all subjects.
        sub (str): The subject ID to extract.

    Returns:
        pd.DataFrame: The data for the specified subject.
    """
    data_sub = data[data["sona_id"] == sub]
    return data_sub


def instantiate_model(
    model: Union[Type[FeatureRL], Type[DecayFeatureRL]],
    subject_data: pd.DataFrame,
    subject_params: pd.DataFrame,
) -> Union[FeatureRL, DecayFeatureRL]:
    """
    Instantiate a model with the given parameters and data.

    Args:
        model (Union[Type[FeatureRL], Type[DecayFeatureRL]]): The model class to instantiate.
        subject_data (pd.DataFrame): The data for the subject.
        subject_params (pd.DataFrame): The parameters for the subject.

    Returns:
        Union[FeatureRL, DecayFeatureRL]: The instantiated model.
    """
    # param_cols = [col for col in subject_params.columns if "param" in col]
    param_cols = subject_params.columns[1:-2]
    params = subject_params[param_cols].values[0]
    curr_model = model(*params, subject_data)
    return curr_model


def fit_model_to_record_RPE_LL_and_uncertainty(
    model: Union[Type[FeatureRL], Type[DecayFeatureRL]],
    subject_data: pd.DataFrame,
    subject_params: pd.DataFrame,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Fit the model to the subject data and record the RPE, log likelihood, and uncertainty.

    Args:
        model (Union[Type[FeatureRL], Type[DecayFeatureRL]]): The model class to fit.
        subject_data (pd.DataFrame): The data for the subject.
        subject_params (pd.DataFrame): The parameters for the subject.

    Returns:
        Tuple[List[float], List[float], List[float]]: The RPE, log likelihood, and uncertainty.
    """
    curr_model = instantiate_model(model, subject_data, subject_params)
    curr_model.fit()
    return curr_model.rpe, curr_model.trial_by_trial_loglik, curr_model.uncertainties


def combine_RPE_LL_and_uncertainty_with_sub_strat_data(
    sub_strat_data: pd.DataFrame,
    rpe: List[float],
    ll: List[float],
    uncertainty: List[float],
) -> pd.DataFrame:
    """
    Recombines the RPE, log likelihood, and uncertainty with the subject data.

    Args:
        sub_strat_data (pd.DataFrame): The subject data.
        rpe (List[float]): The RPE values.
        ll (List[float]): The log likelihood values.
        uncertainty (List[float]): The uncertainty values.

    Returns:
        pd.DataFrame: The subject data with the RPE, log likelihood, and uncertainty.
    """

    sub_strat_data["rpe"] = rpe
    sub_strat_data["trial_by_trial_loglik"] = ll
    sub_strat_data["uncertainty"] = -uncertainty
    return sub_strat_data


def single_subj_proc(
    model: Union[Type[FeatureRL], Type[DecayFeatureRL]],
    strat_data: pd.DataFrame,
    fit_data: pd.DataFrame,
    sub: str,
) -> pd.DataFrame:
    """
    Runs the fitting process for a single subject.

    Args:
        model (Union[Type[FeatureRL], Type[DecayFeatureRL]): The model class to fit.
        strat_data (pd.DataFrame): The strat data.
        fit_data (pd.DataFrame): The fit data.
        sub (str): The subject ID.

    Returns:
        pd.DataFrame: The subject data with the RPE, log likelihood, and uncertainty.
    """
    sub_strat_data = get_single_subject_data(strat_data, sub)
    sub_fit_data = get_single_subject_data(fit_data, sub)
    rpe, ll, uncertainty = fit_model_to_record_RPE_LL_and_uncertainty(
        model, sub_strat_data, sub_fit_data
    )
    sub_strat_data = combine_RPE_LL_and_uncertainty_with_sub_strat_data(
        sub_strat_data, rpe, ll, uncertainty
    )
    return sub_strat_data


def fit_model_to_all_subjects(
    model: Union[Type[FeatureRL], Type[DecayFeatureRL]],
    strat_data: pd.DataFrame,
    fit_data: pd.DataFrame,
) -> List[pd.DataFrame]:
    subs = strat_data["sona_id"].unique()
    args = [(model, strat_data, fit_data, sub) for sub in subs]

    results = [single_subj_proc(*arg) for arg in args]
    return results


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-f", "--fit_input_path", required=True, help="path to the fit data"
    )
    parser.add_argument(
        "-s", "--strat_input_path", required=True, help="path to the strat data"
    ),
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the new strat data will be saved",
    )
    args = parser.parse_args()
    fit_input_path = args.fit_input_path
    strat_input_path = args.strat_input_path
    output_path = args.output_path
    assert os.path.exists(fit_input_path), "fit input path does not exist!"
    assert os.path.exists(strat_input_path), "strat input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"

    fit_data = load_data(fit_input_path)
    strat_data = load_data(strat_input_path)

    # model name should be extracted from the fit_input_path the first part of the filename is the modelname
    model_name = fit_input_path.split("/")[-1].split("_")[0]

    model_dict = MODELS

    assert model_name.lower() in model_dict.keys(), "Model not recognized!"

    model = model_dict[model_name.lower()][0]

    strat_data_with_rpe = fit_model_to_all_subjects(model, strat_data, fit_data)
    # turn strat_data back into a dataframe
    strat_data_with_rpe_df = pd.concat(strat_data_with_rpe)
    output_filename = f"{model_name}_strat_data_rpe.csv"
    output_path = os.path.join(output_path, output_filename)
    strat_data_with_rpe_df.to_csv(output_path, index=False)
    print(f"Finished in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
