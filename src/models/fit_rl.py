import numpy as np
from scipy.optimize import minimize
import pandas as pd
from src.models.RL_constants import MODELS
import multiprocessing as mp
import argparse
import os
import time


# fit all subjects
def _MAP_fit(params, data, model, prior_distributions):
    log_likelihoods = []
    for i, param in enumerate(params):
        log_likelihoods.append(np.log(prior_distributions[i].pdf(param)))

    curr_model = model(*params, data)
    curr_model.fit()

    return curr_model.loglik + sum(log_likelihoods)


def fit_subject(args):
    data, worker_id, model, prior_distributions, bounds = args
    # get data for this subject
    data_sub = data[data["sona_id"] == worker_id]

    def handle(x):
        return -_MAP_fit(x, data_sub, model, prior_distributions)

    best_res = None
    for _ in range(10):
        initial_params = [np.random.uniform(low, high) for low, high in bounds]

        res = minimize(handle, initial_params, bounds=bounds)

        if best_res is None or res.fun < best_res.fun:
            best_res = res
    # save
    result = [worker_id] + list(best_res.x) + [-best_res.fun]
    return result


def parallel_fit_all_subjects(data, model, prior_distributions, bounds):
    # initialize
    num_params = len(bounds)

    # prepare arguments for each process
    args = [
        (data, worker_id, model, prior_distributions, bounds)
        for worker_id in data["sona_id"].unique()
    ]

    # create a pool of processes and map the function over the arguments
    with mp.Pool(8) as pool:
        results = pool.map(fit_subject, args)

    # make a nice dataframe
    columns = ["sona_id"] + [f"param_{i}" for i in range(num_params)] + ["loglik"]
    return pd.DataFrame(results, columns=columns)


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path where the raw transcripts are stored",
    )
    parser.add_argument("-m", "--model", required=True, help="which model to fit"),
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the models should be saved to",
    )
    parser.add_argument(
        "-r",
        "--no_resp_st",
        action="store_true",
        help="Whether to not include resp_st in the model",
        default=False,
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model.lower()
    resp_st = not args.no_resp_st
    assert model_name in MODELS.keys(), "Model not recognized!"
    assert os.path.exists(input_path), "Input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"
    df = pd.read_csv(input_path)
    model, prior_distributions, bounds, modelname, parameter_names = MODELS[model_name]
    if not resp_st:
        # the last bound is always stickiness so just force it to be (1,1)
        bounds[-1] = (1, 1)
        modelname += "_no_resp_st"
    results = parallel_fit_all_subjects(df, model, prior_distributions, bounds)

    # Add aic column to results
    if resp_st:
        results["aic"] = -2 * results["loglik"] + 2 * len(bounds)
    else:
        results["aic"] = -2 * results["loglik"] + 2 * (len(bounds) - 1)
    # Create a mapping dictionary
    rename_dict = {f"param_{i}": name for i, name in enumerate(parameter_names)}

    # Rename the DataFrame columns
    results.rename(columns=rename_dict, inplace=True)
    results.to_csv(os.path.join(output_path, f"{modelname}_results.csv"), index=False)
    print(f"Finished fitting {modelname} in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
