from fl_exp import run_fl_exp
from plot_utils.plot import plot_experimental_results
from experimental_arguments import ExperimentalArguments
import sys
import os
from itertools import product
import pandas as pd
import gc
from ray.util.multiprocessing import Pool
import tensorflow as tf
import numpy as np

def redirect_output(debug: bool, dataset_name: str, exp_var_map: dict) -> None:

    # debugging
    debug_prefix = "debug_" if debug else "exp_"

    # file names specific to experiment
    results_name = f"{debug_prefix}{dataset_name}"
    for key in exp_var_map:
        temp = key
        # shortens output file name
        if key == "total_clients":
            temp = "nclients"
        if key == "client_opt_params/momentum":
            temp = "opt_momentum"
        if key == "lr_schedule_params/initial_learning_rate":
            temp = "init_lr"
        if key == "lr_schedule_params/power":
            temp = "lr_pow"
        results_name += f"_{temp}:{exp_var_map[key]}"
    results_name = results_name.replace(".", "-")
    results_name = results_name.replace("/", ":")
    # redirect print output to file
    results = open('output/' + results_name + '.log', 'w')
    sys.stdout = results
    errs = open('output/' + results_name + '.err', 'w')
    sys.stderr = errs


def redirect_and_run(exp_var_map, exp_args: ExperimentalArguments, seed: int
                     ) -> pd.DataFrame:
    redirect_output(exp_args._debug, exp_args.get_arg(
        "dataset_name"), exp_var_map)

    # set SEEDS for tf and np
    tf.random.set_seed(seed)

    result_row = run_fl_exp(**(exp_args.get_full_args(seed)))
    gc.collect()
    return result_row


def gen_lst(key: str, vals: str) -> list:
    if "/" in key:
        key = key.split("/")[1]
    return [ExperimentalArguments.get_var_from_string(key.strip(), val.strip())
            for val in vals.split(" ")]


def gen_experimental_args() -> tuple[ExperimentalArguments,
                                     list[tuple],
                                     dict[str, list],
                                     list[str],
                                     dict,
                                     int,
                                     bool]:
    experimental_arguments_dict = {}
    exp_vars_val, seed, plot_avgs = "", None, False
    for x in sys.stdin:
        varname, varval = x.split("=")
        if varname != "exp_vars":
            varval = ExperimentalArguments.get_var_from_string(
                varname.strip(), varval.strip())
            if varname == "seed":
                seed = varval
            elif varname == "plot_avgs":
                plot_avgs = varval
            else:
                experimental_arguments_dict[varname] = varval
        else:
            exp_vars_val = varval
    exp_var_lsts = [exp_lst.split(":") for exp_lst in exp_vars_val.split(",")]
    exp_var_lsts = {key: gen_lst(key, vals) for key, vals in exp_var_lsts}
    exp_var_keys = list(exp_var_lsts.keys())
    experimental_arguments_dict["exp_var_map"] = {
        key: exp_var_lsts[key][0] for key in exp_var_keys}

    exp_args = ExperimentalArguments(**experimental_arguments_dict)
    exp_var_map = exp_args.get_arg("exp_var_map")

    experimental_combos = list(product(*list(exp_var_lsts.values())))
    return (exp_args, experimental_combos, exp_var_lsts, exp_var_keys, exp_var_map,
            seed, plot_avgs)


def update_exp_args(exp_val_settings: tuple, exp_var_keys: list[str],
                    exp_var_map: dict, exp_args: ExperimentalArguments
                    ) -> tuple[dict, ExperimentalArguments]:
    for i in range(len(exp_var_keys)):
        key = exp_var_keys[i]
        val = exp_val_settings[i]
        exp_var_map[key] = val
        exp_args.update_arg(key, val)
    return exp_var_map, exp_args


def plot_exp(exp_args: ExperimentalArguments, results_df: pd.DataFrame,
             exp_var_lsts: dict, plot_avgs: bool,
             suffix: str = "") -> None:
    dataset_name = exp_args.get_arg("dataset_name")
    results_fname = "_".join(exp_var_lsts.keys()).replace("/", ":")

    if "total_clients" in results_fname:
        results_fname = results_fname.replace("total_clients", "nclients")
    if "client_opt_params:momentum" in results_fname:
        results_fname = results_fname.replace("client_opt_params:momentum", "opt_momentum")
    if "lr_schedule_params:initial_learning_rate" in results_fname:
        results_fname = results_fname.replace("lr_schedule_params:initial_learning_rate", "init_lr")

    results_fname = dataset_name + "_" + results_fname
    results_df.to_csv(results_fname + "_full_results_df" +
                      suffix+".csv", index=False)

    plot_experimental_results(results_df, exp_var_lsts, plot_avgs, results_fname,
                              exp_args.get_collected_metrics())


def multi_main(n_processes: int = 40, suffix: str = "", existing: str = "", pattern: str = ".csv") -> None:
    pool = Pool(processes=n_processes)
    results = []

    args = gen_experimental_args()
    exp_args, experimental_combos, exp_var_lsts, exp_var_keys, exp_var_map, seed, plot_avgs = args

    # create list containing num_runs seeds
    rng = np.random.default_rng(seed=seed if seed != 0 else None)
    seeds = rng.integers(2**32, size=exp_args.num_runs)

    rows = []
    for exp_val_settings in experimental_combos:
        exp_var_map, exp_args = update_exp_args(
            exp_val_settings, exp_var_keys, exp_var_map, exp_args)
        for run in range(exp_args.num_runs):
            exp_var_map["run#"] = run + 1
            res = pool.apply_async(redirect_and_run, (exp_var_map, exp_args, seeds[run]))
            results.append(res)
    pool.close()
    pool.join()

    for res in results:
        try:
            row = res.get()
        except Exception as err:
            print(f"failed to get result due to error: {err}")
        else:
            rows.append(row)
    results_df = pd.concat(rows, ignore_index=True).reset_index(drop=True)
    print(results_df.to_string())
    plot_exp(exp_args, results_df, exp_var_lsts, plot_avgs, suffix=suffix)


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    existing = str(sys.argv[3]) if len(sys.argv) > 3 else ""
    pattern = str(sys.argv[4]) if len(sys.argv) > 4 else ".csv"
    suffix = str(sys.argv[5]) if len(sys.argv) > 5 else ""
    multi_main(n, suffix=suffix, existing=existing, pattern=pattern)
