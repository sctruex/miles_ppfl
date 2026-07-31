import matplotlib.pyplot as plt
import numpy as np
from os.path import exists as path_exists
from os import makedirs
from pandas import DataFrame, read_csv, concat
from typing import Any, Union, Mapping, Hashable, Callable
from itertools import product


def save_plot(path: str, file: str) -> None:
    if not path_exists(path):
        makedirs(path)
    plt.savefig(path+file)
    plt.close()


def make_plot(y: list[list[Union[int, float]]],
              labels: list[str], axis_label: str,
              path: str = "", file: str = "", title: str = "",
              loss_max: Union[int, float] = 100
              ) -> None:
    if labels is None:
        labels = [axis_label for _ in y]
    # NOTE: relies on loss label always being set to "loss"
    if axis_label == "loss" and max([max(y[m]) for m in range(len(y))]) > loss_max:
        plt.ylim(top=loss_max)
    for m in range(len(y)):
        plt.plot(range(len(y[m])), y[m], label=labels[m])
        plt.plot(range(len(y[m])), y[m], label=labels[m])
    plt.xlabel('Rounds')
    plt.ylabel(axis_label)
    plt.legend()
    plt.title(title)
    plt.title(title)
    if path != "":
        save_plot(path, file)


def select_final_itrs(exp_df: DataFrame,
                      priv_params: list[Union[int, float]],
                      priv_param_colname: str = "priv_param"):
    # NOTE: requires the privacy parameter column to be called priv_param
    keep_rows = [
        exp_df[exp_df[priv_param_colname] == sigma].iloc[-1] for sigma in priv_params
    ]
    exp_df = DataFrame(keep_rows)
    return exp_df


def plot_all_rows(results_df: DataFrame, exp_var_lsts: dict,
                  collected_metrics: list[str], plot_name: str) -> None:
    dataset_name = plot_name.split("_")[0]

    for run_num in np.unique(results_df["run#"]):
        for exp_settings in product(*exp_var_lsts.values()):
            exp_df = results_df[results_df["run#"] == run_num]
            varmap_str = ""
            for i, key in zip(range(len(exp_var_lsts)), exp_var_lsts.keys()):
                # always keep privacy parameter fluctuation on the same graph
                if key != "priv_param":
                    val = exp_settings[i]
                    exp_df = exp_df[exp_df[key] == val]
                    varmap_str += f"_{key}:{val}"
            if len(exp_df) > 0:
                priv_params = np.unique(
                    exp_df["priv_param"]) if "priv_param" in exp_df.columns else None
                if len(exp_df) > len(priv_params):
                    exp_df = select_final_itrs(
                        exp_df=exp_df, priv_params=priv_params)
                loss_lsts, additional_metrics_lsts = get_metrics_lsts(
                    exp_df, collected_metrics)
                loss_lsts = [loss[~np.isnan(loss)] for loss in loss_lsts]
                additional_metrics_lsts = [[
                    addl_metric[~np.isnan(addl_metric)] for addl_metric in additional_metrics]
                    for additional_metrics in additional_metrics_lsts]

                # remove lengthy param names
                print("varmap_str", varmap_str)
                if "total_clients" in varmap_str:
                    varmap_str = varmap_str.replace("total_clients", "nclients")
                if "client_opt_params/momentum" in varmap_str:
                    varmap_str = varmap_str.replace(
                        "client_opt_params/momentum", "opt_momentum")
                if "lr_schedule_params/initial_learning_rate" in varmap_str:
                    varmap_str = varmap_str.replace(
                        "lr_schedule_params/initial_learning_rate", "init_lr")

                varmap_str = varmap_str.replace("/", ":")
                varmap_str = varmap_str.replace(".", "-")
                make_plot(loss_lsts, priv_params, "loss",
                          path=f"results/individual_runs/",
                          file=f"loss_run{run_num}{varmap_str}_{dataset_name}")
                for additional_metrics, metric_name in zip(additional_metrics_lsts,
                                                           collected_metrics):
                    make_plot(additional_metrics, priv_params, metric_name,
                              path=f"results/individual_runs/",
                              file=f"{metric_name}_run{run_num}{varmap_str}_{dataset_name}")


def plot_experimental_results(results_df: DataFrame, exp_var_lsts: dict, plot_avgs: bool,
                              plot_name: str, collected_metrics: list[str]
                              ) -> None:
    # average results for the key across all other keys
    # for all runs as well as
    # across all other keys for each individual run
    if plot_avgs:
        for key in exp_var_lsts:
            for run_val in np.unique(results_df["run#"]):
                run_df = results_df[results_df['run#'] == run_val]
                plot_avg_results(
                    run_df, key, exp_var_lsts[key], f"{run_val}_{plot_name}", collected_metrics)
            plot_avg_results(
                results_df, key, exp_var_lsts[key], plot_name, collected_metrics)
    # plot each run individually
    else:
        print("plotting individuals")
        plot_all_rows(results_df, exp_var_lsts, collected_metrics, plot_name)


def plot_avg_results(results_df: DataFrame, key: str, var_lst: list,
                     plot_name: str, collected_metrics: list[str]) -> None:
    avg_additional_metrics, avg_loss = [[] for _ in collected_metrics], []
    for val in var_lst:
        loss, additional_metrics = create_average_metrics(
            results_df, key, val, collected_metrics)
        avg_loss.append(loss)
        for i in range(len(collected_metrics)):
            avg_additional_metrics[i].append(additional_metrics[i])

    key = key.replace("/", ":")
    make_plot(avg_loss, var_lst, "loss",
              path=f"results/{key}/", file=f"Avg_loss_" + plot_name.replace(key, ""),
              title=key)
    for i in range(len(collected_metrics)):
        name = collected_metrics[i]
        avg_metrics = avg_additional_metrics[i]
        make_plot(avg_metrics, var_lst, name,
                  path=f"results/{key}/", file=f"Avg_{name}_" + plot_name.replace(key, ""),
                  title=key)


def get_metrics_lsts(df: DataFrame, collected_metrics: list[str],
                     loss_name: str = "sparse_categorical_crossentropy"
                     ) -> tuple[list[Union[int, float], list[list[Union[int, float]]]]]:
    """NOTE:probably want more loss flexibility here"""
    loss_lst = np.array([np.array(df[loss_name].iloc[i])
                        for i in range(len(df))])
    additional_metrics_lsts = [np.array([np.array(df[name].iloc[i])
                                         for i in range(len(df))]) for name in collected_metrics]
    return loss_lst, additional_metrics_lsts


def average_metrics(loss_lst: list[Union[int, float]],
                    additional_metrics_lsts: list[list[Union[int, float]]]
                    ) -> tuple[list[Union[int, float], list[list[Union[int, float]]]]]:
    loss_avg = np.nanmean(loss_lst, axis=0)
    additional_metrics_avgs = [np.nanmean(
        metric_lst, axis=0) for metric_lst in additional_metrics_lsts]
    return loss_avg, additional_metrics_avgs


def create_average_metrics(results_df: DataFrame, key: str, val: Any,
                           collected_metrics: list[str]
                           ) -> tuple[list[Union[int, float], Union[int, float]]]:
    df_mean = results_df[results_df[key] == val]
    loss_lst, additional_metrics_lsts = get_metrics_lsts(
        df_mean, collected_metrics)
    loss_avg, additional_metrics_avgs = average_metrics(
        loss_lst, additional_metrics_lsts)
    return loss_avg, additional_metrics_avgs


def create_average_metrics(results_df: DataFrame, key: str, val: Any,
                           collected_metrics: list[str]
                           ) -> tuple[list[Union[int, float], Union[int, float]]]:
    df_mean = results_df[results_df[key] == val]
    loss_lst, additional_metrics_lsts = get_metrics_lsts(
        df_mean, collected_metrics)
    loss_avg, additional_metrics_avgs = average_metrics(
        loss_lst, additional_metrics_lsts)
    loss_avg = loss_avg[~np.isnan(loss_avg)]
    additional_metrics_avgs = [metric_avg[~np.isnan(
        metric_avg)] for metric_avg in additional_metrics_avgs]

    return loss_avg, additional_metrics_avgs


def col_to_float_lst(s: str):
    return [float(x) for x in s.strip('[]').split(',')]


def plot_from_csv(filepath: str,
                  selection_tuples: list[tuple],
                  converters: Mapping[Hashable, Callable],
                  exp_var_lsts: dict,
                  plot_avgs: bool,
                  plot_name: str,
                  collected_metrics: list[str],
                  top_only=False):
    df = read_csv(filepath, converters=converters)

    for col, val in selection_tuples:
        df = df[df[col] == val]

    if not top_only:
        plot_experimental_results(results_df=df,
                                  exp_var_lsts=exp_var_lsts,
                                  plot_avgs=plot_avgs,
                                  plot_name=plot_name,
                                  collected_metrics=collected_metrics)

    accuracies = [[val for val in accs if val != float(
        "nan")][-1] for accs in df["sparse_categorical_accuracy"]]
    losses = [[val for val in accs if val != float(
        "nan")][-1] for accs in df["sparse_categorical_crossentropy"]]
    top3_rows = np.argsort(accuracies)[-5:][::-1]
    for i in top3_rows:
        print(df.iloc[i])
        print(accuracies[i])
        print(losses[i])
        print()