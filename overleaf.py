
from shutil import copy2
from os import listdir
from os.path import isfile
from itertools import product
from typing import Mapping, Hashable, Callable
from pandas import read_csv, concat, DataFrame


def read_in_results(results_loc: str,
                    pattern: str = ".csv",
                    ignore_pattern: str = "",
                    converters: Mapping[Hashable, Callable] = {},
                    existing: list[str] = [],
                    results: list[DataFrame] = []) -> DataFrame:
    """
    read in data from files matching specified pattern (ex: csv) within results_loc.
    files containing ignore_pattern will be disregarded when specified.

    Args:
        results_loc (str):              folder containing results to read in
        pattern (str, optional):        only read in files with names containing specified pattern. 
                                        Defaults to ".csv".
        ignore_pattern (str, optional): when specified, files with names containing pattern 
                                        will not be read into results. 
                                        Defaults to "".
        converters (Mapping[Hashable, Callable], optional): 
                                        column converters to create dataframe from csv. 
                                        Defaults to {}.
        existing (list[str], optional): list of strings specifying results already read in.
                                        used for error message when files cannot be read in.
                                        Defaults to [].
        results (list, optional):       list of data frames already read in.
                                        will be concatenated with results in folder.
                                        Defaults to [].

    Returns:
        DataFrame: complete data frame included results in results as well as those in results_loc
    """
    files = [file for file in listdir(
        results_loc) if pattern in file and (not ignore_pattern or ignore_pattern not in file)]

    for file in files:
        try:
            curr_df = read_csv(f"{results_loc}/{file}", converters=converters)
        except Exception as e:
            print(
                f"Error {e} reading in data frame in {file}. Results not considered.")
        else:
            results.append(curr_df)

    df = concat(results)
    if len(df) == 0:
        print(
            f"No results successful read in from data frames in {results_loc}.")
        print(f"Only considering existing results: {existing}")
    return df


def gen_filename(
        values: list,
        names: list[str]) -> str:
    """
    create overleaf filename for accuracy plot given values of corresponding variables
    assumes run1

    Args:
        values (list): list of values experimental variables have for the given experiment
        names (list[str]): list of names for the corresponding experimental values

    Returns:
        str: overleaf filename
    """
    file = f"accuracy_run1_"
    for varname, varval in zip(names, values):
        file += f"{varname}{varval}_"
    file += "fmnist"
    file = file.replace(".", "-")
    return file


def caption_str(
        values: list,
        names: list[str],
        caption_map: dict[str, str]) -> str:
    """
    generates caption for overleaf subfigure
    displaying accuracy results given values of corresponding variables

    Args:
        values (list): list of values experimental variables have for the given experiment
        names (list[str]): list of names for the corresponding experimental values
        caption_map (dict[str, str]):
            maps variable name in names to how experimental variable should be specified
            within caption of overleaf subfigure

    Returns:
        str: latex caption code (footnotesize) for overleaf subfigure
    """
    caption_str = ""
    if len(names) > 0:
        caption_str = f"{caption_map[names[0]]}: {values[0]}"
        for varname, varval in zip(names[1:], values[1:]):
            caption_str += f", {caption_map[varname]}: {varval}"
    caption_str = f"\t\t\\caption{{\\footnotesize{{{caption_str}}}}}"
    return caption_str


def gen_overleaf_figs(
        overleaf_loc: str,
        results_loc: str,
        colnames_dict: dict[str, str],
        experimental_vars: dict[str, list],
        caption_map: dict[str, str],
        overleaf_fig_caption: str,
        num_per_page: int = 9,
        existing: list[str] = [],
        converters: Mapping[Hashable, Callable] = {}):

    experimental_combos = list(product(*experimental_vars.values()))
    counter, last_settings = 0, experimental_combos[-1]
    df = read_in_results(results_loc=results_loc,
                         converters=converters,
                         existing=existing)

    print("\\begin{figure*}[h!]")

    for vars in product(*experimental_vars.values()):
        file = gen_filename(values=vars, names=list(experimental_vars.keys()))
        # default to including image (assume in results)
        img_cmt = ""
        if f"{file}.png" not in existing:
            exp_df = df
            for key, val in zip(experimental_vars.keys(), vars):
                colname = colnames_dict[key]
                exp_df = exp_df[exp_df[colname] == val]
            if len(exp_df) == 0:
                # commenting out if image is not in results
                img_cmt = "%"
        # TODO: note that the latex comments denoting the change in variables
        #       is not implemented here anymore
        print(f"\t\\begin{{subfigure}}[t]{{0.32\\textwidth}}")
        print(f"\t\t\\centering")

        print(
            f"\t\t{img_cmt}\\includegraphics[width =\\linewidth, height =\\linewidth]{{{overleaf_loc}/{file}.png}}")

        # if image was not in results, use miles photo as stand in
        if img_cmt:
            print(
                "\t\t\\includegraphics[width =\\linewidth, height =\\linewidth]{x.png}")

        print(caption_str(values=vars,
                          names=list(experimental_vars.keys()),
                          caption_map=caption_map))
        print(f"\t\\end{{subfigure}}%")

        counter += 1
        if counter % num_per_page == 0 and vars != last_settings:
            print("\t\\hfill")
            print(overleaf_fig_caption)
            print("\\end{figure*}")
            print("\n\\clearpage\n")
            print("\\begin{figure*}[h!]")
        else:
            print("\t\\hfill")

    print("\t\\hfill")
    print(
        "\t\\caption{\\footnotesize{Dataset: FMNIST, Clients Per Round: 1, ", end="")
    print(
        "Total Rounds: 50, Local Rounds: 1, Clip: 100, Client-Selection: uniform (1 bucket)}}")
    print("\\end{figure*}")
    print("\\clearpage")


def rename_files(src: str, dest_dir: str,
                 replacements: list[tuple[str, str]],
                 avg_prefix: str = "",
                 vars: list[tuple[str, str]] = [],
                 accuracy_only: bool = True):
    """
    Move results from src directory to dest_dir.
    Only accuracy results will be moved when accuracy_only is set to True.
    When avg_prefix is not the empty string,
        move average plots from folders in vars to dest_dir using names specified in vars

    Args:
        src (str):
            location of results, expected to contain and individual_runs folder.
            if avg_prefis is not the empty string, src is expected to contain all folders in vars.
        dest_dir (str):
            location to move renamed results.
            if avg_prefix is not the empty string, dest_dir is expected to contain and avgs folder
        replacements (list[tuple[str, str]]):
            list of tuples with each tuple containing
            a string in automatically generated filenames
            and a shorter replacement for new fileanmes
        avg_prefix (str, optional):
            if moving and renaming average plots,
            what should succeed Avg in new name and preceed title of result.
            Defaults to "" indicating not to consider averages.
        vars (list[tuple[str, str]], optional):
            if moving and renaming average plots (avg_prefix is not ""),
            which folders contain results to be moved.
            for each folder, varname in a tuple in vars,
                move and rename files in folder using varname
            Defaults to [].
        accuracy_only (bool, optional):
            indicates whether to only consider results for accuracy.
            Defaults to True.
    """
    # move individual runs
    src_dir = f"{src}/individual_runs"
    try:
        files = listdir(src_dir)
    except Exception as e:
        print(f"Error {e} finding files in {src_dir}. No files moved")
    else:
        for file in files:
            if accuracy_only and "accuracy" in file:
                new_fname = file
                for original_str, new_str in replacements:
                    new_fname = new_fname.replace(original_str, new_str)
                copy2(f"{src_dir}/{file}", f"{dest_dir}/{new_fname}")

    # move averages
    if avg_prefix:
        dest_dir = f"{dest_dir}/avgs"
        for (dir, var) in vars:
            src_dir = f"{src}/{dir}"
            try:
                files = listdir(src_dir)
            except Exception as e:
                print(f"Error {e} finding files in {src_dir}. No files moved")
            else:
                for file in files:
                    if "accuracy" in file:
                        new_fname = f"Avg{avg_prefix}_{var}.png"
                    copy2(f"{src_dir}/{file}", f"{dest_dir}/{new_fname}")


def find_missing(experimental_vars: dict[str, list],
                 results_loc: str,
                 colnames_dict: dict[str, str],
                 converters: Mapping[Hashable, Callable] = {},
                 selection_dict: dict[str, list] = {},
                 existing: list[str] = []):
    """Determine which experimental settings do not have corresponding accuracy results

    Args:
        experimental_vars (dict):
            key should correspond to experimental variable name
            in the results png filenames
            values should be list of values expected for the key
        results_loc (str): folder where results dataframe is stored
        colnames_dict (dict[str, str]):
            key should correspond to experimental variable name
            in the results png filenames
            values should be the corresponding column name in the data frame
        converters (Mapping[Hashable, Callable], optional):
            any necessary converters for reading in the dataframe
            Defaults to {}.
        selection_dict (dict[str, list], optional):
            key should correspond to experimental variable name
            in the results filenames in results_loc
            values should be list of values to look for (subset of experimental vars)
            Defaults to {}.
        existing (list, optional):
            any results that are stored in location other than results_loc
            that should not be considered missing.
            Defaults to [].
    """
    experimental_vars = {
        key: selection_dict[key] if key in selection_dict else experimental_vars[key] for key in experimental_vars}

    df = read_in_results(
        results_loc=results_loc,
        converters=converters,
        existing=existing
    )
    count, tot = 0, 0
    missing_vars = {key: [] for key in experimental_vars}

    for vars in product(*experimental_vars.values()):
        file = gen_filename(values=vars, names=list(experimental_vars.keys()))
        if f"{file}.png" not in existing:
            exp_df = df
            for key, val in zip(experimental_vars.keys(), vars):
                colname = colnames_dict[key]
                exp_df = exp_df[exp_df[colname] == val]
            if len(exp_df) == 0:
                print("missing", {colnames_dict[key]: val for key, val in zip(
                    experimental_vars.keys(), vars)})
                print()
                count += 1
                for key, val in zip(experimental_vars.keys(), vars):
                    if val not in missing_vars[key]:
                        missing_vars[key] = missing_vars[key] + [val]
        tot += 1

    print(f"missing total of {count}/{tot} results")
    print()
    if count > 0:
        print("exp_vars=", end="")

        def key_str(key, missing_vars):
            key_vals = f"{colnames_dict[key]}:"
            for val in missing_vars[key]:
                key_vals += f"{val} "
            return f"{key_vals[:-1]},"

        exp_vars_str = ""
        for key in missing_vars:
            exp_vars_str += key_str(key, missing_vars)
        print(f"{exp_vars_str[:-1]}")


def main(run_rename=True,
         run_find_missing=True,
         run_gen_figs=True):

    # files in the overleaf but not expected to be in the local folder (previously uploaded)
    existing = ["accuracy_run1_nclients30_batch_size8_opt_momentum0-25_init_lr0-001_lr_power0_clip_schedule_paramsinitial_clip3-0_clip_schedule_paramspower0_fmnist.png",
                "accuracy_run1_nclients30_batch_size8_opt_momentum0-25_init_lr0-001_lr_power0_clip_schedule_paramsinitial_clip5-0_clip_schedule_paramspower0_fmnist.png"]

    # replacements should map names autogenerated in original results png files
    # to shorter names for png files that will be uploaded to overleaf
    replacements = [
        ("sparse_categorical_accuracy", "accuracy"),
        ("total_clients", "nclients"),
        ("client_opt_params:momentum:", "opt_momentum"),
        ("lr_schedule_params:initial_learning_rate:", "init_lr"),
        ("lr_schedule_params:power:", "lr_pow"),
        ("clip_schedule_params:initial_clip", "clip"),
        ("clip_schedule_params:power", "clip_pow"),
        ("0-0_", "0_"),
        (":", "")
    ]

    # folders containing plots for averages
    # paired with name for average plot when moved to overleaf folder
    avg_vars = [("batch_size", "batchsize"),
                ("client_opt_params:momentum", "momentum"),
                ("lr_schedule_params:initial_learning_rate", "init_lr"),
                ("lr_schedule_params:power", "lr_pow"),
                ("clip_schedule_params/initial_clip", "clip"),
                ("clip_schedule_params/power", "clip_pow")]

    # move files from results folder to folder with files to be uploaded to overleaf
    # only consider accuracy results (bool) and
    # rename to shorter names using mappings in replacements
    # also move averages when avg_prefix is non-empty from the avg_vars folders
    # and name appropriately as Avg{avg_prefix}_varname.png
    if run_rename:
        rename_files(src="results",
                     dest_dir="overleaf_data/exp3-ppfl",
                     replacements=replacements,
                     # NOTE: BE CAREFUL HERE
                     #  avg_prefix="30parties",
                     avg_prefix="",
                     vars=avg_vars,
                     accuracy_only=True)

    # use variable name in renamed file name
    # note that priv_param are plotted on same plt
    # keep in order from exp_vars in txt file to produce a compatible new exp_vars
    experimental_vars = {
        "nclients": [10, 30],
        "batch_size": [8, 32, 128],
        "opt_momentum": [.25, .5],
        "priv_param": [0.1, 0.25],
        "init_lr": [0.001, 0.01],
        "lr_pow": [0, 0.75],
        "clip": [1., 3., 5.],
        "clip_pow": [0, 0.75]
    }

    # limit a parameter when looking for missing
    selection_dict = {
        "nclients": [10]
    }
    # map the keys in experimental_vars_with_priv to their column names in the dataframe
    colnames_dict = {
        "nclients": "total_clients",
        "batch_size": "batch_size",
        "opt_momentum": "client_opt_params/momentum",
        "init_lr": "lr_schedule_params/initial_learning_rate",
        "lr_pow": "lr_schedule_params/power",
        "clip": "clip_schedule_params/initial_clip",
        "clip_pow": "clip_schedule_params/power",
        "priv_param": "priv_param"
    }
    # print list of parameter settings for which there are no results in results_loc or existing
    if run_find_missing:
        find_missing(experimental_vars=experimental_vars,
                     selection_dict=selection_dict,
                     colnames_dict=colnames_dict,
                     # where are the data frame csv files located?
                     results_loc="overleaf_data/exp3-ppfl/",
                     existing=existing)

    # how to describe variable in overleaf figure caption
    caption_map = {
        "nclients": "parties",
        "batch_size": "batch size",
        "opt_momentum": "momentum",
        "init_lr": "initial learning rate",
        "lr_pow": "learning rate decay power",
        "clip": "l2 norm clip",
        "clip_pow": "l2 norm clip decay power"
    }

    # remove priv_param (always on same png)
    _ = experimental_vars.pop("priv_param")

    # figure caption (not subfigure - aka does not vary)
    overleaf_fig_caption = "\t\\caption{\\footnotesize{Dataset: FMNIST, Clients Per Round: 100\\%, Total Rounds: 50, Local Rounds: 1, Client-Selection: uniform (1 bucket)}}"

    # create latex code for the overleaf figures
    # assume png files are stored in the overleaf project in the folder of overleaf_loc
    # assume individual csv files are stored locally in the folder of results_loc
    if run_gen_figs:
        gen_overleaf_figs(
            overleaf_loc="results/mar10-mar17graphs/exp3-ppfl/individual_runs",
            results_loc="overleaf_data/exp3-ppfl",
            colnames_dict=colnames_dict,
            experimental_vars=experimental_vars,
            caption_map=caption_map,
            overleaf_fig_caption=overleaf_fig_caption,
            existing=existing)


def col_to_float_lst(s: str):
    return [float(x) for x in s.strip('[]').split(',')]


def append_csv(src_folder="./",
               pattern=".csv",
               ignore_pattern="",
               converters={
                   'sparse_categorical_crossentropy': col_to_float_lst,
                   'sparse_categorical_accuracy': col_to_float_lst,
                   'reject_metrics': col_to_float_lst
               },
               dest_file="./overleaf_data/exp3-ppfl/drt_fmnist_exp3.csv"):
    res_dfs = []
    if isfile(dest_file):
        try:
            res_dfs = [read_csv(dest_file, converters=converters)]
        except Exception as e:
            print(f"Error {e} reading in data from {dest_file}")
    df = read_in_results(results_loc=src_folder,
                         pattern=pattern,
                         ignore_pattern=ignore_pattern,
                         converters=converters,
                         results=res_dfs)
    df.to_csv(dest_file, index=False)


def remove_duplicate_results(src_filepath: str,
                             dest_filepath: str,
                             exp_cols: list[str] = [
                                 "total_clients",
                                 "batch_size",
                                 "client_opt_params/momentum",
                                 "lr_schedule_params/initial_learning_rate",
                                 "lr_schedule_params/power",
                                 "priv_param",
                                 "clip_schedule_params/initial_clip",
                                 "clip_schedule_params/power",
                                 "run#"
                             ],
                             keep: str = "last",
                             converters={
                                 'sparse_categorical_crossentropy': col_to_float_lst,
                                 'sparse_categorical_accuracy': col_to_float_lst,
                                 'reject_metrics': col_to_float_lst
                             }) -> None:
    fname = src_filepath.split("/")[-1]
    src_folder = src_filepath[:src_filepath.find(fname)]
    df = read_in_results(results_loc=src_folder,
                         pattern=fname,
                         converters=converters)
    df = df.drop_duplicates(subset=exp_cols,
                            keep=keep,
                            ignore_index=True)
    df.to_csv(dest_filepath, index=False)


if __name__ == "__main__":
    remove_duplicate_results(
        src_filepath="./overleaf_data/exp3-ppfl/drt_fmnist_exp3.csv",
        dest_filepath="./overleaf_data/exp3-ppfl/drt_fmnist_exp3_dropduplicates.csv")
    # append_csv(src_folder="./",
    #            dest_file="./overleaf_data/exp3-ppfl/drt_fmnist_exp3.csv")
    # main(run_rename=True,
    #      run_find_missing=False,
    #      run_gen_figs=False)
    # main(run_rename=False,
    #      run_find_missing=True,
    #      run_gen_figs=False)
    # main(run_rename=False,
    #      run_find_missing=False,
    #      run_gen_figs=True)
