from model_utils.clip_scheduler import ExponentialClipDecay, StepClipDecay
from model_utils.min_lr_step import PolynomialDecay, MinCapableStepLR
from tensorflow.keras.optimizers.schedules import LearningRateSchedule  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore
from tensorflow_privacy import DPKerasSGDOptimizer as DPSGD
from tensorflow_privacy import DPKerasAdamOptimizer as DPAdam
# from model_utils.custom_dpsgd import CustomDPSGDOptimizer as CustomDPSGD
from typing import Union
from pandas import DataFrame as pandas_dataframe
from data_utils.dataset import get_dataset
from numpy.random import default_rng
from sys import stderr


class ExperimentalArguments:

    int_vars = ["total_clients", "training_rounds", "num_runs", "local_epochs",
                "batch_size", "decay_rounds", "verbose", "num_microbatches", "decay_steps",
                "num_buckets", "seed", 'bucket_approach', 'non_iid_method']
    float_vars = ["clients_per_round", "initial_clip", "decay_rate", "min_clip", "power", "first_round_clip", "momentum", "initial_learning_rate",
                  "end_learning_rate", "priv_param", "target_delta", "min_eps", "max_mult", "min_mult", "eps_distrib_scale"]
    str_vars = ["sampling_approach", "dataset_name",
                "name", "priv_approach", 'clip_round_metric']
    bool_vars = ["sim_smpc", "debug", "sort_eps_lst", "plot_avgs"]
    list_vars = ["additional_df_cols", "bucket_prob"]
    list_type_fns = {"additional_df_cols": str, "bucket_prob": float}
    dict_vars = ["exp_var_map", "server_model_params", "clip_schedule_params", "client_fit_params",
                 "client_model_params", "client_opt_params", "lr_schedule_params", "accountant_params"]
    optimizer_vars = ["optimizer", "client_opt"]
    optimizer_options = {"dpsgd": DPSGD, "adam": "adam",
                         "dpadam": DPAdam}  # NOTE: doesn't use seeded generator - "custom_dpsgd": CustomDPSGD}
    lr_scheduler_options = {
        "PolynomialDecay": PolynomialDecay, "MinCapableStepLR": MinCapableStepLR}
    clip_scheduler_options = {
        "ExponentialClipDecay": ExponentialClipDecay,
        "StepClipDecay": StepClipDecay
    }

    def __init__(self,
                 exp_var_map: dict,
                 additional_df_cols: list[str] = [],
                 num_runs: int = 5,
                 total_clients: int = 30,
                 clients_per_round: float = 0.2,
                 training_rounds: int = 80,
                 local_epochs: int = 1,
                 sampling_approach: str = "uniform",
                 dataset_name: str = "mnist",
                 seed: int = 0,                             # NEW
                 server_model_params: dict = {
                     'optimizer': 'adam'
                 },
                 batch_size: int = 32,
                 clip_scheduler: Union[ExponentialClipDecay,
                                       StepClipDecay] = ExponentialClipDecay,
                 clip_schedule_params: dict = {
                     'initial_clip': 5.,
                     'decay_rate': .6,
                     'decay_rounds': 10,
                     'min_clip': .001,
                     'power': 0,
                     'first_round_clip': 0
                 },
                 client_fit_params: dict = {
                     'verbose': 2
                 },
                 client_model_params: dict = {},
                 client_opt: Optimizer = DPSGD,
                 client_opt_params: dict = {
                     'num_microbatches': 1,
                 },
                 lr_scheduler: LearningRateSchedule = PolynomialDecay,
                 lr_schedule_params: dict = {
                     'initial_learning_rate': .01,
                     'decay_steps': 2000,
                     'end_learning_rate': 1e-3,
                     'power': 1.5,
                     'name': 'PolynomialDecay'
                 },
                 priv_approach: str = "dropout",
                 priv_param: Union[int, float] = 0.0,
                 accountant_params: dict = {
                     'target_delta': 1e-5,
                     'min_eps': 0.01,
                     'num_buckets': 5,
                     'bucket_prob': [.05, .1, .2, .25, .4],
                     'max_mult': 50,
                     'min_mult': 1e-5,
                     'eps_distrib_scale': 0.5,
                     'sort_eps_lst': 1,
                     'bucket_approach': 0,
                     'non_iid_method': 0,
                     'clip_round_metric': 'client_rounds'
                 },
                 sim_smpc: bool = True,
                 debug: bool = False
                 ):

        self.seed = seed
        if sampling_approach == "bucket":
            accountant_params["bucket_prob"] = [1/accountant_params["num_buckets"]
                                                for _ in range(accountant_params["num_buckets"])]
        self.num_runs = num_runs
        self._args = {
            "exp_var_map": exp_var_map,
            "total_clients": total_clients,
            "clients_per_round": max(1, int(clients_per_round*total_clients)),
            "training_rounds": training_rounds,
            "local_epochs": local_epochs,
            "sampling_approach": sampling_approach,
            "dataset_name": dataset_name,
            "server_model_params": server_model_params,
            "batch_size": batch_size,
            "clip_scheduler": clip_scheduler(**clip_schedule_params),
            "client_fit_params": client_fit_params,
            "client_model_params": client_model_params,
            "client_opt": client_opt,
            "client_opt_params": client_opt_params,
            "lr_scheduler": lr_scheduler,
            "lr_schedule_params": lr_schedule_params,
            "priv_approach": priv_approach,
            "priv_param": priv_param,
            "accountant_params": accountant_params,
            "sim_smpc": sim_smpc,
        }
        self._additional_df_cols = additional_df_cols
        self.set_up_dataset()

        # self._clip_schedule_fn = ExponentialClipDecay
        self._clip_schedule_fn = clip_scheduler
        self._clip_schedule_params = clip_schedule_params
        self._debug = debug

    def set_up_dataset(self, np_rng=default_rng()):
        (X_train, y_train), (X_test,
                             y_test), loss, metrics, class_metrics = get_dataset(self._args["dataset_name"], np_rng)
        self.collected_metrics = [metric.name for metric in metrics]
        # results_df = pandas_dataframe(
        #     columns=list(self._args["exp_var_map"].keys())+self._additional_df_cols+["loss"]+self.collected_metrics)

        # self._args["results_df"] = results_df
        self._args["dataset"] = ((X_train, y_train), (X_test, y_test))
        self._args["loss"] = loss
        self._args["metrics"] = metrics
        self._args["class_metrics"] = class_metrics
        if class_metrics:
            num_classes = 10  # len(np.unique(y_train))
            if "class_precision" in class_metrics:
                for class_num in range(num_classes):
                    self.collected_metrics.append(f"precision{class_num}")
            if "class_recall" in class_metrics:
                for class_num in range(num_classes):
                    self.collected_metrics.append(f"recall{class_num}")
            if "f1_score" in class_metrics:
                for class_num in range(num_classes):
                    self.collected_metrics.append(f"f1_score{class_num}")

    def get_full_args(self, seed):
        self._args["np_rng"] = default_rng(seed=seed)
        self.set_up_dataset(self._args["np_rng"])

        return self._args

    def get_arg(self, key):
        return self._args[key]

    def update_arg(self, key, val):
        if "/" in key:
            dict_key, arg_key = key.split("/")
            if dict_key == "clip_schedule_params":
                self._clip_schedule_params[arg_key] = val
                self.update_arg("clip_scheduler", self._clip_schedule_fn(
                    **self._clip_schedule_params))
            else:
                self._args[dict_key][arg_key] = val
        elif key == "clients_per_round":
            self._args["clients_per_round"] = max(
                1, int(val*self._args["total_clients"]))
        elif key == "total_clients":
            self._args['clients_per_round'] = max(
                1, self._args["clients_per_round"] * (val/self._args["total_clients"]))
            self._args[key] = val
        else:
            self._args[key] = val

    def get_collected_metrics(self):
        return self.collected_metrics

    def __str__(self):
        args_str = "NAME OF FILE:\n"
        debug_prefix = "debug_" if self._debug else "exp_"

        # file names specific to experiment
        results_name = f'{debug_prefix}{self._args["dataset_name"]}'
        for key in self._args["exp_var_map"]:
            results_name += f'_{key}:{self._args["exp_var_map"][key]}'
        results_name = results_name.replace(".", "-")
        args_str += f'{results_name}\n'

        # parameter prints
        args_str += f'Dataset: {self._args["dataset_name"]}\n'
        args_str += f'- Total Clients: {self._args["total_clients"]}\n - Training Rounds: {self._args["training_rounds"]}\n'
        args_str += f' - Local Epochs: {self._args["local_epochs"]}\n - Clients Per Round: {self._args["clients_per_round"]}\n - Batch Size: {self._args["batch_size"]}\n'
        args_str += f' - SMPC: {self._args["sim_smpc"]}\n - clip scheduler: {self._clip_schedule_fn}\n'

        for param_name in self._clip_schedule_params.keys():
            args_str += f' - {param_name}: {self._clip_schedule_params[param_name]}\n'

        for param_name in self._args["client_opt_params"].keys():
            args_str += f' - {param_name}: {self._args["client_opt_params"][param_name]}\n'

        args_str += f'Privacy Parameters:\n - Target Delta: {self._args["accountant_params"]["target_delta"]}\n'
        args_str += f'- Min Epsilon: {self._args["accountant_params"]["min_eps"]}\n'
        args_str += f' - Max Multiplier: {self._args["accountant_params"]["max_mult"]}\n - Min Multiplier: {self._args["accountant_params"]["min_mult"]}\n\n'
        args_str += f'Approach: {self._args["priv_approach"]}\n'
        if self._args["priv_approach"] == 'dropout':
            args_str += f' - Multiplier: {self._args["priv_param"]}\n\n'
        args_str += f'Sampling Approach: {self._args["sampling_approach"]}\n'
        if self._args["sampling_approach"] == 'tiered':
            args_str += f' - Bucket Number: {self._args["accountant_params"]["num_buckets"]}\n'
            args_str += f'- Bucket Probability: {self._args["accountant_params"]["bucket_prob"]}\n\n'
        args_str += f'Learning Rate Scheduler Parameters: {self._args["lr_schedule_params"]["name"]}\n'
        for param_name in self._args["lr_schedule_params"].keys():
            args_str += f' - {param_name}: {self._args["lr_schedule_params"][param_name]}\n'

        return args_str

    def get_var_from_string(var_name: str, var_val: str):
        """
        Given a variable name and its corresponding value as a string, return the appropriately typed variable value
        NOTES:  - dictionaries are expected to be comma separated in var_val with a colon separating key and value
                    example:    get_var_from_string("client_opt_params", "num_microbatches:1,momentum:.5") 
                                will return the dictionary {"num_microbatches":1, "momentum":0.5}
                - lists are expected to be space separated in var_val
                    example:    get_var_from_string("bucket_prob", ".05 .1 .2 .25 .4")
                                will return the list [0.5 .1 .2 .25 .4]

        Args:
            var_name (str): name of the argument
            var_val (str): value of the argument as a string

        Returns:
            appropriately typed value for the given variable
        """
        if var_name in ExperimentalArguments.int_vars:
            return int(var_val)
        if var_name in ExperimentalArguments.float_vars:
            return float(var_val)
        if var_name in ExperimentalArguments.bool_vars:
            return bool(int(var_val))
        if var_name in ExperimentalArguments.str_vars:
            return var_val
        if var_name in ExperimentalArguments.list_vars:
            type_fn = ExperimentalArguments.list_type_fns[var_name]
            return [type_fn(key) for key in var_val.split(" ")]
        if var_name in ExperimentalArguments.dict_vars:
            if len(var_val) == 0:
                return {}
            dict_pairs = var_val.split(",")
            dict_pairs = [pair.split(":") for pair in dict_pairs]
            return {key: ExperimentalArguments.get_var_from_string(key, data) for key, data in dict_pairs}
        if var_name in ExperimentalArguments.optimizer_vars:
            return ExperimentalArguments.optimizer_options[var_val]
        if var_name == "lr_scheduler":
            return ExperimentalArguments.lr_scheduler_options[var_val]
        if var_name == "clip_scheduler":
            if var_val not in ExperimentalArguments.clip_scheduler_options.keys():
                print(f"{var_val} not an available clip scheduler. options are: {ExperimentalArguments.clip_scheduler_options.keys()}", file=stderr)
                print("using ExponentialClipDecay", file=stderr)
                var_val = "ExponentialClipDecay"
            return ExperimentalArguments.clip_scheduler_options[var_val]
