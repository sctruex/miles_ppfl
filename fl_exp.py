from server_utils.server import Server
from model_utils.model import get_model
from server_utils.selection import *
from client_utils.create_clients import create_client_lst
from privacy_utils.privacy import *
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from numpy.random import Generator
from statistics import median

from model_utils.min_lr_step import *
from model_utils.clip_scheduler import ExponentialClipDecay

from typing import Union
from keras.losses import Loss
from keras.metrics import Metric


def run_fl_exp(total_clients: int, clients_per_round: int, training_rounds: int, local_epochs: int, sampling_approach: str,
                dataset_name: str, dataset: tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]], loss: Loss, metrics: list[Metric],
                server_model_params: dict, class_metrics: list[str], batch_size: int, clip_scheduler: ExponentialClipDecay,
                client_fit_params: dict, client_model_params: dict, client_opt: tf.keras.optimizers.Optimizer, client_opt_params: dict,
                lr_scheduler: tf.keras.optimizers.schedules.LearningRateSchedule, lr_schedule_params: dict,
                priv_approach: str, priv_param: Union[int, float], accountant_params: dict, sim_smpc: bool,
                exp_var_map: dict, np_rng: Generator) -> pd.DataFrame:
    """Run one experiment of federated learning given the setting specified by the inputs.

    Args:
        total_clients (int):            total number of clients
        clients_per_round (int):        average number of clients to query per round
                                        (determines sampling probabilities for individual clients)
        training_rounds (int):          total number of training rounds
        local_epochs (int):             number of local epochs of training done by an individual client per round of training
        sampling_approach (str):        approach to sampling clients at each round
                                        available approaches: uniform, normal, tiered
        dataset_name (str):             name of dataset on which to simulate federated training
        dataset 
        (tuple[tuple[ndarray, ndarray], 
            tuple[ndarray, ndarray]]):  set of tuples X_train, y_train and X_test, y_test
        loss (Loss):                    definition of loss for given dataset
        metrics (list[Metric]):         list of metrics to evaluate model performance
        server_model_params (dict):     dictionary specifying parameters (other than loss and metrics) that should be passed to model.compile(...) by the server
        class_metrics (list[str]):      list of class level metrics to evaluate model performance at server
        batch_size (int):               batch size used at local training
        clip_scheduler 
            (ExponentialClipDecay):     scheduler to determine l2_norm clip value for a given round of training
                                        must implement clip_scheduler.get_round_clip(curr_round) functionality
        client_fit_params (dict):       additional parameters (other than x,y) to be passed to model.fit(...) by local clients
                                        NOT expected to include (will be set using other inputs):
                                                                 'batch_size', 'epochs'
        client_model_params (dict):     additional parameters (other than optimizer, loss, and metrics) to be passed to model.compile(...) by local clients
        client_opt (Optimizer):         tensorflow.keras.optimizers.Optiimzer to use at local clients for training
        client_opt_params (dict):       additional parameters (other than learning_rate and noise_multiplier) to be passed to client_opt(...) by local clients
        priv_approach (str):            approach determining how to handle heterogeneity in client privacy budgets
                                        available approaches:   tightest (entire training process adheres to lowest privacy budget)
                                                                dropout (individual clients stop participating in training once their budget has been spent)
        priv_param (int|float):         depending on priv_approach, either the averge epsilon value (tightest) or the noise multiplier (dropout)
        accountant_params (dict):       parameters necessary to complete necessary privacy accounting.
                                        expected to include:    target_delta,
                                                                min_eps - minimum allowable privacy budget)
                                                                num_buckets - number of tiers, necessary only when using tiered approach
                                                                bucket_prob - probability of selecting each privacy tier
                                                                max_mult -  upper bound for the search space in finding a multiplier corresponding to a target epsilon
                                                                            only necessary when priv_approach is tightest
                                                                min_mult -  lower bound for the search space in finding a multiplier corresponding to a target epsilon
                                                                            only necessary when priv_approach is tightest
                                                                eps_distrib_scale - used when generating heterogeneous privacy budgets
                                                                                    epsilon values are sampled from the gaussian distribution
                                                                                        the distribution will be centered on the target epsilon (according to privacy_param) 
                                                                                        the distribution scale is determined by target_epsilon * eps_distrib_scale
                                        NOT expected to include (will be set using other inputs):
                                                                'total_clients', 'rounds', 'clients_per_round', 'min_selection', 'batch_size', 'len_data', 'noise_multiplier'
        sim_smpc (bool):                whether or not secure aggregation is assumed 
                                        if True: allows noise to be decreased locally by a sqrt(k) at each round where k = number of round participants
        exp_var_map (dict):             dictionary of values being tested across experiments
        np_rng (Generator):             seeded numpy random generator 

    Returns:
        pd.DataFrame: results_df with results from current experiment appended
    """
    # set tensorflow seed using pre-seeded np_rng
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(np_rng.integers(2**32))

    # add necessary additional values to accountant_params
    accountant_params['total_clients'] = total_clients
    accountant_params['rounds'] = training_rounds
    accountant_params['clients_per_round'] = clients_per_round
    accountant_params['min_selection'] = 1/training_rounds
    accountant_params['batch_size'] = batch_size
    accountant_params['local_epochs'] = local_epochs

    client_fit_params['batch_size'] = batch_size
    client_fit_params['epochs'] = local_epochs

    server_model_params['loss'] = loss
    server_model_params['metrics'] = metrics
    client_model_params['loss'] = loss
    client_model_params['metrics'] = metrics

    my_model = get_model(dataset_name)

    (X_train, y_train), (X_test, y_test) = dataset

    accountant_params['len_data'] = len(X_train)

    run_exp_dict = {col: exp_var_map[col] for col in exp_var_map}

    mult, eps_lst = get_noise(
        priv_approach, priv_param, accountant_params, np_rng)
    accountant_params['noise_multiplier'] = mult
    client_opt_params['noise_multiplier'] = mult

    # for clients to set the avail_rounds correctly, they need to know their INDIVIVDUAL sampling probability...

    print("-" * 25)
    client_lst = create_client_lst(data=(X_train, y_train), eps_lst=eps_lst, np_rng=np_rng, accountant_params=accountant_params,
                                   my_model=my_model, client_opt=client_opt, client_opt_params=client_opt_params,
                                   client_model_params=client_model_params, client_fit_params=client_fit_params,
                                   lr_scheduler=lr_scheduler, lr_schedule_params=lr_schedule_params,
                                   sampling_app=sampling_approach, clip_scheduler=clip_scheduler, dataset_name=dataset_name)
    server = Server((X_test, y_test), np_rng, my_model,
                    server_model_params, class_metrics, client_lst,
                    sim_smpc, accountant_params)

    # print current multiplier and epsilon list
    if priv_approach == 'dropout':
        param = 'multiplier'
    else:
        param = 'epsilon'
    print(f"\nBegin training with {param}: {priv_param}\n")
    print(f"Multiplier = {mult}, Epsilon = {eps_lst}\n")

    server.train(training_rounds, np_rng)
    print("\nFinished training")
    
    metrics_dict = server.get_metrics()
    for metric in metrics_dict:
        run_exp_dict[metric] = metrics_dict[metric]
    run_exp_dict['reject_metrics'] = server.get_rejections()
    run_exp_dict['eps'] = median(eps_lst)

    row = pd.Series(run_exp_dict)
    df_row = pd.DataFrame([row], columns=row.index)
    print(f"final df: {df_row}")

    tf.keras.backend.clear_session()

    return df_row


if __name__ == "__main__":
    run_fl_exp(5, 2, 'mnist', [0.0, 0.5])
