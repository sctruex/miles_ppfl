from ray import client
from privacy_utils.privacy import find_avail_rounds
import tensorflow as tf
import math
import numpy as np


class Client:

    def __init__(self, data, eps, sampling_rate, accountant_params, model_architecture,
                 np_rng: np.random.Generator, opt, opt_params, model_params, fit_params,
                 lr_scheduler, lr_schedule_params, clip_scheduler):
        self._client_x, self._client_y = data
        self._client_model = model_architecture(np_rng)
        self._opt_fn = opt
        self._opt_params = opt_params
        self._total_steps = 0
        self._lr_scheduler = lr_scheduler
        self._lr_params = lr_schedule_params
        self._init_noise_multiplier = self._opt_params['noise_multiplier']
        self._model_params = model_params
        self._clip_scheduler = clip_scheduler
        self._fit_params = fit_params
        self._avail_rounds = find_avail_rounds(
            **accountant_params, eps=eps, q=sampling_rate, client_data_len=len(self._client_x))
        self._sample_prob = sampling_rate
        self._client_clip_lst = []
        self._client_labels = np.unique(self._client_y)
        print(f"client label:{self._client_labels}")
        print(f"client participating in {self._avail_rounds}")

    def compile_model(self, curr_round, k, sim_smpc):
        if not sim_smpc:
            k = 1  # consider local client as only client if smpc is not turned on
        self._opt_params['noise_multiplier'] = self._init_noise_multiplier / \
            math.sqrt(k)
        self._opt_params['l2_norm_clip'] = self._clip_scheduler.get_round_clip(
            curr_round)
        self._lr_params['initial_step'] = self._total_steps
        self._opt = self._opt_fn(learning_rate=self._lr_scheduler(
            **self._lr_params), **self._opt_params)
        self._curr_lr = self._opt._decayed_lr(tf.float32)
        print(
            f"learning_rate at round {curr_round+1} is {self._curr_lr}")
        print(
            f"l2_norm_clip at round {curr_round+1} is {self._opt_params['l2_norm_clip']}")
        self._client_model.compile(optimizer=self._opt, **self._model_params)
        # client clip list test
        self._client_clip_lst.append(self._opt_params['l2_norm_clip'])

    def get_sample_prob(self):
        return self._sample_prob

    def get_client_labels(self):
        return self._client_labels

    def get_client_clip(self):
        return self._opt_params['l2_norm_clip']

    def train_round(self, params, curr_round, k, sim_smpc):
        # currently not accounting for local epochs greater than 1 - may need to do curr_round x local epochs
        if curr_round >= self._avail_rounds:
            return None

        self.compile_model(curr_round=curr_round, k=k, sim_smpc=sim_smpc)
        self._client_model.set_weights(params)
        self._client_model.fit(
            self._client_x, self._client_y, **self._fit_params)

        self._total_steps += self._opt.iterations
        return self._client_model.get_weights()

    def avail_train(self, r):
        return r < self._avail_rounds

    def get_row(self):
        return [self._client_labels, self._client_clip_lst]
