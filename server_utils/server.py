import tensorflow as tf
from numpy.random import Generator
from numpy import nan as NA
from sklearn.metrics import classification_report as report
import numpy as np
import re
import pandas as pd
from client_utils.client import Client


class Server:

    def __init__(self, test_data, np_rng: np.random.Generator, model_architecture, model_params, class_metrics, client_lst, sim_smpc, accountant_params):
        self.client_lst = client_lst
        self.test_data = test_data
        self.model = model_architecture(np_rng)
        # self.model.build((None,32, 32, 3))
        # [[] for _ in range(len(model_params['metrics'])+1)]
        self.metrics = Server.set_up_metrics(
            model_params['metrics'], class_metrics)
        self.class_metrics = class_metrics
        self.agg_metrics = [metric.name for metric in model_params['metrics']]
        self.model.compile(**model_params)
        self._sim_smpc = sim_smpc
        self.accountant_params = accountant_params
        self.bucket_round_count = [-1 for i in range(
            accountant_params['num_buckets'])]

    def create_label_df(self, num_rounds):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 for fmnist dataset
        row = []
        for label in labels:
            # 0 place holder - needs to be run#
            temp_row = [label, 0, [np.nan for _ in range(num_rounds)]]
            row.append(temp_row)
        column = ["label", "run#", "clip_lst"]
        clip_df = pd.DataFrame(row, columns=column)
        print(clip_df.to_string())
        return clip_df

    def set_up_metrics(model_metrics, class_metrics):
        """NOTE: non-class metrics must come first"""
        metrics = {'sparse_categorical_crossentropy': []}
        for metric in model_metrics:
            metrics[metric.name] = []
        # metrics = {metric.name:[] for metric in model_metrics}
        if class_metrics:
            if "class_precision" in class_metrics:
                num_classes = 10
                for class_num in range(num_classes):
                    metrics[f"precision{class_num}"] = []
            if "class_recall" in class_metrics:
                num_classes = 10
                for class_num in range(num_classes):
                    metrics[f"recall{class_num}"] = []
            if "f1_score" in class_metrics:
                num_classes = 10
                for class_num in range(num_classes):
                    metrics[f"f1_score{class_num}"] = []
        return metrics

    def select_clients(self, np_rng: Generator):
        # if self.accountant_params['bucket_approach'] == 1:
        # NEW NON IID CODE
        bucket = np_rng.choice(self.accountant_params['num_buckets'])
        bucket_clients = self.client_lst[bucket]
        self.bucket_round_count[bucket] += 1
        return [bucket_clients[i]
                for i in range(len(bucket_clients))
                if np_rng.random() < bucket_clients[i].get_sample_prob()], self.bucket_round_count[bucket]
        # return [self.client_lst[i]
        #     for i in range(len(self.client_lst))
        #     if rand() < self.client_lst[i].get_sample_prob()]

    def average_params(param_lst):
        avg_weights = [
            tf.reduce_mean(layer_weight_tensors, axis=0)
            for layer_weight_tensors in zip(*param_lst)
        ]
        return avg_weights

    def set_client_clip_average(self, r: int, participating_clients: list[Client]):
        for client in participating_clients:
            label_lst = client.get_client_labels()
            sum_clip = client.get_client_clip()
            for label in label_lst:
                label_clips = self.label_df.loc[self.label_df["label"]
                                                == label, "clip_lst"].iloc[0]
                if np.isnan(label_clips[r]).any():
                    self.label_df.loc[self.label_df["label"] ==
                                      label, "clip_lst"].iloc[0][r] = (sum_clip, 1)
                else:
                    curr_sum, curr_count = label_clips[r]
                    self.label_df.loc[self.label_df["label"] == label, "clip_lst"].iloc[0][r] = (
                        curr_sum + sum_clip, curr_count+1)

    def train(self, num_rounds: int, np_rng: Generator):
        self.label_df = self.create_label_df(num_rounds)
        self.rejections = []
        for r in range(num_rounds):
            print(f"Round {r+1} of {num_rounds}:")
            # if self.accountant_params['bucket_approach'] == 1:
            sample_clients, bucket_round_count = self.select_clients(np_rng)
            # else:
            #     sample_clients = self.select_clients()
            curr_weights = self.model.get_weights()

            if self.accountant_params['clip_round_metric'] == 'client_rounds':
                participating_clients = [
                    client for client in sample_clients if client.avail_train(r)]
                param_lst = [client.train_round(curr_weights, r, len(
                    participating_clients), self._sim_smpc) for client in participating_clients]
                self.set_client_clip_average(r, participating_clients)

            if self.accountant_params['clip_round_metric'] == 'bucket_rounds':
                participating_clients = [
                    client for client in sample_clients if client.avail_train(bucket_round_count)]
                param_lst = [client.train_round(curr_weights, bucket_round_count, len(
                    participating_clients), self._sim_smpc) for client in participating_clients]
                self.set_client_clip_average(r, participating_clients)

            param_lst = [p for p in param_lst if p is not None]
            if len(sample_clients) > 0:
                print(
                    f"Number of clients who rejected training request: {len(sample_clients)-len(param_lst)}")
                self.rejections.append(len(sample_clients)-len(param_lst))

            # param_lst = [curr_weights if p is None else p for p in param_lst]
            if len(param_lst) > 0:
                # params = np_sum(param_lst,axis=0)/len(param_lst)
                avg_params = Server.average_params(param_lst)
                self.model.set_weights(avg_params)
                self.eval_model()

        #         tf.print("*"*200, output_stream="file://l2_norm_vals.txt")
        # tf.print("-"*200, output_stream="file://l2_norm_vals.txt")
        self.fill_metrics_with_na(num_rounds)

    # def train(self, num_rounds):

    #     self.rejections = []
    #     for r in range(num_rounds):
    #         print(f"Round {r+1} of {num_rounds}:")
    #         # sample_clients = self.select_fn(**self.select_params)
    #         # sample_clients = self.select_clients()
    #         sample_clients, bucket_round_count = self.select_clients()
    #         curr_weights = self.model.get_weights()
    #         # participating_clients = [
    #         #     client for client in sample_clients if client.avail_train(r)]
    #         participating_clients = [
    #              client for client in sample_clients if client.avail_train(bucket_round_count)]
    #         # param_lst = [client.train_round(curr_weights, r, len(
    #         #     participating_clients), self._sim_smpc) for client in participating_clients]
    #         param_lst = [client.train_round(curr_weights, bucket_round_count, len(
    #             participating_clients), self._sim_smpc) for client in participating_clients]
    #         param_lst = [p for p in param_lst if p is not None]
    #         if len(sample_clients) > 0:
    #             print(
    #                 f"Number of clients who rejected training request: {len(sample_clients)-len(param_lst)}")
    #             self.rejections.append(len(sample_clients)-len(param_lst))

    #         # param_lst = [curr_weights if p is None else p for p in param_lst]
    #         if len(param_lst) > 0:
    #             # params = np_sum(param_lst,axis=0)/len(param_lst)
    #             avg_params = Server.average_params(param_lst)
    #             self.model.set_weights(avg_params)
    #             self.eval_model()

    #     #         tf.print("*"*200, output_stream="file://l2_norm_vals.txt")
    #     # tf.print("-"*200, output_stream="file://l2_norm_vals.txt")
    #     self.fill_metrics_with_na(num_rounds)

    def eval_model(self):
        x_test, y_test = self.test_data
        result = self.model.evaluate(x_test, y_test, verbose=0)

        print(f"Result: {result}\n")
        # print("self metrics", self.metrics)
        # print(len(self.metrics))
        # print(len(result))
        # for i in range(len(self.metrics)):
        #     self.metrics[i].append(result[i])
        if "sparse_categorical_crossentropy" in self.metrics:
            self.metrics["sparse_categorical_crossentropy"].append(result[0])
        else:
            self.metrics["loss"].append(result[0])

        # if "sparse_categorical_accuracy" in self.metrics:
        #     self.metrics["sparse_categorical_accuracy"].append(result[1])
        # else:
        for metric, i in zip(self.agg_metrics, range(len(self.agg_metrics))):
            self.metrics[metric].append(result[i+1])

        # print("self metrics", self.metrics)

        if self.class_metrics:
            pred = self.model.predict(x_test)
            preds = [np.argmax(x) for x in pred]
            metrics_report = report(y_test, preds)

            print("\nmetrics report", metrics_report)
            cleaned_report = re.sub(
                r"f1-score", "", metrics_report, flags=re.DOTALL)
            print("\n", cleaned_report)
            vals_report = re.findall(r'\d+\.\d+|\d+', cleaned_report)
            parsed_report = [float(num) if '.' in num else int(num)
                             for num in vals_report]
            print(parsed_report)
            # hard coded for 10 classes
            cleaned_metrics = [
                parsed_report[i * 5:(i + 1) * 5] for i in range(10)]
            print("final lst", cleaned_metrics)

            def parse_metrics(metric_name, report_lst):
                """precision metric: index 1, recall metric: index 2, f1_score metric: index 3"""
                class_num = 10
                metric_lst = []
                if metric_name == "class_precision":
                    for i in range(class_num):
                        metric_lst.append(report_lst[i][1])
                    return metric_lst
                if metric_name == "class_recall":
                    for i in range(class_num):
                        metric_lst.append(report_lst[i][2])
                    return metric_lst
                if metric_name == "f1_score":
                    for i in range(class_num):
                        metric_lst.append(report_lst[i][3])
                    return metric_lst

            # parse report for class metrics
            if "class_precision" in self.class_metrics:
                class_precisions = parse_metrics(
                    "class_precision", cleaned_metrics)
                print("class precision list", class_precisions)
                """parse report to get num_classes class precisions"""
                for i in range(len(class_precisions)):
                    self.metrics[f"precision{i}"].append(class_precisions[i])
            if "class_recall" in self.class_metrics:
                class_recall = parse_metrics("class_recall", cleaned_metrics)
                print("class recall list", class_recall)
                # set equal to class_recall lst from parsed report
                for i in range(len(class_recall)):
                    self.metrics[f"recall{i}"].append(class_recall[i])
            if "f1_score" in self.class_metrics:
                f1_score = parse_metrics("f1_score", cleaned_metrics)
                print("class f1_score list", f1_score)
                # set equal to f1_score lst from parsed report
                for i in range(len(f1_score)):
                    self.metrics[f"f1_score{i}"].append(f1_score[i])
            print("Metrics list", self.metrics)

        # # ATTEMPT FOR NEW METRIC
        # pred = self.model.predict(x_test)
        # preds = [np.argmax(x) for x in pred]
        # print("REPORT:", report(y_test, preds))

        # # loss metric
        # self.metrics[0].append(result[0])
        # # accuracy metric
        # self.metrics[1].append(result[1])

    def get_metrics(self):
        # return self.metrics[0], self.metrics[1]
        return self.metrics

    def get_rejections(self):
        return self.rejections

    def fill_metrics_with_na(self, num_rounds):
        for metric in self.metrics:
            # metric.extend([NA]*(num_rounds-len(metric)))
            # self.metrics[metric].extend([NA]*(num_rounds-len(metric)))
            self.metrics[metric].extend(
                [NA]*(num_rounds-len(self.metrics[metric])))
