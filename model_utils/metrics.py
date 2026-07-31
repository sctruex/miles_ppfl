
import tensorflow as tf
from keras.src import backend
from keras.src.metrics import base_metric
from keras.src.utils import metrics_utils
from keras.src.utils.generic_utils import to_list
import numpy as np
import sys


class ProbabilityFBetaScore(base_metric.Metric):
    def __init__(
        self,
        num_classes,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "{None, 'micro', 'macro', 'weighted'}. "
                f"Received: average={average}"
            )

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )

        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.built = False
        self._num_classes = num_classes

        if self.average != "micro":
            self.axis = 0

    def build(self, y_true_shape, y_pred_shape):
        if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
            raise ValueError(
                "FBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "FBetaScore expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        # num_classes = y_pred_shape[-1]
        num_classes = self._num_classes
        if self.average != "micro":
            init_shape = [num_classes]
        else:
            init_shape = []

        def _add_zeros_weight(name):
            return self.add_weight(
                name,
                shape=init_shape,
                initializer="zeros",
                dtype=self.dtype,
            )

        self.true_positives = _add_zeros_weight("true_positives")
        self.false_positives = _add_zeros_weight("false_positives")
        self.false_negatives = _add_zeros_weight("false_negatives")
        self.intermediate_weights = _add_zeros_weight("intermediate_weights")
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true[:, 0], self._num_classes)
        y_true = tf.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = tf.math.argmax(y_pred, axis=1)
        y_pred = tf.one_hot(y_pred, self._num_classes)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        if not self.built:
            self.build(y_true.shape, y_pred.shape)

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-9)
        else:
            y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(
            _weighted_sum(y_pred * y_true, sample_weight)
        )
        self.false_positives.assign_add(
            _weighted_sum(y_pred * (1 - y_true), sample_weight)
        )
        self.false_negatives.assign_add(
            _weighted_sum((1 - y_pred) * y_true, sample_weight)
        )
        self.intermediate_weights.assign_add(
            _weighted_sum(y_true, sample_weight)
        )

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.intermediate_weights,
                tf.reduce_sum(self.intermediate_weights),
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "average": self.average,
            "beta": self.beta,
            "threshold": self.threshold,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))


class ProbabilityF1Score(ProbabilityFBetaScore):
    def __init__(
        self,
        num_classes,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(
            num_classes=num_classes,
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


class ProbabilityPrecision(base_metric.Metric):

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=(len(self.thresholds),),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print(tf.unique_with_counts(tf.math.argmax(y_pred, axis=1)),
        #          output_stream="file://preds.txt")
        # tf.print("\n\n", output_stream="file://preds.txt")
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
            },
            y_true,
            tf.math.argmax(y_pred, axis=1),
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_positives),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_positives, self.false_positives)
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProbabilityRecall(base_metric.Metric):

    def __init__(
        self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.true_positives = self.add_weight(
            "true_positives", shape=(len(self.thresholds),), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=(len(self.thresholds),),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            y_true,
            tf.math.argmax(y_pred, axis=1),
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        result = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives),
        )
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [
                (v, np.zeros((num_thresholds,)))
                for v in (self.true_positives, self.false_negatives)
            ]
        )

    def get_config(self):
        config = {
            "thresholds": self.init_thresholds,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
