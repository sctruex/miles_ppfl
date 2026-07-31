from keras.datasets import mnist, fashion_mnist
from medmnist.dataset import PneumoniaMNIST
from numpy.random import Generator
from numpy import ndarray
from data_utils.csv_datasets.adult.read_adult_data import read_adult_data
from data_utils.parse_csv import get_csv_data
import os
import os.path
from keras.losses import Loss, SparseCategoricalCrossentropy, MeanAbsoluteError
from keras.losses import BinaryCrossentropy
from keras.metrics import Metric, SparseCategoricalAccuracy, MeanAbsoluteError
from keras.metrics import MeanSquaredError, RootMeanSquaredError, Precision, Recall
from keras.metrics import BinaryAccuracy, F1Score

"""
    Every dataset needs to additionally return a list of desired class metrics (None or [] if none desired)
    options for class metrics include class_precision, class_recall, class_f1
"""


def get_dataset(dataset: str, np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    if dataset == 'mnist':
        return mnist_exp(np_rng)
    if dataset == 'fmnist':
        return fmnist()
    if dataset == 'adult' or dataset == 'non_iid_adult':
        return adult(np_rng)
    if dataset == 'non_iid_fmnist':
        return non_iid_fmnist()
    if dataset == 'non_iid_mnist':
        return non_iid_mnist(np_rng)
    if dataset == 'pneumonia':
        return pneumonia(np_rng)


# shuffle data
def data_shuffle(train_data: tuple[ndarray, ndarray], np_rng: Generator) -> tuple[ndarray, ndarray]:
    X_train, y_train = train_data
    perm = np_rng.permutation(len(X_train))
    return X_train[perm], y_train[perm]


# load mnist dataset
def mnist_exp(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()], None)


# load fashion_mnist dataset
def fmnist() -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return ((X_train, y_train),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()],
            None)


# load pnemonia dataset
def pneumonia(np_rng) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    train_data = PneumoniaMNIST(split='train')
    X_train, y_train = train_data.imgs, train_data.labels.flatten().astype('float32')
    test_data = PneumoniaMNIST(split='test')
    X_test, y_test = test_data.imgs, test_data.labels.flatten().astype('float32')
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            BinaryCrossentropy(),
            [BinaryAccuracy(), F1Score(), Precision(), Recall()],
            None)


# load adult dataset
def adult(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    path = "./data_utils/csv_datasets/adult/adult.csv"
    if not os.path.isfile(path):
        read_adult_data("./data_utils/csv_datasets/adult/",)
    (X_train, y_train), (X_test, y_test) = get_csv_data('adult', np_rng)
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            BinaryCrossentropy(),
            [BinaryAccuracy(), F1Score(), Precision(), Recall()],
            None)


def non_iid_fmnist() -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return ((X_train, y_train), (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()],
            ['class_precision', 'class_recall', 'f1_score'])


def non_iid_mnist(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (data_shuffle((X_train, y_train), np_rng), (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()],
            ['class_precision', 'class_recall', 'f1_score'])
