from keras.datasets import cifar10, mnist, fashion_mnist, cifar100, boston_housing
from medmnist.dataset import PneumoniaMNIST, BloodMNIST, OrganAMNIST
from numpy.random import Generator
from numpy import ndarray
from data_utils.csv_datasets.schools.read_school_data import read_school_data
from data_utils.csv_datasets.adult.read_adult_data import read_adult_data
from data_utils.csv_datasets.gleam.read_gleam_data import read_gleam_data
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
    if dataset in ['cifar10', 'cifar10_xfer']:
        return cifar10_data(np_rng)
    if dataset == 'mnist':
        return mnist_exp(np_rng)
    if dataset == 'fmnist':
        return fmnist()
    if dataset == 'schools':
        return schools(np_rng)
    if dataset == 'adult' or dataset == 'non_iid_adult':
        return adult(np_rng)
    if dataset == 'cifar100':
        return cifar100_data(np_rng)
    if dataset == 'boston':
        return boston(np_rng)
    if dataset == 'gleam':
        return gleam(np_rng)
    if dataset == 'non_iid_fmnist':
        return non_iid_fmnist()
    if dataset == 'non_iid_mnist':
        return non_iid_mnist(np_rng)
    if dataset == 'pneumonia':
        return pneumonia(np_rng)
    if dataset == 'organ':
        return organ(np_rng)
    if dataset == 'blood':
        return blood(np_rng)


# shuffle data
def data_shuffle(train_data: tuple[ndarray, ndarray], np_rng: Generator) -> tuple[ndarray, ndarray]:
    X_train, y_train = train_data
    perm = np_rng.permutation(len(X_train))
    return X_train[perm], y_train[perm]


# load cifar10 dataset
def cifar10_data(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train/255., X_test/255.
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()],
            None)


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


# load organ dataset
def organ(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    train_data = OrganAMNIST(split='train')
    X_train, y_train = train_data.imgs, train_data.labels.flatten().astype('float32')
    test_data = OrganAMNIST(split='test')
    X_test, y_test = test_data.imgs, test_data.labels.flatten().astype('float32')
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()], None)


# load blood dataset
def blood(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    train_data = BloodMNIST(split='train')
    X_train, y_train = train_data.imgs, train_data.labels.flatten().astype('float32')
    test_data = BloodMNIST(split='test')
    X_test, y_test = test_data.imgs, test_data.labels.flatten().astype('float32')
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()], None)


# load school dataset
def schools(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    path = "./data_utils/csv_datasets/schools/schools.csv"
    if not os.path.isfile(path):
        read_school_data()
    (X_train, y_train), (X_test, y_test) = get_csv_data('schools', np_rng)
    # BASELINE MAE: 0.1469840025782094
    # MSE from FedAvg in Silos paper: .025-ish
    # baseline = np.abs(y_test - y_train.mean())
    # print(f"baseline: {baseline.mean()}")
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            MeanAbsoluteError(),
            [MeanAbsoluteError(), MeanSquaredError()],
            None)


# load adult dataset
def adult(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    path = "./data_utils/csv_datasets/adult/adult.csv"
    if not os.path.isfile(path):
        read_adult_data("./data_utils/csv_datasets/adult/",)
    (X_train, y_train), (X_test, y_test) = get_csv_data('adult', np_rng)
    # BASELINE ACC: 0.7591904425539756
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            BinaryCrossentropy(),
            [BinaryAccuracy(), F1Score(), Precision(), Recall()],
            # [BinaryAccuracy(), Precision(), Recall()],
            None)
    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # [tf.keras.metrics.SparseCategoricalAccuracy(),
    #  ProbabilityPrecision(),
    #  ProbabilityRecall()])


# load cifar100 dataset
def cifar100_data(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train, X_test = X_train/255., X_test/255.
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            SparseCategoricalCrossentropy(from_logits=True),
            [SparseCategoricalAccuracy()],
            None)


# load boston housing dataset
def boston(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    (X_train, y_train), (X_test, y_test) = boston_housing.load_data(
        test_split=0.2, seed=113)
    train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train, X_test = X_train - train_mean, X_test - train_mean
    X_train, X_test = X_train/train_std, X_test/train_std
    # BASELINE MAE: 6.533042127742185
    # baseline = np.abs(y_test - y_train.mean())
    # print(f"baseline: {baseline.mean()}")
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            MeanAbsoluteError(),
            [MeanAbsoluteError(), RootMeanSquaredError(), MeanSquaredError()],
            None)


def gleam(np_rng: Generator) -> tuple[tuple[ndarray, ndarray], Loss, list[Metric]]:
    path = "./data_utils/csv_datasets/gleam/gleam.csv"
    if not os.path.isfile(path):
        read_gleam_data("./data_utils/csv_datasets/gleam/")
    (X_train, y_train), (X_test, y_test) = get_csv_data('gleam', np_rng)
    # BASELINE ACC: 0.874537256918
    # print("TEST", (np.unique(y_test,return_counts=True)), len(y_test))
    # print(f"GLEAM_MEAN = {list(X_train.mean(axis=0))}")
    # print(f"GLEAM_STD = {list(X_train.std(axis=0))}")
    return (data_shuffle((X_train, y_train), np_rng),
            (X_test, y_test),
            BinaryCrossentropy(),
            [BinaryAccuracy(), Precision(), Recall()],
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
