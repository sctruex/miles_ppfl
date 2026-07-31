from keras import layers, models, Input
from keras.initializers import GlorotUniform
from keras.applications import MobileNet as TransferNet
from keras.applications import VGG16
from model_utils.adult_norms import ADULT_STD, ADULT_MEAN
from numpy.random import Generator


def get_model(model_name):
    if 'fmnist' in model_name:
        return fmnist
    if 'mnist' in model_name and 'fmnist' not in model_name:
        return mnist
    if 'pneumonia' in model_name:
        return pneumonia
    if 'adult' in model_name:
        return adult

# mnist model
def mnist(np_rng: Generator):
    mnist_model = models.Sequential()
    mnist_model.add(layers.Flatten(input_shape=(28, 28, 1)))
    mnist_model.add(layers.Dense(
        200,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    mnist_model.add(layers.Dense(
        100,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    mnist_model.add(layers.Dense(
        10,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return mnist_model


def pneumonia(np_rng: Generator):
    pneumonia_model = models.Sequential()
    pneumonia_model.add(layers.Flatten(input_shape=(28, 28, 1)))
    pneumonia_model.add(layers.Dense(
        200,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    pneumonia_model.add(layers.Dense(
        100,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    pneumonia_model.add(layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return pneumonia_model


def fmnist(np_rng: Generator):
    """NOTE: setting weights to 0 initially"""
    fmnist_model = models.Sequential()
    fmnist_model.add(layers.Flatten(input_shape=(28, 28, 1)))
    fmnist_model.add(layers.Dense(
        200,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    fmnist_model.add(layers.Dense(
        100,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    fmnist_model.add(layers.Dense(
        10,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return fmnist_model

# adult model
def adult(np_rng: Generator):
    adult_model = models.Sequential()
    adult_model.add(Input(shape=(88,)))
    adult_model.add(layers.Normalization(
        mean=ADULT_MEAN,
        variance=ADULT_STD))
    adult_model.add(layers.Dense(
        20,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    adult_model.add(layers.Dense(
        10,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    adult_model.add(layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return adult_model
    