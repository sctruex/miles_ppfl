from keras import layers, models, Input
from keras.initializers import GlorotUniform
from keras.applications import MobileNet as TransferNet
from keras.applications import VGG16
from model_utils.adult_norms import ADULT_STD, ADULT_MEAN, GLEAM_MEAN, GLEAM_STD
from numpy.random import Generator


def get_model(model_name):
    if 'cifar10' in model_name:
        return cifar10
    if 'cifar10_xfer' in model_name:
        return cifar10_vgg16
    if 'fmnist' in model_name:
        return fmnist_m
    if 'mnist' in model_name and 'fmnist' not in model_name:
        return mnist
    if 'pneumonia' in model_name:
        return pneumonia
    if 'blood' in model_name:
        return blood
    if 'organ' in model_name:
        return organ
    if 'schools' in model_name:
        return schools
    if 'adult' in model_name:
        return adult
    if 'cifar100' in model_name:
        return cifar100_m
    if 'boston' in model_name:
        return boston_m
    if 'gleam' in model_name:
        return gleam_m


def cifar10(np_rng: Generator):
    cifar10_model = models.Sequential()
    cifar10_model.add(Input(shape=(32, 32, 3)))
    # ,input_shape=(32, 32, 3))
    cifar10_model.add(layers.RandomFlip(
        mode='horizontal',
        seed=np_rng.integers(2**32)))
    # cifar10_model.add(layers.RandomCrop(size=32, padding=4))
    # cifar10_model.add(layers.RandomCrop(32,32)) # no padding?
    cifar10_model.add(layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229, 0.224, 0.225]))
    # cifar10_model = layers.RandomFlip(mode='horizontal')
    # cifar10_model = layers.RandomCrop(32, 4)
    # cifar10_model = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])

    cifar10_model.add(layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
        name="cifar10_conv2d_1"))
    # cifar10_model.add(layers.MaxPooling2D((2, 2),name="cifar10_maxpool_1"))
    cifar10_model.add(layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
        name="cifar10_conv2d_2"))
    cifar10_model.add(layers.MaxPooling2D(
        (2, 2),
        name="cifar10_maxpool_2"))
    cifar10_model.add(layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
        name="cifar10_conv2d_3"))
    cifar10_model.add(layers.Flatten(name="cifar10_flatten"))
    cifar10_model.add(layers.Dense(
        64,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
        name="cifar10_dense_1"))
    cifar10_model.add(layers.Dense(10,
                                   kernel_initializer=GlorotUniform(
                                       np_rng.integers(2**32)),
                                   name="cifar10_dense_2"))
    return cifar10_model


def cifar10_vgg16(np_rng: Generator):
    cifar10_model = models.Sequential()
    cifar10_model.add(Input(shape=(32, 32, 3)))
    cifar10_model.add(layers.RandomFlip(
        mode='horizontal',
        seed=np_rng.integers(2**32)))
    cifar10_model.add(layers.RandomRotation(
        0.2,
        seed=np_rng.integers(2**32)))
    cifar10_model.add(layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229, 0.224, 0.225]))

    xfer_model = VGG16(
        include_top=False,
        weights='imagenet',
        classes=10,
        input_shape=(32, 32, 3),
        pooling="avg")
    xfer_model.trainable = False
    cifar10_model.add(xfer_model)

    cifar10_model.add(layers.BatchNormalization())
    # cifar10_model.add(layers.Dense(256, activation='relu',
    #                                kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    # cifar10_model.add(layers.Dense(256, activation='relu',
    #                                kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    cifar10_model.add(layers.Dense(
        500,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    cifar10_model.add(layers.Dense(
        10,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))

    return cifar10_model


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


def organ(np_rng: Generator):
    organ_model = models.Sequential()
    organ_model.add(layers.Flatten(input_shape=(28, 28, 1)))
    organ_model.add(layers.Dense(
        200,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    organ_model.add(layers.Dense(
        100,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    organ_model.add(layers.Dense(
        11,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return organ_model


def blood(np_rng: Generator):
    blood_model = models.Sequential()
    blood_model.add(layers.Flatten(input_shape=(28, 28, 1)))
    blood_model.add(layers.Dense(
        200,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    blood_model.add(layers.Dense(
        100,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    blood_model.add(layers.Dense(
        8,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return blood_model


def fmnist_m(np_rng: Generator):
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


# school model
def schools(np_rng: Generator):
    schools_model = models.Sequential()
    schools_model.add(layers.Flatten(input_shape=(28,)))
    # schools_model.add(layers.Dense(20, activation='relu'))
    schools_model.add(layers.Dense(
        10,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    schools_model.add(layers.Dense(
        1,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return schools_model


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

# cifar100 model
# def cifar100_m():
#     cifar100_model = models.Sequential()
#     cifar100_model.add(Input(shape=(32,32,3)))
#     cifar100_model.add(layers.RandomFlip(mode='horizontal'))
#     cifar100_model.add(layers.RandomCrop(32,32))
#     cifar100_model.add(layers.Normalization(mean=[(0.4914, 0.4822, 0.4465)], variance=[0.2023, 0.1994, 0.2010]))

#     cifar100_model.add(layers.Conv2D(
#         32, (3, 3), activation='relu',name="cifar100_conv2d_1"))
#     cifar100_model.add(layers.Conv2D(64, (3, 3), activation='relu',name="cifar100_conv2d_2"))
#     cifar100_model.add(layers.MaxPooling2D((2, 2),name="cifar100_maxpool"))
#     cifar100_model.add(layers.Conv2D(64, (3, 3), activation='relu',name="cifar100_conv2d_3"))
#     cifar100_model.add(layers.Flatten(name="cifar100_flatten"))
#     cifar100_model.add(layers.Dense(64, activation='relu', name="cifar100_dense_1"))
#     cifar100_model.add(layers.Dense(100, name="cifar100_dense_2"))

#     return cifar100_model


def cifar100_m(np_rng: Generator):
    cifar100_model = models.Sequential()
    cifar100_model.add(Input(shape=(32, 32, 3)))
    cifar100_model.add(layers.RandomFlip(
        mode='horizontal',
        seed=np_rng.integers(2**32)))
    cifar100_model.add(layers.Normalization(
        mean=[(0.4914, 0.4822, 0.4465)],
        variance=[0.2023, 0.1994, 0.2010]))
    cifar100_model.add(layers.Resizing(224, 224))
    # cifar100_model.add(layers.Rescaling(1./127.5, offset=-1))

    xfer_model = TransferNet(
        include_top=False,
        weights='imagenet',
        classes=100,
        input_shape=(224, 224, 3))
    # xfer_model.trainable = False
    cifar100_model.add(xfer_model)

    cifar100_model.add(layers.GlobalAveragePooling2D(
        name="cifar100_global_pool"))
    cifar100_model.add(layers.Dropout(
        0.5,
        seed=np_rng.integers(2**32),
        name="cifar100_dropout"))
    cifar100_model.add(layers.Dense(
        100,
        kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
        name="cifar100_dense"))

    return cifar100_model


# boston housing model
def boston_m(np_rng: Generator):
    boston_model = models.Sequential([
        layers.Dense(
            20,
            activation="relu",
            kernel_initializer=GlorotUniform(np_rng.integers(2**32)),
            input_shape=(13, )),
        # layers.Dense(64, activation="relu"),
        layers.Dense(
            1,
            kernel_initializer=GlorotUniform(np_rng.integers(2**32)))])
    return boston_model


def gleam_m(np_rng: Generator):
    gleam_model = models.Sequential()
    gleam_model.add(Input(shape=(18,)))
    gleam_model.add(layers.Normalization(
        mean=GLEAM_MEAN,
        variance=GLEAM_STD))
    # gleam_model.add(layers.Dense(20, activation='relu'))
    gleam_model.add(layers.Dense(
        10,
        activation='relu',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    gleam_model.add(layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=GlorotUniform(np_rng.integers(2**32))))
    return gleam_model


def test_models():
    from keras.datasets import cifar10 as cifar10_data
    from numpy.random import default_rng
    from tensorflow_privacy import DPKerasSGDOptimizer as DPSGD
    from model_utils.min_lr_step import PolynomialDecay
    (x_train, y_train), (x_test, y_test) = cifar10_data.load_data()
    rng = default_rng()
    model = cifar10_vgg16(rng)
    opt_fn = DPSGD
    opt_params = {
        "l2_norm_clip": 3.,
        "noise_multiplier": 0.5,
        "num_microbatches": 1,
        "momentum": .9
    }
    lr_fn = PolynomialDecay
    lr_params = {
        "initial_learning_rate": 1e-4,
        "decay_steps": 2000,
        "end_learning_rate": 1e-4,
        "power": 1.5
    }
    opt = opt_fn(learning_rate=lr_fn(**lr_params), **opt_params)
    from keras.losses import SparseCategoricalCrossentropy as scc
    from keras.metrics import SparseCategoricalAccuracy as sca
    model_params = {
        "loss": scc(from_logits=True),
        "metrics": [sca()]
    }
    model.compile(optimizer=opt, **model_params)
    fit_params = {
        "batch_size": 64,
        "epochs": 10,
        "validation_data": (x_test, y_test)
    }
    model.fit(x_train, y_train, **fit_params)


if __name__ == "__main__":
    test_models()
