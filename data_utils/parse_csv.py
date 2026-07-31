import numpy as np


def get_csv_data(name: str, np_rng: np.random.Generator):
    if name == 'adult':
        split = .85
        return parse_file("./data_utils/csv_datasets/adult/adult.csv", np_rng, split, header_rows=1)


def parse_file(path: str, np_rng: np.random.Generator, split: float, header_rows=0):
    X = np.genfromtxt(path, delimiter=',', skip_header=header_rows)
    """TO-DO: read about axis in generators"""
    np_rng.shuffle(X)
    y = X[:, -1]
    X = X[:, :-1]
    X_train = X[:int(len(X) * split)]
    y_train = y[:int(len(y) * split)]
    X_test = X[int(len(X) * split):]
    y_test = y[int(len(y) * split):]
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    get_csv_data("adult", np.random.default_rng())
