import os
import numpy as np
import scipy.io
import tensorflow as tf


def read_school_data(data_dir='data_utils/csv_datasets/schools', test_frac=0.3, seed=None, bias=False, standardize=True, **__kwargs):
    """Read School dataset."""
    x_trains, y_trains = [], []
    mat = scipy.io.loadmat(os.path.join(data_dir, 'school.mat'))
    raw_x, raw_y = mat['X'][0], mat['Y'][0]  # y is exam score

    for i in range(len(raw_x)):   # For each client
        features, label = raw_x[i], raw_y[i].flatten()
        features = features.astype(float)
        label = label.astype(float)

        if standardize:
            # Preprocessing using mean/std from training examples, within each silo
            mean_train = np.mean(features)
            features -= mean_train
            std_train = np.std(features)
            features /= std_train
            min_y, max_y = 1, 70    # Hardcode stats from dataset.
            label = (label - min_y) / (max_y - min_y)
        if bias:
            features = np.c_[features, np.ones(len(features))]

        # features / exam scores should be float (if not standardized)
        x_trains.append(features.astype(float))
        y_trains.append(label.astype(float))

    x_train = np.vstack(x_trains)
    y_train = np.concatenate(y_trains)
    y_train = y_train.reshape((len(y_train), 1))
    train = np.hstack((x_train, y_train))

    # save to csv
    np.savetxt('data_utils/csv_datasets/schools/schools.csv',
               train, delimiter=",")
    return
