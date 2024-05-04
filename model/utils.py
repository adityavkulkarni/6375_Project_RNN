import numpy as np


def softmax(x):
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def sparse_categorical_crossentropy(y_true, y_pred):
    y_pred1 = np.clip(y_pred, 1e-7, 1 - 1e-7)
    y_true1 = np.eye(y_pred.shape[1])[y_true.reshape(-1)]
    cross_entropy = -np.sum(y_true1 * np.log(y_pred1), axis=-1)
    cross_entropy = np.mean(cross_entropy)
    # scce = SparseCategoricalCrossentropy()
    # err = scce(y_true, y_pred)
    return cross_entropy


def sparse_categorical_crossentropy_gradient(y_true, y_pred):
    y_pred1 = np.clip(y_pred, 1e-7, 1 - 1e-7)
    y_true1 = np.eye(y_pred.shape[1])[y_true.reshape(-1)]
    gradients = y_pred1 - y_true1
    return gradients
