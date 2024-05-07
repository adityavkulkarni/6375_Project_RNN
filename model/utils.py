import numpy as np


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    :param x:
    :return:
    """
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def sparse_categorical_crossentropy(y_true, y_pred):
    """
    Compute sparse categorical crossentropy loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred1 = np.clip(y_pred, 1e-7, 1 - 1e-7)
    y_true1 = np.eye(y_pred.shape[1])[y_true.reshape(-1)]
    cross_entropy = -np.sum(y_true1 * np.log(y_pred1), axis=-1)
    cross_entropy = np.mean(cross_entropy)
    return cross_entropy


def sparse_categorical_crossentropy_gradient(y_true, y_pred):
    """
    Compute gradient of sparse categorical crossentropy loss.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred1 = np.clip(y_pred, 1e-7, 1 - 1e-7)
    y_true1 = np.eye(y_pred.shape[1])[y_true.reshape(-1)]
    gradients = y_pred1 - y_true1
    return gradients


def sparse_categorical_accuracy(y_true, y_pred):
    """
    Compute sparse categorical accuracy.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    correct_predictions = np.equal(y_true, y_pred_classes)
    accuracy = np.mean(correct_predictions)
    return accuracy
