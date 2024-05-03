import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def softmax_grad(z):
    y = softmax(z)
    return y * (1 - y)


def sparse_categorical_crossentropy(y_true, y_pred):
    loss = []
    for i in range(len(y_true)):
        loss.append(y_pred[i][y_true[i]])
    loss = np.array(loss)
    return loss
