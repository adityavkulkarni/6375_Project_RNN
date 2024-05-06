import numpy as np
from sklearn.model_selection import train_test_split

from model.layers.input import Input
from model.layers.recurrent import Recurrent
from model.layers.dense import Dense
from model import utils


class RNN:
    def __init__(self, input_shape, output_shape):
        self.input_layer = Input(input_shape)
        self.recurrent_layer = Recurrent(input_shape, input_shape)
        self.dense = Dense(input_shape, output_shape)

    def train(self, x, y, epochs=100, learning_rate=0.001):
        X_train, X_test, y, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        # print(f"Loss: {self.loss(x, y)}")
        for epoch in range(epochs):
            l_epoch = 0
            a = 0
            for i in range(len(X_train)):
                # Forward Pass
                _y = self.predict(X_train[i])
                # Backward Pass
                dw = utils.sparse_categorical_crossentropy_gradient(y[i], _y)
                dw_3 = self.dense.backward(dw, learning_rate)
                self.recurrent_layer.backward(dw_3, learning_rate)
            for i in range(len(X_test)):
                _y = self.predict(X_train[i])
                l_epoch += utils.sparse_categorical_crossentropy(y_test[i], _y)
                a += self.sparse_categorical_accuracy(y_test[i], _y)
            print(f"epoch: {epoch + 1} Loss: {l_epoch/len(X_test)} Accuracy: {a/ len(X_test)}")

    def predict(self, x):
        y_1 = self.input_layer.forward(x)
        # y_2 = self.recurrent_layer.forward(y_1)
        y_3 = self.dense.forward(y_1)
        return y_3

    def sparse_categorical_accuracy(self, y_true, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        correct_predictions = np.equal(y_true, y_pred_classes)
        accuracy = np.mean(correct_predictions)
        return accuracy

    def loss(self, x, y):
        s = 0
        for i in range(len(x)):
            _y = self.predict(x[i])
            s += utils.sparse_categorical_crossentropy(y[i], _y)
        return s
