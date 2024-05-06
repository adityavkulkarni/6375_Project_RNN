import numpy as np
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.metrics import Accuracy
from sklearn.model_selection import train_test_split

from model.layers.input import Input
from model.layers.recurrent import Recurrent
from model.layers.dense import Dense
from model import utils


class RNN:
    def __init__(self, input_shape, output_shape):
        self.input_layer = Input(input_shape)
        self.recurrent_layer1 = Recurrent(input_shape, input_shape)
        # self.recurrent_layer2 = Recurrent(input_shape, input_shape)
        self.dense = Dense(input_shape, output_shape)

    def train(self, x, y, epochs=10, learning_rate=0.0001):
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
                # dw_2 = self.recurrent_layer2.backward(dw_3, learning_rate)
                self.recurrent_layer1.backward(dw_3, learning_rate)
            _y = []
            m = Accuracy()
            for i in range(len(X_test)):
                _y.append(self.predict(X_test[i]))
                m.update_state(y_test[i], np.argmax(_y[i], axis=1))
                # l_epoch += utils.sparse_categorical_crossentropy(y_test[i], _y)
            _y = np.array(_y)
            scce = SparseCategoricalCrossentropy()
            err = scce(y_test, _y)
            acc = m.result()
            print(f"epoch: {epoch + 1} Loss: {err} Accuracy: {acc}")

    def predict(self, x):
        y_1 = self.input_layer.forward(x)
        y_2 = self.recurrent_layer1.forward(y_1)
        y_3 = self.dense.forward(y_2)
        return y_3
