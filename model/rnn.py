import numpy as np

from model.layers.input import Input
from model.layers.recurrent import Recurrent
from model.layers.dense import Dense
from model import utils


class RNN:
    def __init__(self, input_shape, output_shape):
        self.input_layer = Input(input_shape)
        # self.recurrent_layer = Recurrent(output_shape)
        self.dense = Dense(input_shape, output_shape)

    def train(self, x, y):
        print(f"Loss: {self.loss(x, y)}")
        for _ in range(100):
            for i in range(len(x)):
                # Forward Pass
                _y = self.predict(x[i])
                # Backward Pass
                dw = self.__gradient__(y[i], _y)
                dw_3 = self.dense.backward(dw)
                # dw_2 = self.recurrent_layer.backward(dw_3)
                self.input_layer.backward(dw_3)
            print(f"Loss: {self.loss(x, y)}")

    def predict(self, x):
        y_1 = self.input_layer.forward(x)
        # y_2 = self.recurrent_layer.forward(y_1)
        y_3 = self.dense.forward(y_1)
        return y_3

    def loss(self, x, y):
        s = 0
        for i in range(len(x)):
            _y = self.predict(x[i])
            s += utils.sparse_categorical_crossentropy(y[i], _y)
        return s

    @staticmethod
    def __gradient__(y, _y):
        pass
