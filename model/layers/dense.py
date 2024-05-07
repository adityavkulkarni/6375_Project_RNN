import numpy as np
from model import utils


class Dense:
    def __init__(self, input_size, output_size):
        """
        Class for dense layer
        :param input_size:
        :param output_size:
        """
        self.input = None
        self.z = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size[1], output_size) * np.sqrt(2.0 / input_size[1])
        self.biases = np.zeros(output_size)

    def forward(self, x):
        """
        Forward pass of the dense layer
        :param x:
        :return:
        """
        self.input = x
        output = np.dot(x, self.weights) + self.biases
        self.z = utils.softmax(output)
        return self.z

    def backward(self, grad, learning_rate=0.01):
        """
        Backward pass of the dense layer
        :param grad: gradient from downstream layers
        :param learning_rate:
        :return:
        """
        dw = np.dot(self.input.T, grad)
        db = np.sum(grad, axis=0)
        d_next = np.dot(grad, self.weights.T)

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        return d_next
