import numpy as np

from model import utils


class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input = None
        self.z = None
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size[1], output_size)
        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        output = x.dot(self.weights) + self.biases
        self.z = utils.softmax(output)
        return self.z

    def backward(self, grad):
        pass
