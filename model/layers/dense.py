import numpy as np
from model import utils


class Dense:
    def __init__(self, input_size, output_size):
        self.input = None
        self.z = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size[1], output_size) * np.sqrt(2.0 / input_size[1])
        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights) + self.biases
        self.z = utils.softmax(output)
        return self.z

    def backward(self, grad, learning_rate=0.01):
        # Calculate gradients for weights and biases
        weights_error = np.dot(self.input.T, grad)
        bias_error = np.sum(grad, axis=0)
        # Calculate input error for the next layer
        input_error = np.dot(grad, self.weights.T)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * bias_error
        return input_error
