import numpy as np
import tensorflow as tf
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
        output = np.dot(x, self.weights) + self.biases
        self.z = utils.softmax(output)
        return self.z

    def backward(self, grad):
        # Calculate gradients for weights and biases
        weights_error = np.dot(self.input.T, grad)
        bias_error = np.sum(grad, axis=0)
        # Calculate input error for the next layer
        input_error = np.dot(grad, self.weights.T)

        # update parameters
        self.weights -= self.learning_rate * weights_error
        self.biases -= self.learning_rate * bias_error
        return input_error
