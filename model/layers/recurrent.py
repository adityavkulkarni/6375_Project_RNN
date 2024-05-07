import numpy as np


class Recurrent:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size[0]
        self.hidden_size = hidden_size[0]
        self.output_size = 1
        self.weights_x = np.random.randn(input_size[0], 1) * np.sqrt(2.0 / input_size[0])
        self.weights_h = np.random.randn(input_size[0], 1) * np.sqrt(2.0 / input_size[0])
        self.b = np.zeros((hidden_size[0], 1))
        self.inputs = []
        self.outputs = []
        self.h = []

    def forward(self, x):
        """
        Forward pass of the recurrent layer.
        :param x:
        :return:
        """
        T = len(x)
        self.inputs = x
        self.outputs = []
        self.h = np.zeros((self.hidden_size+1, 1))
        for t in range(1, T+1):
            h_t = np.dot(self.weights_x[t-1], x[t-1]) + np.dot(self.weights_h[t-1], self.h[t-1]) + self.b[t-1]
            self.outputs.append(np.tanh(h_t))
            self.h[t] = h_t
        self.outputs.append(self.outputs[-1])
        return np.array(self.outputs[1:])

    def backward(self, grad, learning_rate=0.01):
        """
        Backward(BPTT) pass of the dense layer
        :param grad: gradient from downstream layers
        :param learning_rate:
        :return:
        """
        T = len(self.inputs)
        grad_weights_x = np.zeros_like(self.weights_x)
        grad_weights_h = np.zeros_like(self.weights_h)
        grad_b = np.zeros_like(self.b)
        grad_h_next = np.zeros((1, 1))
        for t in reversed(range(T)):
            grad_output = grad[t]
            grad_h_t = grad_output * (1 - self.outputs[t] ** 2) + grad_h_next
            grad_b += grad_h_t
            grad_weights_x[t] = np.dot(grad_h_t, self.inputs[t])
            grad_weights_h[t] = np.dot(grad_h_t.T, self.h[t])

        np.clip(grad_weights_x, -0.001, 0.001)
        np.clip(grad_weights_h, -0.001, 0.001)
        np.clip(grad_b, -0.001, 0.001)
        self.weights_x -= learning_rate * grad_weights_x/T
        self.weights_h -= learning_rate * grad_weights_h/T
        self.b -= learning_rate * grad_b/T
