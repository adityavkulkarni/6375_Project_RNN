import numpy as np


class Recurrent:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size[0]
        self.hidden_size = hidden_size[0]
        self.output_size = 1
        self.weights_x = np.random.randn(input_size[0], hidden_size[0]) * np.sqrt(2.0 / input_size[0])
        self.weights_h = np.random.randn(input_size[0], hidden_size[0]) * np.sqrt(2.0 / input_size[0])
        self.b = np.zeros((hidden_size[0], 1))
        self.inputs = []
        self.h = []

    def forward(self, x):
        T = len(x)
        self.inputs = x
        self.h = [np.zeros((self.hidden_size, 1))]
        for t in range(T):
            h_t = np.tanh(np.dot(self.weights_x, x[t]) + np.dot(self.weights_h, self.hidden_states[-1]) + self.b)
            self.h.append(h_t)
        return np.array(self.h[1:])

    def backward(self, grad, learning_rate=0.01):
        dWx = np.zeros_like(self.weights_x)
        dWh = np.zeros_like(self.weights_h)
        db = np.zeros_like(self.b)
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.inputs))):
            dh = grad[t] + dh_next
            dtanh = (1 - self.h[t + 1] ** 2) * dh

            dtanh = np.clip(dtanh, -1, 1)

            dWx += np.dot(dtanh, self.inputs[t].T)
            dWh += np.dot(dtanh, self.h[t].T)
            db += dtanh
            dh_next = np.dot(self.weights_h.T, dtanh)
        self.weights_x -= learning_rate * dWx
        self.weights_h -= learning_rate * dWh
        self.b -= learning_rate * db
