import numpy as np
import pandas as pd
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.metrics import SparseCategoricalAccuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model.layers.input import Input
from model.layers.recurrent import Recurrent
from model.layers.dense import Dense
from model import utils


class RNN:
    def __init__(self, input_shape, output_shape):
        """
        RNN model
        :param input_shape:
        :param output_shape:
        """
        self.input_layer = Input(input_shape)
        self.recurrent_layer1 = Recurrent(input_shape, input_shape)
        self.dense = Dense(input_shape, output_shape)

    def train(self, x, y, epochs=10, learning_rate=0.0001):
        """
        Method for training model
        :param x: input tokens
        :param y: targets
        :param epochs:
        :param learning_rate:
        :return:
        """
        graph = []
        X_train, X_test, y, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        for epoch in range(epochs):
            y_tr = []
            for i in range(len(X_train)):
                # Forward Pass
                _y = self.predict(X_train[i])
                y_tr.append(_y)
                # Backward Pass
                dw = utils.sparse_categorical_crossentropy_gradient(y[i], _y)
                dw_3 = self.dense.backward(dw, learning_rate)
                self.recurrent_layer1.backward(dw_3, learning_rate)
            _y = []
            m = SparseCategoricalAccuracy()
            for i in range(len(X_test)):
                _y.append(self.predict(X_test[i]))
                m.update_state(y_test[i], np.argmax(_y[i], axis=1))
            _y = np.array(_y)
            y_tr = np.array(y_tr)
            scce = SparseCategoricalCrossentropy()
            scce_tr = SparseCategoricalCrossentropy()
            err = scce(y_test, _y)
            err_tr = scce_tr(y, y_tr)
            acc = m.result()
            graph.append((epoch+1, float(err_tr), float(err)))
            print(f"epoch: {epoch + 1} Loss: {err}")
        print(f"Accuracy: {acc}")
        df = pd.DataFrame(graph, columns=["Epochs", "Training loss", "Validation loss"])
        df.to_csv(f'results/rnn_{learning_rate}_{epochs}.csv', index=False)
        fig1 = plt.figure(figsize=(12, 7))
        ax = fig1.add_subplot(1, 1, 1)
        ax.plot(df["Epochs"], df["Training loss"])
        ax.plot(df["Epochs"], df["Validation loss"])
        plt.legend(["Training Loss", "Validation Loss"])
        plt.title("Loss vs Epochs")
        txt = f"Learning Rate: {learning_rate}"
        plt.figtext(0.5, 0.01,
                    txt + "\nFinal Loss:{:.2f}".format(graph[-1][1]),
                    wrap=True, horizontalalignment='center', fontsize=10)
        fig1.savefig(f"results/loss_{learning_rate}_{epochs}.png")

    def predict(self, x):
        """
        Method for predicting results
        :param x: tokens
        :return: logits prediction
        """
        y_1 = self.input_layer.forward(x)
        y_2 = self.recurrent_layer1.forward(y_1)
        y_3 = self.dense.forward(y_2)
        return y_3
