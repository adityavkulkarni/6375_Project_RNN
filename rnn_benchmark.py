import keras
from keras.models import Model
from keras.layers import SimpleRNN, Input, Dense
from keras.losses import sparse_categorical_crossentropy


class RNNBenchmark:
    def __init__(self, input_shape, output_shape):
        """
        Keras RNN benchmark class.
        :param input_shape:
        :param output_shape:
        """
        input_layer = Input(input_shape, name='InputLayer')
        hidden_layer = SimpleRNN(output_shape, return_sequences=True, name='RNNLayer1')(input_layer)
        output_layer = Dense(output_shape, activation='softmax', name='Dense')(hidden_layer)
        self._model = Model(input_layer, output_layer)
        self._model.compile(loss=sparse_categorical_crossentropy,
                            metrics=['sparse_categorical_accuracy'])
        self._model.summary()
        """keras.utils.plot_model(
            self._model, show_shapes=True, show_layer_names=True,
            to_file="model.png")"""

    def train(self, x, y, epochs=10):
        """
        Method for training model
        :param x: input tokens
        :param y: targets
        :param epochs:
        :return:
        """
        return self._model.fit(x, y, epochs=epochs)

    def predict(self, x):
        """
        Method for predicting results
        :param x: tokens
        :return: logits prediction
        """
        return self._model.predict(x)[0]
