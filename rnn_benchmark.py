from keras.models import Model
from keras.layers import SimpleRNN, Input, Dense
from keras.losses import sparse_categorical_crossentropy


class RNNBenchmark:
    def __init__(self, input_shape, output_shape):
        input_layer = Input(input_shape, name='InputLayer')
        hidden_layer = SimpleRNN(output_shape, return_sequences=True, dropout=0.5, name='RNNLayer1')(input_layer)
        hidden_layer = SimpleRNN(output_shape, return_sequences=True, dropout=0.5, name='RNNLayer2')(hidden_layer)
        output_layer = Dense(output_shape, activation='softmax', name='Dense1')(hidden_layer)
        self._model = Model(input_layer, output_layer)
        self._model.compile(loss=sparse_categorical_crossentropy,
                            metrics=['accuracy'])
        self._model.summary()

    def train(self, x, y, epochs=10):
        return self._model.fit(x, y, epochs=epochs)

    def predict(self, x):
        return self._model.predict(x)[0]
