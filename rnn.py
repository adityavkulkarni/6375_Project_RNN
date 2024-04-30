import numpy as np
from keras.models import Model
from keras.layers import SimpleRNN, Input, Dense, TimeDistributed
from keras.losses import sparse_categorical_crossentropy, BinaryCrossentropy


class RNN:
    def __init__(self, input_shape, output_sequence_length,
                 spanish_tokenizer, spanish_vocab_size, learning_rate=0.01):
        self.spanish_tokenizer = spanish_tokenizer
        inputs = Input(shape=input_shape[1:])
        #hidden_layer = GRU(output_sequence_length, return_sequences=True)(inputs)
        hidden_layer = SimpleRNN(output_sequence_length, return_sequences=True)(inputs)
        #outputs = TimeDistributed(Dense(spanish_vocab_size + 1, activation='softmax'))(hidden_layer)
        outputs = Dense(spanish_vocab_size + 1, activation='softmax')(hidden_layer)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(metrics=['accuracy'], loss=sparse_categorical_crossentropy)

    def train(self, x, y, epochs=100, batch_size=1024, validation_split=0.2):
        self.model.fit(x, y, epochs=epochs,
                       batch_size=batch_size, validation_split=validation_split)
        self.model.summary()
        return self.model.evaluate(x, y, batch_size=batch_size)

    def predict(self, x):
        return self.emb_to_text(self.model.predict(x)[0])

    def emb_to_text(self, logits):
        index_to_words = {id: word for word, id in self.spanish_tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
