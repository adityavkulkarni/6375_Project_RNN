import numpy as np
from keras.models import Model
from keras.layers import SimpleRNN, Input, Dense, TimeDistributed, GRU
from keras.losses import sparse_categorical_crossentropy, BinaryCrossentropy


class RNN:
    def __init__(self, input_shape, output_sequence_length,
                 spanish_tokenizer, spanish_vocab_size, learning_rate=0.01):
        self.spanish_tokenizer = spanish_tokenizer
        inputs = Input(shape=input_shape[1:])
        # hidden_layer = GRU(output_sequence_length, return_sequences=True)(inputs)
        hidden_layer = SimpleRNN(output_sequence_length, return_sequences=True)(inputs)
        # outputs = TimeDistributed(Dense(spanish_vocab_size + 1, activation='softmax'))(hidden_layer)
        outputs = Dense(spanish_vocab_size + 1, activation='softmax')(hidden_layer)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(metrics=['accuracy'], loss=sparse_categorical_crossentropy)
        self.model.summary()

    def train(self, x, y, epochs=100, batch_size=1024, validation_split=0.2):
        self.model.fit(x, y, epochs=epochs,
                       batch_size=batch_size, validation_split=validation_split)
        # return self.model.evaluate(x, y, batch_size=batch_size)

    def predict(self, x):
        return self.emb_to_text(self.model.predict(x)[0])

    def emb_to_text(self, logits, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.spanish_tokenizer
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'
        n =np.argmax(logits, 1)
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    def get_word_indices(self, predictions, tokenizer=None):
          """
          Extracts word indices from model predictions and a word index dictionary.

          Args:
              predictions: A 3D NumPy array representing model output.
              word_index: A dictionary mapping integer indices to words in the vocabulary.

          Returns:
              A list of lists containing the predicted word indices for each input sequence
              in the batch.
          """
          if tokenizer is None:
              tokenizer = self.spanish_tokenizer
          index_to_words = {id: word for word, id in tokenizer.word_index.items()}
          index_to_words[0] = '<PAD>'
          predicted_word_indices = []
          # Loop through each sample in the batch (Dimension 1)
          for sequence_predictions in predictions:
              # Get the most likely word index at each timestep (Dimension 2)
              word_indices = np.argmax(sequence_predictions, axis=1)
              predicted_word_indices.append(" ".join([index_to_words[x] for x in word_indices.tolist()]))
          return predicted_word_indices