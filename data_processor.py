import re

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataProcessor:
    def __init__(self, data_file, sentence_count=1000):
        """
        Class for data preparation, prerprocessing and handling of the data.
        :param data_file: file containing sentence pairs.
        :param sentence_count: number of sentence pairs to load
        """
        with open(data_file, "r", encoding='utf-8') as f:
            raw = f.read()
        self.max_sentence_length = 10
        raw_data = list(set(raw.split('\n')))
        pairs = [sentence.split('\t') for sentence in raw_data]
        pairs = [[x[0], x[1]] for x in pairs if 4 < len(x[0].split()) < 12]
        pairs = sorted(pairs, key=lambda x: len(x[1]))
        pairs = pairs[:sentence_count]

        self.english_sentences = [re.sub('[^A-Za-z0-9 ]+', '', pair[0].lower()) for pair in pairs]
        self.spanish_sentences = [re.sub('[^A-Za-z0-9 ]+', '', pair[1].lower()) for pair in pairs]

        spa_text_tokenized, self.spa_text_tokenizer = self.tokenize(self.spanish_sentences)
        eng_text_tokenized, self.eng_text_tokenizer = self.tokenize(self.english_sentences)

        self.spanish_vocab_len = len(self.spa_text_tokenizer.word_index) + 1
        self.english_vocab_len = len(self.eng_text_tokenizer.word_index) + 1

        spa_pad_sentence = pad_sequences(spa_text_tokenized, self.max_sentence_length, padding="post")
        eng_pad_sentence = pad_sequences(eng_text_tokenized, self.max_sentence_length, padding="post")

        self.spa_pad_sentence = spa_pad_sentence.reshape(*spa_pad_sentence.shape, 1)
        self.eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

    @staticmethod
    def tokenize(sentences):
        """
        This function tokenizes sentences.
        :param sentences:
        :return: tokenized sentences and tokenizer object
        """
        text_tokenizer = Tokenizer()
        text_tokenizer.fit_on_texts(sentences)
        return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

    def logits_to_sentence(self, logits):
        """
        Converts a logit vector to a sentence.
        :param logits: output logits
        :return: string
        """
        index_to_words = {idx: word for word, idx in self.spa_text_tokenizer.word_index.items()}
        index_to_words[0] = '<empty>'
        return ' '.join([index_to_words[prediction]
                         for prediction in np.argmax(logits, axis=1)]).replace(index_to_words[0], '')


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')
