import random
import re

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataProcessor:
    def __init__(self, data_file, sentence_count=1000, offset=20000):
        with open(data_file, "r", encoding='utf-8') as f:
            raw = f.read()

        raw_data = list(set(raw.split('\n')))
        pairs = [sentence.split('\t') for sentence in raw_data]
        pairs = pairs[offset: offset + sentence_count + 1]

        self.english_sentences = [re.sub('[^A-Za-z0-9 ]+', '', pair[0].lower()) for pair in pairs]
        self.spanish_sentences = [re.sub('[^A-Za-z0-9 ]+', '', pair[1].lower()) for pair in pairs]

        spa_text_tokenized, self.spa_text_tokenizer = self.tokenize(self.spanish_sentences)
        eng_text_tokenized, self.eng_text_tokenizer = self.tokenize(self.english_sentences)
        self.max_sentence_length = max(len(max(spa_text_tokenized, key=len)), len(max(eng_text_tokenized, key=len)))
        # print('Maximum length spanish sentence: {}'.format(len(max(spa_text_tokenized, key=len))))
        # print('Maximum length english sentence: {}'.format(len(max(eng_text_tokenized, key=len))))

        self.spanish_vocab_len = len(self.spa_text_tokenizer.word_index) + 1
        self.english_vocab_len = len(self.eng_text_tokenizer.word_index) + 1
        # print("Spanish vocabulary is of {} unique words".format(self.spanish_vocab))
        # print("English vocabulary is of {} unique words".format(self.english_vocab))

        spa_pad_sentence = pad_sequences(spa_text_tokenized, self.max_sentence_length, padding="post")
        eng_pad_sentence = pad_sequences(eng_text_tokenized, self.max_sentence_length, padding="post")

        self.spa_pad_sentence = spa_pad_sentence.reshape(*spa_pad_sentence.shape, 1)
        self.eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

    @staticmethod
    def tokenize(sentences):
        text_tokenizer = Tokenizer()
        text_tokenizer.fit_on_texts(sentences)
        return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

    def logits_to_sentence(self, logits):
        index_to_words = {idx: word for word, idx in self.spa_text_tokenizer.word_index.items()}
        index_to_words[0] = '<empty>'
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)]).replace(index_to_words[0], '')


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')
