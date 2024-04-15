import re
import string
from string import digits
from unidecode import unidecode

import pandas as pd


class DataProcessor:
    def __init__(self, file_name):
        data = pd.read_csv(file_name,
                           delimiter="\t", encoding='utf-8',
                           names=['english', 'spanish'], header=None)
        data.drop_duplicates(inplace=True)
        data['english'] = data['english'].apply(lambda x: x.lower())
        data['spanish'] = data['spanish'].apply(lambda x: x.lower())
        data['english'] = data['english'].apply(lambda x: re.sub("'", '', x))
        data['spanish'] = data['spanish'].apply(lambda x: re.sub("'", '', x))
        sp_chars = set(string.punctuation)  # Set of all special characters
        # Remove all the special characters
        data['english'] = data['english'].apply(lambda x: ''.join(ch for ch in x if ch not in sp_chars))
        data['spanish'] = data['spanish'].apply(lambda x: ''.join(ch for ch in x if ch not in sp_chars))
        data['spanish'] = data['spanish'].apply(lambda x: unidecode(x))
        # Remove all numbers from text
        remove_digits = str.maketrans('', '', digits)
        data['english'] = data['english'].apply(lambda x: x.translate(remove_digits))
        data['spanish'] = data['spanish'].apply(lambda x: x.translate(remove_digits))
        # Remove extra spaces
        data['english'] = data['english'].apply(lambda x: x.strip())
        data['spanish'] = data['spanish'].apply(lambda x: x.strip())
        data['english'] = data['english'].apply(lambda x: re.sub(" +", " ", x))
        data['spanish'] = data['spanish'].apply(lambda x: re.sub(" +", " ", x))
        # data['spanish'] = data['spanish'].apply(lambda x: 'START_ ' + x + ' _END')
        print("Sentence count:", len(data))
        # Word Count
        data['len_english'] = data['english'].apply(lambda x: len(x.split(" ")))
        data['len_spanish'] = data['spanish'].apply(lambda x: len(x.split(" ")))

        data['english'] = data['english'].apply(lambda x: str(x).replace(u'\xa0', u' '))
        data['spanish'] = data['spanish'].apply(lambda x: str(x).replace(u'\xa0', u' '))
        max_eng_len = max(data['len_english'])
        max_span_len = max(data['len_spanish'])
        print("Maximum length of English Sentence: ", max_eng_len)
        print("Maximum length of Spanish Sentence: ", max_span_len)

        self.english_words = set()
        for eng in data['english']:
            for word in eng.split():
                if word not in self.english_words:
                    self.english_words.add(word)

        self.spanish_words = set()
        for hin in data['spanish']:
            for word in hin.split():
                if word not in self.spanish_words:
                    self.spanish_words.add(word)

        self.english_vocab, index = {}, 1  # start indexing from 1
        self.english_vocab['<pad>'] = 0  # add a padding token
        for token in self.english_words:
            if token not in self.english_vocab:
                self.english_vocab[token] = index
                index += 1
        self.inverse_english_vocab = {index: token for token, index in self.english_vocab.items()}

        self.spanish_vocab, index = {}, 1  # start indexing from 1
        self.spanish_vocab['<pad>'] = 0  # add a padding token
        for token in self.spanish_words:
            if token not in self.spanish_vocab:
                self.spanish_vocab[token] = index
                index += 1
        self.inverse_spanish_vocab = {index: token for token, index in self.spanish_vocab.items()}
        self.english_vectors = []
        self.spanish_vectors = []
        for i in range(len(data)):
            self.english_vectors.append([self.english_vocab[x] for x in data.loc[i, 'english'].split(" ")])
            self.spanish_vectors.append([self.spanish_vocab[x] for x in data.loc[i, 'spanish'].split(" ")])
        fill = [0] * max(max_span_len, max_eng_len)
        result = [sublist + fill[len(sublist):] for sublist in self.english_vectors]
        self.english_vectors = result
        result = [sublist + fill[len(sublist):] for sublist in self.spanish_vectors]
        self.spanish_vectors = result
        print("Vocab created")


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')