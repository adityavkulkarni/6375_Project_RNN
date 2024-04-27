import pickle
import re
import string
import random
from string import digits
from unidecode import unidecode

import pandas as pd
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras.layers import TextVectorization


"""class DataProcessor:
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
        # data['english'] = data['english'].apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))
        # data['spanish'] = data['spanish'].apply(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x))
        max_eng_len = max(data['len_english'])
        max_span_len = max(data['len_spanish'])
        print("Maximum length of English Sentence: ", max_eng_len)
        print("Maximum length of Spanish Sentence: ", max_span_len)
        min_eng_len = min(data['len_english'])
        min_span_len = min(data['len_spanish'])
        print("Minimum length of English Sentence: ", min_eng_len)
        print("Minimum length of Spanish Sentence: ", min_span_len)
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
                self.english_vocab[re.sub('[^A-Za-z0-9 ]+', '', token)] = index
                index += 1
        self.inverse_english_vocab = {index: token for token, index in self.english_vocab.items()}

        self.spanish_vocab, index = {}, 1  # start indexing from 1
        self.spanish_vocab['<pad>'] = 0  # add a padding token
        for token in self.spanish_words:
            if token not in self.spanish_vocab:
                self.spanish_vocab[re.sub('[^A-Za-z0-9 ]+', '', token)] = index
                index += 1
        self.inverse_spanish_vocab = {index: token for token, index in self.spanish_vocab.items()}
        self.english_vectors = []
        self.spanish_vectors = []
        self.vector_map = []
        for i in range(len(data)):
            self.english_vectors.append([self.english_vocab[re.sub('[^A-Za-z0-9 ]+', '', x)] for x in data.loc[i, 'english'].split(" ")])
            self.spanish_vectors.append([self.spanish_vocab[re.sub('[^A-Za-z0-9 ]+', '', x)] for x in data.loc[i, 'spanish'].split(" ")])
            self.vec_len = max(max_span_len, max_eng_len)
        fill = [0] * self.vec_len
        result = [sublist + fill[len(sublist):] for sublist in self.english_vectors]
        self.english_vectors = result
        result = [sublist + fill[len(sublist):] for sublist in self.spanish_vectors]
        self.spanish_vectors = result
        self.vector_map = [[self.english_vectors[i],
                            self.spanish_vectors[i]]
                           for i in range(len(self.english_vectors))]
        print("Vocab created")

    def sentence2vec(self, sentence, language='english'):
        fill = [0] * self.vec_len
        sentence = sentence.lower()
        if language == 'english':
            results = [self.english_vocab[re.sub('[^A-Za-z0-9 ]+', '', x)] for x in sentence.split(" ")]
        else:
            results = [self.spanish_vocab[re.sub('[^A-Za-z0-9 ]+', '', x)] for x in sentence.split(" ")]
        return results + fill[len(results):]

    def vec2sentence(self, vec, language='english'):
        if language == 'english':
            results = " ".join([self.inverse_english_vocab[x] for x in vec])
        else:
            results = " ".join([self.inverse_spanish_vocab[x] for x in vec])
        return results.replace("<pad>", "")
"""
class DataProcessor:
    def __init__(self, data_path):
        with open(data_path) as f:
            lines = f.read().split("\n")[:-1]
        text_pairs = []
        for line in lines:
            eng, spa = line.split("\t")
            spa = "[start] " + spa + " [end]"
            text_pairs.append((eng, spa))
        random.shuffle(text_pairs)
        num_val_samples = int(0.15 * len(text_pairs))
        num_train_samples = len(text_pairs) - 2 * num_val_samples
        train_pairs = text_pairs[:num_train_samples]
        val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
        test_pairs = text_pairs[num_train_samples + num_val_samples:]

        print(f"{len(text_pairs)} total pairs")
        print(f"{len(train_pairs)} training pairs")
        print(f"{len(val_pairs)} validation pairs")
        print(f"{len(test_pairs)} test pairs")

        strip_chars = string.punctuation + "¿"
        strip_chars = strip_chars.replace("[", "")
        strip_chars = strip_chars.replace("]", "")

        self.vocab_size = 15000
        self.sequence_length = 20
        self.batch_size = 64

        def custom_standardization(input_string):
            lowercase = tf_strings.lower(input_string)
            return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

        self.eng_vectorization = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )
        self.spa_vectorization = TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length + 1,
            standardize=custom_standardization,
        )
        train_eng_texts = [pair[0] for pair in train_pairs]
        train_spa_texts = [pair[1] for pair in train_pairs]
        self.eng_vectorization.adapt(train_eng_texts)
        self.spa_vectorization.adapt(train_spa_texts)

        self.train_ds = self.make_data(train_pairs)
        self.val_ds = self.make_data(val_pairs)

        for inputs, targets in self.train_ds.take(1):
            print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
            print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
            print(f"targets.shape: {targets.shape}")

    def make_data(self, pairs):
        eng_texts, spa_texts = zip(*pairs)
        eng_texts = list(eng_texts)
        spa_texts = list(spa_texts)
        dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self.format_dataset)
        return dataset.cache().shuffle(2048).prefetch(16)

    def format_dataset(self, eng, spa):
        eng = self.eng_vectorization(eng)
        spa = self.spa_vectorization(spa)
        return (
            {
                "encoder_inputs": eng,
                "decoder_inputs": spa[:, :-1],
            },
            spa[:, 1:],
        )


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')
