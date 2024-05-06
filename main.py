import os
import urllib.request

from data_processor import DataProcessor
from model.rnn import RNN
from rnn_benchmark import RNNBenchmark

DATA_PATH = "data/spa.txt"
DATA_URL = "https://github.com/adityavkulkarni/6375_Project_RNN/raw/master/data/spa.txt"

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    data_processor = DataProcessor(DATA_PATH, sentence_count=1000)
    index = 10
    eng_sentence = data_processor.english_sentences[index]
    spa_sentence = data_processor.spanish_sentences[index]

    rnn = RNN(input_shape=(data_processor.max_sentence_length, 1),
              output_shape=data_processor.spanish_vocab_len)
    rnn.train(data_processor.eng_pad_sentence, data_processor.spa_pad_sentence)

    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn.predict(data_processor.eng_pad_sentence[index]))}")

    rnn_benchmark = RNNBenchmark(input_shape=(data_processor.max_sentence_length, 1),
                                 output_shape=data_processor.spanish_vocab_len)
    rnn_benchmark.train(data_processor.eng_pad_sentence, data_processor.spa_pad_sentence)
    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn_benchmark.predict(data_processor.eng_pad_sentence[index:index + 1]))}")
