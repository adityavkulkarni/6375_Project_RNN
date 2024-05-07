import argparse
import os
import urllib.request

from data_processor import DataProcessor
from model.rnn import RNN
from rnn_benchmark import RNNBenchmark

DATA_PATH = "data/spa.txt"
DATA_URL = "https://github.com/adityavkulkarni/6375_Project_RNN/raw/master/data/spa.txt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train a neural network.')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate for model', default=0.001)
    parser.add_argument('--epochs', type=int,
                        help='Epochs for model', default=100)
    parser.add_argument('--samples', type=int,
                        help='Number of sentences for training', default=1000)
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    data_processor = DataProcessor(DATA_PATH, sentence_count=args.samples)
    index = 10
    eng_sentence = data_processor.english_sentences[index]
    spa_sentence = data_processor.spanish_sentences[index]

    rnn = RNN(input_shape=(data_processor.max_sentence_length, 1),
              output_shape=data_processor.spanish_vocab_len)
    rnn.train(data_processor.eng_pad_sentence,
              data_processor.spa_pad_sentence,
              epochs=args.epochs, learning_rate=args.learning_rate)

    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn.predict(data_processor.eng_pad_sentence[index]))}")

    rnn_benchmark = RNNBenchmark(input_shape=(data_processor.max_sentence_length, 1),
                                 output_shape=data_processor.spanish_vocab_len)
    rnn_benchmark.train(data_processor.eng_pad_sentence, data_processor.spa_pad_sentence,
                        epochs=args.epochs)
    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn_benchmark.predict(data_processor.eng_pad_sentence[index:index + 1]))}")
