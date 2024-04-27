import pickle

from data_processor import DataProcessor
from encoder_decoder import Transformer


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')
    model = Transformer(data_processor)
    model.train(epochs=1)
