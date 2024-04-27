import pickle

from data_processor import DataProcessor
from encoder_decoder import Transformer


if __name__ == '__main__':
    try:
        data_processor = pickle.load(open('data/processed.pkl', 'rb'))
    except FileNotFoundError:
        data_processor = DataProcessor('data/spa.txt')
    model = Transformer(data_processor)
    model.train()
