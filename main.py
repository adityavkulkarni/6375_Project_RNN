import random

from data_processor import DataProcessor
from encoder_decoder import Transformer


if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt')
    model = Transformer(data_processor)
    model.train(epochs=10)
    for _ in range(30):
        input_sentence, spanish_sentence = random.choice(data_processor.test_pairs)
        translated = model.decode_sequence(input_sentence)
        print(f"English:{input_sentence}\n "
              f"Original:{spanish_sentence.replace('[start]','').replace('[end]', '')}\n "
              f"Translated: {translated.replace('[start]','').replace('[end]', '')}\n")
