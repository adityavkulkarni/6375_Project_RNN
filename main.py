import random

from data_processor import DataProcessor, DataProcessorV1
from encoder_decoder import Transformer
from rnn import RNN

if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt', sentence_count=1000)
    rnn = RNN(data_processor.training_data_x.shape,
              data_processor.max_spanish_sequence_length,
              data_processor.spanish_tk,
              data_processor.spanish_vocab_size)
    rnn.train(data_processor.training_data_x,
              data_processor.training_data_y,
              batch_size=1024, epochs=100, validation_split=0.2)
    for t in data_processor.training_data_x:
        print(rnn.predict(t))
    """ 
    data_processor = DataProcessor('data/spa.txt')
    model = Transformer(data_processor)
    model.train(epochs=10)
    for _ in range(30):
        input_sentence, spanish_sentence = random.choice(data_processor.test_pairs)
        translated = model.decode_sequence(input_sentence)
        print(f"English:{input_sentence}\n "
              f"Original:{spanish_sentence.replace('[start]','').replace('[end]', '')}\n "
              f"Translated: {translated.replace('[start]','').replace('[end]', '')}\n")
    """
