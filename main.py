from data_processor import DataProcessor
from model.rnn import RNN
from rnn_benchmark import RNNBenchmark

if __name__ == '__main__':
    data_processor = DataProcessor('data/spa.txt', sentence_count=1000)
    index = 10
    eng_sentence = data_processor.english_sentences[index]
    spa_sentence = data_processor.spanish_sentences[index]

    rnn = RNN(input_shape=(data_processor.max_sentence_length, 1),
              output_shape=data_processor.spanish_vocab_len)
    rnn.train(data_processor.eng_pad_sentence, data_processor.spa_pad_sentence)

    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn.predict(data_processor.eng_pad_sentence[index:index + 1])[0])}")

    rnn_benchmark = RNNBenchmark(input_shape=(data_processor.max_sentence_length, 1),
                                 output_shape=data_processor.spanish_vocab_len)
    rnn_benchmark.train(data_processor.eng_pad_sentence, data_processor.spa_pad_sentence)
    print(f"English sentence: {eng_sentence}")
    print(f"Spanish sentence: {spa_sentence}")
    print(f"Predicted sentence: "
          f"{data_processor.logits_to_sentence(rnn_benchmark.predict(data_processor.eng_pad_sentence[index:index + 1]))}")
