# CS6375 - Machine Learning
## Recurrent Neural Network With Backpropogation Through Time For Machine Translation
___
### Dataset: 
#### ANKI: English to Spanish dataset
- Source: [ANKI language translation dataset](https://www.manythings.org/anki/)
- Available in repository: [6375_Project_RNN/data/spa.tx](https://github.com/adityavkulkarni/6375_Project_RNN/blob/master/data/spa.txt)

___
### Execution Instructions:
- Execution:
  ```bash
  python main.py --learning-rate 0.01 --epochs 100 --samples 3000
  ```
  ```
  --learning-rate LEARNING_RATE
                        Learning rate for model
  --epochs EPOCHS       Epochs for model
  --samples SAMPLES     Number of sentences for training
  ```
- All necessary packages are listed in [requirements.txt](requirements.txt)  

___
### File Structure:
- ReadME.md 
- [data](data): directory for datasets 
- [rnn_benchmark.py](rnn_benchmark.py) : Keras model used for benchmarking
- [main.py](main.py) : main file 
- [data_processor.py](data_processor.py): file containing class for data preprocessing
- [model](model): directory containing RNN code
  - [layers](model/layers): directory containing code for each layer
    - [dense.py](model/layers/dense.py): file containing dense layer
    - [input.py](model/layers/input.py): file containing input layer
    - [recurrent.py](model/layers/recurrent.py): file containing recurrent layer
  - [rnn.py](model/rnn.py): file containing the main RNN model
  - [utils](model/utils.py): file containing additional functions 
- [results](results) : directory for storing output 
