class Input:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, x):
        return x

    def backward(self, grad):
        pass
