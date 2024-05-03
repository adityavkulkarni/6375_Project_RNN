class Input:
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward(self, x):
        if x.shape == self.input_size:
            return x
        raise ValueError(f'Input shape {x.shape} does not match {self.input_size}')

    def backward(self, grad):
        pass
