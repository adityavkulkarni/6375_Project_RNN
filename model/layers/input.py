class Input:
    def __init__(self, input_size):
        """
        Class for input layer
        :param input_size:
        """
        self.input_size = input_size
        self.output_size = input_size

    @staticmethod
    def forward(x):
        """
        Forward pass for input layer
        :param x: input tokens
        :return: same input tokens
        """
        return x
