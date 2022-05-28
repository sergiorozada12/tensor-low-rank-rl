import torch

class Actor(torch.nn.Module):
    def __init__(self, input_size, layers_data, output_size):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for size in layers_data:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(torch.nn.ReLU())
            input_size = size
        self.output_layer = torch.nn.Linear(input_size, output_size)
        self.output_activation = torch.nn.Softmax()

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        output_data = self.output_layer(input_data)
        logits = self.output_activation(output_data)
        return logits
