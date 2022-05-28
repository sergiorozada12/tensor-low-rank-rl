import torch

class Critic(torch.nn.Module):
    def __init__(self, input_size, layers_data):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for size in layers_data:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(torch.nn.ReLU())
            input_size = size
        self.output_layer = torch.nn.Linear(input_size, 1)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        value = self.output_layer(input_data)
        return value
