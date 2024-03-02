import torch

class Mlp(torch.nn.Module):
    def __init__(
            self,
            input_size,
            layers_data,
            output_size,
            activation_fn=torch.nn.Relu,
            scale_bias=1.0,
            scale_weights=1.0,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for size in layers_data:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(activation_fn())
            input_size = size
        self.output_layer = torch.nn.Linear(input_size, output_size)
        self.output_layer.weight.data.mul_(scale_weights)
        self.output_layer.bias.data.mul_(scale_bias)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        output_data = self.output_layer(input_data)
        return output_data
