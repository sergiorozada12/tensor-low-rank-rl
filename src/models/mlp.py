import torch

class Mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_1_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_1_size, hidden_2_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_2_size, output_size)

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)
        output = self.fc3(relu2)
        return output