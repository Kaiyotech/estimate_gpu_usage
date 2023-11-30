import torch
import torch.nn as nn
import torch.optim as optim

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleFeedForwardNN, self).__init__()

        # Define the layers using nn.ModuleList
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # Apply LeakyReLU activation except for the output layer
            if i < len(sizes) - 2:
                layers.append(nn.LeakyReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
