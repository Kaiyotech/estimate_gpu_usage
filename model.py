import torch.nn as nn

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dtype):
        super(SimpleFeedForwardNN, self).__init__()

        # Define the layers using nn.ModuleList
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=dtype))
            # Apply LeakyReLU activation except for the output layer
            if i < len(sizes) - 2:
                # layers.append(nn.LeakyReLU())
                # changed to ReLU
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
