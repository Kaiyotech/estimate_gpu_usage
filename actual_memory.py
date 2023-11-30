import sys

import torch as th

import model
import torch.nn as nn
import torch.optim as optim

def parse_input(tuple_str, int_str):
    tuple_values = tuple(map(int, tuple_str.replace("(", "").replace(")", "").split(",")))

    # Convert the second part to an integer
    single_int_value = int(int_str.replace("_", ""))

    return tuple_values, single_int_value


if __name__ == "__main__":
    network_shape = sys.argv[1]
    batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        data_bytes = sys.argv[3]
    else:
        data_bytes = 4

    (network_shape, batch_size) = parse_input(network_shape, batch_size)
    input_size = network_shape[0]
    output_size = network_shape[-1]
    hidden_sizes = network_shape[1:-1]

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = model.SimpleFeedForwardNN(input_size, list(hidden_sizes), output_size)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(10):
        input_data = th.randn(batch_size, input_size).to(device)
        target_data = th.randn((batch_size, output_size), dtype=th.float32).to(device)

        output = model(input_data)
        loss = criterion(output, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    cuda_memory_allocated = th.cuda.memory_allocated(device)
    cuda_memory_cached = th.cuda.memory_reserved(device)

    print("CUDA Memory Allocated:", cuda_memory_allocated / (1024 ** 3), "GB")
    print("CUDA Memory Cached:", cuda_memory_cached / (1024 ** 3), "GB")
    print(th.cuda.memory_summary(device))
