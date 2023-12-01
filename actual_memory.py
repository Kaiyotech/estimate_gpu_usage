import sys

import torch as th

from model import SimpleFeedForwardNN
import torch.nn as nn
import torch.optim as optim
from parse_input import parse_input
from test_optimizer import Adam

def run_actual(network_shape, batch_size, data_bytes):
    if data_bytes == 4:
        dtype = th.float32
    elif data_bytes == 2:
        dtype = th.float16
    elif data_bytes == 8:
        dtype = th.float64
    else:
        dtype = th.float32
        print("Warning: unsupported datatype provided. Defaulting to float32")
    input_size = network_shape[0]
    output_size = network_shape[-1]
    hidden_sizes = network_shape[1:-1]
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = SimpleFeedForwardNN(input_size, list(hidden_sizes), output_size, dtype)
    model.to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = Adam(model.parameters(), lr=1e-4)
    num_steps = 5
    for i in range(num_steps):
        input_data = th.randn(batch_size, input_size, dtype=dtype).to(device)
        target_data = th.randn((batch_size, output_size), dtype=dtype).to(device)

        output = model(input_data)
        loss = criterion(output, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"On Step {i}")
        cuda_memory_allocated = th.cuda.memory_allocated(device)
        cuda_memory_cached = th.cuda.memory_reserved(device)
        cuda_max_allocated = th.cuda.max_memory_allocated(device)
        print("CUDA Memory Allocated:", cuda_memory_allocated / (1024 ** 3), "GB")
        print("CUDA Memory Cached:", cuda_memory_cached / (1024 ** 3), "GB")
        print("CUDA Max Allocated Since Start", cuda_max_allocated / (1024 ** 3), "GB")

        # add some forward passes to simulate reinforcement learning batched on the GPU
        for _ in range(1000):
            input_data = th.randn(10_000, input_size, dtype=dtype).to(device)
            model.forward(input_data)
    # cuda_memory_allocated = th.cuda.memory_allocated(device)
    # cuda_memory_cached = th.cuda.memory_reserved(device)
    # print("CUDA Memory Allocated:", cuda_memory_allocated / (1024 ** 3), "GB")
    # print("CUDA Memory Cached:", cuda_memory_cached / (1024 ** 3), "GB")
    # print(th.cuda.memory_summary(device))


if __name__ == "__main__":
    network_shape = sys.argv[1]
    batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        data_bytes = sys.argv[3]
    else:
        data_bytes = 4

    (network_shape, batch_size) = parse_input(network_shape, batch_size)
    run_actual(network_shape, batch_size, data_bytes)
