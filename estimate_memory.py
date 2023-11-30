import sys
from model import SimpleFeedForwardNN
from prettytable import PrettyTable
from parse_input import parse_input
import torch as th

def run_estimate(network_shape, batch_size, data_bytes):
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
    model = SimpleFeedForwardNN(input_size, list(hidden_sizes), output_size, dtype)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    mem_usage_backprop = 0
    for layer in (list(hidden_sizes) + [output_size]):
        mem_usage_backprop += layer * data_bytes * batch_size
    mem_usage_optimizer = 3 * total_params * data_bytes
    mem_model = total_params * data_bytes
    print('Total memory estimate for training: {0:5.2f}GB'.format((mem_model +
                                                                   mem_usage_optimizer + mem_usage_backprop) / 1e9))


if __name__ == "__main__":
    network_shape = sys.argv[1]
    batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        data_bytes = int(sys.argv[3])
    else:
        data_bytes = 4

    (network_shape, batch_size) = parse_input(network_shape, batch_size)
    run_estimate(network_shape, batch_size, data_bytes)
