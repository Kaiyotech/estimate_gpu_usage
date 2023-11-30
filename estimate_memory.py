import sys
import model
from prettytable import PrettyTable


def parse_input(tuple_str, int_str):
    tuple_values = tuple(map(int, tuple_str.replace("(", "").replace(")", "").split(",")))

    # Convert the second part to an integer
    single_int_value = int(int_str.replace("_", ""))

    return tuple_values, single_int_value


if __name__ == "__main__":
    network_shape = sys.argv[1]
    batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        data_bytes = int(sys.argv[3])
    else:
        data_bytes = 4

    (network_shape, batch_size) = parse_input(network_shape, batch_size)
    input_size = network_shape[0]
    output_size = network_shape[-1]
    hidden_sizes = network_shape[1:-1]

    model = model.SimpleFeedForwardNN(input_size, list(hidden_sizes), output_size)

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    critic_params = 0
    actor_params = 0
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
