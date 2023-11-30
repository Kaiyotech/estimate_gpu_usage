import sys
from estimate_memory import run_estimate
from actual_memory import run_actual
from parse_input import parse_input

if __name__ == "__main__":
    network_shape = sys.argv[1]
    batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        data_bytes = int(sys.argv[3])
    else:
        data_bytes = 4

    (network_shape, batch_size) = parse_input(network_shape, batch_size)
    run_estimate(network_shape, batch_size, data_bytes)
    run_actual(network_shape, batch_size, data_bytes)
