def parse_input(tuple_str, int_str):
    tuple_values = tuple(map(int, tuple_str.replace("(", "").replace(")", "").split(",")))

    # Convert the second part to an integer
    single_int_value = int(int_str.replace("_", ""))

    return tuple_values, single_int_value
