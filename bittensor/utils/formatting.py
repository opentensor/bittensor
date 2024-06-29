import math


def get_human_readable(num, suffix="H"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Y{suffix}"


def millify(n: int):
    millnames = ["", " K", " M", " B", " T"]
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.2f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def float_to_u16(value):
    # Ensure the input is within the expected range
    if not (0 <= value <= 1):
        raise ValueError("Input value must be between 0 and 1")

    # Calculate the u16 representation
    u16_max = 65535
    return int(value * u16_max)


def u16_to_float(value):
    # Ensure the input is within the expected range
    if not (0 <= value <= 65535):
        raise ValueError("Input value must be between 0 and 65535")

    # Calculate the float representation
    u16_max = 65535
    return value / u16_max
