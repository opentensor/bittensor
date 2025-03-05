import math


def get_human_readable(num, suffix="H"):
    """Convert a number into a human-readable format with suffixes."""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Y{suffix}"


def millify(n: int):
    """Converts a number into a more readable format with suffixes."""
    mill_names = ["", " K", " M", " B", " T"]
    n = float(n)
    mill_idx = max(
        0,
        min(
            len(mill_names) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    return "{:.2f}{}".format(n / 10 ** (3 * mill_idx), mill_names[mill_idx])
