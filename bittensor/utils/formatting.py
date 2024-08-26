import math
from typing import List


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


def convert_blocks_to_time(blocks: int, block_time: int = 12) -> tuple[int, int, int]:
    """
    Converts number of blocks into number of hours, minutes, seconds.
    :param blocks: number of blocks
    :param block_time: time per block, by default this is 12
    :return: tuple containing number of hours, number of minutes, number of seconds
    """
    seconds = blocks * block_time
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


def float_to_u16(value: int) -> int:
    # Ensure the input is within the expected range
    if not (0 <= value <= 1):
        raise ValueError("Input value must be between 0 and 1")

    # Calculate the u16 representation
    u16_max = 65535
    return int(value * u16_max)


def u16_to_float(value: int) -> float:
    # Ensure the input is within the expected range
    if not (0 <= value <= 65535):
        raise ValueError("Input value must be between 0 and 65535")

    # Calculate the float representation
    u16_max = 65535
    return value / u16_max


def float_to_u64(value: float) -> int:
    # Ensure the input is within the expected range
    if not (0 <= value < 1):
        raise ValueError("Input value must be between 0 and 1")

    # Convert the float to a u64 value, take the floor value
    return int(math.floor((value * (2**64 - 1)))) - 1


def u64_to_float(value: int) -> float:
    u64_max = 2**64 - 1
    # Allow for a small margin of error (e.g., 1) to account for potential rounding issues
    if not (0 <= value <= u64_max + 1):
        raise ValueError(
            f"Input value ({value}) must be between 0 and {u64_max} (2^64 - 1)"
        )
    return min(value / u64_max, 1.0)  # Ensure the result is never greater than 1.0


def normalize_u64_values(values: List[int]) -> List[int]:
    """
    Normalize a list of u64 values so that their sum equals u64::MAX (2^64 - 1).
    """
    if not values:
        raise ValueError("Input list cannot be empty")

    if any(v < 0 for v in values):
        raise ValueError("Input values must be non-negative")

    total = sum(values)
    if total == 0:
        raise ValueError("Sum of input values cannot be zero")

    u64_max = 2**64 - 1
    normalized = [int((v / total) * u64_max) for v in values]

    # Adjust values to ensure sum is exactly u64::MAX
    current_sum = sum(normalized)
    diff = u64_max - current_sum

    for i in range(abs(diff)):
        if diff > 0:
            normalized[i % len(normalized)] += 1
        else:
            normalized[i % len(normalized)] = max(
                0, normalized[i % len(normalized)] - 1
            )

    # Final check and adjustment
    final_sum = sum(normalized)
    if final_sum > u64_max:
        normalized[-1] -= final_sum - u64_max

    assert (
        sum(normalized) == u64_max
    ), f"Sum of normalized values ({sum(normalized)}) is not equal to u64::MAX ({u64_max})"

    return normalized
