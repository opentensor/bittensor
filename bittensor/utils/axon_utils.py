from typing import Optional

ALLOWED_DELTA = 4_000_000_000  # Delta of 4 seconds for nonce validation
NANOSECONDS_IN_SECOND = 1_000_000_000


def allowed_nonce_window_ns(
    current_time_ns: int, synapse_timeout: Optional[float] = None
) -> int:
    """
    Calculates the allowed window for a nonce in nanoseconds.

    Args:
        current_time_ns (int): The current time in nanoseconds.
        synapse_timeout (Optional[float]): The optional timeout for the synapse in seconds. If None, it defaults to 0.

    Returns:
        int: The allowed nonce window in nanoseconds.
    """
    synapse_timeout_ns = (synapse_timeout or 0) * NANOSECONDS_IN_SECOND
    allowed_window_ns = current_time_ns - ALLOWED_DELTA - synapse_timeout_ns
    return allowed_window_ns


def calculate_diff_seconds(
    current_time: int, synapse_timeout: Optional[float], synapse_nonce: int
):
    """
    Calculates the difference in seconds between the current time and the synapse nonce,
    and also returns the allowed delta in seconds.

    Args:
        current_time (int): The current time in nanoseconds.
        synapse_timeout (Optional[float]): The optional timeout for the synapse in seconds.
        synapse_nonce (int): The nonce value for the synapse in nanoseconds.

    Returns:
        tuple: A tuple containing the difference in seconds (float) and the allowed delta in seconds (float).
    """
    synapse_timeout_ns = (synapse_timeout or 0) * NANOSECONDS_IN_SECOND
    diff_seconds = (current_time - synapse_nonce) / NANOSECONDS_IN_SECOND
    allowed_delta_seconds = (ALLOWED_DELTA + synapse_timeout_ns) / NANOSECONDS_IN_SECOND
    return diff_seconds, allowed_delta_seconds
