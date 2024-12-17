# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
