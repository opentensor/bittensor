# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import Optional

from bittensor.constants import ALLOWED_DELTA, NANOSECONDS_IN_SECOND


def allowed_nonce_window_ns(current_time_ns: int, synapse_timeout: Optional[float]):
    synapse_timeout_ns = (synapse_timeout or 0) * NANOSECONDS_IN_SECOND
    allowed_window_ns = current_time_ns - ALLOWED_DELTA - synapse_timeout_ns
    return allowed_window_ns


def calculate_diff_seconds(
    current_time: int, synapse_timeout: Optional[float], synapse_nonce: int
):
    synapse_timeout_ns = (synapse_timeout or 0) * NANOSECONDS_IN_SECOND
    diff_seconds = (current_time - synapse_nonce) / NANOSECONDS_IN_SECOND
    allowed_delta_seconds = (ALLOWED_DELTA + synapse_timeout_ns) / NANOSECONDS_IN_SECOND
    return diff_seconds, allowed_delta_seconds
