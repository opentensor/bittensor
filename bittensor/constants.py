# The MIT License (MIT)
# Copyright © 2023 OpenTensor Foundation

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

# Standard Library
import asyncio
from typing import Dict, Type

# 3rd Party
import aiohttp


ALLOWED_DELTA = 4_000_000_000  # Delta of 4 seconds for nonce validation
V_7_2_0 = 7002000
NANOSECONDS_IN_SECOND = 1_000_000_000

#### Dendrite ####
DENDRITE_ERROR_MAPPING: Dict[Type[Exception], tuple] = {
    aiohttp.ClientConnectorError: ("503", "Service unavailable"),
    asyncio.TimeoutError: ("408", "Request timeout"),
    aiohttp.ClientResponseError: (None, "Client response error"),
    aiohttp.ClientPayloadError: ("400", "Payload error"),
    aiohttp.ClientError: ("500", "Client error"),
    aiohttp.ServerTimeoutError: ("504", "Server timeout error"),
    aiohttp.ServerDisconnectedError: ("503", "Service disconnected"),
    aiohttp.ServerConnectionError: ("503", "Service connection error"),
}

DENDRITE_DEFAULT_ERROR = ("422", "Failed to parse response")
#### End Dendrite ####
