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

import logging
from symbol import return_stmt

import numpy as np
from bittensor import utils, ss58_address_to_bytes
import pytest


def test_ss58_to_vec_u8(mocker):
    """Tests `utils.ss58_to_vec_u8` function."""
    # Prep
    test_ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    fake_return = b'2\xa6?'
    mocked_ss58_address_to_bytes = mocker.patch.object(utils, "ss58_address_to_bytes", return_value=fake_return)

    # Call
    result = utils.ss58_to_vec_u8(test_ss58_address)

    # Asserts
    mocked_ss58_address_to_bytes.assert_called_once_with(test_ss58_address)
    assert result == [int(byte) for byte in fake_return]
