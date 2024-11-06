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

import math

from bittensor.utils import formatting


def test_get_human_readable():
    """Tests the `get_human_readable` function in the `formatting` module."""
    num1 = 1000
    num2 = 1_000_000
    num3 = 1_000_000_000
    num4 = 150
    negative_num = -1000

    # Test for default output
    assert formatting.get_human_readable(num1) == "1.0KH"

    # Test for different quantities
    assert formatting.get_human_readable(num2) == "1.0MH"
    assert formatting.get_human_readable(num3) == "1.0GH"

    # Test for numbers less than 1000
    assert formatting.get_human_readable(num4) == "150.0H"

    # Test for negative numbers
    assert formatting.get_human_readable(negative_num) == "-1.0KH"

    # Test for different suffix
    assert formatting.get_human_readable(num1, suffix="B") == "1.0KB"
    assert formatting.get_human_readable(num2, suffix="B") == "1.0MB"
    assert formatting.get_human_readable(num3, suffix="B") == "1.0GB"
    assert formatting.get_human_readable(num4, suffix="B") == "150.0B"
    assert formatting.get_human_readable(negative_num, suffix="B") == "-1.0KB"


def test_millify():
    """Test millify function with various cases."""
    # Testing with value 0
    assert formatting.millify(0) == "0.00"
    # Testing with a number in the tens
    assert formatting.millify(10) == "10.00"
    # Testing with a number in the hundreds
    assert formatting.millify(100) == "100.00"
    # Testing with a number in the thousands
    assert formatting.millify(1000) == "1.00 K"
    # Testing with a number in the millions
    assert formatting.millify(1000000) == "1.00 M"
    # Testing with a number in the billions
    assert formatting.millify(1000000000) == "1.00 B"
    # Testing with a number in the trillions
    assert formatting.millify(1000000000000) == "1.00 T"
    # Testing with maximum limit
    mill_names = ["", " K", " M", " B", " T"]
    n = 10 ** (3 * (len(mill_names) - 1) + 1)
    mill_idx = max(
        0,
        min(
            len(mill_names) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    assert formatting.millify(n) == "{:.2f}{}".format(
        n / 10 ** (3 * mill_idx), mill_names[mill_idx]
    )
