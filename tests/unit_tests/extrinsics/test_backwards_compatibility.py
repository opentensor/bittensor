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

import sys


def test_mock_import():
    """
    Tests that `bittensor.mock` can be imported and is the same as `bittensor.utils.mock`.
    """
    import bittensor.mock as redirected_mock
    import bittensor.utils.mock as real_mock

    assert "bittensor.mock" in sys.modules
    assert redirected_mock is real_mock


def test_extrinsics_import():
    """Tests that `bittensor.extrinsics` can be imported and is the same as `bittensor.utils.backwards_compatibility.extrinsics`."""
    import bittensor.extrinsics as redirected_extrinsics
    import bittensor.utils.backwards_compatibility.extrinsics as real_extrinsics

    assert "bittensor.extrinsics" in sys.modules
    assert redirected_extrinsics is real_extrinsics
