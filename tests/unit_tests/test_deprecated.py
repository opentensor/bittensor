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
    """Tests that `bittensor.extrinsics` can be imported and is the same as `bittensor.utils.deprecated.extrinsics`."""
    import bittensor.extrinsics as redirected_extrinsics
    import bittensor.core.extrinsics as real_extrinsics

    assert "bittensor.extrinsics" in sys.modules
    assert redirected_extrinsics is real_extrinsics


def test_object_aliases_are_correctly_mapped():
    """Ensures all object aliases correctly map to their respective classes in Bittensor package."""
    import bittensor

    assert issubclass(bittensor.axon, bittensor.Axon)
    assert issubclass(bittensor.config, bittensor.Config)
    assert issubclass(bittensor.dendrite, bittensor.Dendrite)
    assert issubclass(bittensor.keyfile, bittensor.Keyfile)
    assert issubclass(bittensor.metagraph, bittensor.Metagraph)
    assert issubclass(bittensor.wallet, bittensor.Wallet)
    assert issubclass(bittensor.synapse, bittensor.Synapse)
