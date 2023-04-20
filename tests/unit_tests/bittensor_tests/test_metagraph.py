# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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

import unittest
import bittensor
from tests.helpers import get_mock_neuron, get_mock_hotkey, get_mock_coldkey

class TestMetagraph(unittest.TestCase):
    """
    Tests metagraph class methods.
    """
    
    def test_from_neurons(self):
        mock_info_dict = {k: 0 for k in list(bittensor.SubnetInfo.__annotations__.keys())}
        mock_info_dict['burn'] = bittensor.Balance(0)

        metagraph = bittensor.metagraph.from_neurons(
            network = "mock",
            netuid = -1, 
            block = 0,
            neurons = [
                get_mock_neuron(
                    uid = i,
                    hotkey = get_mock_hotkey(i + 1), # +1 to avoid 0 as this gives null neuron
                    coldkey = get_mock_coldkey(i + 1),
                )
            for i in range(2000)],
            info = bittensor.SubnetInfo(
                **mock_info_dict
            )
        )

        # Test each property.
        self.assertEqual(metagraph.network, "mock")
        self.assertEqual(metagraph.netuid, -1)
        self.assertEqual(metagraph.n, 2000)
        self.assertEqual(len(metagraph.hotkeys), 2000)
        self.assertEqual(len(metagraph.coldkeys), 2000)
        self.assertEqual(len(metagraph.uids), 2000)
