# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
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

import bittensor
import torch
import unittest
from bittensor._subtensor.subtensor_mock import mock_subtensor


class TestMetagraph(unittest.TestCase):
    
    def setUp(self):
        mock_subtensor.kill_global_mock_process()
        self.sub = bittensor.subtensor(_mock=True)
        self.metagraph = bittensor.metagraph( netuid = 3, network="mock")

    def test_print_empty(self):
        print (self.metagraph)

    def test_sync(self):
        self.metagraph.sync(subtensor=self.sub)
        self.metagraph.sync(subtensor=self.sub, block=0)

    def test_load_sync_save(self):
        self.metagraph.sync(subtensor=self.sub)
        self.metagraph.save()
        self.metagraph.load()
        self.metagraph.save()

    def test_factory(self):
        self.metagraph.load().sync(subtensor=self.sub).save()

    def test_state_dict(self):
        self.metagraph.load()
        state = self.metagraph.state_dict()
        assert 'uids' in state
        assert 'total_stake' in state
        assert 'last_update' in state
        assert 'block' in state
        assert 'tau' in state
        assert 'weights' in state
        assert 'endpoints' in state

    def test_properties(self):
        metagraph = self.metagraph
        metagraph.hotkeys
        metagraph.coldkeys
        metagraph.endpoints
        metagraph.R
        metagraph.T
        metagraph.S
        metagraph.D
        metagraph.C

    def test_to_dataframe(self):
        df = self.metagraph.to_dataframe()
