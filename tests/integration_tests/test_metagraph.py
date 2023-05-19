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
import pytest
from bittensor._subtensor.subtensor_mock import mock_subtensor


@pytest.fixture(autouse=True)
def setup():
    mock_subtensor.kill_global_mock_process()


class TestMetagraph:
    def setup_method(self):
        self.sub = bittensor.subtensor(_mock=True)
        self.metagraph = bittensor.metagraph(netuid=3, network="mock")

    def test_print_empty(self):
        print(self.metagraph)

    def test_lite_sync(self):
        self.metagraph.sync(lite=True)

    def test_full_sync(self):
        self.metagraph.sync(lite=False)

    def test_sync_block_0(self):
        self.metagraph.sync(lite=True, block=0)

    def test_load_sync_save(self):
        self.metagraph.sync(lite=True)
        self.metagraph.save()
        self.metagraph.load()
        self.metagraph.save()

    def test_state_dict(self):
        self.metagraph.load()
        state = self.metagraph.state_dict()
        assert 'version' in state
        assert 'n' in state
        assert 'block' in state
        assert 'stake' in state
        assert 'total_stake' in state
        assert 'ranks' in state
        assert 'trust' in state
        assert 'consensus' in state
        assert 'validator_trust' in state
        assert 'incentive' in state
        assert 'emission' in state
        assert 'dividends' in state
        assert 'active' in state
        assert 'last_update' in state
        assert 'validator_permit' in state
        assert 'weights' in state
        assert 'bonds' in state
        assert 'uids' in state

    def test_properties(self):
        metagraph = self.metagraph
        metagraph.hotkeys
        metagraph.coldkeys
        metagraph.addresses
        metagraph.validator_trust
        metagraph.S
        metagraph.R
        metagraph.I
        metagraph.E
        metagraph.C
        metagraph.T
        metagraph.Tv
        metagraph.D
        metagraph.B
        metagraph.W

    def test_parameters(self):
        params = list(self.metagraph.parameters())
        assert len(params) > 0
        assert isinstance(params[0], torch.nn.parameter.Parameter)
