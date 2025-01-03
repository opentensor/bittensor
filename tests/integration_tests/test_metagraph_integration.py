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

import os

import torch

from bittensor.core.metagraph import METAGRAPH_STATE_DICT_NDARRAY_KEYS, get_save_dir
from bittensor.core.metagraph import Metagraph
from bittensor.utils.mock import MockSubtensor

_subtensor_mock: MockSubtensor = MockSubtensor()


def setUpModule():
    _subtensor_mock.reset()
    _subtensor_mock.create_subnet(netuid=3)
    _subtensor_mock.set_difficulty(netuid=3, difficulty=0)  # Set diff 0


class TestMetagraph:
    def setup_method(self):
        self.sub = MockSubtensor()
        self.metagraph = Metagraph(netuid=3, network="mock", sync=False)

    def test_print_empty(self):
        print(self.metagraph)

    def test_lite_sync(self):
        self.metagraph.sync(lite=True, subtensor=self.sub)

    def test_full_sync(self):
        self.metagraph.sync(lite=False, subtensor=self.sub)

    def test_sync_block_0(self):
        self.metagraph.sync(lite=True, block=0, subtensor=self.sub)

    def test_load_sync_save(self):
        self.metagraph.sync(lite=True, subtensor=self.sub)
        self.metagraph.save()
        self.metagraph.load()
        self.metagraph.save()

    def test_load_sync_save_from_torch(self):
        self.metagraph.sync(lite=True, subtensor=self.sub)

        def deprecated_save_torch(metagraph):
            save_directory = get_save_dir(metagraph.network, metagraph.netuid)
            os.makedirs(save_directory, exist_ok=True)
            graph_filename = save_directory + f"/block-{metagraph.block.item()}.pt"
            state_dict = metagraph.state_dict()
            for key in METAGRAPH_STATE_DICT_NDARRAY_KEYS:
                state_dict[key] = torch.nn.Parameter(
                    torch.tensor(state_dict[key]), requires_grad=False
                )
            torch.save(state_dict, graph_filename)

        deprecated_save_torch(self.metagraph)
        self.metagraph.load()

    def test_state_dict(self):
        self.metagraph.load()
        state = self.metagraph.state_dict()
        assert "version" in state
        assert "n" in state
        assert "block" in state
        assert "total_stake" in state
        assert "alpha_stake" in state
        assert "tao_stake" in state
        assert "ranks" in state
        assert "trust" in state
        assert "consensus" in state
        assert "validator_trust" in state
        assert "incentive" in state
        assert "emission" in state
        assert "dividends" in state
        assert "active" in state
        assert "last_update" in state
        assert "validator_permit" in state
        assert "weights" in state
        assert "bonds" in state
        assert "uids" in state

    def test_properties(self):
        metagraph = self.metagraph
        metagraph.hotkeys
        metagraph.coldkeys
        metagraph.addresses
        metagraph.validator_trust
        metagraph.R
        metagraph.I
        metagraph.E
        metagraph.C
        metagraph.T
        metagraph.Tv
        metagraph.D
        metagraph.B
        metagraph.W
        metagraph.Ts
        metagraph.AS
        metagraph.S
