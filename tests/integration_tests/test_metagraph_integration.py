from unittest import mock

import bittensor
import torch
import os
from bittensor.utils.mock import MockSubtensor
from bittensor.core.metagraph import METAGRAPH_STATE_DICT_NDARRAY_KEYS, get_save_dir

_subtensor_mock: MockSubtensor = MockSubtensor()


def setUpModule():
    _subtensor_mock.reset()
    _subtensor_mock.create_subnet(netuid=3)
    _subtensor_mock.set_difficulty(netuid=3, difficulty=0)  # Set diff 0


class TestMetagraph:
    def setup_method(self):
        self.sub = MockSubtensor()
        self.metagraph = bittensor.Metagraph(netuid=3, network="mock", sync=False)

    def test_print_empty(self):
        print(self.metagraph)

    def test_lite_sync(self):
        self.metagraph.sync(lite=True, subtensor=self.sub)

    def test_full_sync(self):
        self.metagraph.sync(lite=False, subtensor=self.sub)

    def test_sync_block_0(self):
        self.metagraph.sync(lite=True, block=0, subtensor=self.sub)

    def test_load_sync_save(self):
        with mock.patch.object(self.sub, "neurons_lite", return_value=[]):
            self.metagraph.sync(lite=True, subtensor=self.sub)
            self.metagraph.save()
            self.metagraph.load()
            self.metagraph.save()

    def test_load_sync_save_from_torch(self):
        with mock.patch.object(self.sub, "neurons_lite", return_value=[]):
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
        assert "stake" in state
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
