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

import unittest
from unittest.mock import MagicMock, patch

import pytest
from bittensor_wallet import Keypair

import bittensor
from bittensor.core import settings
from bittensor.core.extrinsics import transfer
from bittensor.utils.balance import Balance
from bittensor.utils.mock import MockSubtensor
from tests.helpers import (
    get_mock_coldkey,
    MockConsole,
    get_mock_keypair,
    get_mock_wallet,
)


class TestSubtensor(unittest.TestCase):
    _mock_console_patcher = None
    _mock_subtensor: MockSubtensor
    subtensor: MockSubtensor

    def setUp(self):
        self.wallet = get_mock_wallet(
            hotkey=get_mock_keypair(0, self.id()),
            coldkey=get_mock_keypair(1, self.id()),
        )
        self.balance = Balance.from_tao(1000)
        self.mock_neuron = MagicMock()  # NOTE: this might need more sophistication
        self.subtensor = MockSubtensor()  # own instance per test

    @classmethod
    def setUpClass(cls) -> None:
        # mock rich console status
        mock_console = MockConsole()
        cls._mock_console_patcher = patch(
            "bittensor.core.settings.bt_console", mock_console
        )
        cls._mock_console_patcher.start()
        # Keeps the same mock network for all tests. This stops the network from being re-setup for each test.
        cls._mock_subtensor = MockSubtensor()
        cls._do_setup_subnet()

    @classmethod
    def _do_setup_subnet(cls):
        # reset the mock subtensor
        cls._mock_subtensor.reset()
        # Setup the mock subnet 3
        cls._mock_subtensor.create_subnet(netuid=3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._mock_console_patcher.stop()

    def test_network_overrides(self):
        """Tests that the network overrides the chain_endpoint."""
        # Argument importance: chain_endpoint (arg) > network (arg) > config.subtensor.chain_endpoint > config.subtensor.network
        config0 = bittensor.Subtensor.config()
        config0.subtensor.network = "finney"
        config0.subtensor.chain_endpoint = "wss://finney.subtensor.io"  # Should not match bittensor.core.settings.FINNEY_ENTRYPOINT
        assert config0.subtensor.chain_endpoint != settings.FINNEY_ENTRYPOINT

        config1 = bittensor.Subtensor.config()
        config1.subtensor.network = "local"
        config1.subtensor.chain_endpoint = None

        # Mock network calls
        with patch("substrateinterface.SubstrateInterface.connect_websocket"):
            with patch("substrateinterface.SubstrateInterface.reload_type_registry"):
                print(bittensor.Subtensor, type(bittensor.Subtensor))
                # Choose network arg over config
                sub1 = bittensor.Subtensor(config=config1, network="local")
                self.assertEqual(
                    sub1.chain_endpoint,
                    settings.LOCAL_ENTRYPOINT,
                    msg="Explicit network arg should override config.network",
                )

                # Choose network config over chain_endpoint config
                sub2 = bittensor.Subtensor(config=config0)
                self.assertNotEqual(
                    sub2.chain_endpoint,
                    settings.FINNEY_ENTRYPOINT,  # Here we expect the endpoint corresponding to the network "finney"
                    msg="config.network should override config.chain_endpoint",
                )

                sub3 = bittensor.Subtensor(config=config1)
                # Should pick local instead of finney (default)
                assert sub3.network == "local"
                assert sub3.chain_endpoint == settings.LOCAL_ENTRYPOINT

    def test_get_current_block(self):
        block = self.subtensor.get_current_block()
        assert type(block) is int

    def test_do_block_step(self):
        self.subtensor.do_block_step()
        block = self.subtensor.get_current_block()
        assert type(block) is int

    def test_do_block_step_query_previous_block(self):
        self.subtensor.do_block_step()
        block = self.subtensor.get_current_block()
        self.subtensor.query_subtensor("NetworksAdded", block)

    def test_transfer(self):
        fake_coldkey = get_mock_coldkey(1)

        transfer.do_transfer = MagicMock(return_value=(True, "0x", None))
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_balance = MagicMock(return_value=self.balance)
        success = self.subtensor.transfer(
            self.wallet,
            fake_coldkey,
            amount=200,
        )
        self.assertTrue(success, msg="Transfer should succeed")

    def test_transfer_inclusion(self):
        fake_coldkey = get_mock_coldkey(1)
        transfer.do_transfer = MagicMock(return_value=(True, "0x", None))
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        success = self.subtensor.transfer(
            self.wallet, fake_coldkey, amount=200, wait_for_inclusion=True
        )
        self.assertTrue(success, msg="Transfer should succeed")

    def test_transfer_failed(self):
        fake_coldkey = get_mock_coldkey(1)
        transfer.do_transfer = MagicMock(
            return_value=(False, None, "Mock failure message")
        )

        fail = self.subtensor.transfer(
            self.wallet, fake_coldkey, amount=200, wait_for_inclusion=True
        )
        self.assertFalse(fail, msg="Transfer should fail")

    def test_transfer_invalid_dest(self):
        fake_coldkey = get_mock_coldkey(1)

        fail = self.subtensor.transfer(
            self.wallet,
            fake_coldkey[:-1],  # invalid dest
            amount=200,
            wait_for_inclusion=True,
        )
        self.assertFalse(fail, msg="Transfer should fail because of invalid dest")

    def test_transfer_dest_as_bytes_fails(self):
        fake_coldkey = get_mock_coldkey(1)
        with patch(
            "bittensor.core.extrinsics.transfer.do_transfer",
            return_value=(True, "0x", None),
        ):
            self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                return_value=self.mock_neuron
            )
            self.subtensor.get_balance = MagicMock(return_value=self.balance)

            dest_as_bytes: bytes = Keypair(fake_coldkey).public_key

            with pytest.raises(TypeError):
                self.subtensor.transfer(
                    self.wallet,
                    dest_as_bytes,  # invalid dest
                    amount=200,
                    wait_for_inclusion=True,
                )

    def test_set_weights(self):
        chain_weights = [0]

        self.subtensor.set_weights = MagicMock(return_value=True)
        self.subtensor.do_set_weights = MagicMock(return_value=(True, None))

        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=3,
            uids=[1],
            weights=chain_weights,
        )
        assert success is True

    def test_set_weights_inclusion(self):
        chain_weights = [0]
        self.subtensor.do_set_weights = MagicMock(return_value=(True, None))
        self.subtensor.set_weights = MagicMock(return_value=True)

        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=1,
            uids=[1],
            weights=chain_weights,
            wait_for_inclusion=True,
        )
        assert success is True

    def test_set_weights_failed(self):
        chain_weights = [0]
        self.subtensor.do_set_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )
        self.subtensor.set_weights = MagicMock(return_value=False)

        fail = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=3,
            uids=[1],
            weights=chain_weights,
            wait_for_inclusion=True,
        )
        assert fail is False

    def test_get_balance(self):
        fake_coldkey = get_mock_coldkey(0)
        balance = self.subtensor.get_balance(address=fake_coldkey)
        assert type(balance) is bittensor.utils.balance.Balance

    def test_defaults_to_finney(self):
        sub = bittensor.Subtensor()
        assert sub.network == "finney"
        assert sub.chain_endpoint == settings.FINNEY_ENTRYPOINT


if __name__ == "__main__":
    unittest.main()
