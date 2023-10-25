# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Technologies Inc

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


import random
import unittest
from queue import Empty as QueueEmpty
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import bittensor
from bittensor.mock import MockSubtensor
import pytest
from bittensor.utils.balance import Balance
from substrateinterface import Keypair
from tests.helpers import (
    _get_mock_hotkey,
    _get_mock_coldkey,
    MockConsole,
    _get_mock_keypair,
    _get_mock_wallet,
)


class TestSubtensor(unittest.TestCase):
    _mock_console_patcher = None
    _mock_subtensor: MockSubtensor
    subtensor: MockSubtensor

    def setUp(self):
        self.wallet = _get_mock_wallet(
            hotkey=_get_mock_keypair(0, self.id()),
            coldkey=_get_mock_keypair(1, self.id()),
        )
        self.balance = Balance.from_tao(1000)
        self.mock_neuron = MagicMock()  # NOTE: this might need more sophistication
        self.subtensor = MockSubtensor()  # own instance per test

    @classmethod
    def setUpClass(cls) -> None:
        # mock rich console status
        mock_console = MockConsole()
        cls._mock_console_patcher = patch("bittensor.__console__", mock_console)
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
        config0 = bittensor.subtensor.config()
        config0.subtensor.network = "finney"
        config0.subtensor.chain_endpoint = "wss://finney.subtensor.io"  # Should not match bittensor.__finney_entrypoint__
        assert config0.subtensor.chain_endpoint != bittensor.__finney_entrypoint__

        config1 = bittensor.subtensor.config()
        config1.subtensor.network = "local"
        config1.subtensor.chain_endpoint = None

        # Mock network calls
        with patch("substrateinterface.SubstrateInterface.connect_websocket"):
            with patch("substrateinterface.SubstrateInterface.reload_type_registry"):
                print(bittensor.subtensor, type(bittensor.subtensor))
                # Choose network arg over config
                sub1 = bittensor.subtensor(config=config1, network="local")
                self.assertEqual(
                    sub1.chain_endpoint,
                    bittensor.__local_entrypoint__,
                    msg="Explicit network arg should override config.network",
                )

                # Choose network config over chain_endpoint config
                sub2 = bittensor.subtensor(config=config0)
                self.assertNotEqual(
                    sub2.chain_endpoint,
                    bittensor.__finney_entrypoint__,  # Here we expect the endpoint corresponding to the network "finney"
                    msg="config.network should override config.chain_endpoint",
                )

                sub3 = bittensor.subtensor(config=config1)
                # Should pick local instead of finney (default)
                assert sub3.network == "local"
                assert sub3.chain_endpoint == bittensor.__local_entrypoint__

    def test_get_current_block(self):
        block = self.subtensor.get_current_block()
        assert type(block) == int

    def test_unstake(self):
        self.subtensor._do_unstake = MagicMock(return_value=True)

        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value={"partialFee": 100}
        )

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        success = self.subtensor.unstake(self.wallet, amount=200)
        self.assertTrue(success, msg="Unstake should succeed")

    def test_unstake_inclusion(self):
        self.subtensor._do_unstake = MagicMock(return_value=True)

        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value={"partialFee": 100}
        )

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=self.balance)
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        success = self.subtensor.unstake(
            self.wallet, amount=200, wait_for_inclusion=True
        )
        self.assertTrue(success, msg="Unstake should succeed")

    def test_unstake_failed(self):
        self.subtensor._do_unstake = MagicMock(return_value=False)

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        fail = self.subtensor.unstake(self.wallet, amount=200, wait_for_inclusion=True)
        self.assertFalse(fail, msg="Unstake should fail")

    def test_stake(self):
        self.subtensor._do_stake = MagicMock(return_value=True)

        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value={"partialFee": 100}
        )

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        self.subtensor.get_hotkey_owner = MagicMock(
            return_value=self.wallet.coldkeypub.ss58_address
        )
        success = self.subtensor.add_stake(self.wallet, amount=200)
        self.assertTrue(success, msg="Stake should succeed")

    def test_stake_inclusion(self):
        self.subtensor._do_stake = MagicMock(return_value=True)

        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value={"partialFee": 100}
        )

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        self.subtensor.get_hotkey_owner = MagicMock(
            return_value=self.wallet.coldkeypub.ss58_address
        )
        success = self.subtensor.add_stake(
            self.wallet, amount=200, wait_for_inclusion=True
        )
        self.assertTrue(success, msg="Stake should succeed")

    def test_stake_failed(self):
        self.subtensor._do_stake = MagicMock(return_value=False)

        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value={"partialFee": 100}
        )

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_balance = MagicMock(return_value=Balance.from_rao(0))

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_stake_for_coldkey_and_hotkey = MagicMock(
            return_value=Balance.from_tao(500)
        )
        self.subtensor.get_hotkey_owner = MagicMock(
            return_value=self.wallet.coldkeypub.ss58_address
        )
        fail = self.subtensor.add_stake(
            self.wallet, amount=200, wait_for_inclusion=True
        )
        self.assertFalse(fail, msg="Stake should fail")

    def test_transfer(self):
        fake_coldkey = _get_mock_coldkey(1)

        self.subtensor._do_transfer = MagicMock(return_value=(True, "0x", None))
        self.subtensor.register = MagicMock(return_value=True)
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
        fake_coldkey = _get_mock_coldkey(1)
        self.subtensor._do_transfer = MagicMock(return_value=(True, "0x", None))
        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        success = self.subtensor.transfer(
            self.wallet, fake_coldkey, amount=200, wait_for_inclusion=True
        )
        self.assertTrue(success, msg="Transfer should succeed")

    def test_transfer_failed(self):
        fake_coldkey = _get_mock_coldkey(1)
        self.subtensor._do_transfer = MagicMock(
            return_value=(False, None, "Mock failure message")
        )

        fail = self.subtensor.transfer(
            self.wallet, fake_coldkey, amount=200, wait_for_inclusion=True
        )
        self.assertFalse(fail, msg="Transfer should fail")

    def test_transfer_invalid_dest(self):
        fake_coldkey = _get_mock_coldkey(1)

        fail = self.subtensor.transfer(
            self.wallet,
            fake_coldkey[:-1],  # invalid dest
            amount=200,
            wait_for_inclusion=True,
        )
        self.assertFalse(fail, msg="Transfer should fail because of invalid dest")

    def test_transfer_dest_as_bytes(self):
        fake_coldkey = _get_mock_coldkey(1)
        self.subtensor._do_transfer = MagicMock(return_value=(True, "0x", None))

        self.subtensor.register = MagicMock(return_value=True)
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=self.mock_neuron
        )
        self.subtensor.get_balance = MagicMock(return_value=self.balance)

        dest_as_bytes: bytes = Keypair(fake_coldkey).public_key
        success = self.subtensor.transfer(
            self.wallet,
            dest_as_bytes,  # invalid dest
            amount=200,
            wait_for_inclusion=True,
        )
        self.assertTrue(success, msg="Transfer should succeed")

    def test_set_weights(self):
        chain_weights = [0]

        class success:
            def __init__(self):
                self.is_success = True

            def process_events(self):
                return True

        self.subtensor._do_set_weights = MagicMock(return_value=(True, None))

        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=3,
            uids=[1],
            weights=chain_weights,
        )
        assert success == True

    def test_set_weights_inclusion(self):
        chain_weights = [0]
        self.subtensor._do_set_weights = MagicMock(return_value=(True, None))

        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=1,
            uids=[1],
            weights=chain_weights,
            wait_for_inclusion=True,
        )
        assert success == True

    def test_set_weights_failed(self):
        chain_weights = [0]
        self.subtensor._do_set_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )

        fail = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=3,
            uids=[1],
            weights=chain_weights,
            wait_for_inclusion=True,
        )
        assert fail == False

    def test_get_balance(self):
        fake_coldkey = _get_mock_coldkey(0)
        balance = self.subtensor.get_balance(address=fake_coldkey)
        assert type(balance) == bittensor.utils.balance.Balance

    def test_get_balances(self):
        balances = self.subtensor.get_balances()
        assert type(balances) == dict
        for i in balances:
            assert type(balances[i]) == bittensor.utils.balance.Balance

    def test_get_uid_by_hotkey_on_subnet(self):
        mock_coldkey_kp = _get_mock_keypair(0, self.id())
        mock_hotkey_kp = _get_mock_keypair(100, self.id())

        # Register on subnet 3
        mock_uid = self.subtensor.force_register_neuron(
            netuid=3,
            hotkey=mock_hotkey_kp.ss58_address,
            coldkey=mock_coldkey_kp.ss58_address,
        )

        uid = self.subtensor.get_uid_for_hotkey_on_subnet(
            mock_hotkey_kp.ss58_address, netuid=3
        )
        self.assertIsInstance(
            uid, int, msg="get_uid_for_hotkey_on_subnet should return an int"
        )
        self.assertEqual(
            uid,
            mock_uid,
            msg="get_uid_for_hotkey_on_subnet should return the correct uid",
        )

    def test_is_hotkey_registered(self):
        mock_coldkey_kp = _get_mock_keypair(0, self.id())
        mock_hotkey_kp = _get_mock_keypair(100, self.id())

        # Register on subnet 3
        _ = self.subtensor.force_register_neuron(
            netuid=3,
            hotkey=mock_hotkey_kp.ss58_address,
            coldkey=mock_coldkey_kp.ss58_address,
        )

        registered = self.subtensor.is_hotkey_registered(
            mock_hotkey_kp.ss58_address, netuid=3
        )
        self.assertTrue(registered, msg="Hotkey should be registered")

    def test_is_hotkey_registered_not_registered(self):
        mock_hotkey_kp = _get_mock_keypair(100, self.id())

        # Do not register on subnet 3

        registered = self.subtensor.is_hotkey_registered(
            mock_hotkey_kp.ss58_address, netuid=3
        )
        self.assertFalse(registered, msg="Hotkey should not be registered")

    def test_defaults_to_finney(self):
        sub = bittensor.subtensor()
        assert sub.network == "finney"
        assert sub.chain_endpoint == bittensor.__finney_entrypoint__


if __name__ == "__main__":
    unittest.main()
