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
import socket
import os
import unittest
from queue import Empty as QueueEmpty
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import numpy as np
import pytest
from substrateinterface import Keypair

import bittensor
from bittensor.mock import MockSubtensor
from bittensor.utils import weight_utils
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

        self.subtensor.set_weights = MagicMock(return_value=True)
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
        self.subtensor.set_weights = MagicMock(return_value=True)

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
        self.subtensor.set_weights = MagicMock(return_value=False)

        fail = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=3,
            uids=[1],
            weights=chain_weights,
            wait_for_inclusion=True,
        )
        assert fail == False

    def test_commit_weights(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )
        commit_hash = bittensor.utils.weight_utils.generate_weight_hash(
            address=self.wallet.hotkey.ss58_address,
            netuid=3,
            uids=weight_uids,
            values=weight_vals,
            salt=salt.tolist(),
            version_key=0,
        )

        self.subtensor.commit_weights = MagicMock(
            return_value=(True, "Successfully committed weights.")
        )
        self.subtensor._do_commit_weights = MagicMock(return_value=(True, None))

        success, message = self.subtensor.commit_weights(
            wallet=self.wallet, netuid=3, uids=uids, weights=weights, salt=salt
        )
        assert success is True
        assert message == "Successfully committed weights."

    def test_commit_weights_inclusion(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        commit_hash = bittensor.utils.weight_utils.generate_weight_hash(
            address=self.wallet.hotkey.ss58_address,
            netuid=1,
            uids=weight_uids,
            values=weight_vals,
            salt=salt.tolist(),
            version_key=0,
        )

        self.subtensor._do_commit_weights = MagicMock(return_value=(True, None))
        self.subtensor.commit_weights = MagicMock(
            return_value=(True, "Successfully committed weights.")
        )

        success, message = self.subtensor.commit_weights(
            wallet=self.wallet,
            netuid=1,
            uids=uids,
            weights=weights,
            salt=salt,
            wait_for_inclusion=True,
        )
        assert success is True
        assert message == "Successfully committed weights."

    def test_commit_weights_failed(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        commit_hash = bittensor.utils.weight_utils.generate_weight_hash(
            address=self.wallet.hotkey.ss58_address,
            netuid=3,
            uids=weight_uids,
            values=weight_vals,
            salt=salt.tolist(),
            version_key=0,
        )

        self.subtensor._do_commit_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )
        self.subtensor.commit_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )

        success, message = self.subtensor.commit_weights(
            wallet=self.wallet,
            netuid=3,
            uids=uids,
            weights=weights,
            salt=salt,
            wait_for_inclusion=True,
        )
        assert success is False
        assert message == "Mock failure message"

    def test_reveal_weights(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        self.subtensor.reveal_weights = MagicMock(
            return_value=(True, "Successfully revealed weights.")
        )
        self.subtensor._do_reveal_weights = MagicMock(return_value=(True, None))

        success, message = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=3,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=0,
        )
        assert success is True
        assert message == "Successfully revealed weights."

    def test_reveal_weights_inclusion(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        self.subtensor._do_reveal_weights = MagicMock(return_value=(True, None))
        self.subtensor.reveal_weights = MagicMock(
            return_value=(True, "Successfully revealed weights.")
        )

        success, message = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=1,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=0,
            wait_for_inclusion=True,
        )
        assert success is True
        assert message == "Successfully revealed weights."

    def test_reveal_weights_failed(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        self.subtensor._do_reveal_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )
        self.subtensor.reveal_weights = MagicMock(
            return_value=(False, "Mock failure message")
        )

        success, message = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=3,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=0,
            wait_for_inclusion=True,
        )
        assert success is False
        assert message == "Mock failure message"

    def test_commit_and_reveal_weights(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        version_key = 0

        # Mock the commit_weights and reveal_weights functions
        self.subtensor.commit_weights = MagicMock(
            return_value=(True, "Successfully committed weights.")
        )
        self.subtensor._do_commit_weights = MagicMock(return_value=(True, None))
        self.subtensor.reveal_weights = MagicMock(
            return_value=(True, "Successfully revealed weights.")
        )
        self.subtensor._do_reveal_weights = MagicMock(return_value=(True, None))

        # Commit weights
        commit_success, commit_message = self.subtensor.commit_weights(
            wallet=self.wallet,
            netuid=3,
            uids=uids,
            weights=weights,
            salt=salt,
        )
        assert commit_success is True
        assert commit_message == "Successfully committed weights."

        # Reveal weights
        reveal_success, reveal_message = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=3,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=version_key,
        )
        assert reveal_success is True
        assert reveal_message == "Successfully revealed weights."

    def test_commit_and_reveal_weights_inclusion(self):
        weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        uids = np.array([1, 2, 3, 4], dtype=np.int64)
        salt = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        version_key = 0

        # Mock the commit_weights and reveal_weights functions
        self.subtensor.commit_weights = MagicMock(
            return_value=(True, "Successfully committed weights.")
        )
        self.subtensor._do_commit_weights = MagicMock(return_value=(True, None))
        self.subtensor.reveal_weights = MagicMock(
            return_value=(True, "Successfully revealed weights.")
        )
        self.subtensor._do_reveal_weights = MagicMock(return_value=(True, None))

        # Commit weights with wait_for_inclusion
        commit_success, commit_message = self.subtensor.commit_weights(
            wallet=self.wallet,
            netuid=1,
            uids=uids,
            weights=weights,
            salt=salt,
            wait_for_inclusion=True,
        )
        assert commit_success is True
        assert commit_message == "Successfully committed weights."

        # Reveal weights with wait_for_inclusion
        reveal_success, reveal_message = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=1,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=version_key,
            wait_for_inclusion=True,
        )
        assert reveal_success is True
        assert reveal_message == "Successfully revealed weights."

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

    def test_registration_multiprocessed_already_registered(self):
        workblocks_before_is_registered = random.randint(5, 10)
        # return False each work block but return True after a random number of blocks
        is_registered_return_values = (
            [False for _ in range(workblocks_before_is_registered)]
            + [True]
            + [True, False]
        )
        # this should pass the initial False check in the subtensor class and then return True because the neuron is already registered

        mock_neuron = MagicMock()
        mock_neuron.is_null = True

        # patch solution queue to return None
        with patch(
            "multiprocessing.queues.Queue.get", return_value=None
        ) as mock_queue_get:
            # patch time queue get to raise Empty exception
            with patch(
                "multiprocessing.queues.Queue.get_nowait", side_effect=QueueEmpty
            ) as mock_queue_get_nowait:
                wallet = _get_mock_wallet(
                    hotkey=_get_mock_keypair(0, self.id()),
                    coldkey=_get_mock_keypair(1, self.id()),
                )
                self.subtensor.is_hotkey_registered = MagicMock(
                    side_effect=is_registered_return_values
                )

                self.subtensor.difficulty = MagicMock(return_value=1)
                self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                    side_effect=mock_neuron
                )
                self.subtensor._do_pow_register = MagicMock(return_value=(True, None))

                with patch("bittensor.__console__.status") as mock_set_status:
                    # Need to patch the console status to avoid opening a parallel live display
                    mock_set_status.__enter__ = MagicMock(return_value=True)
                    mock_set_status.__exit__ = MagicMock(return_value=True)

                    # should return True
                    assert (
                        self.subtensor.register(
                            wallet=wallet, netuid=3, num_processes=3, update_interval=5
                        )
                        == True
                    )

                # calls until True and once again before exiting subtensor class
                # This assertion is currently broken when difficulty is too low
                assert (
                    self.subtensor.is_hotkey_registered.call_count
                    == workblocks_before_is_registered + 2
                )

    def test_registration_partly_failed(self):
        do_pow_register_mock = MagicMock(
            side_effect=[(False, "Failed"), (False, "Failed"), (True, None)]
        )

        def is_registered_side_effect(*args, **kwargs):
            nonlocal do_pow_register_mock
            return do_pow_register_mock.call_count < 3

        current_block = [i for i in range(0, 100)]

        wallet = _get_mock_wallet(
            hotkey=_get_mock_keypair(0, self.id()),
            coldkey=_get_mock_keypair(1, self.id()),
        )

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=bittensor.NeuronInfo._null_neuron()
        )
        self.subtensor.is_hotkey_registered = MagicMock(
            side_effect=is_registered_side_effect
        )

        self.subtensor.difficulty = MagicMock(return_value=1)
        self.subtensor.get_current_block = MagicMock(side_effect=current_block)
        self.subtensor._do_pow_register = do_pow_register_mock

        # should return True
        self.assertTrue(
            self.subtensor.register(
                wallet=wallet, netuid=3, num_processes=3, update_interval=5
            ),
            msg="Registration should succeed",
        )

    def test_registration_failed(self):
        is_registered_return_values = [False for _ in range(100)]
        current_block = [i for i in range(0, 100)]
        mock_neuron = MagicMock()
        mock_neuron.is_null = True

        with patch(
            "bittensor.extrinsics.registration.create_pow", return_value=None
        ) as mock_create_pow:
            wallet = _get_mock_wallet(
                hotkey=_get_mock_keypair(0, self.id()),
                coldkey=_get_mock_keypair(1, self.id()),
            )

            self.subtensor.is_hotkey_registered = MagicMock(
                side_effect=is_registered_return_values
            )

            self.subtensor.get_current_block = MagicMock(side_effect=current_block)
            self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                return_value=mock_neuron
            )
            self.subtensor.substrate.get_block_hash = MagicMock(
                return_value="0x" + "0" * 64
            )
            self.subtensor._do_pow_register = MagicMock(return_value=(False, "Failed"))

            # should return True
            self.assertIsNot(
                self.subtensor.register(wallet=wallet, netuid=3),
                True,
                msg="Registration should fail",
            )
            self.assertEqual(mock_create_pow.call_count, 3)

    def test_registration_stale_then_continue(self):
        # verify that after a stale solution, the solve will continue without exiting

        class ExitEarly(Exception):
            pass

        mock_is_stale = MagicMock(side_effect=[True, False])

        mock_do_pow_register = MagicMock(side_effect=ExitEarly())

        mock_subtensor_self = MagicMock(
            neuron_for_pubkey=MagicMock(
                return_value=MagicMock(is_null=True)
            ),  # not registered
            _do_pow_register=mock_do_pow_register,
            substrate=MagicMock(
                get_block_hash=MagicMock(return_value="0x" + "0" * 64),
            ),
        )

        mock_wallet = MagicMock()

        mock_create_pow = MagicMock(return_value=MagicMock(is_stale=mock_is_stale))

        with patch("bittensor.extrinsics.registration.create_pow", mock_create_pow):
            # should create a pow and check if it is stale
            # then should create a new pow and check if it is stale
            # then should enter substrate and exit early because of test
            self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                return_value=bittensor.NeuronInfo._null_neuron()
            )
            with pytest.raises(ExitEarly):
                bittensor.subtensor.register(mock_subtensor_self, mock_wallet, netuid=3)
            self.assertEqual(
                mock_create_pow.call_count, 2, msg="must try another pow after stale"
            )
            self.assertEqual(mock_is_stale.call_count, 2)
            self.assertEqual(
                mock_do_pow_register.call_count,
                1,
                msg="only tries to submit once, then exits",
            )

    def test_defaults_to_finney(self):
        sub = bittensor.subtensor()
        assert sub.network == "finney"
        assert sub.chain_endpoint == bittensor.__finney_entrypoint__


if __name__ == "__main__":
    unittest.main()
