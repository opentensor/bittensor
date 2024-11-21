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

import random
import unittest
from queue import Empty as QueueEmpty
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
        # Keeps the same mock network for all tests. This stops the network from being re-setup for each test.
        cls._mock_subtensor = MockSubtensor()
        cls._do_setup_subnet()

    @classmethod
    def _do_setup_subnet(cls):
        # reset the mock subtensor
        cls._mock_subtensor.reset()
        # Setup the mock subnet 3
        cls._mock_subtensor.create_subnet(netuid=3)

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
        with patch("websockets.sync.client.connect"):
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

    def test_registration_multiprocessed_already_registered(self):
        work_blocks_before_is_registered = random.randint(5, 10)
        # return False each work block but return True after a random number of blocks
        is_registered_return_values = (
            [False for _ in range(work_blocks_before_is_registered)]
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
                wallet = get_mock_wallet(
                    hotkey=get_mock_keypair(0, self.id()),
                    coldkey=get_mock_keypair(1, self.id()),
                )
                self.subtensor.is_hotkey_registered = MagicMock(
                    side_effect=is_registered_return_values
                )

                self.subtensor.difficulty = MagicMock(return_value=1)
                self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                    side_effect=mock_neuron
                )
                self.subtensor._do_pow_register = MagicMock(return_value=(True, None))

                # should return True
                assert self.subtensor.register(
                    wallet=wallet, netuid=3, num_processes=3, update_interval=5
                )

                # calls until True and once again before exiting subtensor class
                # This assertion is currently broken when difficulty is too low
                assert (
                    self.subtensor.is_hotkey_registered.call_count
                    == work_blocks_before_is_registered + 2
                )

    def test_registration_partly_failed(self):
        do_pow_register_mock = MagicMock(
            side_effect=[(False, "Failed"), (False, "Failed"), (True, None)]
        )

        def is_registered_side_effect(*args, **kwargs):
            nonlocal do_pow_register_mock
            return do_pow_register_mock.call_count < 3

        current_block = [i for i in range(0, 100)]

        wallet = get_mock_wallet(
            hotkey=get_mock_keypair(0, self.id()),
            coldkey=get_mock_keypair(1, self.id()),
        )

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
            return_value=bittensor.NeuronInfo.get_null_neuron()
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
            "bittensor.core.extrinsics.registration.create_pow", return_value=None
        ) as mock_create_pow:
            wallet = get_mock_wallet(
                hotkey=get_mock_keypair(0, self.id()),
                coldkey=get_mock_keypair(1, self.id()),
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
        # verify that after a stale solution, to solve will continue without exiting

        class ExitEarly(Exception):
            pass

        mock_is_stale = MagicMock(side_effect=[True, False])

        mock_do_pow_register = MagicMock(side_effect=ExitEarly())

        mock_subtensor_self = MagicMock(
            neuron_for_pubkey=MagicMock(
                return_value=MagicMock(is_null=True)
            ),  # not registered
            substrate=MagicMock(
                get_block_hash=MagicMock(return_value="0x" + "0" * 64),
            ),
        )

        mock_wallet = MagicMock()

        mock_create_pow = MagicMock(return_value=MagicMock(is_stale=mock_is_stale))

        with patch(
            "bittensor.core.extrinsics.registration.create_pow", mock_create_pow
        ), patch(
            "bittensor.core.extrinsics.registration._do_pow_register",
            mock_do_pow_register,
        ):
            # should create a pow and check if it is stale
            # then should create a new pow and check if it is stale
            # then should enter substrate and exit early because of test
            self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(
                return_value=bittensor.NeuronInfo.get_null_neuron()
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


if __name__ == "__main__":
    unittest.main()
