# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import time
import unittest
from queue import Empty as QueueEmpty
from unittest.mock import MagicMock, patch

import bittensor
import pytest
from bittensor._subtensor.subtensor_mock import mock_subtensor
from bittensor.utils.balance import Balance
from substrateinterface import Keypair
from tests.helpers import get_mock_neuron, get_mock_hotkey, get_mock_coldkey, get_mock_neuron_by_uid

class TestSubtensor(unittest.TestCase):
    def setUp(self):
        self.subtensor = bittensor.subtensor( network = 'nobunaga' )
        self.wallet = bittensor.wallet(_mock=True)
        self.mock_neuron = get_mock_neuron_by_uid(0)
        self.balance = Balance.from_tao(1000)

    def test_defaults_to_nobunaga( self ):
        assert self.subtensor.chain_endpoint == bittensor.__nobunaga_entrypoint__

    def test_networks( self ):
        assert self.subtensor.chain_endpoint == bittensor.__nobunaga_entrypoint__

    def test_network_overrides( self ):
        config = bittensor.subtensor.config()
        subtensor = bittensor.subtensor(network='nobunaga', config=config, )
        assert subtensor.chain_endpoint == bittensor.__nobunaga_entrypoint__

    def test_neurons( self ):
        def mock_get_neuron_by_uid(_):
            return get_mock_neuron_by_uid(1)

        with patch.object(self.subtensor.substrate, 'rpc_request'):
            with patch('bittensor.Subtensor.get_uid_for_hotkey_on_subnet', return_value=1):
                with patch('bittensor.NeuronInfo.from_json', side_effect=mock_get_neuron_by_uid):

                    neuron = self.subtensor.neuron_for_uid( 1, netuid = -1 )
                    assert type(neuron.axon_info.ip) == int
                    assert type(neuron.axon_info.port) == int
                    assert type(neuron.axon_info.ip_type) == int
                    assert type(neuron.uid) == int
                    assert type(neuron.axon_info.protocol) == int
                    assert type(neuron.hotkey) == str
                    assert type(neuron.coldkey) == str

                    neuron = self.subtensor.get_neuron_for_pubkey_and_subnet(neuron.hotkey, netuid = -1)
                    assert type(neuron.axon_info.ip) == int
                    assert type(neuron.axon_info.port) == int
                    assert type(neuron.axon_info.ip_type) == int
                    assert type(neuron.uid) == int
                    assert type(neuron.axon_info.protocol) == int
                    assert type(neuron.hotkey) == str
                    assert type(neuron.coldkey) == str


    def test_get_current_block( self ):
        block = self.subtensor.get_current_block()
        assert (type(block) == int)

    def test_unstake( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            success= self.subtensor.unstake(self.wallet,
                                amount = 200
                                )
            assert success == True

    def test_unstake_inclusion( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            success= self.subtensor.unstake(self.wallet,
                                amount = 200,
                                wait_for_inclusion = True
                                )
            assert success == True

    def test_unstake_failed( self ):
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message = 'Mock'
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            fail= self.subtensor.unstake(self.wallet,
                                amount = 200,
                                wait_for_inclusion = True
                                )
            assert fail == False

    def test_stake(self):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            success= self.subtensor.add_stake(self.wallet,
                                amount = 200
                                )
            assert success == True

    def test_stake_inclusion(self):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            success= self.subtensor.add_stake(self.wallet,
                                amount = 200,
                                wait_for_inclusion = True
                                )
            assert success == True

    def test_stake_failed( self ):
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message = 'Mock'
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.get_payment_info = MagicMock(
            return_value = { 'partialFee': 100 }
        )
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_balance = MagicMock(return_value = Balance.from_rao(0))

        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        with patch('bittensor.Subtensor.get_stake_for_coldkey_and_hotkey', return_value=Balance.from_tao(500)):
            fail= self.subtensor.add_stake(self.wallet,
                                amount = 200,
                                wait_for_inclusion = True
                                )
            assert fail == False
        
    def test_transfer( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
            block_hash: str = '0x'

        fake_coldkey = get_mock_coldkey(1)
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
        success= self.subtensor.transfer(self.wallet,
                            fake_coldkey,
                            amount = 200,
                            )
        assert success == True

    def test_transfer_inclusion( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
            block_hash: str = '0x'

        fake_coldkey = get_mock_coldkey(1)
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
        
        success= self.subtensor.transfer(self.wallet,
                            fake_coldkey,
                            amount = 200,
                            wait_for_inclusion = True
                            )
        assert success == True

    def test_transfer_failed(self ):
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message = 'Mock'
            def process_events(self):
                return True

        fake_coldkey = get_mock_coldkey(1)
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

        fail= self.subtensor.transfer(self.wallet,
                            fake_coldkey,
                            amount = 200,
                            wait_for_inclusion = True
                            )
        assert fail == False

    def test_transfer_invalid_dest(self ):
        fake_coldkey = get_mock_coldkey(1)

        fail = self.subtensor.transfer(self.wallet,
                            fake_coldkey[:-1], # invalid dest
                            amount = 200,
                            wait_for_inclusion = True
                            )
        assert fail == False

    def test_transfer_dest_as_bytes(self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
            block_hash: str = '0x'

        fake_coldkey = get_mock_coldkey(1)
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)

        dest_as_bytes: bytes = Keypair(fake_coldkey).public_key
        success = self.subtensor.transfer(self.wallet,
                            dest_as_bytes, # invalid dest
                            amount = 200,
                            wait_for_inclusion = True
                            )
        assert success == True

    def test_set_weights( self ):
        chain_weights = [0]
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()

        success= self.subtensor.set_weights(wallet=self.wallet,
                            netuid = -1,
                            uids=[1],
                            weights=chain_weights,
                            )
        assert success == True

    def test_set_weights_inclusion( self ):
        chain_weights = [0]
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()

        success = self.subtensor.set_weights(wallet=self.wallet,
                            netuid = 1,
                            uids=[1],
                            weights=chain_weights,
                            wait_for_inclusion = True
                            )
        assert success == True

    def test_set_weights_failed( self ):
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message = 'Mock'
            def process_events(self):
                return True
        
        chain_weights = [0]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 
        self.subtensor.substrate.compose_call = MagicMock()
        self.subtensor.substrate.create_signed_extrinsic = MagicMock()

        fail= self.subtensor.set_weights(wallet=self.wallet,
                            netuid = -1, 
                            uids=[1],
                            weights=chain_weights,
                            wait_for_inclusion = True
                            )
        assert fail == False

    def test_get_balance( self ):
        fake_coldkey = get_mock_coldkey(0)
        balance= self.subtensor.get_balance(address=fake_coldkey)
        assert type(balance) == bittensor.utils.balance.Balance

    def test_get_balances( self ):
        balance= self.subtensor.get_balances()
        assert type(balance) == dict
        for i in balance:
            assert type(balance[i]) == bittensor.utils.balance.Balance

    def test_get_uid_by_hotkey_on_subnet( self ):
        fake_hotkey = get_mock_hotkey(0)
        with patch('bittensor.Subtensor.query_paratensor', return_value=MagicMock( value=0 )):
            uid = self.subtensor.get_uid_for_hotkey_on_subnet(fake_hotkey, netuid = -1)
            assert isinstance(uid, int)

    def test_hotkey_register( self ):
        fake_hotkey = get_mock_hotkey(0)
        self.subtensor.get_uid_for_hotkey_on_subnet = MagicMock(return_value = 0)
        register= self.subtensor.is_hotkey_registered(fake_hotkey, netuid = -1)
        assert register == True

    def test_hotkey_register_failed( self ):
        self.subtensor.get_uid_for_hotkey_on_subnet = MagicMock(return_value = None) 
        register= self.subtensor.is_hotkey_registered('mock', netuid = -1)
        assert register == False

    def test_registration_multiprocessed_already_registered( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
            
        workblocks_before_is_registered = random.randint(5, 10)
        # return False each work block but return True after a random number of blocks
        is_registered_return_values = [False for _ in range(workblocks_before_is_registered)] + [True] + [True, False]
        # this should pass the initial False check in the subtensor class and then return True because the neuron is already registered

        mock_neuron = MagicMock()           
        mock_neuron.is_null = True

        with patch('bittensor.Subtensor.difficulty'):
            # patch solution queue to return None
            with patch('multiprocessing.queues.Queue.get', return_value=None) as mock_queue_get:
                # patch time queue get to raise Empty exception
                with patch('multiprocessing.queues.Queue.get_nowait', side_effect=QueueEmpty) as mock_queue_get_nowait:

                    wallet = bittensor.wallet(_mock=True)
                    wallet.is_registered = MagicMock( side_effect=is_registered_return_values )

                    self.subtensor.difficulty= MagicMock(return_value=1)
                    self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock( side_effect=mock_neuron )
                    self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success())

                    with patch('bittensor.__console__.status') as mock_set_status:
                        # Need to patch the console status to avoid opening a parallel live display
                        mock_set_status.__enter__ = MagicMock(return_value=True)
                        mock_set_status.__exit__ = MagicMock(return_value=True)

                        # should return True
                        assert self.subtensor.register(wallet=wallet, netuid = -1, num_processes=3, update_interval=5 ) == True

                    # calls until True and once again before exiting subtensor class
                    # This assertion is currently broken when difficulty is too low
                    assert wallet.is_registered.call_count == workblocks_before_is_registered + 2 
    
    def test_registration_partly_failed( self ):
        
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message ='Failed'
            def process_events(self):
                return False
            
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        submit_extrinsic_mock = MagicMock( side_effect = [failed(), failed(), success()])

        def is_registered_side_effect(*args, **kwargs):
            nonlocal submit_extrinsic_mock 
            return submit_extrinsic_mock.call_count < 3
            
        current_block = [i for i in range(0,100)]

        with patch('bittensor.Subtensor.get_neuron_for_pubkey_and_subnet', return_value = bittensor.NeuronInfo._null_neuron()):
            with patch('bittensor.Subtensor.difficulty'):
                wallet = bittensor.wallet(_mock=True)
                wallet.is_registered = MagicMock(side_effect=is_registered_side_effect)

                self.subtensor.difficulty = MagicMock(return_value=1)
                self.subtensor.get_current_block = MagicMock(side_effect=current_block)
                self.subtensor.substrate.submit_extrinsic = submit_extrinsic_mock

                with patch('bittensor.__console__.status') as mock_set_status:
                    # Need to patch the console status to avoid opening a parallel live display
                    mock_set_status.__enter__ = MagicMock(return_value=True)
                    mock_set_status.__exit__ = MagicMock(return_value=True)

                    # should return True
                    assert self.subtensor.register(wallet=wallet, netuid = -1, num_processes=3, update_interval=5) == True

    def test_registration_failed( self ):
        class failed():
            def __init__(self):
                self.is_success = False
                self.error_message ='Failed'
            def process_events(self):
                return False
            

        is_registered_return_values = [False for _ in range(100)]
        current_block = [i for i in range(0,100)]
        mock_neuron = MagicMock()           
        mock_neuron.is_null = True

        with patch('bittensor.utils.create_pow' ):
            bittensor.utils.create_pow = MagicMock(return_value=None)
            wallet = bittensor.wallet(_mock=True)
            wallet.is_registered = MagicMock( side_effect=is_registered_return_values )

            self.subtensor.get_current_block = MagicMock(side_effect=current_block)
            self.subtensor.get_neuron_for_pubkey_and_subnet = MagicMock( return_value=mock_neuron )
            self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed())

            # should return True
            assert self.subtensor.register(wallet=wallet, netuid = -1 ) == False
            assert bittensor.utils.create_pow.call_count == 3 

    def test_registration_stale_then_continue( self ):
        # verifty that after a stale solution, the solve will continue without exiting

        class ExitEarly(Exception):
            pass

        mock_not_stale = MagicMock(
            side_effect = [False, True]
        )

        mock_substrate_enter = MagicMock(
                    side_effect=ExitEarly()
        )

        mock_subtensor_self = MagicMock(
            neuron_for_pubkey = MagicMock( return_value = MagicMock(is_null = True) ), # not registered
            substrate=MagicMock(
                __enter__ = mock_substrate_enter
            )
        )

        mock_wallet = MagicMock()

        mock_create_pow = MagicMock(
            return_value = MagicMock()
        )

        with patch('bittensor.Subtensor.get_neuron_for_pubkey_and_subnet', return_value=bittensor.NeuronInfo._null_neuron() ):
            with patch('bittensor.utils.create_pow', mock_create_pow):
                with patch('bittensor.utils.POWNotStale', mock_not_stale):
                    # should create a pow and check if it is stale
                    # then should create a new pow and check if it is stale
                    # then should enter substrate and exit early because of test
                    with pytest.raises(ExitEarly):
                        bittensor.Subtensor.register(mock_subtensor_self, mock_wallet, netuid = -1)
                    assert mock_create_pow.call_count == 2 # must try another pow after stale
                    assert mock_not_stale.call_count == 2
                    assert mock_substrate_enter.call_count == 1 # only tries to submit once, then exits


def test_subtensor_mock():
    mock_subtensor.kill_global_mock_process()
    sub = bittensor.subtensor(_mock=True)
    assert mock_subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    del(sub)
    assert not mock_subtensor.global_mock_process_is_running()

def test_create_mock_process():
    mock_subtensor.kill_global_mock_process()
    mock_subtensor.create_global_mock_process()
    assert mock_subtensor.global_mock_process_is_running()
    mock_subtensor.kill_global_mock_process()
    assert not mock_subtensor.global_mock_process_is_running()

def test_mock_from_mock_arg():
    sub = bittensor.subtensor(_mock=True)
    assert mock_subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    sub.optionally_kill_owned_mock_instance()
    assert not mock_subtensor.global_mock_process_is_running()
    del(sub)
    assert not mock_subtensor.global_mock_process_is_running()

def test_mock_from_network_arg():
    mock_subtensor.kill_global_mock_process()
    sub = bittensor.subtensor(network='mock')
    assert sub.network == 'mock'
    assert mock_subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    sub.__del__()
    assert not mock_subtensor.global_mock_process_is_running()

def test_create_from_config():
    mock_subtensor.kill_global_mock_process()
    config = bittensor.subtensor.config()
    config.subtensor.network = 'mock'
    sub = bittensor.subtensor(config=config)
    assert mock_subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    del(sub)
    assert not mock_subtensor.global_mock_process_is_running()

def test_two_subtensor_ownership():
    mock_subtensor.kill_global_mock_process()
    sub1 = bittensor.subtensor(_mock=True)
    sub2 = bittensor.subtensor(_mock=True)
    assert sub1._is_mocked == True
    assert sub2._is_mocked == True
    assert sub1._owned_mock_subtensor_process != None
    assert sub2._owned_mock_subtensor_process == None
    assert mock_subtensor.global_mock_process_is_running()
    del( sub2 )
    assert mock_subtensor.global_mock_process_is_running()
    del ( sub1 )
    time.sleep(2)
    assert not mock_subtensor.global_mock_process_is_running()

def test_subtensor_mock_functions():
    sub = bittensor.subtensor(_mock=True)
    sub.total_issuance
    sub.total_stake
    sub.immunity_period(netuid = -1)
    sub.rho(netuid = -1)
    sub.kappa(netuid = -1)
    sub.blocks_since_epoch(netuid = -1)
    sub.max_n(netuid = -1)
    sub.min_allowed_weights(netuid = -1)
    sub.validator_epoch_length(netuid = -1)
    sub.validator_epochs_per_reset(netuid = -1)
    sub.validator_sequence_length(netuid = -1)
    sub.validator_batch_size(netuid = -1)
    sub.difficulty(netuid = -1)

if __name__ == "__main__":
    unittest.main()
