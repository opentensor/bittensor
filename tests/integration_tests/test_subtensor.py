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

from typing import DefaultDict
from unittest import mock
import bittensor
import pytest
import psutil  
import unittest
import time
from unittest.mock import MagicMock
from bittensor.utils.balance import Balance
from substrateinterface import Keypair

class TestSubtensor(unittest.TestCase):
    def setUp(self):
        self.subtensor = bittensor.subtensor( network = 'nobunaga' )
        self.wallet = bittensor.wallet()
        coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        self.wallet.set_coldkey(coldkey, encrypt=False, overwrite=True)
        self.wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=True)
        self.wallet.set_hotkey(Keypair.create_from_mnemonic(Keypair.generate_mnemonic()), encrypt=False, overwrite=True)
        self.mock_neuron = self.subtensor._neuron_dict_to_namespace(
            dict({
                "version":1,
                "ip":0,
                "port":0,
                "ip_type":0,
                "uid":1,
                "modality":0,
                "hotkey":'some_hotkey',
                "coldkey":'some_coldkey',
                "active":0,
                "last_update":0,
                "priority":0,
                "stake":1000000000000.0,
                "rank":0.0,
                "trust":0.0,
                "consensus":0.0,
                "incentive":0.0,
                "dividends":0.0,
                "emission":0.0,
                "bonds":[],
                "weights":[],
                "is_null":False
            })
        )
        self.neurons = self.subtensor.neurons()
        self.balance = Balance.from_tao(1000)
        assert True

    def test_defaults_to_nobunaga( self ):
        assert self.subtensor.endpoint_for_network() in bittensor.__nobunaga_entrypoints__

    def test_networks( self ):
        assert self.subtensor.endpoint_for_network() in bittensor.__nobunaga_entrypoints__

    def test_network_overrides( self ):
        config = bittensor.subtensor.config()
        subtensor = bittensor.subtensor(network='nobunaga', config=config, )
        assert subtensor.endpoint_for_network() in bittensor.__nobunaga_entrypoints__

    def test_connect_no_failure( self ):
        self.subtensor.connect(timeout = 1, failure=False)


    def test_connect_success( self ):
        success = self.subtensor.connect()
        assert success == True


    def test_connect_fail( self ):
        self.subtensor.substrate=None
        with pytest.raises(RuntimeError):
            self.subtensor.connect()


    def test_neurons( self ):
        assert len(self.neurons) > 0
        assert type(self.neurons[0].ip) == int
        assert type(self.neurons[0].port) == int
        assert type(self.neurons[0].ip_type) == int
        assert type(self.neurons[0].uid) == int
        assert type(self.neurons[0].modality) == int
        assert type(self.neurons[0].hotkey) == str
        assert type(self.neurons[0].coldkey) == str

        neuron = self.subtensor.neuron_for_uid( 1 )
        assert type(neuron.ip) == int
        assert type(neuron.port) == int
        assert type(neuron.ip_type) == int
        assert type(neuron.uid) == int
        assert type(neuron.modality) == int
        assert type(neuron.hotkey) == str
        assert type(neuron.coldkey) == str

        neuron = self.subtensor.neuron_for_pubkey(neuron.hotkey)
        assert type(neuron.ip) == int
        assert type(neuron.port) == int
        assert type(neuron.ip_type) == int
        assert type(neuron.uid) == int
        assert type(neuron.modality) == int
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
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
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
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 

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
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
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
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
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
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = Balance.from_tao(0))

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

        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
        success= self.subtensor.transfer(self.wallet,
                            neuron.hotkey,
                            amount = 200,
                            )
        assert success == True

    def test_transfer_inclusion( self ):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True

        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        self.subtensor.register = MagicMock(return_value = True) 
        self.subtensor.neuron_for_pubkey = MagicMock(return_value = self.mock_neuron) 
        self.subtensor.get_balance = MagicMock(return_value = self.balance)
        
        success= self.subtensor.transfer(self.wallet,
                            neuron.hotkey,
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

        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

        fail= self.subtensor.transfer(self.wallet,
                            neuron.hotkey,
                            amount = 200,
                            wait_for_inclusion = True
                            )
        assert fail == False

    def test_set_weights( self ):
        chain_weights = [0]
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        success= self.subtensor.set_weights(wallet=self.wallet,
                            uids=[neuron.uid],
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
        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        success= self.subtensor.set_weights(wallet=self.wallet,
                            uids=[neuron.uid],
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
        neuron = self.neurons[ 1 ]
        chain_weights = [0]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

        fail= self.subtensor.set_weights(wallet=self.wallet,
                            uids=[neuron.uid],
                            weights=chain_weights,
                            wait_for_inclusion = True
                            )
        assert fail == False

    def test_get_balance( self ):
        neuron = self.neurons[ 1 ]
        balance= self.subtensor.get_balance(address=neuron.hotkey)
        assert type(balance) == bittensor.utils.balance.Balance

    def test_get_balances( self ):
        balance= self.subtensor.get_balances()
        assert type(balance) == dict
        for i in balance:
            assert type(balance[i]) == bittensor.utils.balance.Balance

    def test_get_uid_for_hotkey( self ):
        neuron = self.neurons[ 1 ]
        uid= self.subtensor.get_uid_for_hotkey(neuron.hotkey)
        assert type(uid) == int

    def test_hotkey_register( self ):
        neuron = self.neurons[ 1 ]
        register= self.subtensor.is_hotkey_registered(neuron.hotkey)
        assert register == True

    def test_hotkey_register_failed( self ):
        self.subtensor.get_uid_for_hotkey = MagicMock(return_value = -1) 
        register= self.subtensor.is_hotkey_registered('mock')
        assert register == False


def test_subtensor_mock():
    bittensor.subtensor.kill_global_mock_process()
    sub = bittensor.subtensor.mock()
    assert bittensor.subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    del(sub)
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_create_mock_process():
    bittensor.subtensor.kill_global_mock_process()
    bittensor.subtensor.create_global_mock_process()
    assert bittensor.subtensor.global_mock_process_is_running()
    bittensor.subtensor.kill_global_mock_process()
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_mock_from_mock_arg():
    sub = bittensor.subtensor(_mock=True)
    assert bittensor.subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    sub.optionally_kill_owned_mock_instance()
    assert not bittensor.subtensor.global_mock_process_is_running()
    del(sub)
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_mock_from_network_arg():
    bittensor.subtensor.kill_global_mock_process()
    sub = bittensor.subtensor(network='mock')
    assert sub.network == 'mock'
    assert bittensor.subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    sub.__del__()
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_create_from_config():
    bittensor.subtensor.kill_global_mock_process()
    config = bittensor.subtensor.config()
    config.subtensor.network = 'mock'
    sub = bittensor.subtensor(config=config)
    assert bittensor.subtensor.global_mock_process_is_running()
    assert sub._is_mocked == True
    assert sub._owned_mock_subtensor_process != None
    del(sub)
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_two_subtensor_ownership():
    bittensor.subtensor.kill_global_mock_process()
    sub1 = bittensor.subtensor.mock()
    sub2 = bittensor.subtensor.mock()
    assert sub1._is_mocked == True
    assert sub2._is_mocked == True
    assert sub1._owned_mock_subtensor_process != None
    assert sub2._owned_mock_subtensor_process == None
    assert bittensor.subtensor.global_mock_process_is_running()
    del( sub2 )
    assert bittensor.subtensor.global_mock_process_is_running()
    del ( sub1 )
    time.sleep(2)
    assert not bittensor.subtensor.global_mock_process_is_running()

def test_subtensor_mock_functions():
    sub = bittensor.subtensor()
    sub.n
    sub.total_issuance
    sub.total_stake
    sub.immunity_period
    sub.rho
    sub.kappa
    sub.blocks_per_epoch
    sub.blocks_since_epoch
    sub.max_n

test_subtensor_mock()
test_create_mock_process()
test_mock_from_mock_arg()
test_mock_from_network_arg()
test_create_from_config()
test_two_subtensor_ownership()