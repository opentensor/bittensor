from typing import DefaultDict
import bittensor
import pytest
import unittest
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

    def test_timeout_set_weights( self ):
        chain_weights = [0]
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
        neuron = self.neurons[ 1 ]
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
        success= self.subtensor.timeout_set_weights(wallet=self.wallet,
                            uids=[neuron.uid],
                            weights=chain_weights,
                            timeout=10,
                            )
        assert success == True

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
# def test_stake( ):
#     assert(type(subtensor.get_stake_for_uid(0)) == bittensor.utils.balance.Balance)
# def test_weight_uids( ):
#     weight_uids = subtensor.weight_uids_for_uid(0)
#     assert(type(weight_uids) == list)
#     assert(type(weight_uids[0]) == int)

# def test_weight_vals( ):
#     weight_vals = subtensor.weight_vals_for_uid(0)
#     assert(type(weight_vals) == list)
#     assert(type(weight_vals[0]) == int)

# def test_last_emit( ):
#     last_emit = subtensor.get_last_emit_data_for_uid(0)
#     assert(type(last_emit) == int)

# def test_get_active():
#     active = subtensor.get_active()
#     assert (type(active) == list)
#     assert (type(active[0][0]) == str)
#     assert (type(active[0][1]) == int)

# def test_get_stake():
#     stake = subtensor.get_stake()
#     assert (type(stake) == list)
#     assert (type(stake[0][0]) == int)
#     assert (type(stake[0][1]) == int)

# def test_get_last_emit():
#     last_emit = subtensor.get_stake()
#     assert (type(last_emit) == list)
#     assert (type(last_emit[0][0]) == int)
#     assert (type(last_emit[0][1]) == int)

# def test_get_weight_vals():
#     weight_vals = subtensor.get_weight_vals()
#     assert (type(weight_vals) == list)
#     assert (type(weight_vals[0][0]) == int)
#     assert (type(weight_vals[0][1]) == list)
#     assert (type(weight_vals[0][1][0]) == int)

# def test_get_weight_uids():
#     weight_uids = subtensor.get_weight_vals()
#     assert (type(weight_uids) == list)
#     assert (type(weight_uids[0][0]) == int)
#     assert (type(weight_uids[0][1]) == list)
#     assert (type(weight_uids[0][1][0]) == int)
