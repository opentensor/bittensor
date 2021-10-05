from typing import DefaultDict
import bittensor
import pytest
from unittest.mock import MagicMock

def test_create():
    subtensor = bittensor.subtensor()

def test_defaults_to_akatsuki( ):
    subtensor = bittensor.subtensor()
    assert subtensor.endpoint_for_network() in bittensor.__akatsuki_entrypoints__


def test_networks():
    subtensor = bittensor.subtensor( network = 'kusanagi' )
    assert subtensor.endpoint_for_network() in bittensor.__kusanagi_entrypoints__
    subtensor = bittensor.subtensor( network = 'akatsuki' )
    assert subtensor.endpoint_for_network() in bittensor.__akatsuki_entrypoints__

def test_network_overrides():
    config = bittensor.subtensor.config()
    subtensor = bittensor.subtensor(network='kusanagi',config=config)
    assert subtensor.endpoint_for_network() in bittensor.__kusanagi_entrypoints__
    subtensor = bittensor.subtensor(network='akatsuki', config=config)
    assert subtensor.endpoint_for_network() in bittensor.__akatsuki_entrypoints__

def test_connect_no_failure( ):
     subtensor = bittensor.subtensor(
         network = "kusanagi"
     )
     subtensor.connect(timeout = 1, failure=False)

subtensor = bittensor.subtensor(
     network = 'akatsuki'
)

def test_connect_success( ):
    subtensor = bittensor.subtensor(
     network = 'akatsuki'
    )
    success = subtensor.connect()
    assert success == True


def test_connect_fail( ):
    subtensor = bittensor.subtensor(
     network = 'akatsuki'
    )
    subtensor.substrate=None
    with pytest.raises(RuntimeError):
        subtensor.connect()


def test_neurons( ):
     neurons = subtensor.neurons()
     assert len(neurons) > 0
     assert type(neurons[0].ip) == int
     assert type(neurons[0].port) == int
     assert type(neurons[0].ip_type) == int
     assert type(neurons[0].uid) == int
     assert type(neurons[0].modality) == int
     assert type(neurons[0].hotkey) == str
     assert type(neurons[0].coldkey) == str

     neuron = subtensor.neuron_for_uid( 0, neurons[0].hotkey )
     assert type(neuron.ip) == str
     assert type(neuron.port) == int
     assert type(neuron.ip_type) == int
     assert type(neuron.uid) == int
     assert type(neuron.modality) == int
     assert type(neuron.hotkey) == str
     assert type(neuron.coldkey) == str

     neuron = subtensor.neuron_for_pubkey(neuron.hotkey)
     assert type(neuron.ip) == str
     assert type(neuron.port) == int
     assert type(neuron.ip_type) == int
     assert type(neuron.uid) == int
     assert type(neuron.modality) == int
     assert type(neuron.hotkey) == str
     assert type(neuron.coldkey) == str


def test_get_current_block():
     block = subtensor.get_current_block()
     assert (type(block) == int)


wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 

def test_subscribe():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.subscribe(wallet,
                        ip='127.0.0.1',
                        port=8080,
                        modality=0
                        )
    assert success == True


def test_subscribe_failed():
    class failed():
        def __init__(self):
            self.is_success = False
            self.error_message = 'Mock'
        def process_events(self):
            return True

    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

    fail= subtensor.subscribe(wallet,
                        ip='127.0.0.1',
                        port=8080,
                        modality=0
                        )
    assert fail == False

def test_unstake():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.unstake(wallet,
                        amount = 200
                        )
    assert success == True

def test_unstake_inclusion():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.unstake(wallet,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert success == True

def test_unstake_failed():
    class failed():
        def __init__(self):
            self.is_success = False
            self.error_message = 'Mock'
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

    fail= subtensor.unstake(wallet,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert fail == False

def test_stake():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.add_stake(wallet,
                        amount = 200
                        )
    assert success == True

def test_stake_inclusion():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.add_stake(wallet,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert success == True

def test_stake_failed():
    class failed():
        def __init__(self):
            self.is_success = False
            self.error_message = 'Mock'
        def process_events(self):
            return True

    subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

    fail= subtensor.add_stake(wallet,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert fail == False
    
def test_transfer():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.transfer(wallet,
                        neuron.hotkey,
                        amount = 200,
                        )
    assert success == True

def test_transfer_inclusion():
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True

    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.transfer(wallet,
                        neuron.hotkey,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert success == True

def test_transfer_failed():
    class failed():
        def __init__(self):
            self.is_success = False
            self.error_message = 'Mock'
        def process_events(self):
            return True

    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

    fail= subtensor.transfer(wallet,
                        neuron.hotkey,
                        amount = 200,
                        wait_for_inclusion = True
                        )
    assert fail == False

def test_set_weights():
    chain_weights = [0]
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.set_weights(wallet=wallet,
                        uids=[neuron.uid],
                        weights=chain_weights,
                        )
    assert success == True

def test_set_weights_inclusion():
    chain_weights = [0]
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.set_weights(wallet=wallet,
                        uids=[neuron.uid],
                        weights=chain_weights,
                        wait_for_inclusion = True
                        )
    assert success == True

def test_set_weights_failed():
    class failed():
        def __init__(self):
            self.is_success = False
            self.error_message = 'Mock'
        def process_events(self):
            return True
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    chain_weights = [0]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = failed()) 

    fail= subtensor.set_weights(wallet=wallet,
                        uids=[neuron.uid],
                        weights=chain_weights,
                        wait_for_inclusion = True
                        )
    assert fail == False

def test_timeout_set_weights():
    chain_weights = [0]
    class success():
        def __init__(self):
            self.is_success = True
        def process_events(self):
            return True
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    subtensor.substrate.submit_extrinsic = MagicMock(return_value = success()) 
    success= subtensor.timeout_set_weights(wallet=wallet,
                        uids=[neuron.uid],
                        weights=chain_weights,
                        timeout=10,
                        )
    assert success == True

def test_get_balance():

    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    balance= subtensor.get_balance(address=neuron.hotkey)
    assert type(balance) == bittensor.utils.balance.Balance

def test_get_balances():
    subtensor = bittensor.subtensor()
    balance= subtensor.get_balances()
    assert type(balance) == dict
    for i in balance:
        assert type(balance[i]) == bittensor.utils.balance.Balance

def test_get_uid_for_hotkey():
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    uid= subtensor.get_uid_for_hotkey(neuron.hotkey)
    assert type(uid) == int

def test_hotkey_register():
    neurons = subtensor.neurons()
    neuron = neurons[ 0 ]
    register= subtensor.is_hotkey_registered(neuron.hotkey)
    assert register == True

def test_hotkey_register_failed():
    subtensor.get_uid_for_hotkey = MagicMock(return_value = -1) 
    register= subtensor.is_hotkey_registered('mock')
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
