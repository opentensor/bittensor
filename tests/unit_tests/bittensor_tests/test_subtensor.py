import bittensor
import pytest
from munch import Munch

def test_create():
    subtensor = bittensor.subtensor.Subtensor()

def test_defaults_to_akira( ):
    subtensor = bittensor.subtensor.Subtensor()
    assert subtensor.endpoint_for_network() in bittensor.__akira_entrypoints__

def test_endpoint_overides():
    subtensor = bittensor.subtensor.Subtensor(
        chain_endpoint = "this is the endpoint"
    )
    assert subtensor.endpoint_for_network() == "this is the endpoint"

def test_networks():
    subtensor = bittensor.subtensor.Subtensor()
    subtensor.config.subtensor.network = 'akira'
    assert subtensor.endpoint_for_network()  in bittensor.__akira_entrypoints__
    subtensor.config.subtensor.network = 'boltzmann'
    assert subtensor.endpoint_for_network()  in bittensor.__boltzmann_entrypoints__
    subtensor.config.subtensor.network = 'kusanagi'
    assert subtensor.endpoint_for_network() in bittensor.__kusanagi_entrypoints__

def test_connect_failure( ):
    subtensor = bittensor.subtensor.Subtensor(
        chain_endpoint = "this is the endpoint"
    )
    with pytest.raises(ValueError):
        subtensor.connect(timeout = 1)

def test_connect_no_failure( ):
    subtensor = bittensor.subtensor.Subtensor(
        network = "kusanagi"
    )
    subtensor.connect(timeout = 1, failure=False)

subtensor = bittensor.subtensor.Subtensor( 
    network = 'kusanagi'
)
def test_connect_success( ):
    subtensor.connect()

def test_neurons( ):
    neurons = subtensor.neurons()
    assert len(neurons) > 0
    assert type(neurons[0][0]) == int
    assert type(neurons[0][1]['ip']) == int
    assert type(neurons[0][1]['port']) == int
    assert type(neurons[0][1]['ip_type']) == int
    assert type(neurons[0][1]['uid']) == int
    assert type(neurons[0][1]['modality']) == int
    assert type(neurons[0][1]['hotkey']) == str
    assert type(neurons[0][1]['coldkey']) == str

    neuron = subtensor.get_neuron_for_uid( 0 )
    assert neurons[0][1]['ip'] == neuron['ip']
    assert neurons[0][1]['port'] == neuron['port']
    assert neurons[0][1]['ip_type'] == neuron['ip_type']
    assert neurons[0][1]['uid'] == neuron['uid']
    assert neurons[0][1]['modality'] == neuron['modality']
    assert neurons[0][1]['hotkey'] == neuron['hotkey']
    assert neurons[0][1]['coldkey'] == neuron['coldkey']

def test_stake( ):
    assert(type(subtensor.get_stake_for_uid(0)) == bittensor.utils.balance.Balance)

def test_weight_uids( ):
    weight_uids = subtensor.weight_uids_for_uid(0)
    assert(type(weight_uids) == list)
    assert(type(weight_uids[0]) == int)

def test_weight_vals( ):
    weight_vals = subtensor.weight_vals_for_uid(0)
    assert(type(weight_vals) == list)
    assert(type(weight_vals[0]) == int)

def test_last_emit( ):
    last_emit = subtensor.get_last_emit_data_for_uid(0)
    assert(type(last_emit) == int)

def test_get_current_block():
    block = subtensor.get_current_block()
    assert (type(block) == int)

def test_get_active():
    active = subtensor.get_active()
    assert (type(active) == list)
    assert (type(active[0][0]) == str)
    assert (type(active[0][1]) == int)

def test_get_stake():
    stake = subtensor.get_stake()
    assert (type(stake) == list)
    assert (type(stake[0][0]) == int)
    assert (type(stake[0][1]) == int)

def test_get_last_emit():
    last_emit = subtensor.get_stake()
    assert (type(last_emit) == list)
    assert (type(last_emit[0][0]) == int)
    assert (type(last_emit[0][1]) == int)

def test_get_weight_vals():
    weight_vals = subtensor.get_weight_vals()
    assert (type(weight_vals) == list)
    assert (type(weight_vals[0][0]) == int)
    assert (type(weight_vals[0][1]) == list)
    assert (type(weight_vals[0][1][0]) == int)

def test_get_weight_uids(): 
    weight_uids = subtensor.get_weight_vals()
    assert (type(weight_uids) == list)
    assert (type(weight_uids[0][0]) == int)
    assert (type(weight_uids[0][1]) == list)
    assert (type(weight_uids[0][1][0]) == int)