import bittensor
from munch import Munch

def test_create():
    subtensor = bittensor.subtensor.Subtensor()

def test_check_config_network_not_exists_defaults_akira( ):
    config = bittensor.metagraph.Subtensor.build_config()
    config.subtensor.network = None
    config.subtensor.chain_endpoint = None
    bittensor.metagraph.Subtensor.check_config(config)
    assert config.subtensor.network == 'akira'
    assert config.subtensor.chain_endpoint in bittensor.__akira_entrypoints__

def test_check_config_network_to_endpoint():
    config = bittensor.metagraph.Subtensor.build_config()
    config.subtensor.network = 'akira'
    bittensor.metagraph.Subtensor.check_config(config)
    assert config.subtensor.chain_endpoint in bittensor.__akira_entrypoints__
    config.subtensor.network = 'boltzmann'
    bittensor.metagraph.Subtensor.check_config(config)
    assert config.subtensor.chain_endpoint in bittensor.__boltzmann_entrypoints__
    config.subtensor.network = 'kusanagi'
    bittensor.metagraph.Subtensor.check_config(config)
    assert config.subtensor.chain_endpoint in bittensor.__kusanagi_entrypoints__

config = bittensor.metagraph.Subtensor.build_config()
config.chain_endpoint = 'feynman.boltzmann.bittensor.com:9944'
subtensor = bittensor.subtensor.Subtensor( config )

def test_connect( ):
    assert subtensor.connect() == True

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

def test_uid_for_public_key( ):
    assert subtensor.get_uid_for_pubkey("0x2ebbc6812171f4cff93927319ccda80cc3101fb5dbc283821d1ff9cede03893d") == 0

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