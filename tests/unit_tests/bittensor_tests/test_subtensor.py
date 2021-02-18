import bittensor

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