
from loguru import logger
from munch import Munch

import bittensor
from bittensor.subtensor.interface import Keypair

def test_create_neuron():
    neuron = bittensor.neuron.Neuron()

def test_neuron_start():
    config = bittensor.neuron.Neuron.build_config()
    config.metagraph.chain_endpoint = "feynman.kusanagi.bittensor.com:9944"
    neuron = bittensor.neuron.Neuron( config )
    neuron.start()

def test_neuron_with():
    config = bittensor.neuron.Neuron.build_config()
    config.metagraph.chain_endpoint = "feynman.kusanagi.bittensor.com:9944"
    neuron = bittensor.neuron.Neuron( config )
    with neuron:
        pass

def test_neuron_serve():
    config = bittensor.neuron.Neuron.build_config()
    config.metagraph.chain_endpoint = "feynman.kusanagi.bittensor.com:9944"
    neuron = bittensor.neuron.Neuron( config )
    neuron.serve( bittensor.synapse.Synapse() )

if __name__ == "__main__":
    test_create_neuron()