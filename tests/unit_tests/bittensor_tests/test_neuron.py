
from loguru import logger
from munch import Munch

import bittensor
import pytest

def test_create_neuron():
    neuron = bittensor.neuron.Neuron()

def test_boltzmann_subscribe_success():
    neuron = bittensor.neuron.Neuron()
    neuron.config.subtensor.network = 'akira'
    with neuron:
        logger.success("Success")

def test_boltzmann_subscribe_failure():
    neuron = bittensor.neuron.Neuron()
    neuron.config.subtensor.chain_endpoint = 'not an endpoint'
    with pytest.raises(ValueError):
        with neuron:
            logger.success("Success")
    
if __name__ == "__main__":
    test_boltzmann_subscribe_success()
    # test_boltzmann_subscribe_failure()