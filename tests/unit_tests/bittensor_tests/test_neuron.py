
from loguru import logger
from munch import Munch

import bittensor
import pytest
from unittest.mock import MagicMock

def test_create_neuron():
    neuron = bittensor.neuron.Neuron()

def test_boltzmann_subscribe_success():
    neuron = bittensor.neuron.Neuron()
    neuron.subtensor.connect = MagicMock(return_value = True)    
    neuron.subtensor.subscribe = MagicMock(return_value = True) 
    neuron.metagraph.set_weights = MagicMock()   
    neuron.metagraph.sync = MagicMock()  
    with neuron:
        assert True

if __name__ == "__main__":
    test_boltzmann_subscribe_success()
