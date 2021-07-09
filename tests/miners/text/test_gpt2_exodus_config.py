import os, sys
from unittest.mock import MagicMock
import bittensor
import torch
import numpy
import pathlib

from miners.text.gpt2_exodus import neuron

def test_run_gpt2_config():
    PATH = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'test_config.yml'
    sys.argv = [sys.argv[0], '--neuron.config',PATH]
    config = neuron.config()
    assert config['neuron']['n_epochs'] == 1
    assert config['neuron']['epoch_length'] == 2
    gpt2_exodus_miner = neuron( config = config )
    gpt2_exodus_miner.subtensor.connect = MagicMock(return_value = True)  
    gpt2_exodus_miner.subtensor.is_connected = MagicMock(return_value = True)      
    gpt2_exodus_miner.subtensor.subscribe = MagicMock(return_value = True)  
    gpt2_exodus_miner.metagraph.set_weights = MagicMock(return_value = True) 
    gpt2_exodus_miner.run()

if __name__ == "__main__":
    test_run_gpt2_config()