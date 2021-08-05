import os, sys
from unittest.mock import MagicMock
import bittensor
import torch
import numpy
import pathlib

from miners.text.template_miner import Miner

def test_run_template_config():
    PATH = str(pathlib.Path(__file__).parent.resolve()) + '/' + 'test_config.yml'
    sys.argv = [sys.argv[0], '--miner.config',PATH]
    config = Miner.config()
    assert config['miner']['n_epochs'] == 1
    assert config['miner']['epoch_length'] == 2
    gpt2_exodus_miner = Miner( config = config )
    bittensor.neuron.subtensor.connect = MagicMock(return_value = True)  
    bittensor.neuron.subtensor.is_connected = MagicMock(return_value = True)      
    bittensor.neuron.subtensor.subscribe = MagicMock(return_value = True)  
    bittensor.neuron.metagraph.set_weights = MagicMock(return_value = True) 
    gpt2_exodus_miner.run()

if __name__ == "__main__":
    test_run_template_config()