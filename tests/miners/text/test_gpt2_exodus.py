import os, sys
from unittest.mock import MagicMock
import bittensor
import torch
import numpy
from miners.text.template_miner import Miner

def test_run_template():

    config = Miner.config()
    config.miner.n_epochs = 1
    config.miner.epoch_length = 2
    gpt2_exodus_miner = Miner( config = config )
    bittensor.neuron.subtensor.connect = MagicMock(return_value = True)  
    bittensor.neuron.subtensor.is_connected = MagicMock(return_value = True)      
    bittensor.neuron.subtensor.subscribe = MagicMock(return_value = True)  
    bittensor.neuron.metagraph.set_weights = MagicMock(return_value = True) 
    gpt2_exodus_miner.run()

if __name__ == "__main__":
    test_run_template()