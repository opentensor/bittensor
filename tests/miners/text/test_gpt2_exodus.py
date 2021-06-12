import os, sys
from unittest.mock import MagicMock
import bittensor
import torch
import numpy

from miners.text.gpt2_exodus import miner

def test_run_gpt2():
    config = miner.config()
    config.n_epochs = 1
    config.epoch_length = 2
    gpt2_exodus_miner = miner( config = config )
    gpt2_exodus_miner.subtensor.connect = MagicMock(return_value = True)  
    gpt2_exodus_miner.subtensor.is_connected = MagicMock(return_value = True)      
    gpt2_exodus_miner.subtensor.subscribe = MagicMock(return_value = True)  
    gpt2_exodus_miner.metagraph.set_weights = MagicMock(return_value = True) 
    gpt2_exodus_miner.run()

if __name__ == "__main__":
    test_run_gpt2()