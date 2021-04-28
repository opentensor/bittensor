import os, sys
from unittest.mock import MagicMock
<<<<<<< HEAD
import bittensor
import torch
import numpy

from miners.gpt2_genesis import Miner
=======
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")

from gpt2_genesis.gpt2_genesis import Miner
>>>>>>> 2bd62712ca4c5755ac2f7a70065b77f79eb2dc81

def test_run_gpt2():
    miner = Miner(
        n_epochs = 1,
        epoch_length = 1
    )
    miner.subtensor.connect = MagicMock(return_value = True)  
    miner.subtensor.is_connected = MagicMock(return_value = True)      
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock(return_value = True) 
    miner.run()
test_run_gpt2()