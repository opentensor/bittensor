import os, sys, time
from unittest.mock import MagicMock
import bittensor
import torch
import numpy
from gpt2_genesis.gpt2_genesis import Miner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/text/")
from gpt2_genesis import Miner

def test_run_gpt2():
    miner = Miner()
    miner.subtensor.connect = MagicMock(return_value = True)    
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock()   
    miner.metagraph.sync = MagicMock()  
    miner.run()
test_run_gpt2()