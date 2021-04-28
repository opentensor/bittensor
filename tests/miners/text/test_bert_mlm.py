import os, sys
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")

from bert_mlm.bert_mlm import Miner


def test_run_bert_mlm():
    miner = Miner(
        n_epochs = 1,
        epoch_length = 1
    )
    miner.subtensor.connect = MagicMock(return_value = True) 
    miner.subtensor.is_connected = MagicMock(return_value = True)         
    miner.subtensor.subscribe = MagicMock(return_value = True)  
    miner.metagraph.set_weights = MagicMock()   
    miner.run()

test_run_bert_mlm()