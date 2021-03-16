import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")
import bittensor
import torch
import numpy
from bert_mlm.bert_mlm import Miner

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

def test_run_bert_mlm():
    miner = Miner(
        epoch_length = 1,
        n_epochs = 1
    )
    bittensor.subtensor.connect = MagicMock(return_value = True)    
    bittensor.subtensor.subscribe = MagicMock(return_value = True) 
    bittensor.metagraph.set_weights = MagicMock()   
    bittensor.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = bittensor.wallet.hotkey.public_key,
        address = bittensor.config.axon.external_ip,
        port = bittensor.config.axon.external_port,
        uid = 0,
    )
    bittensor.metagraph.uid = 0
    bittensor.metagraph.state.n = 1
    bittensor.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    bittensor.metagraph.state.neurons = [neuron]
    bittensor.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    bittensor.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    bittensor.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    bittensor.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    bittensor.metagraph.state.uid_for_pubkey[bittensor.wallet.hotkey.public_key] = 0
    bittensor.metagraph.state.index_for_uid[0] = 0
    bittensor.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    miner.run()

test_run_bert_mlm()