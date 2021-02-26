import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")
import bittensor
import torch
import numpy
from xlm_wiki import Miner

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)
        
def test_run_xlm_clm():
    xlm_clm_session_config = Miner.build_config()
    xlm_clm_session_config.miner.n_epochs = 1
    xlm_clm_session_config.miner.epoch_length = 1
    xlm_clm_session_config.synapse.causal = True
    xlm_clm_session = Miner(xlm_clm_session_config)
    xlm_clm_session.neuron.subtensor.connect = MagicMock(return_value = True)    
    xlm_clm_session.neuron.metagraph.subtensor.subscribe = MagicMock(return_value = True)   
    xlm_clm_session.neuron.metagraph.set_weights = MagicMock()   
    xlm_clm_session.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = xlm_clm_session.neuron.wallet.hotkey.public_key,
        address = xlm_clm_session_config.axon.external_ip,
        port = xlm_clm_session_config.axon.external_port,
        uid = 0,
    )
    xlm_clm_session.neuron.metagraph.uid = 0
    xlm_clm_session.neuron.metagraph.state.n = 1
    xlm_clm_session.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    xlm_clm_session.neuron.metagraph.state.neurons = [neuron]
    xlm_clm_session.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    xlm_clm_session.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    xlm_clm_session.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    xlm_clm_session.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    xlm_clm_session.neuron.metagraph.state.uid_for_pubkey[xlm_clm_session.neuron.wallet.hotkey.public_key] = 0
    xlm_clm_session.neuron.metagraph.state.index_for_uid[0] = 0
    xlm_clm_session.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    xlm_clm_session.run()

test_run_xlm_clm()