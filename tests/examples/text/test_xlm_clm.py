import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/TEXT/")
import bittensor
import torch
import numpy
from xlm import Miner

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)
        
def test_run_xlm_clm():
    xlm_clm_session_config = Miner.build_config()
    xlm_clm_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    xlm_clm_session_config.session.n_epochs = 1
    xlm_clm_session_config.session.epoch_length = 1
    xlm_clm_session_config.synapse.causal = True
    xlm_clm_session = Miner(xlm_clm_session_config)

    xlm_clm_session.neuron.metagraph.connect = MagicMock(return_value = (bittensor.metagraph.Metagraph.ConnectSuccess, ""))    
    xlm_clm_session.neuron.metagraph.subscribe = MagicMock(return_value = (bittensor.metagraph.Metagraph.SubscribeSuccess, ""))   
    xlm_clm_session.neuron.metagraph.set_weights = MagicMock()   
    xlm_clm_session.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = xlm_clm_session.neuron.wallet.keypair.public_key,
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
    xlm_clm_session.neuron.metagraph.state.uid_for_pubkey[xlm_clm_session.neuron.wallet.keypair.public_key] = 0
    xlm_clm_session.neuron.metagraph.state.index_for_uid[0] = 0
    xlm_clm_session.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    xlm_clm_session.run()

test_run_xlm_clm()