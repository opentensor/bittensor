import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")
import bittensor
import torch
import numpy
from bert_mlm import Miner

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

def test_run_bert_mlm():
    bert_mlm_session_config = Miner.build_config()
    bert_mlm_session_config.metagraph.chain_endpoint = 'feynman.akira.bittensor.com:9944'
    bert_mlm_session_config.miner.n_epochs = 1
    bert_mlm_session_config.miner.epoch_length = 1
    bert_mlm_session = Miner(bert_mlm_session_config)
    bert_mlm_session.neuron.metagraph.connect = MagicMock(return_value = (bittensor.metagraph.Metagraph.ConnectSuccess, ""))    
    bert_mlm_session.neuron.metagraph.subscribe = MagicMock(return_value = (bittensor.metagraph.Metagraph.SubscribeSuccess, ""))   
    bert_mlm_session.neuron.metagraph.set_weights = MagicMock()   
    bert_mlm_session.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = bert_mlm_session.neuron.wallet.keypair.public_key,
        address = bert_mlm_session_config.axon.external_ip,
        port = bert_mlm_session_config.axon.external_port,
        uid = 0,
    )
    bert_mlm_session.neuron.metagraph.uid = 0
    bert_mlm_session.neuron.metagraph.state.n = 1
    bert_mlm_session.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    bert_mlm_session.neuron.metagraph.state.neurons = [neuron]
    bert_mlm_session.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    bert_mlm_session.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    bert_mlm_session.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    bert_mlm_session.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    bert_mlm_session.neuron.metagraph.state.uid_for_pubkey[bert_mlm_session.neuron.wallet.keypair.public_key] = 0
    bert_mlm_session.neuron.metagraph.state.index_for_uid[0] = 0
    bert_mlm_session.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    bert_mlm_session.run()

test_run_bert_mlm()