import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")
import bittensor
import torch
import numpy
from gpt2_wiki import Miner


def test_run_gpt2():
    gpt2_session_config = Miner.build_config()
    gpt2_session_config.miner.n_epochs = 1
    gpt2_session_config.miner.epoch_length = 1
    gpt2_session = Miner(gpt2_session_config)
    gpt2_session.neuron.subtensor.connect = MagicMock(return_value = True)    
    gpt2_session.neuron.metagraph.subtensor.subscribe = MagicMock(return_value = True)  
    gpt2_session.neuron.metagraph.set_weights = MagicMock()   
    gpt2_session.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = gpt2_session.neuron.wallet.hotkey.public_key,
        address = gpt2_session_config.axon.external_ip,
        port = gpt2_session_config.axon.external_port,
        uid = 0,
    )
    gpt2_session.neuron.metagraph.uid = 0
    gpt2_session.neuron.metagraph.state.n = 1
    gpt2_session.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    gpt2_session.neuron.metagraph.state.neurons = [neuron]
    gpt2_session.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    gpt2_session.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    gpt2_session.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    gpt2_session.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    gpt2_session.neuron.metagraph.state.uid_for_pubkey[gpt2_session.neuron.wallet.hotkey.public_key] = 0
    gpt2_session.neuron.metagraph.state.index_for_uid[0] = 0
    gpt2_session.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    gpt2_session.run()
test_run_gpt2()