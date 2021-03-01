import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/TEXT/")
import bittensor
import torch
import numpy
from gpt2_wiki.gpt2_wiki import Miner


def test_run_gpt2():
    miner = Miner(
        epoch_length = 1,
        n_epochs = 1
    )
    miner.neuron.subtensor.connect = MagicMock(return_value = True)    
    miner.neuron.metagraph.subtensor.subscribe = MagicMock(return_value = True)  
    miner.neuron.metagraph.set_weights = MagicMock()   
    miner.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = miner.neuron.wallet.hotkey.public_key,
        address = miner.config.axon.external_ip,
        port = miner.config.axon.external_port,
        uid = 0,
    )
    miner.neuron.metagraph.uid = 0
    miner.neuron.metagraph.state.n = 1
    miner.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    miner.neuron.metagraph.state.neurons = [neuron]
    miner.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    miner.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    miner.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    miner.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    miner.neuron.metagraph.state.uid_for_pubkey[miner.neuron.wallet.hotkey.public_key] = 0
    miner.neuron.metagraph.state.index_for_uid[0] = 0
    miner.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    miner.run()
test_run_gpt2()