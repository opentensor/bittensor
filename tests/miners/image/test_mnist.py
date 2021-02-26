import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("miners/IMAGE/")
import bittensor
import torch
import numpy
from mnist import Miner

def test_run_mnist():
    mnist_miner_config = Miner.build_config()
    mnist_miner_config.miner.n_epochs = 1
    mnist_miner_config.miner.epoch_length = 1
    mnist_miner = Miner(mnist_miner_config)
    mnist_miner.neuron.subtensor.connect = MagicMock(return_value = True)    
    mnist_miner.neuron.subtensor.subscribe = MagicMock(return_value = True) 
    mnist_miner.neuron.metagraph.set_weights = MagicMock()   
    mnist_miner.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = mnist_miner.neuron.wallet.hotkey.public_key,
        address = mnist_miner_config.axon.external_ip,
        port = mnist_miner_config.axon.external_port,
        uid = 0,
    )
    mnist_miner.neuron.metagraph.uid = 0
    mnist_miner.neuron.metagraph.state.n = 1
    mnist_miner.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    mnist_miner.neuron.metagraph.state.neurons = [neuron]
    mnist_miner.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    mnist_miner.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    mnist_miner.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    mnist_miner.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    mnist_miner.neuron.metagraph.state.uid_for_pubkey[mnist_miner.neuron.wallet.hotkey.public_key] = 0
    mnist_miner.neuron.metagraph.state.index_for_uid[0] = 0
    mnist_miner.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    mnist_miner.run()
test_run_mnist()