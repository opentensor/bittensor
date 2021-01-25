from bittensor.synapses.ffnn import FFNNSynapse
from bittensor.miner import Miner
import os, sys, time
from unittest.mock import MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("examples/IMAGE/")
import bittensor
import torch
import numpy
from mnist import Session

def test_run_mnist():
    miner = Miner(FFNNSynapse)
    mnist_session_config = miner.build_config()
    mnist_session_config.miner.n_epochs = 1
    mnist_session_config.miner.epoch_length = 1
    mnist_session = Session(FFNNSynapse, mnist_session_config)
    mnist_session.neuron.metagraph.connect = MagicMock(return_value = (bittensor.metagraph.Metagraph.ConnectSuccess, ""))    
    mnist_session.neuron.metagraph.subscribe = MagicMock(return_value = (bittensor.metagraph.Metagraph.SubscribeSuccess, ""))   
    mnist_session.neuron.metagraph.set_weights = MagicMock()   
    mnist_session.neuron.metagraph.sync = MagicMock()  
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = mnist_session.neuron.wallet.keypair.public_key,
        address = mnist_session_config.axon.external_ip,
        port = mnist_session_config.axon.external_port,
        uid = 0,
    )
    mnist_session.neuron.metagraph.uid = 0
    mnist_session.neuron.metagraph.state.n = 1
    mnist_session.neuron.metagraph.state.tau = torch.tensor([0.5], dtype = torch.float32)
    mnist_session.neuron.metagraph.state.neurons = [neuron]
    mnist_session.neuron.metagraph.state.indices = torch.tensor([0], dtype=torch.int64)
    mnist_session.neuron.metagraph.state.uids = torch.tensor([0], dtype=torch.int64)
    mnist_session.neuron.metagraph.state.lastemit = torch.tensor([0], dtype=torch.int64)
    mnist_session.neuron.metagraph.state.stake = torch.tensor([0], dtype=torch.float32)
    mnist_session.neuron.metagraph.state.uid_for_pubkey[mnist_session.neuron.wallet.keypair.public_key] = 0
    mnist_session.neuron.metagraph.state.index_for_uid[0] = 0
    mnist_session.neuron.metagraph.state.W = torch.tensor( numpy.ones( (1, 1) ), dtype=torch.float32)
    mnist_session.run()
test_run_mnist()