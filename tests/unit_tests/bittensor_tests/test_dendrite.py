

import torch
import bittensor
import pytest

from bittensor.config import Config
from bittensor.subtensor.interface import Keypair
from bittensor import bittensor_pb2
import time
from munch import Munch

config = {'neuron':
              {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
               'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
               'name': 'mnist', 'trial_id': '1608070667'},
          'synapse': {'target_dim': 10},
          'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True, 'timeout': 0.5,
                       'do_backoff': True, 'max_backoff': 100}, 'axon': {'port': 8091, 'remote_ip': '191.97.53.53'},
          'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
          'metagraph': {'chain_endpoint': '206.189.254.5:12345', 'stale_emit_filter': 10000},
          'meta_logger': {'log_dir': 'data/'},
          'session': {'keyfile': None, 'keypair': None }
          }

config = Munch.fromDict(config)


config.dendrite.do_backoff = False
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)

config.session.keypair = keypair

dendrite = bittensor.dendrite.Dendrite(config)
neuron = bittensor_pb2.Neuron(
    version = bittensor.__version__,
    public_key = keypair.public_key,
    address = '0.0.0.0',
    port = 12345,
)


def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron], [x])

def test_dendrite_forward_image_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron], [x])

def test_dendrite_forward_text_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron], [x])

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops = dendrite.forward_text( [neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, ops = dendrite.forward_image( [neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out[0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = dendrite.forward_tensor( [neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]


def test_dendrite_backoff():
    _config = Munch.fromDict(config.copy())
    _config.dendrite.do_backoff = True
    _config.dendrite.max_backoff = 1
    _mnemonic = Keypair.generate_mnemonic()
    _keypair = Keypair.create_from_mnemonic(_mnemonic)
    _dendrite = bittensor.dendrite.Dendrite(_config)
    _neuron = bittensor_pb2.Neuron(
        version = bittensor.__version__,
        public_key = _keypair.public_key,
        address = '0.0.0.0',
        port = 12345,
    )
    
    # Add a quick sleep here, it appears that this test is intermittent, likely based on the asyncio operations of past tests.
    # This forces the test to sleep a little while until the dust settles. 
    # Bandaid patch, TODO(unconst): fix this.

    time.sleep(5)
    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Backoff call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Backoff
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Backoff call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron], [x])
    assert ops[0].item() == bittensor_pb2.ReturnCode.Backoff
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]


if __name__ == "__main__":
    test_dendrite_forward_tensor_shape_error ()
    test_dendrite_forward_image_shape_error ()
    test_dendrite_forward_text_shape_error ()
    test_dendrite_forward_text ()
    test_dendrite_forward_image ()
    test_dendrite_forward_tensor ()
    test_dendrite_backoff ()