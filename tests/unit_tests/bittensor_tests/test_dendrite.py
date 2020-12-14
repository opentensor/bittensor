

import torch
import bittensor
import pytest

from bittensor.config import Config
from bittensor.subtensor import Keypair
from bittensor import bittensor_pb2
import time


config = Config.load()
config.dendrite.do_backoff = False
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)
dendrite = bittensor.dendrite.Dendrite(config, keypair)
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
    _config = Config.load()
    _config.dendrite.do_backoff = True
    _config.dendrite.max_backoff = 1
    _mnemonic = Keypair.generate_mnemonic()
    _keypair = Keypair.create_from_mnemonic(_mnemonic)
    _dendrite = bittensor.dendrite.Dendrite(_config, _keypair)
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