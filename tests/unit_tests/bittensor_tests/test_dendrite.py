

import torch
import pytest
import time
from munch import Munch
import bittensor

dendrite = bittensor.dendrite.Dendrite()
dendrite.config.receptor.do_backoff = False
neuron_pb2 = bittensor.proto.Neuron(
    version = bittensor.__version__,
    public_key = dendrite.wallet.hotkey.public_key,
    address = '0.0.0.0',
    port = 12345,
)

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_pb2], [x])

def test_dendrite_forward_image_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron_pb2], [x])

def test_dendrite_forward_text_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron_pb2], [x])

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops = dendrite.forward_text( [neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, ops = dendrite.forward_image( [neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = dendrite.forward_tensor( [neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

def test_dendrite_backoff():
    _dendrite = bittensor.dendrite.Dendrite()
    _dendrite.config.receptor.do_backoff = True
    _dendrite.config.receptor.max_backoff = 1
    _neuron_pb2 = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = _dendrite.wallet.hotkey.public_key,
        address = '0.0.0.0',
        port = 12345,
    )
    
    # Add a quick sleep here, it appears that this test is intermittent, likely based on the asyncio operations of past tests.
    # This forces the test to sleep a little while until the dust settles. 
    # Bandaid patch, TODO(unconst): fix this.

    time.sleep(5)
    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Backoff call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Backoff
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

    # Backoff call.
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops = _dendrite.forward_tensor( [_neuron_pb2], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Backoff
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]


if __name__ == "__main__":
    test_dendrite_forward_tensor_shape_error ()
    test_dendrite_forward_image_shape_error ()
    test_dendrite_forward_text_shape_error ()
    test_dendrite_forward_text ()
    test_dendrite_forward_image ()
    test_dendrite_forward_tensor ()
    test_dendrite_backoff ()