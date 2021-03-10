import bittensor
import torch
import pytest
from unittest.mock import MagicMock

bittensor.init()
bittensor.neuron.dendrite.forward = MagicMock(return_value = [torch.tensor([]), [0]]) 
bittensor.neuron.dendrite.backward = MagicMock(return_value = [torch.tensor([]), [0]]) 

neuron_pb2 = bittensor.proto.Neuron(
    version = bittensor.__version__,
    public_key = bittensor.neuron.wallet.hotkey.public_key,
    address = '0.0.0.0',
    port = 12345,
)

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        bittensor.forward_tensor( neurons=[neuron_pb2], inputs=[x])

def test_dendrite_forward_image_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        bittensor.forward_image( neurons=[neuron_pb2], inputs=[x])

def test_dendrite_forward_text_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        bittensor.forward_image( neurons=[neuron_pb2], inputs=[x])

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    bittensor.neuron.dendrite.forward = MagicMock(return_value = [ [torch.zeros([2, 4, bittensor.__network_dim__])] , [0]]) 
    out, ops = bittensor.forward_text( neurons=[neuron_pb2], inputs=[x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Success
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    bittensor.neuron.dendrite.forward = MagicMock(return_value = [ [torch.zeros([1, 1, bittensor.__network_dim__])] , [0]]) 
    out, ops = bittensor.forward_image( neurons=[neuron_pb2], inputs=[x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Success
    assert list(out[0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    bittensor.neuron.dendrite.forward = MagicMock(return_value = [ [torch.zeros([3, 3, bittensor.__network_dim__])], [0]]) 
    out, ops = bittensor.forward_tensor( neurons=[neuron_pb2], inputs=[x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Success
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]
