from sys import version
from bittensor._endpoint import endpoint
import bittensor
import torch
import pytest
from unittest.mock import MagicMock
from torch.autograd import Variable
import multiprocessing
import time

dendrite = bittensor.dendrite()
dendrite.receptor_pool.forward = MagicMock(return_value = [torch.tensor([]), [1], [0]]) 
dendrite.receptor_pool.backward = MagicMock(return_value = [torch.tensor([]), [1], [0]]) 
endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    hotkey = '',
    ip = '0.0.0.0', 
    ip_type = 4, 
    port = 8080, 
    modality = 0, 
    coldkey = ''
)

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( endpoints=[endpoint], inputs=[x])

def test_dendrite_forward_image_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( endpoints=[endpoint], inputs=[x])

def test_dendrite_forward_text_shape_error():
    x = torch.rand(3, 3, 3)
    with pytest.raises(ValueError):
        dendrite.forward_image( endpoints=[endpoint], inputs=[x])

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [torch.zeros([2, 4, bittensor.__network_dim__])], [1], [0]]) 
    tensors, codes, times = dendrite.forward_text( endpoints=[endpoint], inputs=[x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ], dtype=torch.float32)
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [torch.zeros([1, 1, bittensor.__network_dim__])] , [1], [0]]) 
    tensors, codes, times  = dendrite.forward_image( endpoints=[endpoint], inputs=[x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [torch.zeros([3, 3, bittensor.__network_dim__])], [1], [0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints=[endpoint], inputs=[x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0].shape) == [3, 3, bittensor.__network_dim__]

def test_dendrite_forward_tensor_pass_through_text():
    x = torch.ones((3, 3), dtype=torch.int64)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y, y, y] , [1, 1, 1], [0,0,0]]) 
    tensors, codes, times = dendrite.forward_text( endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1].item() == bittensor.proto.ReturnCode.Success
    assert codes[2].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0].shape == y.shape
    assert tensors[1].shape == y.shape
    assert tensors[2].shape == y.shape

def test_dendrite_forward_tensor_pass_through_image():
    x = torch.rand(3, 3, 3, 3, 3)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y, y, y] , [1, 1, 1], [0,0,0]]) 
    tensors, codes, times = dendrite.forward_image( endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1].item() == bittensor.proto.ReturnCode.Success
    assert codes[2].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0].shape == y.shape
    assert tensors[1].shape == y.shape
    assert tensors[2].shape == y.shape

def test_dendrite_forward_tensor_pass_through_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y, y, y] , [1, 1, 1], [0,0,0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints = [endpoint, endpoint, endpoint], inputs=[x, x, x])
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1].item() == bittensor.proto.ReturnCode.Success
    assert codes[2].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0].shape == y.shape
    assert tensors[1].shape == y.shape
    assert tensors[2].shape == y.shape

def test_dendrite_forward_tensor_stack():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y, y, y] , [1, 1, 1], [0,0,0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints = [endpoint, endpoint, endpoint], inputs = [x, x, x])
    stacked = torch.stack(tensors, dim=2)
    assert stacked.shape == torch.zeros([3, 3, 3, bittensor.__network_dim__ ]).shape
    averaged = torch.mean(stacked, dim=2)
    assert averaged.shape == torch.zeros([3, 3, bittensor.__network_dim__ ]).shape

def test_dendrite_backward():
    x = Variable(torch.rand((1, 1, bittensor.__network_dim__), dtype=torch.float32), requires_grad=True)
    y = torch.ones((1, 1, bittensor.__network_dim__))
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y], [0], [0]]) 
    dendrite.receptor_pool.backward = MagicMock(return_value = [ [y], [0], [0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints = [ endpoint ], inputs=[ x ])
    tensors[0].sum().backward()
    assert x.grad.shape == y.shape

def test_dendrite_backward_large():
    x = Variable(torch.rand((1, 1, bittensor.__network_dim__), dtype=torch.float32), requires_grad=True)
    y = torch.ones((1, 1, bittensor.__network_dim__))
    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y], [0], [0]]) 
    dendrite.receptor_pool.backward = MagicMock(return_value = [ [y], [0], [0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints = [ endpoint ], inputs=[ x ])
    tensors[0].sum().backward()
    assert x.grad.shape == y.shape
    assert x.grad.tolist() == y.tolist()

def test_dendrite_backward_multiple():
    x1 = Variable(torch.rand((1, 1, bittensor.__network_dim__), dtype=torch.float32), requires_grad=True)
    x2 = Variable(torch.rand((1, 1, bittensor.__network_dim__), dtype=torch.float32), requires_grad=True)
    x3 = Variable(torch.rand((1, 1, bittensor.__network_dim__), dtype=torch.float32), requires_grad=True)
    y1 = torch.ones(1, 1, bittensor.__network_dim__)
    y2 = torch.ones(1, 1, bittensor.__network_dim__)
    y3 = torch.ones(1, 1, bittensor.__network_dim__)

    dendrite.receptor_pool.forward = MagicMock(return_value = [ [y1, y2, y3], [1,1,1], [0,0,0]]) 
    dendrite.receptor_pool.backward = MagicMock(return_value = [ [y1, y2, y3], [1,1,1], [0,0,0]]) 
    tensors, codes, times = dendrite.forward_tensor( endpoints = [endpoint, endpoint, endpoint], inputs=[ x1, x2, x3 ])
    tensors[0].sum().backward()
    assert x1.grad.shape == y1.shape
    assert x2.grad.shape == y2.shape
    assert x3.grad.shape == y3.shape
    assert x1.grad.tolist() == y1.tolist()
    assert x2.grad.tolist() == y2.tolist()
    assert x3.grad.tolist() == y3.tolist()


wallet =  bittensor.wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 

wallet.create_new_coldkey( use_password=False, overwrite = True)
wallet.create_new_hotkey( use_password=False, overwrite = True)

def test_axon_receptor_forward_works():
    def forward( inputs_x:torch.FloatTensor):
        time.sleep(0.2)
        return torch.zeros([3, 3, bittensor.__network_dim__])
    axon = bittensor.axon (
        port = 8080,
        ip = '0.0.0.0',
        wallet = wallet,
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()
    endpoints = []
    for i in range(5):
        endpoint = bittensor.endpoint(
            version = bittensor.__version_as_int__,
            uid = 1,
            hotkey = str(i),
            ip = '0.0.0.0', 
            ip_type = 4, 
            port = 8080, 
            modality = 0, 
            coldkey = ''
        )
        endpoints += [endpoint]
    dendrite = bittensor.dendrite(max_active_receptors= 500)
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    tensors, codes, times = dendrite.forward_tensor( endpoints=endpoints, inputs=[x for i in endpoints])
    for i in dendrite.receptor_pool.receptors:
        assert(dendrite.receptor_pool.receptors[i].state() == dendrite.receptor_pool.receptors[i].state().READY)
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0].shape) == [3, 3, bittensor.__network_dim__]


if __name__  == "__main__":
    test_axon_receptor_forward_works()
    #test_dendrite_backward_large()

