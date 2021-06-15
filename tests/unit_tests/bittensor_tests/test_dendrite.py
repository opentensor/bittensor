

import torch
import pytest
import time
import bittensor

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
wallet.create_new_coldkey(use_password=False, overwrite = True)
wallet.create_new_hotkey(use_password=False, overwrite = True)

dendrite = bittensor.dendrite( wallet = wallet )
neuron_obj = bittensor.endpoint(
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = dendrite.wallet.hotkey.public_key,
    coldkey = dendrite.wallet.coldkey.public_key,
    modality = 0
)

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3, dtype=float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_obj], [x])

def test_dendrite_forward_image_shape_error():
    x = torch.rand(3, 3, 3, dtype=float32)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron_obj], [x])

def test_dendrite_forward_text_shape_error():
    x = torch.zeros((3, 3, 3), dtype=int64)
    with pytest.raises(ValueError):
        dendrite.forward_image( [neuron_obj], [x])

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops = dendrite.forward_text( [neuron_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ], dtype=float32)
    out, ops = dendrite.forward_image( [neuron_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=float32)
    out, ops = dendrite.forward_tensor( [neuron_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

def test_dendrite_backoff():
    _dendrite = bittensor.dendrite( wallet = wallet )
    _endpoint_obj = bittensor.endpoint(
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = _dendrite.wallet.hotkey.public_key,
        coldkey = _dendrite.wallet.coldkey.public_key,
        modality = 0
    )
    print (_endpoint_obj)
    
    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=float32)
    out, ops = _dendrite.forward_tensor( [_endpoint_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]


if __name__ == "__main__":
    # test_dendrite_forward_tensor_shape_error ()
    # test_dendrite_forward_image_shape_error ()
    # test_dendrite_forward_text_shape_error ()
    # test_dendrite_forward_text ()
    # test_dendrite_forward_image ()
    # test_dendrite_forward_tensor ()
    test_dendrite_backoff ()