from sys import version
import grpc
import torch
import bittensor
import time

from unittest.mock import MagicMock
import unittest.mock as mock
import asyncio

logging = bittensor.logging()

# --- Receptor Pool ---

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
wallet.create_new_coldkey(use_password=False, overwrite = True)
wallet.create_new_hotkey(use_password=False, overwrite = True)

wallet2 =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest2',
) 
wallet2.create_new_coldkey(use_password=False, overwrite = True)
wallet2.create_new_hotkey(use_password=False, overwrite = True)


neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = wallet.hotkey.public_key,
    coldkey = wallet.coldkey.public_key,
    modality = 0
)

receptor_pool = bittensor.receptor_pool(wallet=wallet)

def test_receptor_pool_forward():
    endpoints = [neuron_obj]
    x = torch.ones( (1,2,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert list(torch.stack(resp1, dim=0).shape) == [1, 2, 2, bittensor.__network_dim__]

def test_receptor_pool_backward():
    endpoints = [neuron_obj]
    x = torch.ones( (1,2,2) )
    receptor_pool.backward( endpoints, x,x, bittensor.proto.Modality.TENSOR, timeout=1)


def test_receptor_pool_max_workers_forward():
    neuron_obj2 = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.1',
        ip_type = 4,
        port = 12345,
        hotkey = wallet2.hotkey.public_key,
        coldkey = wallet2.coldkey.public_key,
        modality = 0
    )
    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    endpoints = [neuron_obj,neuron_obj2]
    x = torch.ones( (2,2,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert list(torch.stack(resp1, dim=0).shape) == [2, 2, 2, bittensor.__network_dim__]

def test_receptor_pool_forward_hang():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2,2,2) )    
    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [])
    
    future = asyncio.Future()
    future.set_result(mock_return_val)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward.future = MagicMock( return_value = future )
    resp1,  codes, _ = receptor_pool.forward( endpoints, x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert codes == [bittensor.proto.ReturnCode.Timeout,bittensor.proto.ReturnCode.Timeout]

def test_receptor_pool_backward_hang():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2,2,2) )
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [])
    
    future = asyncio.Future()
    future.set_result(mock_return_val)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)
    receptor_pool.receptors[neuron_obj.hotkey].stub.Backward.future = MagicMock( return_value = future )
    receptor_pool.backward( endpoints, x,x, bittensor.proto.Modality.TENSOR, timeout=1)

if __name__ == "__main__":
    test_receptor_pool_backward_hang()