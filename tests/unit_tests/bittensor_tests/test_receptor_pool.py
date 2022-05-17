# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

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
wallet = bittensor.wallet.mock()
wallet2 = bittensor.wallet.mock()
wallet2.create_new_coldkey(use_password=False, overwrite = True)
wallet2.create_new_hotkey(use_password=False, overwrite = True)


neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkey.ss58_address,
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
        hotkey = wallet2.hotkey.ss58_address,
        coldkey = wallet2.coldkey.ss58_address,
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
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
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