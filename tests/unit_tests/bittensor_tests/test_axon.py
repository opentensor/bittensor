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

import time
import unittest.mock as mock
import uuid

import grpc
import pytest
import torch

import bittensor
from bittensor.utils.test_utils import get_random_unused_port

wallet = bittensor.wallet.mock()
axon = bittensor.axon(wallet = wallet)

def sign(wallet):
    nounce = str(int(time.time() * 1000))
    receptor_uid = str(uuid.uuid1())
    message  = "{}{}{}".format(nounce, str(wallet.hotkey.ss58_address), receptor_uid)
    spliter = 'bitxx'
    signature = spliter.join([ nounce, str(wallet.hotkey.ss58_address), "0x" + wallet.hotkey.sign(message).hex(), receptor_uid])
    return signature

def test_sign():
    sign(wallet)
    sign(axon.wallet)

def test_forward_wandb():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    axon.update_stats_for_request( request, response, call_time, code )
    print( axon.to_wandb() )


def test_forward_not_implemented():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_tensor_success():
    def forward( inputs_x: torch.FloatTensor):
        return torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_forward_callback( forward, modality=2)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_tensor_success_image():
    def forward( inputs_x: torch.FloatTensor):
        return torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_forward_callback( forward, modality=1)
    inputs_raw = torch.rand(1,1,1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_empty_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[]
    )
    response, code, call_time, message = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ x ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_forward_batch_shape_error():
    inputs_raw = torch.rand(0, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_seq_shape_error():
    inputs_raw = torch.rand(1, 0, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_deserialization_empty():
    def forward( inputs_x: torch.Tensor):
        return None
    axon.attach_forward_callback( forward, modality = bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.EmptyResponse

def test_forward_response_deserialization_error():
    def forward( inputs_x: torch.Tensor):
        return dict()
    axon.attach_forward_callback( forward, modality = bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.ResponseDeserializationException

def test_forward_tensor_exception():
    def forward( inputs_x: torch.FloatTensor):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise Exception('Mock')
    axon.attach_forward_callback( forward, modality=2)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey= '123'
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

def test_forward_tensor_timeout():
    def forward( inputs_x: torch.FloatTensor):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise TimeoutError('Timeout')
    axon.attach_forward_callback( forward, modality=2)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey= '123'
    )

    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Timeout

def test_forward_unknown_error():
    def forward( inputs_x: torch.FloatTensor,modality):
        raise Exception('Unknown')
    with mock.patch.object(axon, '_call_forward', new=forward):
        inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
        serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
        request = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            tensors=[inputs_serialized],
            hotkey= '123'
        )

        response, code, call_time, message = axon._forward( request )
        assert code == bittensor.proto.ReturnCode.UnknownException

#--- backwards ---

def test_backward_invalid_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest

def test_backward_response_not_implemented():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ x, g]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_backward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_grads_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_grad_inputs_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(2, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_response_serialization_error():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ):
        return dict() 
    axon.attach_backward_callback( backward, modality=bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.ResponseSerializationException

def test_backward_response_empty_error():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ):
        return None
    axon.attach_backward_callback( backward,modality=bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.EmptyResponse

def test_backward_response_success_text():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1])
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TEXT )
    inputs_raw = torch.ones((1, 1))
    grads_raw = torch.zeros((1, 1, bittensor.__network_dim__))
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_success_image():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1])
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.IMAGE )
    inputs_raw = torch.ones((1, 1,1,1,1))
    grads_raw = torch.zeros((1, 1, bittensor.__network_dim__))
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_success():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TENSOR )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_timeout():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise TimeoutError('Timeout')
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TENSOR )
    inputs_raw = torch.rand(2, 2, 2)
    grads_raw = torch.rand(2, 2, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Timeout

def test_backward_response_exception():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise Exception('Timeout')
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TENSOR )
    inputs_raw = torch.rand(2, 2, 2)
    grads_raw = torch.rand(2, 2, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

# -- axon priority:

def test_forward_tensor_success_priority():
    
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    axon = bittensor.axon(wallet = wallet, priority= priority)

    def forward( inputs_x: torch.FloatTensor):
        return torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_forward_callback( forward, modality=2)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_success_text_priority():
        
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    axon = bittensor.axon(wallet = wallet, priority= priority)

    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1])
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TEXT )
    inputs_raw = torch.ones((1, 1))
    grads_raw = torch.zeros((1, 1, bittensor.__network_dim__))
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success


def test_grpc_forward_works():
    def forward( inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 7084,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7084',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    response = stub.Forward(request,
                            metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(axon.wallet)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ))

    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()
    assert axon.stats.total_requests == 1 
    axon.to_wandb()


def test_grpc_forward_works_gzip():
    def forward( inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 7082,
        ip = '127.0.0.1',
        wallet = wallet,
        compression= 'gzip'
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7082',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    response = stub.Forward(request,
                            metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(axon.wallet)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ))

    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()
    assert axon.stats.total_requests == 1 
    axon.to_wandb()


def test_grpc_forward_works_deflate():
    def forward( inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 7083,
        ip = '127.0.0.1',
        wallet = wallet,
        compression= 'deflate'
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7083',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    response = stub.Forward(request,
                            metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(axon.wallet)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ))

    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()
    assert axon.stats.total_requests == 1 
    axon.to_wandb()


def test_grpc_backward_works():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])

    axon = bittensor.axon (
        port = 7086,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_backward_callback( backward , modality = bittensor.proto.Modality.TENSOR)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7086',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized, grads_serialized]
    )
    response = stub.Backward(request,
                             metadata = (
                                    ('rpc-auth-header','Bittensor'),
                                    ('bittensor-signature',sign(axon.wallet)),
                                    ('bittensor-version',str(bittensor.__version_as_int__)),
                                    ))
    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()

def test_grpc_forward_fails():
    def forward( inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 7081,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7081',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    try:
        response = stub.Forward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def test_grpc_backward_fails():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])

    axon = bittensor.axon (
        port = 7085,
        ip = '127.0.0.1',
        wallet = wallet
    )
    axon.attach_backward_callback( backward , modality = bittensor.proto.Modality.TENSOR)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7085',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized, grads_serialized]
    )
    
    try:
        response = stub.Backward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        val = s.connect_ex(('localhost', port))
        if val == 0:
            return True
        else:
            return False

def test_axon_is_destroyed():
    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon = bittensor.axon ( port = port )
    assert is_port_in_use( port ) == True
    axon.start()
    assert is_port_in_use( port ) == True
    axon.stop()
    assert is_port_in_use( port ) == False
    axon.__del__()
    assert is_port_in_use( port ) == False

    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon2 = bittensor.axon ( port = port )
    assert is_port_in_use( port ) == True
    axon2.start()
    assert is_port_in_use( port ) == True
    axon2.__del__()
    assert is_port_in_use( port ) == False

    port_3 = get_random_unused_port()
    assert is_port_in_use( port_3 ) == False
    axonA = bittensor.axon ( port = port_3 )
    assert is_port_in_use( port_3 ) == True
    axonB = bittensor.axon ( port = port_3 )
    assert axonA.server != axonB.server
    assert is_port_in_use( port_3 ) == True
    axonA.start()
    assert is_port_in_use( port_3 ) == True
    axonB.start()
    assert is_port_in_use( port_3 ) == True
    axonA.__del__()
    assert is_port_in_use( port ) == False
    axonB.__del__()
    assert is_port_in_use( port ) == False


if __name__ == "__main__":
    #test_backward_response_serialization_error()
    #test_axon_is_destroyed()
    test_forward_wandb()
    test_grpc_forward_works()