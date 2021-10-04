from sys import version
import grpc
import torch
import bittensor

from unittest.mock import MagicMock
import unittest.mock as mock

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
wallet.create_new_coldkey(use_password=False, overwrite = True)
wallet.create_new_hotkey(use_password=False, overwrite = True)

endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 8060,
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkey.ss58_address,
    modality = 0
)
receptor = bittensor.receptor ( 
    endpoint = endpoint, 
    wallet = wallet,
)
channel = grpc.insecure_channel('localhost',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])          
stub = bittensor.grpc.BittensorStub(channel)

def test_print():
    print(receptor)
    print(str(receptor))

#-- dummy testing --

def test_dummy_forward():
    endpoint = bittensor.endpoint.dummy()
    dummy_receptor = bittensor.receptor ( endpoint= endpoint, wallet=wallet)
    assert dummy_receptor.endpoint.uid == -1
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, time = dummy_receptor.forward( x, bittensor.proto.Modality.TEXT, timeout=1)
    assert ops == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

def test_dummy_backward():
    endpoint = bittensor.endpoint.dummy()
    dummy_receptor = bittensor.receptor ( endpoint= endpoint, wallet=wallet)
    assert dummy_receptor.endpoint.uid == -1

    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = torch.ones((x.size(0),x.size(1),bittensor.__network_dim__))
    out, ops, time = dummy_receptor.backward( x,grads,bittensor.proto.Modality.TEXT , timeout=1)
    print (out, ops, time)
    assert ops == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

# -- request serialization --

def test_receptor_forward_request_serialize_error():    
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, time = receptor.forward( x, dict(), timeout=1)
    assert ops == bittensor.proto.ReturnCode.RequestSerializationException


def test_receptor_backward_request_serialize_error():    
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = torch.ones((x.size(0),x.size(1),bittensor.__network_dim__))
    out, ops, time = receptor.backward( x,grads, dict(), timeout=1)
    assert ops == bittensor.proto.ReturnCode.RequestSerializationException

# -- forward testing --

def test_receptor_neuron_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, time = receptor.forward( x, bittensor.proto.Modality.TEXT, timeout=1)
    print (out, ops, time)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

def test_receptor_neuron_image():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, ops, time  = receptor.forward( x, bittensor.proto.Modality.IMAGE, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [1, 1, bittensor.__network_dim__]

def test_receptor_neuron_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward( x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_request_empty():
    x = torch.tensor([])
    out, ops, time  = receptor.forward( x, bittensor.proto.Modality.TEXT, timeout=1)
    assert ops == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [0]

def test_receptor_neuron_mock_server():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_serve_timeout():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Timeout
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_serve_empty():                
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.EmptyResponse
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_deserialization_error():
    y = dict() # bad response
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.ResponseDeserializationException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_shape_error():
    y = torch.rand(1, 3, bittensor.__network_dim__)

    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
   
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.ResponseShapeException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_server_response_with_nans():
    import numpy as np
    y = torch.rand(3, 3, bittensor.__network_dim__)
    y[0][0][0] = np.nan

    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
   
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Success
    assert out[0][0][0] == 0

# -- backwards testing --

def test_receptor_neuron_text_backward():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = torch.ones((x.size(0),x.size(1),bittensor.__network_dim__))
    out, ops, time = receptor.backward( x,grads, bittensor.proto.Modality.TEXT, timeout=1)
    print (out, ops, time)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

def test_receptor_neuron_image_backward():
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, ops, time  = receptor.backward( x,x, bittensor.proto.Modality.IMAGE, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [1, 1, bittensor.__network_dim__]

def test_receptor_neuron_tensor_backward():
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward( x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_request_empty_backward():
    x = torch.tensor([])
    out, ops, time  = receptor.backward( x,x, bittensor.proto.Modality.TEXT, timeout=1)
    assert ops == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [0]

def test_receptor_neuron_mock_server_backward():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.public_key,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_deserialization_error_backward():
    y = dict() # bad response
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.public_key,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.ResponseDeserializationException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_shape_error_backward():
    y = torch.rand(1, 3, bittensor.__network_dim__)

    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
   
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.public_key,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.ResponseShapeException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_server_response_with_nans_backward():
    import numpy as np
    y = torch.rand(3, 3, bittensor.__network_dim__)
    y[0][0][0] = np.nan

    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
   
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.public_key,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.Success
    assert out[0][0][0] == 0
"""
# -- no return code -- 

def test_receptor_forward_no_return():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = None,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.UnknownException

def test_receptor_backward_no_return():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = None,
            tensors = [y_serialized])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.UnknownException
"""
# -- no exception in response -- 

def test_receptor_forward_exception():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.UnknownException,
            tensors = [y_serialized])

    stub.Forward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.UnknownException

def test_receptor_backward_exception():
    y = torch.zeros(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.UnknownException,
            tensors = [y_serialized])

    stub.Backward = MagicMock( return_value=mock_return_val )
    receptor.stub = stub

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert ops == bittensor.proto.ReturnCode.UnknownException

# -- stub erorr -- 

def test_receptor_forward_stub_exception():


    def forward_break():
        raise Exception('Mock')

    with mock.patch.object(receptor.stub, 'Forward', new=forward_break):
        x = torch.rand(3, 3, bittensor.__network_dim__)
        out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
        assert ops == bittensor.proto.ReturnCode.UnknownException

def test_receptor_backward_stub_exception():

    def backward_break():
        raise Exception('Mock')
    with mock.patch.object(receptor.stub, 'Backward', new=backward_break):
        x = torch.rand(3, 3, bittensor.__network_dim__)
        out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
        assert ops == bittensor.proto.ReturnCode.UnknownException


def test_receptor_forward_endpoint_exception():
    
    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )
    
    def forward_break():
        raise Exception('Mock')

    with mock.patch.object(bittensor.proto, 'TensorMessage', new=forward_break):
        x = torch.rand(3, 3, bittensor.__network_dim__)
        out, ops, time  = receptor.forward(x, bittensor.proto.Modality.TENSOR, timeout=1)
        assert ops == bittensor.proto.ReturnCode.UnknownException

def test_receptor_backward_endpoint_exception():
    
    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )
    def backward_break():
        raise Exception('Mock')

    with mock.patch.object(bittensor.proto, 'TensorMessage', new=backward_break):
        x = torch.rand(3, 3, bittensor.__network_dim__)
        out, ops, time  = receptor.backward(x,x, bittensor.proto.Modality.TENSOR, timeout=1)
        assert ops == bittensor.proto.ReturnCode.UnknownException


if __name__ == "__main__":
    test_receptor_neuron_text ()
    test_receptor_neuron_image ()
    test_receptor_neuron_tensor ()
    test_receptor_neuron_request_empty ()
    test_receptor_neuron_mock_server ()
    test_receptor_neuron_mock_server_deserialization_error ()
    test_receptor_neuron_mock_server_shape_error ()
