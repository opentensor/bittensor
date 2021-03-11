import grpc
import torchvision.transforms as transforms
import torch
import unittest
import bittensor
import pytest
import torchvision

from unittest.mock import MagicMock
import bittensor.serialization as serialization
from munch import Munch


neuron = bittensor.proto.Neuron(
    version = bittensor.__version__,
    public_key = "A",
    address = '0',
    port = 1,
)
receptor = bittensor.receptor.Receptor( neuron = neuron )


def test_receptor_create():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0',
        port = 1,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    assert receptor.endpoint == '0:1'

def test_receptor_localhost_endpoint():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = "1.1.1.1",
        port = 1,
    )
    config = bittensor.receptor.Receptor.default_config()
    config.axon = Munch()
    config.axon.external_ip = "1.1.1.1"
    receptor = bittensor.receptor.Receptor( neuron = neuron, config = config )
    assert receptor.endpoint == 'localhost:1'


def test_receptor_neuron_text():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.forward( inputs = x, mode = bittensor.proto.Modality.TEXT)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]

def test_receptor_neuron_image():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    out, op, msg = receptor.forward( x, bittensor.proto.Modality.IMAGE)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [1, 1, bittensor.__network_dim__]

def test_receptor_neuron_tensor():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward( x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_request_empty():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([])
    out, op, msg = receptor.forward( x, bittensor.proto.Modality.TEXT)
    assert op == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [0]

def test_receptor_neuron_mock_server():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )

    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_serve_timeout():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )

    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [y_serialized])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Timeout
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_serve_empty():   
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )             
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.EmptyResponse
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_deserialization_error():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 
    y = dict() # bad response
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.ResponseDeserializationException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_server_shape_error():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 

    y = torch.rand(1, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )


    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.ResponseShapeException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_server_response_with_nans():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 

    import numpy as np
    y = torch.rand(3, 3, bittensor.__network_dim__)
    y[0][0][0] = np.nan
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Forward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    out, op, msg = receptor.forward(x, bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Success
    assert out[0][0][0] == 0


def test_receptor_backward_neuron_text():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TEXT)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [2, 4, bittensor.__network_dim__]


def test_receptor_neuron_backward_image():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([ [ [ [ [ 1 ] ] ] ] ])
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode =bittensor.proto.Modality.IMAGE)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [1, 1, bittensor.__network_dim__]

def test_receptor_neuron_backward_tensor():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Unavailable
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_backward_non_forward_success():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 1, mode = bittensor.proto.Modality.TENSOR)
    assert op == 1
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_backward_request_empty():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    x = torch.tensor([])
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TEXT)
    assert op == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [0]

def test_receptor_neuron_backward_request_empty_grads():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )
    g = torch.tensor([])
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TEXT)
    assert op == bittensor.proto.ReturnCode.EmptyRequest
    assert list(out.shape) == [2, 4, 512]

def test_receptor_neuron_backward_mock_server():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )

    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_backward_serve_timeout():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )

    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [y_serialized])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Timeout
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_backward_serve_empty():   
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron )             
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.EmptyResponse
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_mock_backward_server_deserialization_error():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 
    y = dict() # bad response
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.ResponseDeserializationException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]

def test_receptor_neuron_mock_backward_server_shape_error():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 

    y = torch.rand(1, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.ResponseShapeException
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]


def test_receptor_neuron_server_backward_response_with_nans():
    neuron = bittensor.proto.Neuron(
        version = bittensor.__version__,
        public_key = "A",
        address = '0.0.0.0',
        port = 22424,
    )
    receptor = bittensor.receptor.Receptor( neuron = neuron ) 

    import numpy as np
    y = torch.rand(3, 3, bittensor.__network_dim__)
    y[0][0][0] = np.nan
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version__,
            public_key = "A",
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized])
    receptor.stub.Backward = MagicMock( return_value=mock_return_val )

    x = torch.rand(3, 3, bittensor.__network_dim__)
    g = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, op, msg = receptor.backward( inputs = x, grads = g, code = 0, mode = bittensor.proto.Modality.TENSOR)
    assert op == bittensor.proto.ReturnCode.Success
    assert list(out.shape) == [3, 3, bittensor.__network_dim__]
