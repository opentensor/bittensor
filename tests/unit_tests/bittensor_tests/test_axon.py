import random
import torch
import unittest
from munch import Munch
from unittest.mock import MagicMock
import bittensor
import bittensor.serialization as serialization

axon = bittensor.axon.Axon()

def test_forward_success():
    axon.enqueue_forward_to_nucleus = MagicMock(
        return_value=[torch.tensor([1]), bittensor.proto.ReturnCode.NotImplemented, 'not implemented' ]
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_not_implemented():
    axon.enqueue_forward_to_nucleus = MagicMock(
        return_value=[None, bittensor.proto.ReturnCode.NotImplemented, 'not implemented']
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented


def test_forward_empty_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[]
    )
    response, code, message = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_forward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_forward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_forward_deserialization():
    axon.enqueue_forward_to_nucleus = MagicMock(
        return_value=[None, bittensor.proto.ReturnCode.Success, 'success']
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.ResponseDeserializationException


def test_forward_pipe_timeout():
    axon = bittensor.axon.Axon()
    axon.config.axon.forward_processing_timeout = 0.1
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NucleusTimeout

def test_backward_pipe_timeout():
    axon = bittensor.axon.Axon()
    axon.config.axon.backward_processing_timeout = 0.1
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.NucleusTimeout


def test_backward_invalid_request():
    axon = bittensor.axon.Axon()
    axon.config.axon.backward_processing_timeout = 0.1
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest

def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x, g]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_backward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_grad_inputs_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(2, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_response_deserialization_error():
    axon = bittensor.axon.Axon()
    axon.enqueue_backward_to_nucleus = MagicMock(
        return_value=[None, bittensor.proto.ReturnCode.Success, 'success']
    )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.ResponseSerializationException


def test_backward_response_success():
    axon = bittensor.axon.Axon()
    axon.enqueue_backward_to_nucleus = MagicMock(
        return_value=[torch.tensor([1]), bittensor.proto.ReturnCode.Success, 'Success']
    )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success