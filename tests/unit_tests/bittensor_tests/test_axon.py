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
        return_value=[torch.tensor([1]), 'not implemented', bittensor.proto.ReturnCode.NotImplemented]
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, message, code = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_not_implemented():
    axon.enqueue_forward_to_nucleus = MagicMock(
        return_value=[None, 'not implemented', bittensor.proto.ReturnCode.NotImplemented]
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, message, code = axon._forward( request )
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
    response, message, code = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x ]
    )
    response, message, code  = axon._forward( request )
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
    response, message, code  = axon._forward( request )
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
    response, message, code  = axon._forward( request )
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
    response, message, code  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_forward_deserialization():
    axon.enqueue_forward_to_nucleus = MagicMock(
        return_value=[None, 'Success', bittensor.proto.ReturnCode.Success]
    )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, message, code = axon._forward( request )
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
    response, message, code = axon._forward( request )
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
    response, message, code = axon._backward( request )
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
    response, message, code = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest

def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x, g]
    )
    response, message, code  = axon._backward( request )
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
    response, message, code  = axon._backward( request )
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
    response, message, code  = axon._backward( request )
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
    response, message, code  = axon._backward( request )
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
    response, message, code = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_response_deserialization_error():
    axon = bittensor.axon.Axon()
    axon.enqueue_backward_to_nucleus = MagicMock(
        return_value=[None, 'Success', bittensor.proto.ReturnCode.Success]
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
    response, message, code = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.ResponseSerializationException


def test_backward_response_success():
    axon = bittensor.axon.Axon()
    axon.enqueue_backward_to_nucleus = MagicMock(
        return_value=[torch.tensor([1]), 'Success', bittensor.proto.ReturnCode.Success]
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
    response, message, code = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

# def test_forward_tensor_shape_error():
#     inputs_raw = torch.rand(1, 1, 1, 1)
#     serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
#     inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#         tensors=[ inputs_serialized ]
#     )
#     response, message, code  = axon._forward( request )
#     assert code == bittensor.proto.ReturnCode.RequestShapeException


# def test_forward_success():
#     axon.nucleus = nucleus
#     x = torch.rand(3, 3, bittensor.__network_dim__)
#     serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
#     x_serialized = serializer.serialize(x, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#         tensors=[x_serialized]
#     )
#     axon.nucleus.forward = MagicMock(return_value=[x, 'success', bittensor.proto.ReturnCode.Success])

#     response = axon.Forward(request, None)
#     assert response.return_code == bittensor.proto.ReturnCode.Success
#     assert len(response.tensors) == 1
#     assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
#     assert serialization.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32


# def test_backward_not_serving():
#     axon.nucleus = None
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#     )
#     response = axon.Backward(request, None)
#     assert response.return_code == bittensor.proto.ReturnCode.NotServingNucleus


# def test_empty_backward_request():
#     axon.serve(nucleus)
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#     )
#     response = axon.Backward(request, None)
#     assert response.return_code == bittensor.proto.ReturnCode.InvalidRequest


# def test_single_item_backward_request():
#     axon.serve(nucleus)
#     x = torch.rand(3, 3, bittensor.__network_dim__)
#     serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
#     x_serialized = serializer.serialize(x, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#         tensors=[x_serialized]
#     )
#     response = axon.Backward(request, None)
#     assert response.return_code == bittensor.proto.ReturnCode.InvalidRequest


# def test_backward_deserialization_error():
#     axon.serve(nucleus)
#     x = dict()
#     y = dict()  # Not tensors that can be deserialized.
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#         tensors=[x, y]
#     )
#     response = axon.Backward(request, None)
#     assert response.return_code == bittensor.proto.ReturnCode.RequestDeserializationException


# def test_backward_success():
#     axon.serve(nucleus)
#     x = torch.rand(3, 3, bittensor.__network_dim__)
#     serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
#     x_serialized = serializer.serialize(x, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = axon.wallet.hotkey.public_key,
#         tensors=[x_serialized, x_serialized]
#     )
#     axon.nucleus.backward = MagicMock(return_value=[x, 'success', bittensor.proto.ReturnCode.Success])
#     response = axon.Backward(request, None)

#     assert response.return_code == bittensor.proto.ReturnCode.Success
#     assert len(response.tensors) == 1
#     assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
#     assert serialization.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32

# def test_set_priority():
#     axon = bittensor.axon.Axon()
#     n1 = bittensor.proto.Neuron(
#         version = bittensor.__version__,
# 	    public_key = '12345',
#         address = '10DowningStree',
# 	    port = 666,
#         ip_type = 4,
#         modality = 0,
#         uid = 120
#     )
#     axon.set_priority([n1], torch.tensor([1]))
#     assert axon.priority['12345'] == 1

#     # Now get priority
#     request = bittensor.proto.TensorMessage(
#         version=bittensor.__version__,
#         public_key = '12345',
#     )
#     assert abs(axon.get_call_priority(request) - 1.0) < 0.0001


# def test_priority_never_matches():
#     axon = bittensor.axon.Axon()
#     n1 = bittensor.proto.Neuron(
#         version = bittensor.__version__,
# 	    public_key = '12345',
#         address = '10DowningStree',
# 	    port = 666,
#         ip_type = 4,
#         modality = 0,
#         uid = 120
#     )
#     axon.set_priority([n1], torch.tensor([1]))
#     assert axon.priority['12345'] == 1

#     # Now get priority
#     previously_seen = set()
#     for i in range(50000):
#         request = bittensor.proto.TensorMessage(
#             version=bittensor.__version__,
#             public_key = '12345',
#         )
#         priority = axon.get_call_priority(request)
#         assert priority not in previously_seen
#         previously_seen.add(priority)

# if __name__ == "__main__":    
#     test_priority_never_matches()