import random
import torch
import unittest

from munch import Munch
from unittest.mock import MagicMock

import bittensor
import bittensor.serialization as serialization
import bittensor.utils.serialization_utils as serialization_utils
from bittensor import bittensor_pb2

axon = bittensor.axon.Axon()
synapse = bittensor.synapse.Synapse()

def test_serve():
    assert axon.synapse == None
    for _ in range(0, 10):
        axon.serve(synapse)
    assert axon.synapse != None

def test_forward_not_implemented():
    axon.serve(synapse)
    axon.nucleus.forward = MagicMock(return_value=[None, 'not implemented', bittensor_pb2.ReturnCode.NotImplemented])
    x = torch.rand(3, 3, bittensor.__network_dim__)

    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x_serialized]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotImplemented


def test_forward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse


def test_empty_forward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.EmptyRequest


def test_forward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict()  # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException


def test_forward_success():
    axon.synapse = synapse
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x_serialized]
    )
    axon.nucleus.forward = MagicMock(return_value=[x, 'success', bittensor_pb2.ReturnCode.Success])

    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert serialization_utils.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32


def test_backward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse


def test_empty_backward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_single_item_backward_request():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x_serialized]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_backward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict()  # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException


def test_backward_success():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = serialization.get_serializer( serialzer_type = bittensor_pb2.Serializer.MSGPACK )
    x_serialized = serializer.serialize(x, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)
  
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.keypair.public_key,
        tensors=[x_serialized, x_serialized]
    )
    axon.nucleus.backward = MagicMock(return_value=[x, 'success', bittensor_pb2.ReturnCode.Success])
    response = axon.Backward(request, None)

    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert serialization_utils.bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32


if __name__ == "__main__":    
    test_backward_success()