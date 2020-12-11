from unittest.mock import MagicMock

from bittensor.config import Config
from bittensor.subtensor import Keypair
from bittensor.synapse import Synapse
from bittensor.serializer import PyTorchSerializer, torch_dtype_to_bittensor_dtype, bittensor_dtype_to_torch_dtype
from bittensor import bittensor_pb2
import bittensor
import unittest
import random
import torch

config = Config.load()
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)
axon = bittensor.axon.Axon( config, keypair)
synapse = Synapse( config, None )

def test_serve():
    assert axon.synapse == None
    for _ in range(0, 10):
        axon.serve(synapse)
    assert axon.synapse != None

def test_forward_not_implemented():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[PyTorchSerializer.serialize_tensor(x)]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotImplemented

def test_forward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse

def test_empty_forward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict() # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException

def test_forward_success():
    axon.synapse = synapse
    x = torch.rand(3, 3, bittensor.__network_dim__)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[PyTorchSerializer.serialize_tensor(x)]
    )
    axon.synapse.call_forward = MagicMock(return_value=x)

    response = axon.Forward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32

def test_backward_not_serving():
    axon.synapse = None
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse

def test_empty_backward_request():
    axon.serve(synapse)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_single_item_backward_request():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[PyTorchSerializer.serialize_tensor(x)]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


def test_backward_deserialization_error():
    axon.serve(synapse)
    x = dict()
    y = dict() # Not tensors that can be deserialized.
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[x, y]
    )
    response = axon.Backward(request, None)
    assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException

def test_backward_success():
    axon.serve(synapse)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    request = bittensor_pb2.TensorMessage(
        version=bittensor.__version__,
        public_key=keypair.public_key,
        tensors=[PyTorchSerializer.serialize_tensor(x), PyTorchSerializer.serialize_tensor(x)]
    )
    axon.synapse.call_backward = MagicMock(return_value=x)
    response = axon.Backward(request, None)

    assert response.return_code == bittensor_pb2.ReturnCode.Success
    assert len(response.tensors) == 1
    assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
    assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
