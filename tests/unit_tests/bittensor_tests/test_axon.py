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

class TestAxon(unittest.TestCase):
    def setUp(self):
        self.config = Config.load()
        mnemonic = Keypair.generate_mnemonic()
        self.keypair = Keypair.create_from_mnemonic(mnemonic)
        self.axon = bittensor.axon.Axon(self.config, self.keypair)
        self.synapse = Synapse(self.config, None)

    def test_serve(self):
        assert self.axon.synapse == None
        for _ in range(0, 10):
            self.axon.serve(self.synapse)
        assert self.axon.synapse != None


    def test_forward_not_serving(self):
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
        )
        response = self.axon.Forward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse

    def test_empty_forward_request(self):
        self.axon.serve(self.synapse)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
        )
        response = self.axon.Forward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.EmptyRequest

    def test_forward_deserialization_error(self):
        self.axon.serve(self.synapse)
        x = dict()
        y = dict() # Not tensors that can be deserialized.
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[x, y]
        )
        response = self.axon.Forward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException

    def test_forward_success(self):
        self.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )
        self.axon.synapse.call_forward = MagicMock(return_value=x)

        response = self.axon.Forward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.Success
        assert len(response.tensors) == 1
        assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32

    def test_backward_not_serving(self):
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
        )
        response = self.axon.Backward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.NotServingSynapse

    def test_empty_backward_request(self):
        self.axon.serve(self.synapse)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
        )
        response = self.axon.Backward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


    def test_single_item_backward_request(self):
        self.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )
        response = self.axon.Backward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.InvalidRequest


    def test_backward_deserialization_error(self):
        self.axon.serve(self.synapse)
        x = dict()
        y = dict() # Not tensors that can be deserialized.
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[x, y]
        )
        response = self.axon.Backward(request, None)
        assert response.return_code == bittensor_pb2.ReturnCode.RequestDeserializationException

    def test_backward_success(self):
        self.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x), PyTorchSerializer.serialize_tensor(x)]
        )
        self.axon.synapse.call_backward = MagicMock(return_value=x)
        response = self.axon.Backward(request, None)

        assert response.return_code == bittensor_pb2.ReturnCode.Success
        assert len(response.tensors) == 1
        assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
