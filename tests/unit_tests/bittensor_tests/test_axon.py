from unittest.mock import MagicMock

from bittensor.config import Config
from substrateinterface import Keypair
from bittensor.synapse import Synapse
from bittensor.serializer import PyTorchSerializer, torch_dtype_to_bittensor_dtype, bittensor_dtype_to_torch_dtype
from bittensor import bittensor_pb2
import bittensor
import unittest
import random
import torch

class TestAxon(unittest.TestCase):
    def setUp(self):
        self.config = Config.load(neuron_path='bittensor/neurons/mnist')
        mnemonic = Keypair.generate_mnemonic()
        self.keypair = Keypair.create_from_mnemonic(mnemonic)
        self.session = bittensor.init(self.config, self.keypair)
        self.neuron = bittensor_pb2.Neuron(
            version = bittensor.__version__,
            public_key = self.keypair.public_key,
            address = '0.0.0.0',
            port = 12345,
        )
        self.synapse = Synapse(self.config, self.session)
    
    def test_serve(self):
        assert self.session.axon._synapse == None
        for _ in range(0, 10):
            self.session.axon.serve(self.synapse)
        assert self.session.axon._synapse != None

    def test_forward(self):
        request = bittensor_pb2.TensorMessage()

        # Check for null response by sending a request with no tensors in it
        response = self.session.axon.Forward(request, None)
        assert response == bittensor_pb2.TensorMessage(
                                version=bittensor.__version__,
                                public_key=self.keypair.public_key)

        # Let's add a synapse and call forward on the axon
        self.session.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )

        self.session.axon._synapse.call_forward = MagicMock(return_value=x)
        response = self.session.axon.Forward(request, None)
        assert response.tensors[0].shape == [3, 3, bittensor.__network_dim__]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
    
    def test_backward(self):
        # Let's add a synapse and call backward on the axon
        self.session.axon.serve(self.synapse)
        x = torch.rand(3, 3, bittensor.__network_dim__)
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x)]
        )

        self.session.axon._synapse.call_forward = MagicMock(return_value=x)
        response = self.session.axon.Backward(request, None)

        # We should get back a null response since the number of tensors does not match.
        assert response ==  bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=self.keypair.public_key)

        x2 = torch.rand(3, 3, bittensor.__network_dim__)
        
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x), PyTorchSerializer.serialize_tensor(x2)]
        )
        response = self.session.axon.Backward(request, None)
        assert len(response.tensors) == 1
        assert response.tensors[0].shape == [1,1]
        assert bittensor_dtype_to_torch_dtype(response.tensors[0].dtype) == torch.float32
  
