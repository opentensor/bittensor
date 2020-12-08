
import grpc
import torchvision.transforms as transforms
import torch
import unittest
import bittensor
import pytest
import torchvision

from bittensor.config import Config
from bittensor.subtensor import Keypair
from bittensor.dendrite import RemoteNeuron, _RemoteModuleCall
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from unittest.mock import MagicMock
from bittensor.serializer import PyTorchSerializer

class TestDendrite(unittest.TestCase):

    def setUp(self):
        self.config = Config.load()
        mnemonic = Keypair.generate_mnemonic()
        self.keypair = Keypair.create_from_mnemonic(mnemonic)
        self.session = bittensor.init(self.config, self.keypair)
        self.neuron = bittensor_pb2.Neuron(
            version = bittensor.__version__,
            public_key = self.keypair.public_key,
            address = '0.0.0.0',
            port = 12345,
        )

    def test_dendrite_forward_tensor(self):
        x = torch.rand(3, 3, 3)
        output = self.session.dendrite.forward_tensor([self.neuron], [x])
        assert len(output) == 1
        assert output[0].shape == torch.Size([3, 3, bittensor.__network_dim__])

        # Let's try and break the forward_tensor call
        x = torch.rand(3)
        with pytest.raises(ValueError):
            self.session.dendrite.forward_tensor([self.neuron], [x])

    def test_dendrite_forward_image(self):
        # Let's grab some image data
        data = torchvision.datasets.MNIST(root = "data/datasets/", train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(data, batch_size = 10, shuffle=True, num_workers=2)
        # Let's grab a random image, and give it a crazy type to break the system

        image = None
        for _, (images, _) in enumerate(trainloader):
            image = images
            break
        
        # Add sequence dimension to the image.
        sequenced_image = image.unsqueeze(1)
        output = self.session.dendrite.forward_image([self.neuron], [sequenced_image])
        assert len(output) == 1
        assert output[0].shape == torch.Size([10, 1, bittensor.__network_dim__])

        # Let's try and break the forward_image call
        with pytest.raises(ValueError):
            self.session.dendrite.forward_image([self.neuron], [image])
    
    def test_dendrite_forward_text(self):
        words = ["This", "is", "a", "word", "list"]
        max_l = 0
        ts_list = []
        for w in words:
            ts_list.append(torch.ByteTensor(list(bytes(w, 'utf8'))))
            max_l = max(ts_list[-1].size()[0], max_l)

        word_tensor = torch.zeros((len(ts_list), max_l), dtype=torch.int64)
        for i, ts in enumerate(ts_list):
            word_tensor[i, 0:ts.size()[0]] = ts

        output = self.session.dendrite.forward_text([self.neuron], [word_tensor])
        assert len(output) == 1
        word_tensor_size = list(word_tensor.shape)
        assert output[0].shape == torch.Size([word_tensor_size[0], word_tensor_size[1], bittensor.__network_dim__])

        # Let's try and break the forward_text call
        word_tensor = word_tensor.unsqueeze(1)
        with pytest.raises(ValueError):
            self.session.dendrite.forward_text([self.neuron], [word_tensor])

class TestRemoteModuleCall(unittest.TestCase):

    def setUp(self):
        self.config = Config.load()
        mnemonic = Keypair.generate_mnemonic()
        self.keypair = Keypair.create_from_mnemonic(mnemonic)
        self.session = bittensor.init(self.config, self.keypair)
        self.neuron = bittensor_pb2.Neuron(
            version = bittensor.__version__,
            public_key = self.keypair.public_key,
            address = '0.0.0.0',
            port = 12345,
        )
        self.nounce = None
        self.signature = None
        self.remote_neuron = RemoteNeuron(self.neuron, self.config, self.keypair)
        self.channel = grpc.insecure_channel(
            'localhost',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])          
        self.stub = bittensor_grpc.BittensorStub(self.channel)
        # = RemoteSynapse(self.synapse, self._config)        
        self.dummy = torch.empty(0, requires_grad=True)
    
    def test_remote_module_forward(self):
        # Let's create some tensor
        x = torch.rand(3, 3, bittensor.__network_dim__)
        x.requires_grad = True
        # Mock remote module call
        fwd_return_value = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            public_key=self.keypair.public_key,
            tensors=[PyTorchSerializer.serialize_tensor(x)])

        # Some fake modality that doesn't exist
        modality = bittensor_pb2.DataType.UTF8

        # Since modality is invalid, this should get caught in the inner exception
        # of the remote module's fwd call and we will have outputs = torch.zeros
        output = _RemoteModuleCall.apply(self, self.dummy, x, modality)
        assert torch.all(output[0].data == torch.zeros((x.size(0), x.size(1), bittensor.__network_dim__)))

        # Now let's set up a modality that does exist
        modality = bittensor_pb2.Modality.TENSOR
        self.stub.Forward = MagicMock(return_value=fwd_return_value)
        bittensor.tbwriter = MagicMock(return_value=bittensor.session.tbwriter)
        output = _RemoteModuleCall.apply(self, self.dummy, x, modality)
        assert len(output) == x.size(0)
        assert torch.all(output == x)
