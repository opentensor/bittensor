
from bittensor.serializer import PyTorchSerializer
import grpc
import torchvision.transforms as transforms
import torch
import unittest
import bittensor
import random
import pytest
import torchvision

from bittensor.dendrite import RemoteSynapse, _RemoteModuleCall
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from unittest.mock import MagicMock

def mock_synapse():
    private_key = bittensor.crypto.Crypto.generate_private_ed25519()
    public_key = bittensor.crypto.Crypto.public_key_from_private(private_key)
    synapse = bittensor_pb2.Synapse(
        version = bittensor.__version__,
        neuron_key = bittensor.crypto.Crypto.public_key_to_string(public_key),
        synapse_key = bittensor.crypto.Crypto.public_key_to_string(public_key),
        address = '0.0.0.0',
        port = 12345,
        block_hash = None
    )
    return synapse

class TestDendrite(unittest.TestCase):

    def setUp(self):
        axon_port = random.randint(8000, 9000)
        metagraph_port = random.randint(8000, 9000)
        bp_host = 'localhost'
        bp_port = metagraph_port
        self._config = bittensor.Config(     axon_port = axon_port,
                                        metagraph_port = metagraph_port,
                                        bp_host = bp_host,
                                        bp_port = bp_port
                                    )
        self._config.log()
        self.dendrite = bittensor.Dendrite(self._config)
        self.mocked_synapse = mock_synapse()

    def test_dendrite_forward_tensor(self):
        x = torch.rand(3, 3, 3)
        output = self.dendrite.forward_tensor([self.mocked_synapse], [x])
        assert len(output) == 1
        assert output[0].shape == torch.Size([3, 3, bittensor.__network_dim__])

        # Let's try and break the forward_tensor call
        x = torch.rand(3)
        with pytest.raises(ValueError):
            self.dendrite.forward_tensor([self.mocked_synapse], [x])

    def test_dendrite_forward_image(self):
        # Let's grab some image data
        batch_size = 64
        data = torchvision.datasets.MNIST(root = self._config.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True, num_workers=2)
        # Let's grab a random image, and give it a crazy type to break the system

        image = None
        for _, (images, _) in enumerate(trainloader):
            image = images
            break
        
        # Add sequence dimension to the image.
        sequenced_image = image.unsqueeze(1)
        output = self.dendrite.forward_image([self.mocked_synapse], [sequenced_image])
        assert len(output) == 1
        assert output[0].shape == torch.Size([batch_size, 1, bittensor.__network_dim__])

        # Let's try and break the forward_image call
        with pytest.raises(ValueError):
           self.dendrite.forward_image([self.mocked_synapse], [image])
    
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

        output = self.dendrite.forward_text([self.mocked_synapse], [word_tensor])
        assert len(output) == 1
        word_tensor_size = list(word_tensor.shape)
        assert output[0].shape == torch.Size([word_tensor_size[0], word_tensor_size[1], bittensor.__network_dim__])

        # Let's try and break the forward_text call
        word_tensor = word_tensor.unsqueeze(1)
        with pytest.raises(ValueError):
            self.dendrite.forward_text([self.mocked_synapse], [word_tensor])

class TestRemoteModuleCall(unittest.TestCase):

    def setUp(self):
        axon_port = random.randint(8000, 9000)
        metagraph_port = random.randint(8000, 9000)
        bp_host = 'localhost'
        bp_port = metagraph_port
        self._config = bittensor.Config(     axon_port = axon_port,
                                        metagraph_port = metagraph_port,
                                        bp_host = bp_host,
                                        bp_port = bp_port
                                    )
        self._config.log()
        self.dendrite = bittensor.Dendrite(self._config)
        self.synapse = mock_synapse()
        self.remote_synapse = RemoteSynapse(self.synapse, self._config)
        self.dummy = torch.empty(0, requires_grad=True)
        self.local_neuron_key = self.synapse.neuron_key
        self.nounce = None
        self.signature = None
        self.channel = grpc.insecure_channel(
            'localhost',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor_grpc.BittensorStub(self.channel)
        self.remote_synapse = RemoteSynapse(self.synapse, self._config)        
        self.tblogger = bittensor.TBLogger("./tests/tmp")

    def test_remote_module_forward(self):
        # Let's create some tensor
        x = torch.rand(3, 3, bittensor.__network_dim__)
        x.requires_grad = True
        # Mock remote module call
        fwd_return_value = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            neuron_key=self._config.neuron_key,
            synapse_key=self._config.synapse_key,
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
        bittensor.tbwriter = MagicMock(return_value=self.tblogger)
        output = _RemoteModuleCall.apply(self, self.dummy, x, modality)
        assert len(output) == x.size(0)
        assert torch.all(output == x)