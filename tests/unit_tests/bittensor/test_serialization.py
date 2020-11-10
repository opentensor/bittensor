from bittensor.exceptions.Exceptions import DeserializationException, SerializationException
from bittensor.serializer import PyTorchSerializer, torch_dtype_to_bittensor_dtype, bittensor_dtype_to_torch_dtype
from random import randrange

from datasets import load_dataset
from bittensor.synapses.gpt2 import nextbatch

import bittensor.bittensor_pb2 as bittensor_pb2
import torchvision.transforms as transforms
import torch
import unittest
import bittensor
import torchvision
import pytest

class TestSerialization(unittest.TestCase):
    config = None

    def setUp(self):
        self.config = bittensor.Config()
        self.synapse_key = "test_synapse_key"
        self.neuron_key = "test_neuron_key"

    def test_serialize(self):
        for _ in range(10):
            tensor_a = torch.rand([12, 23])
            content = PyTorchSerializer.serialize(tensor_a, bittensor_pb2.Modality.TENSOR)
            tensor_b = PyTorchSerializer.deserialize(content)
            torch.all(torch.eq(tensor_a, tensor_b))
            
    def test_serialize_modality(self):
        # Let's grab some image data
        data = torchvision.datasets.MNIST(root = self.config.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        
        # Let's grab a random image, and try and de-serialize it incorrectly.
        image = data[randrange(len(data))][0]

        with pytest.raises(SerializationException):
            PyTorchSerializer.serialize(image, 11)
    
    def test_serialize_deserialize_image(self):
        # Let's grab some image data
        data = torchvision.datasets.MNIST(root = self.config.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        
        # Let's grab a random image, and give it a crazy type to break the system
        image = data[randrange(len(data))][0]

        serialized_image_tensor_message = PyTorchSerializer.serialize(image, bittensor_pb2.Modality.IMAGE)
        
        assert image.requires_grad == serialized_image_tensor_message.requires_grad
        assert list(image.shape) == serialized_image_tensor_message.shape
        assert serialized_image_tensor_message.modality == bittensor_pb2.Modality.IMAGE
        assert serialized_image_tensor_message.dtype != bittensor_pb2.DataType.UNKNOWN
        assert serialized_image_tensor_message.buffer == image.cpu().numpy().tobytes()

        deserialized_image_tensor_message = PyTorchSerializer.deserialize(serialized_image_tensor_message)
        assert serialized_image_tensor_message.requires_grad == deserialized_image_tensor_message.requires_grad
        assert serialized_image_tensor_message.shape == list(deserialized_image_tensor_message.shape)
        assert torch_dtype_to_bittensor_dtype(deserialized_image_tensor_message.dtype) != bittensor_pb2.DataType.UNKNOWN


    def test_serialize_deserialize_text(self):
        # Let's create some text data
        words = ["This", "is", "a", "word", "list"]
        max_l = 0
        ts_list = []
        for w in words:
            ts_list.append(torch.ByteTensor(list(bytes(w, 'utf8'))))
            max_l = max(ts_list[-1].size()[0], max_l)

        data = torch.zeros((len(ts_list), max_l), dtype=torch.int64)
        for i, ts in enumerate(ts_list):
            data[i, 0:ts.size()[0]] = ts
        
        serialized_data_tensor_message = PyTorchSerializer.serialize(data, bittensor_pb2.Modality.TEXT)
        assert data.requires_grad == serialized_data_tensor_message.requires_grad
        assert list(data.shape) == serialized_data_tensor_message.shape
        assert serialized_data_tensor_message.modality == bittensor_pb2.Modality.TEXT
        assert serialized_data_tensor_message.dtype != bittensor_pb2.DataType.UNKNOWN
        assert serialized_data_tensor_message.buffer == data.cpu().numpy().tobytes()

        deserialized_data_tensor_message = PyTorchSerializer.deserialize(serialized_data_tensor_message)
        assert serialized_data_tensor_message.requires_grad == deserialized_data_tensor_message.requires_grad
        assert serialized_data_tensor_message.shape == list(deserialized_data_tensor_message.shape)
        assert torch_dtype_to_bittensor_dtype(deserialized_data_tensor_message.dtype) != bittensor_pb2.DataType.UNKNOWN
    
    def test_serialize_deserialize_tensor(self):
        data = torch.rand([12, 23])

        serialized_tensor_message = PyTorchSerializer.serialize(data, bittensor_pb2.Modality.TENSOR)
        assert data.requires_grad == serialized_tensor_message.requires_grad
        assert list(data.shape) == serialized_tensor_message.shape
        assert serialized_tensor_message.modality == bittensor_pb2.Modality.TENSOR
        assert serialized_tensor_message.dtype == bittensor_pb2.DataType.FLOAT32
        assert serialized_tensor_message.buffer == data.cpu().numpy().tobytes()

        deserialized_tensor_message = PyTorchSerializer.deserialize(serialized_tensor_message)
        assert serialized_tensor_message.requires_grad == deserialized_tensor_message.requires_grad
        assert serialized_tensor_message.shape == list(deserialized_tensor_message.shape)
        assert torch_dtype_to_bittensor_dtype(deserialized_tensor_message.dtype) == bittensor_pb2.DataType.FLOAT32

    def test_deserialization_modality_failure(self):
        data = torch.rand([12, 23])
        tensor_message = PyTorchSerializer.serialize_tensor(data)
        tensor_message.modality = 11

        with pytest.raises(DeserializationException):
            PyTorchSerializer.deserialize(tensor_message)
    
    def test_bittensor_dtype_to_torch_dtype(self):
        with pytest.raises(SerializationException):
            bittensor_dtype_to_torch_dtype(11)