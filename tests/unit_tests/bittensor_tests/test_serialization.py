
import torchvision.transforms as transforms
import torch
import unittest
import torchvision
import pytest
import bittensor
import bittensor.serialization as serialization

from random import randrange

class TestSerialization(unittest.TestCase):
    config = None

    def setUp(self):        
        config = bittensor.wallet.Wallet.default_config()

    def test_serialize(self):
        for _ in range(10):
            tensor_a = torch.rand([12, 23])
            serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
            content = serializer.serialize(tensor_a, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            tensor_b = serializer.deserialize(content, to_type = bittensor.proto.TensorType.TORCH)
            torch.all(torch.eq(tensor_a, tensor_b))
            
    def test_serialize_object_type_exception(self):
        # Let's grab a random image, and try and de-serialize it incorrectly.
        image = torch.ones( [1, 28, 28] )

        serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        with pytest.raises(serialization.SerializationTypeNotImplementedException):
            serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = 11)

    def test_deserialization_object_type_exception(self):
        data = torch.rand([12, 23])
        
        serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)

        with pytest.raises(serialization.SerializationTypeNotImplementedException):
            serializer.deserialize(tensor_message, to_type = 11)
    
    def test_serialize_deserialize_image(self):
        # Let's grab some image data
        # Let's grab a random image, and give it a crazy type to break the system
        image = torch.ones( [1, 28, 28] )

        serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_image_tensor_message = serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
        
        assert image.requires_grad == serialized_image_tensor_message.requires_grad
        assert list(image.shape) == serialized_image_tensor_message.shape
        assert serialized_image_tensor_message.modality == bittensor.proto.Modality.IMAGE
        assert serialized_image_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN

        deserialized_image_tensor_message = serializer.deserialize(serialized_image_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_image_tensor_message.requires_grad == deserialized_image_tensor_message.requires_grad
        assert serialized_image_tensor_message.shape == list(deserialized_image_tensor_message.shape)
        assert serialization.torch_dtype_to_bittensor_dtype(deserialized_image_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

        assert torch.all(torch.eq(deserialized_image_tensor_message, image))


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

        serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_data_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_data_tensor_message.requires_grad
        assert list(data.shape) == serialized_data_tensor_message.shape
        assert serialized_data_tensor_message.modality == bittensor.proto.Modality.TEXT
        assert serialized_data_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN

        deserialized_data_tensor_message = serializer.deserialize(serialized_data_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_data_tensor_message.requires_grad == deserialized_data_tensor_message.requires_grad
        assert serialized_data_tensor_message.shape == list(deserialized_data_tensor_message.shape)
        assert serialization.torch_dtype_to_bittensor_dtype(deserialized_data_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

        assert torch.all(torch.eq(deserialized_data_tensor_message, data))

    
    def test_serialize_deserialize_tensor(self):
        data = torch.rand([12, 23])

        serializer = serialization.get_serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_tensor_message.requires_grad
        assert list(data.shape) == serialized_tensor_message.shape
        assert serialized_tensor_message.modality == bittensor.proto.Modality.TENSOR
        assert serialized_tensor_message.dtype == bittensor.proto.DataType.FLOAT32

        deserialized_tensor_message = serializer.deserialize(serialized_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_tensor_message.requires_grad == deserialized_tensor_message.requires_grad
        assert serialized_tensor_message.shape == list(deserialized_tensor_message.shape)
        assert serialization.torch_dtype_to_bittensor_dtype(deserialized_tensor_message.dtype) == bittensor.proto.DataType.FLOAT32

        assert torch.all(torch.eq(deserialized_tensor_message, data))

    
    def test_bittensor_dtype_to_torch_dtype(self):
        with pytest.raises(serialization.DeserializationException):
            serialization.bittensor_dtype_to_torch_dtype(11)