# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import torch
import unittest
import pytest
import bittensor

class TestSerialization(unittest.TestCase):

    def test_serialize(self):
        for _ in range(10):
            tensor_a = torch.rand([12, 23])
            serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
            content = serializer.serialize(tensor_a, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            tensor_b = serializer.deserialize(content, to_type = bittensor.proto.TensorType.TORCH)
            torch.all(torch.eq(tensor_a, tensor_b))
            
    def test_serialize_object_type_exception(self):
        # Let's grab a random image, and try and de-serialize it incorrectly.
        image = torch.ones( [1, 28, 28] )

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        with pytest.raises(bittensor.serializer.SerializationTypeNotImplementedException):
            serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = 11)

    def test_deserialization_object_type_exception(self):
        data = torch.rand([12, 23])
        
        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)

        with pytest.raises(bittensor.serializer.SerializationTypeNotImplementedException):
            serializer.deserialize(tensor_message, to_type = 11)
    
    def test_serialize_deserialize_image(self):
        # Let's grab some image data
        # Let's grab a random image, and give it a crazy type to break the system
        image = torch.ones( [1, 28, 28] )

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_image_tensor_message = serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
        
        assert image.requires_grad == serialized_image_tensor_message.requires_grad
        assert list(image.shape) == serialized_image_tensor_message.shape
        assert serialized_image_tensor_message.modality == bittensor.proto.Modality.IMAGE
        assert serialized_image_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN

        deserialized_image_tensor_message = serializer.deserialize(serialized_image_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_image_tensor_message.requires_grad == deserialized_image_tensor_message.requires_grad
        assert serialized_image_tensor_message.shape == list(deserialized_image_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_image_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

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

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_data_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_data_tensor_message.requires_grad
        assert list(data.shape) == serialized_data_tensor_message.shape
        assert serialized_data_tensor_message.modality == bittensor.proto.Modality.TEXT
        assert serialized_data_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN

        deserialized_data_tensor_message = serializer.deserialize(serialized_data_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_data_tensor_message.requires_grad == deserialized_data_tensor_message.requires_grad
        assert serialized_data_tensor_message.shape == list(deserialized_data_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_data_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

        assert torch.all(torch.eq(deserialized_data_tensor_message, data))

    
    def test_serialize_deserialize_tensor(self):
        data = torch.rand([12, 23])

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        serialized_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_tensor_message.requires_grad
        assert list(data.shape) == serialized_tensor_message.shape
        assert serialized_tensor_message.modality == bittensor.proto.Modality.TENSOR
        assert serialized_tensor_message.dtype == bittensor.proto.DataType.FLOAT32

        deserialized_tensor_message = serializer.deserialize(serialized_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_tensor_message.requires_grad == deserialized_tensor_message.requires_grad
        assert serialized_tensor_message.shape == list(deserialized_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_tensor_message.dtype) == bittensor.proto.DataType.FLOAT32

        assert torch.all(torch.eq(deserialized_tensor_message, data))

    
    def test_bittensor_dtype_to_torch_dtype(self):
        with pytest.raises(bittensor.serializer.DeserializationException):
            bittensor.serializer.bittensor_dtype_to_torch_dtype(11)


class TestCMPSerialization(unittest.TestCase):

    def test_serialize(self):
        for _ in range(10):
            tensor_a = torch.rand([12, 23])
            serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
            content = serializer.serialize(tensor_a, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            tensor_b = serializer.deserialize(content, to_type = bittensor.proto.TensorType.TORCH)
            torch.all(torch.eq(tensor_a, tensor_b))
            
    def test_serialize_object_type_exception(self):
        # Let's grab a random image, and try and de-serialize it incorrectly.
        image = torch.ones( [1, 28, 28] )

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
        with pytest.raises(bittensor.serializer.SerializationTypeNotImplementedException):
            serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = 11)

    def test_deserialization_object_type_exception(self):
        data = torch.rand([12, 23])
        
        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
        tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)

        with pytest.raises(bittensor.serializer.SerializationTypeNotImplementedException):
            serializer.deserialize(tensor_message, to_type = 11)
    
    def test_serialize_deserialize_image(self):
        # Let's grab some image data
        # Let's grab a random image, and give it a crazy type to break the system
        image = torch.ones( [1, 28, 28] )
        data_size = image.element_size()*image.nelement()

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
        serialized_image_tensor_message = serializer.serialize(image, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
        
        assert image.requires_grad == serialized_image_tensor_message.requires_grad
        assert list(image.shape) == serialized_image_tensor_message.shape
        assert serialized_image_tensor_message.modality == bittensor.proto.Modality.IMAGE
        assert serialized_image_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN
        assert serialized_image_tensor_message.ByteSize() < data_size

        deserialized_image_tensor_message = serializer.deserialize(serialized_image_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_image_tensor_message.requires_grad == deserialized_image_tensor_message.requires_grad
        assert serialized_image_tensor_message.shape == list(deserialized_image_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_image_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

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
        data_size = data.element_size()*data.nelement()

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
        serialized_data_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_data_tensor_message.requires_grad
        assert list(data.shape) == serialized_data_tensor_message.shape
        assert serialized_data_tensor_message.modality == bittensor.proto.Modality.TEXT
        assert serialized_data_tensor_message.dtype != bittensor.proto.DataType.UNKNOWN
        assert serialized_data_tensor_message.ByteSize() < data_size

        deserialized_data_tensor_message = serializer.deserialize(serialized_data_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_data_tensor_message.requires_grad == deserialized_data_tensor_message.requires_grad
        assert serialized_data_tensor_message.shape == list(deserialized_data_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_data_tensor_message.dtype) != bittensor.proto.DataType.UNKNOWN

        assert torch.all(torch.eq(deserialized_data_tensor_message, data))

    
    def test_serialize_deserialize_tensor(self):
        data = torch.rand([12, 23])
        data_size = data.element_size()*data.nelement()

        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.CMPPACK )
        serialized_tensor_message = serializer.serialize(data, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
       
        assert data.requires_grad == serialized_tensor_message.requires_grad
        assert list(data.shape) == serialized_tensor_message.shape
        assert serialized_tensor_message.modality == bittensor.proto.Modality.TENSOR
        assert serialized_tensor_message.dtype == bittensor.proto.DataType.FLOAT32
        assert serialized_tensor_message.ByteSize() < data_size


        deserialized_tensor_message = serializer.deserialize(serialized_tensor_message, to_type = bittensor.proto.TensorType.TORCH)
        assert serialized_tensor_message.requires_grad == deserialized_tensor_message.requires_grad
        assert serialized_tensor_message.shape == list(deserialized_tensor_message.shape)
        assert bittensor.serializer.torch_dtype_to_bittensor_dtype(deserialized_tensor_message.dtype) == bittensor.proto.DataType.FLOAT32

        assert torch.all(torch.eq(deserialized_tensor_message, data.to(torch.float16)))

    
    def test_bittensor_dtype_to_torch_dtype(self):
        with pytest.raises(bittensor.serializer.DeserializationException):
            bittensor.serializer.bittensor_dtype_to_torch_dtype(11)