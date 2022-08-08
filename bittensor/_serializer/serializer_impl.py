""" An interface for serializing and deserializing bittensor tensors"""

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
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional


import bittensor

class Serializer(object):
    r""" Bittensor base serialization object for converting between bittensor.proto.Tensor and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    @staticmethod
    def empty():
        """Returns an empty bittensor.proto.Tensor message with the version"""
        torch_proto = bittensor.proto.Tensor(version= bittensor.__version_as_int__)
        return torch_proto

    def serialize (self, tensor_obj: object, modality: bittensor.proto.Modality= bittensor.proto.Modality.TEXT, from_type: int = bittensor.proto.TensorType.TORCH) -> bittensor.proto.Tensor:
        """Serializes a torch object to bittensor.proto.Tensor wire format.

        Args:
            tensor_obj (:obj:`object`, `required`): 
                general tensor object i.e. torch.Tensor or tensorflow.Tensor

            from_type (`obj`: bittensor.proto.TensorType, `Optional`): 
                Serialization from this type. i.e. bittensor.proto.TensorType.TORCH or bittensor.proto.TensorType.TENSORFLOW

        Returns:
            tensor_pb2: (obj: `bittensor.proto.Tensor`, `Optional`): 
                Serialized tensor as bittensor.proto.proto. 

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the passed type and a bittensor.proto.Tensor

            SerializationException: (Exception): 
                Raised when the subclass serialization throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if from_type == bittensor.proto.TensorType.TORCH:
            return self.serialize_from_torch( torch_tensor = tensor_obj, modality = modality)

        elif from_type == bittensor.proto.TensorType.NUMPY:
            return self.serialize_from_numpy( numpy_tensor = tensor_obj, modality = modality)

        elif from_type == bittensor.proto.TensorType.TENSORFLOW:
            return self.serialize_from_tensorflow( tensorflow_tensor = tensor_obj, modality = modality)

        else:
            raise bittensor.serializer.SerializationTypeNotImplementedException("Serialization from type {} not implemented.".format(from_type))

        raise NotImplementedError

    def deserialize (self, tensor_pb2: bittensor.proto.Tensor, to_type: int) -> object:
        """Serializes a torch object to bittensor.proto.Tensor wire format.

        Args:
            tensor_pb2 (`obj`: bittensor.proto.Tensor, `required`): 
                Serialized tensor as bittensor.proto.proto. 

            to_type (`obj`: bittensor.proto.TensorType, `required`): 
                Deserialization to this type. i.e. bittensor.proto.TensorType.TORCH or bittensor.proto.TensorType.TENSORFLOW

        Returns:
            tensor_obj (:obj:`torch.FloatTensor`, `required`): 
                tensor object of type from_type in bittensor.proto.TensorType

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the pb2 and the passed type.
          
            DeserializationException: (Exception): 
                Raised when the subclass deserializer throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if to_type == bittensor.proto.TensorType.TORCH:
            return self.deserialize_to_torch( tensor_pb2 )

        elif to_type == bittensor.proto.TensorType.NUMPY:
            return self.deserialize_to_numpy( tensor_pb2 )

        elif to_type == bittensor.proto.TensorType.TENSORFLOW:
            return self.deserialize_to_tensorflow( tensor_pb2 )

        else:
            raise bittensor.serializer.SerializationTypeNotImplementedException("Deserialization to type {} not implemented.".format(to_type))

    def serialize_from_tensorflow(self, tensorflow_tensor: torch.Tensor, modality: bittensor.proto.Modality) -> bittensor.proto.Tensor:
        """ tensorflow -> bittensor.proto.Tensor """
        raise bittensor.serializer.SerializationTypeNotImplementedException

    def serialize_from_torch(self, torch_tensor: torch.Tensor, modality: bittensor.proto.Modality) -> bittensor.proto.Tensor:
        """ torch -> bittensor.proto.Tensor """
        raise bittensor.serializer.SerializationTypeNotImplementedException
    
    def serialize_from_numpy(self, numpy_tensor: torch.Tensor, modality: bittensor.proto.Modality) -> bittensor.proto.Tensor:
        """ numpy -> bittensor.proto.Tensor """
        raise bittensor.serializer.SerializationTypeNotImplementedException

    def deserialize_to_torch(self, tensor_pb2: bittensor.proto.Tensor) -> torch.Tensor:
        """ bittensor.proto.Tensor -> torch """
        raise bittensor.serializer.SerializationTypeNotImplementedException

    def deserialize_to_tensorflow(self, tensor_pb2: bittensor.proto.Tensor) -> object:
        """ bittensor.proto.Tensor -> tensorflow """
        raise bittensor.serializer.SerializationTypeNotImplementedException

    def deserialize_to_numpy(self, tensor_pb2: bittensor.proto.Tensor) -> object:
        """ bittensor.proto.Tensor -> numpy """
        raise bittensor.serializer.SerializationTypeNotImplementedException


class MSGPackSerializer( Serializer ):
    """ Make conversion between torch and bittensor.proto.torch
    """
    def serialize_from_torch(self, torch_tensor: torch.Tensor, modality: bittensor.proto.Modality) -> bittensor.proto.Tensor:
        """ Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            torch_tensor (torch.Tensor): 
                Torch tensor to serialize.

            modality (bittensor.proto.Modality): 
                Datatype modality. i.e. TENSOR, TEXT, IMAGE

        Returns:
            bittensor.proto.Tensor: 
                The serialized torch tensor as bittensor.proto.proto. 
        """
        dtype = bittensor.serializer.torch_dtype_to_bittensor_dtype(torch_tensor.dtype)
        shape = list(torch_tensor.shape)
        torch_numpy = torch_tensor.cpu().detach().numpy().copy()
        data_buffer = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        torch_proto = bittensor.proto.Tensor (
                                    version = bittensor.__version_as_int__,
                                    buffer = data_buffer,
                                    shape = shape,
                                    dtype = dtype,
                                    serializer = bittensor.proto.Serializer.MSGPACK,
                                    tensor_type = bittensor.proto.TensorType.TORCH,
                                    modality = modality,
                                    requires_grad = torch_tensor.requires_grad
                                )
        return torch_proto

    def deserialize_to_torch(self, torch_proto: bittensor.proto.Tensor) -> torch.Tensor:
        """Deserializes an bittensor.proto.Tensor to a torch.Tensor object.

        Args:
            torch_proto (bittensor.proto.Tensor): 
                Proto containing torch tensor to derserialize.

        Returns:
            torch.Tensor: 
                Deserialized torch tensor.
        """
        dtype = bittensor.serializer.bittensor_dtype_to_torch_dtype(torch_proto.dtype)
        shape = tuple(torch_proto.shape)
        numpy_object = msgpack.unpackb(torch_proto.buffer, object_hook=msgpack_numpy.decode).copy()
        torch_object = torch.as_tensor(numpy_object).view(shape).requires_grad_(torch_proto.requires_grad)
        return torch_object.type(dtype)


class CMPPackSerializer( Serializer ):
    """ Make conversion between torch and bittensor.proto.torch in float16
    """
    def serialize_from_torch(self, torch_tensor: torch.Tensor, modality: bittensor.proto.Modality) -> bittensor.proto.Tensor:
        """ Serializes a torch.Tensor to an bittensor Tensor proto in float16

        Args:
            torch_tensor (torch.Tensor): 
                Torch tensor to serialize.

            modality (bittensor.proto.Modality): 
                Datatype modality. i.e. TENSOR, TEXT, IMAGE

        Returns:
            bittensor.proto.Tensor: 
                The serialized torch tensor as bittensor.proto.proto. 
        """
        dtype = bittensor.serializer.torch_dtype_to_bittensor_dtype(torch_tensor.dtype)
        shape = list(torch_tensor.shape)
        torch_numpy = torch_tensor.cpu().detach().half().numpy().copy()
        data_buffer = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        torch_proto = bittensor.proto.Tensor (
                                    version = bittensor.__version_as_int__,
                                    buffer = data_buffer,
                                    shape = shape,
                                    dtype = dtype,
                                    serializer = bittensor.proto.Serializer.CMPPACK,
                                    tensor_type = bittensor.proto.TensorType.TORCH,
                                    modality = modality,
                                    requires_grad = torch_tensor.requires_grad
                                )
        return torch_proto

    def deserialize_to_torch(self, torch_proto: bittensor.proto.Tensor) -> torch.Tensor:
        """Deserializes an bittensor.proto.Tensor to a torch.Tensor object.

        Args:
            torch_proto (bittensor.proto.Tensor): 
                Proto containing torch tensor to derserialize.

        Returns:
            torch.Tensor: 
                Deserialized torch tensor.
        """
        dtype = bittensor.serializer.bittensor_dtype_to_torch_dtype(torch_proto.dtype)
        shape = tuple(torch_proto.shape)
        numpy_object = msgpack.unpackb(torch_proto.buffer, object_hook=msgpack_numpy.decode).copy()
        torch_object = torch.as_tensor(numpy_object).view(shape).requires_grad_(torch_proto.requires_grad)
        return torch_object.type(dtype)

