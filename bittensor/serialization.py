""" An interface for serializing and deserializing bittensor tensors"""
import numpy as np
import torch
import msgpack
import msgpack_numpy

from loguru import logger
from typing import Dict, List, Any

import bittensor
import bittensor.utils.serialization_utils as serialization_utils

class SerializationException (Exception):
    """ Raised during serialization """
    pass

class DeserializationException (Exception):
    """ Raised during deserialization """
    pass

class NoSerializerForEnum (Exception):
    """ Raised if there is no serializer for the passed type """
    pass

class SerializationTypeNotImplementedException (Exception):
    """ Raised if serialization/deserialization is not implemented for the passed object type """
    pass

class BittensorSerializerBase(object):
    r""" Bittensor base serialization object for converting between bittensor.pb2.Tensor and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    def serialize (self, tensor_obj: object, modality: bittensor.pb2.Modality, from_type: int) -> bittensor.pb2.Tensor:
        """Serializes a torch object to bittensor.pb2.Tensor wire format.

        Args:
            tensor_obj (:obj:`object`, `required`): 
                general tensor object i.e. torch.Tensor or tensorflow.Tensor

            from_type (`obj`: bittensor.pb2.TensorType, `required`): 
                Serialization from this type. i.e. bittensor.pb2.TensorType.TORCH or bittensor.pb2.TensorType.TENSORFLOW

        Returns:
            tensor_pb2: (obj: `bittensor.pb2.Tensor`, `required`): 
                Serialized tensor as bittensor.pb2.proto. 

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the passed type and a bittensor.pb2.Tensor

            SerializationException: (Exception): 
                Raised when the subclass serialization throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if from_type == bittensor.pb2.TensorType.TORCH:
            return self.serialize_from_torch( torch_tensor = tensor_obj, modality = modality)

        elif from_type == bittensor.pb2.TensorType.NUMPY:
            return self.serialize_from_numpy( numpy_tensor = tensor_obj, modality = modality)

        elif from_type == bittensor.pb2.TensorType.TENSORFLOW:
            return self.serialize_from_tensorflow( tensorflow_tensor = tensor_obj, modality = modality)

        else:
            raise SerializationTypeNotImplementedException("Serialization from type {} not implemented.".format(from_type))

        raise NotImplementedError

    def deserialize (self, tensor_pb2: bittensor.pb2.Tensor, to_type: int) -> object:
        """Serializes a torch object to bittensor.pb2.Tensor wire format.

        Args:
            tensor_pb2 (`obj`: bittensor.pb2.Tensor, `required`): 
                Serialized tensor as bittensor.pb2.proto. 

            to_type (`obj`: bittensor.pb2.TensorType, `required`): 
                Deserialization to this type. i.e. bittensor.pb2.TensorType.TORCH or bittensor.pb2.TensorType.TENSORFLOW

        Returns:
            tensor_obj (:obj:`torch.FloatTensor`, `required`): 
                tensor object of type from_type in bittensor.pb2.TensorType

        Raises:
            SerializationTypeNotImplementedException (Exception):
                Raised if the serializer does not implement the conversion between the pb2 and the passed type.
          
            DeserializationException: (Exception): 
                Raised when the subclass deserializer throws an error for the passed object.
        """
        # TODO (const): add deserialization types for torch -> tensorflow 
        if to_type == bittensor.pb2.TensorType.TORCH:
            return self.deserialize_to_torch( tensor_pb2 )

        elif to_type == bittensor.pb2.TensorType.NUMPY:
            return self.deserialize_to_numpy( tensor_pb2 )

        elif to_type == bittensor.pb2.TensorType.TENSORFLOW:
            return self.deserialize_to_tensorflow( tensor_pb2 )

        else:
            raise SerializationTypeNotImplementedException("Deserialization to type {} not implemented.".format(to_type))

    def serialize_from_tensorflow(self, tensorflow_tensor: torch.Tensor, modality: bittensor.pb2.Modality) -> bittensor.pb2.Tensor:
        raise SerializationTypeNotImplementedException

    def serialize_from_torch(self, torch_tensor: torch.Tensor, modality: bittensor.pb2.Modality) -> bittensor.pb2.Tensor:
        raise SerializationTypeNotImplementedException
    
    def serialize_from_numpy(self, numpy_tensor: torch.Tensor, modality: bittensor.pb2.Modality) -> bittensor.pb2.Tensor:
        raise SerializationTypeNotImplementedException

    def deserialize_to_torch(self, tensor_pb2: bittensor.pb2.Tensor) -> torch.Tensor:
        raise SerializationTypeNotImplementedException

    def deserialize_to_tensorflow(self, tensor_pb2: bittensor.pb2.Tensor) -> object:
        raise SerializationTypeNotImplementedException

    def deserialize_to_numpy(self, tensor_pb2: bittensor.pb2.Tensor) -> object:
        raise SerializationTypeNotImplementedException

def get_serializer ( serialzer_type: bittensor.pb2.Serializer ) -> BittensorSerializerBase:
    r"""Returns the correct serializer object for the passed Serializer enum. 

        Args:
            serialzer_type (:obj:`bittensor.pb2.Serializer`, `required`): 
                The serialzer_type ENUM from bittensor.pb2.

        Returns:
            BittensorSerializerBase: (obj: `BittensorSerializerBase`, `required`): 
                The bittensor serializer/deserialzer for the passed type.

        Raises:
            NoSerializerForEnum: (Exception): 
                Raised if the passed there is no serialzier for the passed type. 
    """

    # WARNING: the pickle serializer is not safe. Should be removed in future verions.
    # if serialzer_type == bittensor.pb2.Serializer.PICKLE:
    #     return PyTorchPickleSerializer()
    if serialzer_type == bittensor.pb2.Serializer.MSGPACK:
        return MSGPackSerializer()
    else:
        raise NoSerializerForEnum("No known serialzier for proto type {}".format(serialzer_type))


class MSGPackSerializer( BittensorSerializerBase ):

    def serialize_from_torch(self, torch_tensor: torch.Tensor, modality: bittensor.pb2.Modality) -> bittensor.pb2.Tensor:
        """ Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            torch_tensor (torch.Tensor): 
                Torch tensor to serialize.

            modality (bittensor.pb2.Modality): 
                Datatype modality. i.e. TENSOR, TEXT, IMAGE

        Returns:
            bittensor.pb2.Tensor: 
                The serialized torch tensor as bittensor.pb2.proto. 
        """
        dtype = serialization_utils.torch_dtype_to_bittensor_dtype(torch_tensor.dtype)
        shape = list(torch_tensor.shape)
        torch_numpy = torch_tensor.cpu().numpy().copy()
        data_buffer = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        torch_proto = bittensor.pb2.Tensor(version = bittensor.__version__,
                                    buffer = data_buffer,
                                    shape = shape,
                                    dtype = dtype,
                                    serializer = bittensor.pb2.Serializer.MSGPACK,
                                    tensor_type = bittensor.pb2.TensorType.TORCH,
                                    modality = modality,
                                    requires_grad = torch_tensor.requires_grad)
        return torch_proto

    def deserialize_to_torch(self, torch_proto: bittensor.pb2.Tensor) -> torch.Tensor:
        """Deserializes an bittensor.pb2.Tensor to a torch.Tensor object.

        Args:
            torch_proto (bittensor.pb2.Tensor): 
                Proto containing torch tensor to derserialize.

        Returns:
            torch.Tensor: 
                Deserialized torch tensor.
        """
        dtype = serialization_utils.bittensor_dtype_to_torch_dtype(torch_proto.dtype)
        shape = tuple(torch_proto.shape)
        numpy_object = msgpack.unpackb(torch_proto.buffer, object_hook=msgpack_numpy.decode).copy()
        torch_object = torch.as_tensor(numpy_object).view(shape).requires_grad_(torch_proto.requires_grad)
        return torch_object.type(dtype)


