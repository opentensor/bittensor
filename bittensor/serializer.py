""" An interface for serializing and deserializing bittensor tensors"""
from bittensor import bittensor_pb2
import numpy as np
import torch

import bittensor

class SerializerBase:
    @staticmethod
    def todef(tensor: object) -> bittensor_pb2.TensorDef:
        """ Returns the bittensor_pb2.TensorDef description for this Tensor.

        Args:
            obj (object): Tensor object: i.e. torch.Tensor type.

        Raises:
            NotImplementedError: Must be implemented in a subclass of this object.

        Returns:
            bittensor_pb2.TensorDef: The TensorDef proto describing this Tensor.
        """
        raise NotImplementedError()

    @staticmethod
    def serialize(tensor: object) -> bittensor_pb2.Tensor:
        """ Returns a serialized version of generic tensor obj as an bittensor_pb2.Tensor proto.  

        Args:
            tensor (object): Tensor object: i.e. torch.Tensor.

        Raises:
            NotImplementedError: Must be implemented in the subclass of this object.

        Returns:
            bittensor_pb2.Tensor: The proto version of this object.
        """
        raise NotImplementedError()

    @staticmethod
    def deserialize(proto: bittensor_pb2.Tensor) -> object:
        """ Returns the a generic tensor object from an bittensor_pb2.Tensor proto.

        Args:
            proto (bittensor_pb2.Tensor): The proto to deserialize.

        Raises:
            NotImplementedError: Must be implemented in the subclass of this object.

        Returns:
            object: Generic tensor object.
        """
        raise NotImplementedError()
    
    
def torch_dtype_to_bittensor_dtype(tdtype):
    if tdtype == torch.float32:
        dtype = bittensor_pb2.DataType.FLOAT32
    elif tdtype == torch.float64:
        dtype = bittensor_pb2.DataType.FLOAT64
    elif tdtype == torch.int32:
        dtype = bittensor_pb2.DataType.INT32
    elif tdtype == torch.int64:
        dtype = bittensor_pb2.DataType.INT64
    else:
        dtype = bittensor_pb2.DataType.UNKNOWN
    return dtype


def bittensor_dtype_to_torch_dtype(odtype):
    if odtype == bittensor_pb2.DataType.FLOAT32:
        dtype = torch.float32
    elif odtype == bittensor_pb2.DataType.FLOAT64:
        dtype = torch.float64
    elif odtype == bittensor_pb2.DataType.INT32:
        dtype = torch.int32
    elif odtype == bittensor_pb2.DataType.INT64: 
        dtype = torch.int64
    else:
        # TODO (const): raise error
        dtype = torch.float32
    return dtype

def bittensor_dtype_np_dtype(odtype):
    if odtype == bittensor_pb2.DataType.FLOAT32:
        dtype = np.float32
    elif odtype == bittensor_pb2.DataType.FLOAT64:
        dtype = np.float64
    elif odtype == bittensor_pb2.DataType.INT32:
        dtype = np.int32
    elif odtype == bittensor_pb2.DataType.INT64: 
        dtype = np.int64
    else:
        # TODO(const): raise error.
        dtype = np.float32
    return dtype

class PyTorchSerializer(SerializerBase):
    @staticmethod
    def todef(tensor: torch.Tensor) -> bittensor_pb2.TensorDef:
        """ Returns a bittensor TensorDef proto for a torch.Tensor. 

        Args:
            tensor (torch.Tensor): Any torch.Tensor object.

        Returns:
            bittensor_pb2.TensorDef: An bittensor TensorDef for the passed torch.Tensor.
        """
        dtype = torch_dtype_to_bittensor_dtype(tensor.dtype)
        # NOTE: The first dimension is the batch dimensiondatetime A combination of a date and a time. Attributes: ()
        assert len(tensor.shape) > 1
        shape = list(tensor.shape)
        return bittensor_pb2.TensorDef(
                        version = bittensor.PROTOCOL_VERSION, 
                        shape = shape, 
                        dtype = dtype,
                        requires_grad = tensor.requires_grad)

    @staticmethod
    def serialize(tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        """Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            tensor (torch.Tensor): torch.Tensor to serialize.

        Returns:
            bittensor_pb2.Tensor: Serialized tensor as bittensor_pb2.proto. 
        """
        # Using numpy intermediary because deserializing with pickle can run arbitray code on your machine.
        data_buffer = tensor.numpy().tobytes()
        tensor_def = PyTorchSerializer.todef(tensor)
        proto = bittensor_pb2.Tensor(
                    version = bittensor.PROTOCOL_VERSION,
                    buffer = data_buffer,
                    tensor_def = tensor_def)      
        return proto

    @staticmethod
    def deserialize(proto: bittensor_pb2.Tensor) -> torch.Tensor:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (bittensor_pb2.Tensor): Proto to derserialize.

        Returns:
            torch.Tensor: torch.Tensor to deserialize.
        """
        dtype = bittensor_dtype_np_dtype(proto.tensor_def.dtype)
        # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
        array = np.frombuffer(proto.buffer, dtype=np.dtype(dtype)).copy()
        # NOTE (const): The first dimension is always assumed to be the batch dimension.
        shape = tuple(proto.tensor_def.shape)
        assert len(shape) > 1 
        tensor = torch.as_tensor(array).view(shape).requires_grad_(proto.tensor_def.requires_grad)
        return tensor