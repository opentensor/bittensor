""" An interface for serializing and deserializing opentensor tensors"""
from opentensor_proto import opentensor_pb2
from io import BytesIO
import torch
import pickle

class PyTorchSerializer(SerializerBase):
    @staticmethod
    def todef(tensor: torch.Tensor) -> opentensor_pb2.TensorDef:
        """ Returns a opentensor TensorDef proto for a torch.Tensor. 

        Args:
            tensor (torch.Tensor): Any torch.Tensor object.

        Returns:
            opentensor_pb2.TensorDef: An opentensor TensorDef for the passed torch.Tensor.
        """
        if tensor.dtype == torch.float32:
            dtype = opentensor_pb2.DataType.FLOAT32

        elif tensor.dtype == torch.float64:
            dtype = opentensor_pb2.DataType.FLOAT64

        elif tensor.dtype == torch.int32:
            dtype = opentensor_pb2.DataType.INT32

        elif tensor.dtype == torch.int64:
            dtype = opentensor_pb2.DataType.INT64

        else:
            dtype = opentensor_pb2.DataType.UNKNOWN

        # NOTE: we assume that the first dimension is the batch dimension.
        assert len(tensor.shape) > 1
        shape = tensor.shape[1:]
        return opentensor_pb2.TensorDef(
                        verison = opentensor.PROTOCOL_VERSION, 
                        shape = shape, 
                        dtype = dtype,
                        requires_grad = tensor.requires_grad)

    @staticmethod
    def serialize(tensor: torch.Tensor) -> opentensor_pb2.Tensor:
        """Serializes a torch.Tensor to an opentensor Tensor proto.

        Args:
            tensor (torch.Tensor): torch.Tensor to serialize.

        Returns:
            opentensor_pb2.Tensor: Serialized tensor as opentensor_pb2.proto. 
        """
        # Using numpy intermediary because deserializing with pickle can run arbitray code on your machine.
        data_buffer = tensor.numpy().tobytes()
        tensor_def = PyTorchSerializer.todef(tensor)
        proto = opentensor_pb2.Tensor(
                    version = opentensor.PROTOCOL_VERSION,
                    buffer = data_buffer,
                    tensor_def = tensor_def)            
        return proto

    @staticmethod
    def deserialize(proto: opentensor_pb2.Tensor) -> torch.Tensor:
        """Deserializes an opentensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (opentensor_pb2.Tensor): Proto to derserialize.

        Returns:
            torch.Tensor: torch.Tensor to deserialize.
        """
        # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
        array = np.frombuffer(proto.buffer, dtype=np.dtype(proto.tensor_def.dtype)).copy()
        # NOTE (const): The first dimension is always assumed to be the batch dimension.
        shape = tuple([-1] + proto.tensor_def.shape)
        assert len(shape) > 1 
        tensor = torch.as_tensor(array).view(shape).requires_grad_(proto.tensor_def.requires_grad)
        return tensor

class SerializerBase:
    @staticmethod
    def todef(tensor: object) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef description for this Tensor.

        Args:
            obj (object): Tensor object: i.e. torch.Tensor type.

        Raises:
            NotImplementedError: Must be implemented in a subclass of this object.

        Returns:
            opentensor_pb2.TensorDef: The TensorDef proto describing this Tensor.
        """
        raise NotImplementedError()

    @staticmethod
    def serialize(tensor: object) -> opentensor_pb2.Tensor:
        """ Returns a serialized version of generic tensor obj as an opentensor_pb2.Tensor proto.  

        Args:
            tensor (object): Tensor object: i.e. torch.Tensor.

        Raises:
            NotImplementedError: Must be implemented in the subclass of this object.

        Returns:
            opentensor_pb2.Tensor: The proto version of this object.
        """
        raise NotImplementedError()

    @staticmethod
    def deserialize(proto: opentensor_pb2.Tensor) -> object:
        """ Returns the a generic tensor object from an opentensor_pb2.Tensor proto.

        Args:
            proto (opentensor_pb2.Tensor): The proto to deserialize.

        Raises:
            NotImplementedError: Must be implemented in the subclass of this object.

        Returns:
            object: Generic tensor object.
        """
        raise NotImplementedError()