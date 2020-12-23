""" An interface for serializing and deserializing bittensor tensors"""
import numpy as np
import torch

import bittensor
from bittensor.serialization import BittensorSerializerBase
from bittensor import bittensor_pb2

def torch_dtype_to_bittensor_dtype(tdtype):
    """ Translates between torch.dtypes and bittensor.dtypes.

        Args:
            tdtype (torch.dtype): torch.dtype to translate.

        Returns:
            dtype: (bittensor.dtype): translated bittensor.dtype.
    """
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


def bittensor_dtype_to_torch_dtype(bdtype):
    """ Translates between bittensor.dtype and torch.dtypes.

        Args:
            bdtype (bittensor.dtype): bittensor.dtype to translate.

        Returns:
            dtype: (torch.dtype): translated torch.dtype.
    """
    if bdtype == bittensor_pb2.DataType.FLOAT32:
        dtype = torch.float32
    elif bdtype == bittensor_pb2.DataType.FLOAT64:
        dtype = torch.float64
    elif bdtype == bittensor_pb2.DataType.INT32:
        dtype = torch.int32
    elif bdtype == bittensor_pb2.DataType.INT64:
        dtype = torch.int64
    else:
        raise bittensor.serialization.DeserializationException(
            'Unknown bittensor.Dtype or no equivalent torch.dtype for bittensor.dtype = {}'
            .format(bdtype))
    return dtype


def bittensor_dtype_np_dtype(bdtype):
    """ Translates between bittensor.dtype and np.dtypes.

        Args:
            bdtype (bittensor.dtype): bittensor.dtype to translate.

        Returns:
            dtype: (numpy.dtype): translated np.dtype.
    """
    if bdtype == bittensor_pb2.DataType.FLOAT32:
        dtype = np.float32
    elif bdtype == bittensor_pb2.DataType.FLOAT64:
        dtype = np.float64
    elif bdtype == bittensor_pb2.DataType.INT32:
        dtype = np.int32
    elif bdtype == bittensor_pb2.DataType.INT64:
        dtype = np.int64
    else:
        raise bittensor.serialization.SerializationException(
            'Unknown bittensor.dtype or no equivalent numpy.dtype for bittensor.dtype = {}'
            .format(bdtype))
    return dtype


class PyTorchPickleSerializer( BittensorSerializerBase ):

    ENUM = bittensor_pb2.Serializer.TORCHPICKLE

    @staticmethod
    def serialize_from_torch(torch_tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        """ Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            torch_tensor (torch.Tensor): 
                Torch tensor to serialize.

        Returns:
            bittensor_pb2.Tensor: 
                The serialized torch tensor as bittensor_pb2.proto. 
        """
        if torch_tensor.requires_grad:
            data_buffer = torch_tensor.cpu().detach().numpy().tobytes()
        else:
            data_buffer = torch_tensor.cpu().numpy().tobytes()
        
        dtype = torch_dtype_to_bittensor_dtype(torch_tensor.dtype)
        shape = list(torch_tensor.shape)
        torch_proto = bittensor_pb2.Tensor(version = bittensor.__version__,
                                     buffer = data_buffer,
                                     shape = shape,
                                     dtype = dtype,
                                     serializer = PyTorchPickleSerializer.ENUM,
                                     modality = bittensor_pb2.Modality.TEXT,
                                     requires_grad = torch_tensor.requires_grad)
        return torch_proto


    @staticmethod
    def deserialize_to_torch(torch_proto: bittensor_pb2.Tensor) -> torch.Tensor:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            torch_proto (bittensor_pb2.Tensor): 
                Proto containing torch tensor to derserialize.

        Returns:
            torch.Tensor: 
                Deserialized torch tensor.
        """
        dtype = bittensor_dtype_np_dtype(torch_proto.dtype)
        array = np.frombuffer(torch_proto.buffer, dtype=np.dtype(dtype)).copy()
        shape = tuple(torch_proto.shape)
        tensor = torch.as_tensor(array).view(shape).requires_grad_(
            torch_proto.requires_grad)
        return tensor
