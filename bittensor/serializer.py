""" An interface for serializing and deserializing bittensor tensors"""
from bittensor import bittensor_pb2
import numpy as np
import pickle
import PIL
import torch
from typing import List, Tuple, Dict, Optional

import bittensor
   
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

class PyTorchSerializer():
    
    @staticmethod
    def deserialize(proto: bittensor_pb2.Tensor) -> object:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (bittensor_pb2.Tensor): Proto to derserialize.

        Returns:
            List[object]:
        """
        # Check typing
        if proto.modality == bittensor_pb2.Modality.IMAGE:
            return PyTorchSerializer.deserialize_image( proto )
        
        elif proto.modality == bittensor_pb2.Modality.TEXT:
            return PyTorchSerializer.deserialize_text( proto )
        
        elif proto.modality == bittensor_pb2.Modality.TENSOR:
            return PyTorchSerializer.deserialize_tensor( proto )
        
        else:
            raise NotImplementedError
   
    @staticmethod
    def serialize(tensor: object, modality: bittensor_pb2.Modality) -> bittensor_pb2.Tensor:
        """Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            tensor (object): general object with modality TEXT, IMAGE, TENSOR

        Returns:
            bittensor_pb2.Tensor: Serialized tensor as bittensor_pb2.proto. 
        """
        # Check typing
        if modality == bittensor_pb2.Modality.IMAGE:
            return PyTorchSerializer.serialize_image( tensor )
        
        elif modality == bittensor_pb2.Modality.TEXT:
            return PyTorchSerializer.serialize_text( tensor )
        
        elif modality == bittensor_pb2.Modality.TENSOR:
            return PyTorchSerializer.serialize_tensor( tensor )
        
        else:
            raise NotImplementedError
            

    @staticmethod
    def serialize_text(tensor: List[str]) -> bittensor_pb2.Tensor:
        """Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            tensor (torch.Tensor): torch.Tensor to serialize.

        Returns:
            bittensor_pb2.Tensor: Serialized tensor as bittensor_pb2.proto. 
        """
        data_buffer = pickle.dumps(tensor)
        proto = bittensor_pb2.Tensor(
                    version = bittensor.__version__,
                    buffer = data_buffer,
                    shape = [len(tensor), 1],
                    dtype = bittensor_pb2.DataType.UTF8,
                    modality = bittensor_pb2.Modality.TEXT,
                    requires_grad = False)
        return proto
        
    @staticmethod
    def serialize_image(tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        """Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            tensor (torch.Tensor): torch.Tensor to serialize.

        Returns:
            bittensor_pb2.Tensor: Serialized tensor as bittensor_pb2.proto. 
        """    
        # TODO (const) replace asserts with errors.
        assert len(tensor.shape) > 1
        # Using numpy intermediary because deserializing with pickle can run arbitray code on your machine.
        data_buffer = tensor.cpu().numpy().tobytes()
        dtype = torch_dtype_to_bittensor_dtype (tensor.dtype)
        shape = list(tensor.shape)
        proto = bittensor_pb2.Tensor(
                    version = bittensor.__version__,
                    buffer = data_buffer,
                    shape = shape,
                    dtype = dtype,
                    modality = bittensor_pb2.Modality.IMAGE,
                    requires_grad = tensor.requires_grad)      
        return proto
    
    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> bittensor_pb2.Tensor:
        """Serializes a torch.Tensor to an bittensor Tensor proto.

        Args:
            tensor (torch.Tensor): torch.Tensor to serialize.

        Returns:
            bittensor_pb2.Tensor: Serialized tensor as bittensor_pb2.proto. 
        """

        # TODO (const) replace asserts with errors.
        assert len(tensor.shape) > 1
        # Using numpy intermediary because deserializing with pickle can run arbitray code on your machine.
        data_buffer = tensor.cpu().numpy().tobytes()
        dtype = torch_dtype_to_bittensor_dtype (tensor.dtype)
        shape = list(tensor.shape)
        proto = bittensor_pb2.Tensor(
                    version = bittensor.__version__,
                    buffer = data_buffer,
                    shape = shape,
                    dtype = dtype,
                    modality = bittensor_pb2.Modality.TENSOR,
                    requires_grad = tensor.requires_grad)      
        return proto
    
    
    @staticmethod
    def deserialize_image(proto: bittensor_pb2.Tensor) -> torch.Tensor:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (bittensor_pb2.Tensor): Proto to derserialize.

        Returns:
            List[object]:
        """
        dtype = np.float32
        # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
        array = np.frombuffer(proto.buffer, dtype=np.dtype(dtype)).copy()
        # NOTE (const): The first dimension is always assumed to be the batch dimension.
        shape = tuple(proto.shape)
        assert len(shape) > 1 
        tensor = torch.as_tensor(array).view(shape).requires_grad_(proto.requires_grad)
        return tensor
        
    @staticmethod
    def deserialize_text(proto: bittensor_pb2.Tensor) -> List[str]:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (bittensor_pb2.Tensor): Proto to derserialize.

        Returns:
            List[object]:
        """
        return pickle.loads(proto.buffer)
            
    @staticmethod
    def deserialize_tensor(proto: bittensor_pb2.Tensor) -> torch.Tensor:
        """Deserializes an bittensor_pb2.Tensor to a torch.Tensor object.

        Args:
            proto (bittensor_pb2.Tensor): Proto to derserialize.

        Returns:
            torch.Tensor: torch.Tensor to deserialize.
        """
        dtype = bittensor_dtype_np_dtype(proto.dtype)
        # TODO avoid copying the array (need to silence pytorch warning, because array is not writable)
        array = np.frombuffer(proto.buffer, dtype=np.dtype(dtype)).copy()
        # NOTE (const): The first dimension is always assumed to be the batch dimension.
        shape = tuple(proto.shape)
        assert len(shape) > 1 
        tensor = torch.as_tensor(array).view(shape).requires_grad_(proto.requires_grad)
        return tensor