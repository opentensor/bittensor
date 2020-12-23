import numpy as np
import torch
import bittensor
from bittensor import bittensor_pb2
import bittensor.serialization as serialization

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
        raise serialization.DeserializationException(
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
        raise serialization.SerializationException(
            'Unknown bittensor.dtype or no equivalent numpy.dtype for bittensor.dtype = {}'
            .format(bdtype))
    return dtype