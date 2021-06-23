
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
""" An interface for serializing and deserializing bittensor tensors"""

import torch
import numpy as np
import bittensor

from . import serializer_impl

class serializer:

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
    
    def __new__(cls, serialzer_type: bittensor.proto.Serializer = bittensor.proto.Serializer.MSGPACK ) -> 'bittensor.Serializer':
        r"""Returns the correct serializer object for the passed Serializer enum. 

            Args:
                serialzer_type (:obj:`bittensor.proto.Serializer`, `required`): 
                    The serialzer_type ENUM from bittensor.proto.

            Returns:
                Serializer: (obj: `bittensor.Serializer`, `required`): 
                    The bittensor serializer/deserialzer for the passed type.

            Raises:
                NoSerializerForEnum: (Exception): 
                    Raised if the passed there is no serialzier for the passed type. 
        """
        # WARNING: the pickle serializer is not safe. Should be removed in future verions.
        # if serialzer_type == bittensor.proto.Serializer.PICKLE:
        #     return PyTorchPickleSerializer()
        if serialzer_type == bittensor.proto.Serializer.MSGPACK:
            return serializer_impl.MSGPackSerializer()
        else:
            raise bittensor.serializer.NoSerializerForEnum("No known serialzier for proto type {}".format(serialzer_type))

    @staticmethod
    def torch_dtype_to_bittensor_dtype(tdtype):
        """ Translates between torch.dtypes and bittensor.dtypes.

            Args:
                tdtype (torch.dtype): torch.dtype to translate.

            Returns:
                dtype: (bittensor.dtype): translated bittensor.dtype.
        """
        if tdtype == torch.float32:
            dtype = bittensor.proto.DataType.FLOAT32
        elif tdtype == torch.float64:
            dtype = bittensor.proto.DataType.FLOAT64
        elif tdtype == torch.int32:
            dtype = bittensor.proto.DataType.INT32
        elif tdtype == torch.int64:
            dtype = bittensor.proto.DataType.INT64
        else:
            dtype = bittensor.proto.DataType.UNKNOWN
        return dtype

    @staticmethod
    def bittensor_dtype_to_torch_dtype(bdtype):
        """ Translates between bittensor.dtype and torch.dtypes.

            Args:
                bdtype (bittensor.dtype): bittensor.dtype to translate.

            Returns:
                dtype: (torch.dtype): translated torch.dtype.
        """
        if bdtype == bittensor.proto.DataType.FLOAT32:
            dtype=torch.float32
        elif bdtype == bittensor.proto.DataType.FLOAT64:
            dtype = torch.float64
        elif bdtype == bittensor.proto.DataType.INT32:
            dtype = torch.int32
        elif bdtype == bittensor.proto.DataType.INT64:
            dtype=torch.int64
        else:
            raise bittensor.serializer.DeserializationException(
                'Unknown bittensor.Dtype or no equivalent torch.dtype for bittensor.dtype = {}'
                .format(bdtype))
        return dtype

    @staticmethod
    def bittensor_dtype_np_dtype(bdtype):
        """ Translates between bittensor.dtype and np.dtypes.

            Args:
                bdtype (bittensor.dtype): bittensor.dtype to translate.

            Returns:
                dtype: (numpy.dtype): translated np.dtype.
        """
        if bdtype == bittensor.proto.DataType.FLOAT32:
            dtype = np.float32
        elif bdtype == bittensor.proto.DataType.FLOAT64:
            dtype = np.float64
        elif bdtype == bittensor.proto.DataType.INT32:
            dtype = np.int32
        elif bdtype == bittensor.proto.DataType.INT64:
            dtype = np.int64
        else:
            raise bittensor.serializer.SerializationException(
                'Unknown bittensor.dtype or no equivalent numpy.dtype for bittensor.dtype = {}'
                .format(bdtype))
        return dtype
