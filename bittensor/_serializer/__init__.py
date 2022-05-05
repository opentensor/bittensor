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
import numpy as np
import bittensor
from typing import Tuple, List, Union, Optional

from . import serializer_impl

class serializer:
    """ An interface for serializing and deserializing bittensor tensors"""

    class SerializationException (Exception):
        """ Raised during serialization """

    class DeserializationException (Exception):
        """ Raised during deserialization """

    class NoSerializerForEnum (Exception):
        """ Raised if there is no serializer for the passed type """

    class SerializationTypeNotImplementedException (Exception):
        """ Raised if serialization/deserialization is not implemented for the passed object type """
    
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
        elif serialzer_type == bittensor.proto.Serializer.CMPPACK:
            return serializer_impl.CMPPackSerializer()
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
        elif tdtype == torch.float16:
            dtype = bittensor.proto.DataType.FLOAT16
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
        elif bdtype == bittensor.proto.DataType.FLOAT16:
            dtype=torch.float16
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


class Synapse_Serializer:

    @staticmethod
    def args_to_synapse( args: List[ Tuple[ 'bittensor.proto.Synapse.Type', dict ] ] ) -> 'bittensor.proto.Synapse':
        try:
            return Synapse_Serializer.serialize( 0, synapse_args = args[ 1 ], synapse_type = args[ 0 ] ) 
        except Exception as error:
            raise ValueError( "Failed to serialze synapse arguments {} with error: {}".format( args, error ))

    @staticmethod
    def format_synapses ( synapses: Union[ List[ 'bittensor.proto.Synapse' ], List[ Tuple[ 'bittensor.proto.Synapse.Type', dict ] ]] ) -> List[ 'bittensor.proto.Synapse' ]:
        """ Formats a list of synapses or synapse argument information.

        Args:
            synapses (:obj:`Union[ List[ bittensor.proto.Synapse ], List[ Tuple[ bittensor.proto.Synapse.Type, dict ] ]]` of shape :obj:`(num_synapses)`, `required`):
                Protos specifiying the synapses to call, or synapse types with args. Each corresponds to a synapse function on the axon and args.
                Responses are packed in this ordering. 

        Returns:
            formatted_synapses (:obj:`List[ bittensor.proto.Synapse ]` of shape :obj:`(num_synapses)`, `required`):
                Protos specifiying the synapses to call formatted with a type.
        """
        formatted_synapses = {}
        for syn in synapses:
            # Optionally convert synapse args dict to synapse protos.
            if isinstance( syn, tuple ):
                formatted_synapses.append( Synapse_Serializer.args_to_synapse( syn ) )
        for syn in formatted_synapses:
            if isinstance( syn, bittensor.proto.Synapse.TextLastHiddenState ):
                syn.synapse_type = bittensor.proto.Synapse.SynapseType.TextLastHiddenState
            elif isinstance ( syn, bittensor.proto.Synpase.TextCausalLM ):
                syn.synapse_type = bittensor.proto.Synapse.SynapseType.TextCausalLM
            elif isinstance( syn,  bittensor.proto.Synapse.TEXT_SEQ_2_SEQ ):
                syn.synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ
            else:
                raise ValueError( "Passed synapse object has unknown type: {}".format( type(syn).__name__ ) )    
        return formatted_synapses

    """ Make conversion between torch and bittensor.proto.torch
    """
    @staticmethod
    def serialize(self, tensor_pos: int,  args: dict={}, synapse_type: 'bittensor.proto.SynapseType' = 0 ) -> bittensor.proto.Tensor:
        """ Serializes a dictionary of args to an bittensor Synapse proto.

        Args:
            tensor_pos (int):
                position of the corresponding tensor

            args (dictionary): 
                dictionary of args for synapse 

            synapse_type (bittensor.proto.synapse_type): 
                synapse_type 

        Returns:
            bittensor.proto.Synapse: 
                The serialized torch tensor as bittensor.proto.proto. 
        """
        if synapse_type == bittensor.proto.SynapseType.TEXT_LAST_HIDDEN_STATE:
            arg_proto = bittensor.proto.SynapseArgsTextLastHiddenState(
                            synapse_type = bittensor.proto.SynapseType.TEXT_LAST_HIDDEN_STATE
                        )

        elif synapse_type == bittensor.proto.SynapseType.TEXT_CAUSAL_LM:
            arg_proto = bittensor.proto.SynapseArgsTextCausalLM(
                            synapse_type = bittensor.proto.SynapseType.TEXT_CAUSAL_LM,
                            topk = args['topk'] if 'topk' in args else 5
                        )

        elif synapse_type == bittensor.proto.SynapseType.TEXT_SEQ_2_SEQ:
            arg_proto = bittensor.proto.SynapseArgsTextSeq2Seq(
                            synapse_type = bittensor.proto.SynapseType.TEXT_SEQ_2_SEQ,
                            topk = args['topk'] if 'topk' in args else 5,
                            k_sequence = args['k_sequence'] if 'k_sequence' in args else 2,
                        )
        torch_proto = bittensor.proto.Synapse (
                                    tensor_pos= tensor_pos,
                                    args_data = arg_proto.SerializeToString(),
                                    synapse_type = synapse_type,
                                )
        return torch_proto

    @staticmethod
    def deserialize( torch_proto: 'bittensor.proto.Synapse' ) -> dict:
        """Deserializes an bittensor.proto.Synapse to a bittensor.proto.Synapse_args object.

        Args:
            torch_proto (bittensor.proto.Synapse): 
                Proto containing synapse args to deserialize.

        Returns:
            args: 
                Deserialized Dict containing args.
        """
        synapse_type =  torch_proto.synapse_type

        if synapse_type == bittensor.proto.SynapseType.TEXT_LAST_HIDDEN_STATE:
            args_proto = bittensor.proto.SynapseArgsTextLastHiddenState()

        elif synapse_type == bittensor.proto.SynapseType.TEXT_CAUSAL_LM:
            args_proto = bittensor.proto.SynapseArgsTextCausalLM()

        elif synapse_type == bittensor.proto.SynapseType.TEXT_SEQ_2_SEQ:
            args_proto = bittensor.proto.SynapseArgsTextSeq2Seq()

        args_proto.ParseFromString(torch_proto.args_data)


        return args_proto
