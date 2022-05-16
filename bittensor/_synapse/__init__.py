
from multiprocessing.sharedctypes import Value
from yaml import serialize_all
import bittensor
import torch
from typing import Union, List, Tuple, Optional

from bittensor._serializer import serializer
from .synapse_impl import Synapse
from .text_causallm_impl import TextCausalLM
from .text_lasthiddenstate_impl import TextLastHiddenState
from .text_seq2seq_impl import TextSeq2Seq
class synapse:
    """
    Factory class for the synapse objects. The synapses are designed to work the bittensor protocol and is 
    reponsible for the serialization and deserialization of their contents. They are expected to be included by
    the forwarding neuron when making a call through the bittensor api.

    Examples:
        >>> causallm_synapse = bittensor.synapse.TextCausalLM()
        >>> dendrite.text(endpoints = [..], inputs = [..], synapses= [causallm_synapse] )
    
    """
    @staticmethod
    def TextLastHiddenState (
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextLastHiddenState:
        """ Factory function which returns a TextLastHiddenState synapse adapter given arguments.
            Args:
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextLastHiddenState (:obj:`TextLastHiddenState`, `required`):
                    TextLastHiddenState instance adapter class.
        """
        return TextLastHiddenState (
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def TextCausalLM ( 
        topk:int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextCausalLM:
        """ Factory function which returns a TextCausalLM synapse adapter given arguments.
            Args:
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextCausalLM (:obj:`TextCausalLM`, `required`):
                    TextCausalLM instance adapter class.
        """
        return TextCausalLM ( 
            topk = topk,
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def TextSeq2Seq ( 
        topk:int = 512, 
        num_to_generate: int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextSeq2Seq:
        """ Factory function which returns a TextSeq2Seq synapse adapter given arguments.
            Args:
                topk (:obj:`int` of shape :obj:`(1)`, `optional`, :default: `512`):
                    Number of topk logits to return per item in the sequence.
                num_to_generate (:obj:`int` of shape :obj:`(1)`, `optional`, :default: `512`):
                    Number of topk logits to generate per example in the batch.
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serialzer used to pack torch tensors on backward response.
            Returns:
                TextSeq2Seq (:obj:`TextSeq2Seq`, `required`):
                    TextSeq2Seq instance adapter class.
        """
        return TextSeq2Seq ( 
            topk = topk, 
            num_to_generate = num_to_generate,
            forward_request_serializer_type = forward_request_serializer_type,
            forward_response_serializer_type = forward_response_serializer_type,
            backward_request_serializer_type = backward_request_serializer_type,
            backward_response_serializer_type = backward_response_serializer_type,
        )

    @staticmethod
    def deserialize( synapse_wire_proto: bittensor.proto.Synapse ) -> Synapse:
        if synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            return TextLastHiddenState.deserialize_from_wire_proto ( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:
            return TextCausalLM.deserialize_from_wire_proto( synapse_wire_proto )
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:
            return TextSeq2Seq.deserialize_from_wire_proto( synapse_wire_proto )
        else:
            raise ValueError("Synapse type is unknown.")