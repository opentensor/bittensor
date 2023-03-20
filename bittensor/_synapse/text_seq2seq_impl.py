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

import bittensor
import torch
from typing import Union, List, Tuple, Optional

from .synapse_impl import Synapse

class TextSeq2Seq (Synapse):
    """ TextSeq2Seq Synapse type for sequence generation from language models.
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ
    def __init__( 
        self, 
        topk:int = 50, 
        num_to_generate: int = 256,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 2,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        top_p: float = 0.95, 
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        max_time: float = 150,
        num_beam_groups: int = 1,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> 'TextSeq2Seq':  
        """ TextSeq2Seq Synapse initializer.
        Args:
            Topk (:obj:int, :default: 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering. 
            num_to_generate (:obj: int, :default: 256):
                The number of tokens to generate using the language model
            num_beams (:obj: int, :default: 5):
                The number of beams to keep during beam search
            no_repeat_ngram_size (:obj: int, :default: 2):
                The number of repeat n gram allowed
            early_stopping: (:obj: bool, :default: True):
                If the model should early stop if the probabilty drops a certain threshold
            num_return_sequences: (:obj: int, :default: 1):
                How many sequences should the model return
            do_sample (:obj: bool, :default: False):
                If the model should do sample its probablity during generation
            top_p (:obj: float, :default: 0.95): 
                probability cutoff for top p sampling
            temperature: (:obj: float, :default: 1.0):
                The value used to module the next token probabilities for the softmax calculation
            repetition_penalty (:obj: float, :default: 1.0):
                The parameter for repetition penalty. 1.0 means no penalty.
            length_penalty (:obj: float, :default: 1.0): 
                The parameter for length penalty. 0.0 means no penalty, <0 to encourage longer sequences.
            num_beam_groups (:obj: int, :default: 1):
                Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. 
            max_time (:obj: float, :default: 150): 
                The maximum time that a server can use to generate
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
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.num_to_generate = num_to_generate
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.num_beam_groups = num_beam_groups
        self.max_time = max_time 
        self.synapse_type = TextSeq2Seq.synapse_type

    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str: return "TextSeq2Seq"

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the instance proto to an instance class."""
        return TextSeq2Seq (
            topk = instance_proto.topk,
            num_to_generate = instance_proto.num_to_generate,
            num_beams = instance_proto.num_beams,
            no_repeat_ngram_size = instance_proto.no_repeat_ngram_size,
            early_stopping = instance_proto.early_stopping,
            num_return_sequences = instance_proto.num_return_sequences,
            do_sample = instance_proto.do_sample,
            top_p = instance_proto.top_p,
            temperature = instance_proto.temperature,
            repetition_penalty = instance_proto.repetition_penalty,
            length_penalty = instance_proto.length_penalty,
            num_beam_groups = instance_proto.num_beam_groups,
            max_time = instance_proto.max_time,
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'Synapse':
        """ Deserialzied the wire proto to an instance class. """
        instance_proto = bittensor.proto.Synapse.TextSeq2Seq()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return TextSeq2Seq.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextSeq2Seq':
        """ Serializes the class instance to a Synapse instance proto."""
        return bittensor.proto.Synapse.TextSeq2Seq ( 
            topk = self.topk, 
            num_to_generate = self.num_to_generate,
            num_beams = self.num_beams,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            early_stopping = self.early_stopping,
            num_return_sequences = self.num_return_sequences,
            do_sample = self.do_sample,
            top_p = self.top_p,
            temperature = self.temperature,
            repetition_penalty = self.repetition_penalty,
            length_penalty = self.length_penalty,
            num_beam_groups = self.num_beam_groups,
            max_time = self.max_time,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = ''  ) -> bittensor.proto.Synapse:
        """ Serializes the class instance to a Synapse wire proto. """
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextSeq2Seq.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor     ( self, forward_request_tensor ):
        if len( forward_request_tensor.shape ) != 2 or forward_request_tensor.shape[0] == 0 or forward_request_tensor.shape[1] == 0:
            raise ValueError( "forward_request_tensor.shape must be in [-1, -1], got: {} for synapse: {}".format( list(forward_request_tensor.shape), self ) ) 

    def check_forward_response_tensor    ( self, forward_request_tensor, forward_response_tensor ):
        if forward_response_tensor == None:
            raise ValueError('Empty Response')
        if (
             len( forward_response_tensor.shape ) != 2 or
             forward_response_tensor.size(0) != forward_request_tensor.size(0) or
             forward_response_tensor.size(1) > self.num_to_generate
            ):
            raise ValueError( "forward_response_tensor.shape must be in [{}, <{}], got: {} for synapse: {}".format( forward_request_tensor.size(0) , self.num_to_generate,  list(forward_response_tensor.shape), self ) ) 

    def check_backward_request_gradient  ( self, forward_request_tensor, backward_request_gradient ):
        if len(backward_request_gradient.shape) > 1 or ( torch.numel(backward_request_gradient) >= 1 ): # the gradient for seq2seq should always be torch.tensor([])
            raise ValueError( "backward_request_gradient.shape must be in [0], got: {} for synapse: {}".format( forward_request_tensor.size(0) , forward_request_tensor.size(1), bittensor.__network_dim__, list(backward_request_gradient.shape), self ) ) 

        return

    def encode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor: 
        return forward_request_tensor
        
    def decode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        return forward_request_tensor

    def encode_forward_response_tensor   ( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor: 
        # Apply topk logit encoding.
        return forward_response_tensor

    def decode_forward_response_tensor   ( self, forward_request_tensor: torch.Tensor,
                                           forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        # Decode topk logit encoding.
        return forward_response_tensor  # [batch_size, sequence_len]

    def encode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: 
        # Apply topk logit encoding for gradients.
        return backward_request_gradient

    def decode_backward_request_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor:
        # Decode topk logit encoding for gradients.
        return backward_request_gradient

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite forward call when the call fails."""
        try:
            if forward_request_tensor.size(0) == 0 :
                return torch.tensor([])

            return torch.zeros( ( forward_request_tensor.size(0), self.num_to_generate), dtype=torch.float32)
        except:
            return torch.tensor([])

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns a zeroed tensor used as response to a dendrite backward call when the call fails."""

        return torch.tensor([])