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

class TextCausalLM (Synapse):
    """ TextCausalLM Synapse type for next token prediction from languge models.
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM

    def __init__( 
        self, 
        topk: int = 512,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ):  
        """ TextCausalLM Synapse initializer.
        Args:
            Topk (:obj:int, :default: 512):
                The top k number of logits to compress and send over the wire 
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
        super().__init__ (
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.synapse_type = TextCausalLM.synapse_type

    def __repr__(self) -> str: return self.__str__()
    def __str__(self) -> str: return "TextCausalLM"

    @staticmethod
    def deserialize_from_instance_proto ( instance_proto: bittensor.proto.Synapse ) -> 'TextCausalLM':
        return TextCausalLM ( 
            topk = instance_proto.topk, 
            forward_request_serializer_type = instance_proto.forward_request_serializer_type,
            forward_response_serializer_type = instance_proto.forward_response_serializer_type,
            backward_request_serializer_type = instance_proto.backward_request_serializer_type,
            backward_response_serializer_type = instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto ( wire_proto: bittensor.proto.Synapse ) -> 'TextCausalLM':
        instance_proto = bittensor.proto.Synapse.TextCausalLM()
        instance_proto.ParseFromString( wire_proto.synapse_data )
        return TextCausalLM.deserialize_from_instance_proto( instance_proto )

    def serialize_to_instance_proto( self ) -> 'bittensor.proto.Synapse.TextCausalLM':
        return bittensor.proto.Synapse.TextCausalLM ( 
            topk = self.topk,
            forward_request_serializer_type = self.forward_request_serializer_type,
            forward_response_serializer_type = self.forward_response_serializer_type,
            backward_request_serializer_type = self.backward_request_serializer_type,
            backward_response_serializer_type = self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto ( self, code: 'bittensor.proto.ReturnCode' = 0, message: str = '' ) -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse (
                synapse_data = self.serialize_to_instance_proto().SerializeToString(),
                synapse_type = TextCausalLM.synapse_type,
                return_code = code,
                message = message
            )

    def check_forward_request_tensor     ( self, forward_request_tensor ): 
        if len( forward_request_tensor.shape ) != 2 or forward_request_tensor.shape[0] == 0 or forward_request_tensor.shape[1] == 0:
            raise ValueError( "forward_request_tensor.shape must be in [-1, -1], got: {} for synapse: {}".format( list(forward_request_tensor.shape), self ) ) 

    def check_forward_response_tensor    ( self, forward_request_tensor, forward_response_tensor ):
        if forward_response_tensor == None:
            raise ValueError("Empty Response") 

        if (
             len( forward_response_tensor.shape ) != 3 or
             forward_response_tensor.size(0) != forward_request_tensor.size(0) or
             forward_response_tensor.size(1) != forward_request_tensor.size(1) or
             forward_response_tensor.size(2) != self.topk*2
            ):
            raise ValueError( "forward_response_tensor.shape must be in [{}, {}, {}], got: {} for synapse: {}".format( forward_request_tensor.size(0) , forward_request_tensor.size(1), self.topk*2, list(forward_response_tensor.shape), self ) ) 

    def check_backward_request_gradient  ( self, forward_request_tensor, backward_request_gradient ):
        if ( len( backward_request_gradient.shape ) != 3 or
             backward_request_gradient.size(0) != forward_request_tensor.size(0) or
             backward_request_gradient.size(1) != forward_request_tensor.size(1) or 
             backward_request_gradient.size(2) != bittensor.__vocab_size__ 
            ):   
            raise ValueError( "backward_request_gradient.shape: {} must be equivalent to forward_request_tensor.shape: {} for synapse: {}".format( list( backward_request_gradient.shape ), list(forward_request_tensor.shape), self ) ) 

    def encode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor: return forward_request_tensor
    def decode_forward_request_tensor    ( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor: return forward_request_tensor

    def encode_forward_response_tensor( self, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns topk tokens/probabilities given unnormalized logits as input. """
        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1)  # normalized probabilities: [batch_size, sequence_len, vocab_size]
        topk_values, topk_indices = torch.topk(probs, self.topk) # topk probs and indices: [batch_size, sequence_len, topk]
        encoded_probs = torch.cat((topk_values, topk_indices), dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]

    def decode_forward_response_tensor( self, forward_request_tensor: torch.Tensor, forward_response_tensor: torch.Tensor ) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        batch_size, sequence_len, _ = forward_response_tensor.shape
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        topk_values = encoded_probs[..., :self.topk]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., self.topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

        topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
        remainder_floor = remainder_pmass / (bittensor.__vocab_size__ - self.topk)  # divide remainder: [batch_size, sequence_len]

        logits = torch.ones((batch_size, sequence_len, bittensor.__vocab_size__)).to(topk_values.device)
        logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]
        logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

        return logits  # [batch_size, sequence_len, vocab_size]

    def encode_backward_response_gradient( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient
    def decode_backward_response_gradient ( self, backward_request_gradient: torch.Tensor ) -> torch.Tensor: return backward_request_gradient

    def encode_backward_request_gradient( self, backward_response_gradient: torch.Tensor ) -> torch.Tensor:
        """ Return topk most negative token grads given full logit gradients. """
        values, indices = torch.topk(backward_response_gradient, self.topk) # ascend sort to get most negative gradients - informs on ideal logits
        encoded_grads = torch.cat((values, indices), dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_grads  # [batch_size, sequence_len, topk + topk]

    def decode_backward_request_gradient( self, backward_response_gradient: torch.Tensor ) -> torch.Tensor:
        """ Return full gradients by decoding topk-encoding input. """
        batch_size, sequence_len, _ = backward_response_gradient.shape
        encoded_grads = backward_response_gradient  # encoded gradients: [batch_size, sequence_len, topk + topk]
        topk_values = encoded_grads[..., :self.topk]  # topk grads: [batch_size, sequence_len, topk]
        topk_indices = encoded_grads[..., self.topk:].long()  # topk grads indices: [batch_size, sequence_len, topk]

        gradients = torch.zeros((batch_size, sequence_len, bittensor.__vocab_size__)).to(topk_values.device)
        gradients.scatter_(-1, topk_indices, topk_values)  # insert topk grads: [batch_size, sequence_len, vocab_size]

        return gradients  # [batch_size, sequence_len, vocab_size]

    def nill_forward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        try:
            return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)
        except:
            return torch.tensor([])

    def nill_backward_response_tensor( self, forward_request_tensor: torch.Tensor ) -> torch.Tensor:
        try:
            return torch.zeros( ( forward_request_tensor.size(0), forward_request_tensor.size(1), bittensor.__vocab_size__ ), dtype=torch.float32)
        except:
            return torch.tensor([])
