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


from typing import List, Optional

import bittensor

from .synapse_impl import NullSynapse, Synapse
from .text_causallm_impl import TextCausalLM
from .text_causallmnext_impl import TextCausalLMNext
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
    __synapses_types__ = ['TextLastHiddenState', 'TextCausalLM', 'TextSeq2Seq']

    @staticmethod
    def TextLastHiddenState (
        mask: Optional[List[int]] = None,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextLastHiddenState:
        """ Factory function which returns a TextLastHiddenState synapse adapter given arguments.
            Args:
                mask (:obj:`List[int]` of shape :obj:`(n)`, `optional`, :default: `[]`):
                    An optional response mask over the returned embeddings.               
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
            mask = mask,
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
    def TextCausalLMNext(
        topk: int = 4096,
        forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
        backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ) -> TextCausalLMNext:
        """ Factory function which returns a TextCausalLMNext synapse adapter given arguments.
            Args:
                topk (:obj:`int`):
                    Specifies the number of topk server token phrases to return.
                forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward response.
                backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on forward request.
                backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                    Serializer used to pack torch tensors on backward response.
            Returns:
                TextCausalLMNext (:obj:`TextCausalLMNext`, `required`):
                    TextCausalLMNext instance adapter class.
        """
        return TextCausalLMNext(
            topk=topk,
            forward_request_serializer_type=forward_request_serializer_type,
            forward_response_serializer_type=forward_response_serializer_type,
            backward_request_serializer_type=backward_request_serializer_type,
            backward_response_serializer_type=backward_response_serializer_type,
        )

    @staticmethod
    def TextSeq2Seq ( 
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
    ) -> TextSeq2Seq:
        """ Factory function which returns a TextSeq2Seq synapse adapter given arguments.
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
        return TextSeq2Seq ( 
            topk = topk, 
            num_to_generate = num_to_generate,
            num_beams = num_beams,
            no_repeat_ngram_size = no_repeat_ngram_size,
            early_stopping = early_stopping,
            num_return_sequences = num_return_sequences,
            do_sample = do_sample,
            top_p = top_p, 
            temperature = temperature,
            repetition_penalty = repetition_penalty,
            length_penalty = length_penalty,
            num_beam_groups = num_beam_groups,
            max_time = max_time,
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
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
            return TextCausalLMNext.deserialize_from_wire_proto(synapse_wire_proto)
        elif synapse_wire_proto.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:
            return TextSeq2Seq.deserialize_from_wire_proto( synapse_wire_proto )
        else:
            return NullSynapse()