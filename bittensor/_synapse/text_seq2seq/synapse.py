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

import grpc
import torch
import bittensor

from .. import synapse
from . import call

class TextSeq2SeqSynapse( synapse.Synapse  ):
    """ TextSeq2SeqSynapse: Class for servicing text_seq2seq requests."""

    def __str__(self):
        return 'TextSeq2Seq'
    
    def priority( self, forward_call: 'call.TextSeq2SeqForwardCall' ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, forward_call: 'call.TextSeq2SeqForwardCall'  ) -> torch.FloatTensor:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, forward_call: 'call.TextSeq2SeqForwardCall' ) -> torch.FloatTensor:
        """ forward: Returns the forward pass of the synapse.
            Args:
                forward_call.text_prompt (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text prompt.
                forward_call.timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                    timeout for the request.
                forward_call.topk (:obj:`int`, `optional`, defaults to 50):
                    topk for the request.
                forward_call. num_to_generate (:obj:`int`, `optional`, defaults to 256):
                    num_to_generate for the request.
                forward_call.num_beams (:obj:`int`, `optional`, defaults to 5):
                    num_beams for the request.
                forward_call.no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 2):
                    no_repeat_ngram_size for the request.
                forward_call.early_stopping (:obj:`bool`, `optional`, defaults to False):
                    early_stopping for the request.
                forward_call.num_return_sequences (:obj:`int`, `optional`, defaults to 1):
                    num_return_sequences for the request.
                forward_call.do_sample (:obj:`bool`, `optional`, defaults to False):
                    do_sample for the request.
                forward_call.top_p (:obj:`float`, `optional`, defaults to 0.95):
                    top_p for the request.
                forward_call.temperature (:obj:`float`, `optional`, defaults to 1.0):
                    temperature for the request.
                forward_call.repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                    repetition_penalty for the request.
                forward_call.length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                    length_penalty for the request.
                forward_call.max_time (:obj:`float`, `optional`, defaults to 150):
                    max_time for the request.
                forward_call.num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                    num_beam_groups for the request.
            Returns:
                torch.FloatTensor: torch tensor of text generations.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def ForwardTextSeq2Seq( 
            self, 
            request: bittensor.ForwardTextSeq2SeqRequest, 
            context: grpc.ServicerContext 
        ) -> bittensor.ForwardTextLastHiddenStateResponse:
        """ ForwardTextSeq2Seq
            --------------------
            Args:
                request (bittensor.ForwardTextSeq2SeqRequest): 
                    request.hotkey (string): hotkey of the neuron.
                    request.serialized_text_prompt (string): serialized text prompt.
                    request.text_prompt_serializer_type (bittensor.proto.SerializerType): text prompt serializer type.
                    request.generations_serializer_type (bittensor.proto.SerializerType): generations serializer type.
                    request.timeout (float): timeout for the request.
                    request.topk (int): topk for the request.
                    request.num_to_generate (int): num_to_generate for the request.
                    request.num_beams (int): num_beams for the request.
                    request.no_repeat_ngram_size (int): no_repeat_ngram_size for the request.
                    request.early_stopping (bool): early_stopping for the request.
                    request.num_return_sequences (int): num_return_sequences for the request.
                    request.do_sample (bool): do_sample for the request.
                    request.top_p (float): top_p for the request.
                    request.temperature (float): temperature for the request.
                    request.repetition_penalty (float): repetition_penalty for the request.
                    request.length_penalty (float): length_penalty for the request.
                    request.max_time (float): max_time for the request.
                    request.num_beam_groups (int): num_beam_groups for the request.
                context (grpc.ServicerContext):
                    grpc tcp context.
            Returns:
                response (bittensor.ForwardTextSeq2SeqResponse): 
                    response.serialized_text_outputs (string): serialized text outputs.
        """
        forward_call = call.TextSeq2SeqForwardCall.from_forward_request_proto( request )
        return self._Forward( forward_call = forward_call )

    
    