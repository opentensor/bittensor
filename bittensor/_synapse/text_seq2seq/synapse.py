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
import asyncio
import bittensor

class TextSeq2SeqSynapse( bittensor.TextSeq2SeqServicer ):
    """ TextSeq2SeqSynapse: Class for servicing text_seq2seq requests."""

    def __init__( self ):
        r""" Initializes a new Synapse."""
        self.priority_threadpool = bittensor.prioritythreadpool()
    
    def priority( self, hotkey:str, text_prompt: torch.FloatTensor, request: bittensor.ForwardTextSeq2SeqRequest ) -> float:
        """ priority: Returns the priority of the synapse."""
        raise NotImplementedError('Must implement priority() in subclass.')

    def blacklist( self, hotkey:str, text_prompt: torch.FloatTensor, request: bittensor.ForwardTextSeq2SeqRequest ) -> torch.FloatTensor:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')
    
    def _attach( self, axon: 'bittensor.axon.Axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextSeq2SeqServicer_to_server( self, axon.server )

    def forward( 
            self,
            text_prompt: torch.LongTensor,
            timeout: int = bittensor.__blocktime__,
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
            num_beam_groups: int = 1
        ) -> torch.FloatTensor:
        """ forward: Returns the forward pass of the synapse.
            Args:
                text_prompt (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                    torch tensor of text prompt.
                timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                    timeout for the request.
                topk (:obj:`int`, `optional`, defaults to 50):
                    topk for the request.
                num_to_generate (:obj:`int`, `optional`, defaults to 256):
                    num_to_generate for the request.
                num_beams (:obj:`int`, `optional`, defaults to 5):
                    num_beams for the request.
                no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 2):
                    no_repeat_ngram_size for the request.
                early_stopping (:obj:`bool`, `optional`, defaults to False):
                    early_stopping for the request.
                num_return_sequences (:obj:`int`, `optional`, defaults to 1):
                    num_return_sequences for the request.
                do_sample (:obj:`bool`, `optional`, defaults to False):
                    do_sample for the request.
                top_p (:obj:`float`, `optional`, defaults to 0.95):
                    top_p for the request.
                temperature (:obj:`float`, `optional`, defaults to 1.0):
                    temperature for the request.
                repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                    repetition_penalty for the request.
                length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                    length_penalty for the request.
                max_time (:obj:`float`, `optional`, defaults to 150):
                    max_time for the request.
                num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                    num_beam_groups for the request.
            Returns:
                torch.FloatTensor: torch tensor of text generations.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def ForwardTextSeq2Seq( 
            self, 
            request: bittensor.ForwardTextSeq2SeqRequest,
            context: grpc.ServicerContext,
        ) -> bittensor.ForwardTextSeq2SeqResponse:
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
        # Deserialize text prompt.
        text_prompt_serializer = bittensor.serializer( serializer_type = request.text_prompt_serializer_type )
        text_prompt = text_prompt_serializer.deserialize( request.serialized_text_prompt, from_type = bittensor.proto.TensorType.TORCH )
        
        # Check if the request hotkey is blacklisted.
        if self.blacklist( request.hotkey, text_prompt, request): 
            return bittensor.ForwardTextSeq2SeqResponse()

        # Get the priority of the request.
        priority = self.priority( request.hotkey, text_prompt, request)

        # Submit the request to the threadpool.
        future = self.priority_threadpool.submit(
            self.forward,
            hotkey = request.hotkey,
            text_prompt = text_prompt,
            topk = request.topk,
            num_to_generate = request.num_to_generate,
            num_beams = request.num_beams,
            no_repeat_ngram_size = request.no_repeat_ngram_size,
            early_stopping = request.early_stopping,
            num_return_sequences = request.num_return_sequences,
            do_sample = request.do_sample,
            top_p = request.top_p,
            temperature = request.temperature,
            repetition_penalty = request.repetition_penalty,
            length_penalty = request.length_penalty,
            max_time = request.max_time,
            num_beam_groups = request.num_beam_groups,
            priority = priority,
        )
        # Wait for the request to complete.
        generations = future.result( timeout = request.timeout )

        # Serialize generations.
        generations_serializer = bittensor.serializer( serializer_type = request.generations_serializer_type )
        serialized_generations= generations_serializer.serialize( generations, from_type = bittensor.proto.TensorType.TORCH )

        # Return the response.
        return bittensor.ForwardTextSeq2SeqResponse(
            serialized_generations = serialized_generations,
        )

    

    
    