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

import os
import copy
import bittensor
import argparse

class TextSeq2SeqSynapse( bittensor.Synapse, bittensor.grpc.TextSeq2SeqServicer ):
    """ TextSeq2SeqSynapse: Class for servicing text_seq2seq requests."""

    # Synapse name.
    name: str = 'text_seq2seq'

    def __init__(
            self, 
            config: 'bittensor.Config' = None, 
            metagraph: 'bittensor.metagraph.Metagraph' = None,
        ):
        if config == None: config = TextSeq2SeqSynapse.config()
        TextSeq2SeqSynapse.check_config( config )
        super().__init__( config, metagraph )
        self.config = copy.deepcopy(config)
        self.metagraph = metagraph
        self.priority_threadpool = bittensor.prioritythreadpool( config = config.synapse.text_seq2seq )

    def _attach( self, axon: 'bittensor.axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextSeq2SeqServicer_to_server( self, axon.server )

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Returns the config for this synapse."""
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        super().add_args( parser = parser, prefix = prefix )
        bittensor.prioritythreadpool.add_args( parser, prefix = prefix_str + 'synapse.text_seq2seq' )
        try:
            parser.add_argument('--' + prefix_str + 'synapse.text_seq2seq.blacklist.stake', type=float, help='The amount of stake (tao) required to make a call.', default=10)
            parser.add_argument('--' + prefix_str + 'synapse.text_seq2seq.blacklist.allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=True)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def help(cls):
        """ Print help to stdout """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod   
    def add_defaults(cls, defaults):
        """ Add default values to defaults object"""
        defaults.synapse = bittensor.Config()
        defaults.synapse.text_seq2seq.blacklist.stake = os.getenv('BT_SYNAPSE_TEXT_SEQ2SEQ_BLACKLIST_STAKE') if os.getenv('BT_SYNAPSE_TEXT_SEQ2SEQ_BLACKLIST_STAKE') != None else 10
        defaults.synapse.text_seq2seq.blacklist.allow_non_registered = os.getenv('BT_SYNAPSE_TEXT_SEQ2SEQ_BLACKLIST_ALLOW_NON_REGISTERED') if os.getenv('BT_SYNAPSE_TEXT_SEQ2SEQ_BLACKLIST_ALLOW_NON_REGISTERED') != None else True

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def __str__(self):
        return 'TextSeq2Seq'
    
    def priority( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
        """ priority: Returns the priority of the synapse for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement priority() in subclass.')
    
    def _priority( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
        """ _priority: Returns the priority of the forward call.
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to check.
            Returns:
                float: priority of the forward call.
        """
        return self.priority( forward_call)

    def blacklist( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall'  ) -> bool:
        """ blacklist: Returns True if the synapse should not be called for the given hotkey and text_inputs."""
        raise NotImplementedError('Must implement blacklist() in subclass.')

    def forward( self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> 'bittensor.TextSeq2SeqBittensorCall':
        """ Fills the forward_call with the results of the forward pass.
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
                forward_call (:obj:`bittensor.TextSeq2SeqBittensorCall`, `required`):
                    filled forward call dataclass.
        """
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: bittensor.ForwardTextSeq2SeqRequest 
        ) -> 'bittensor.TextSeq2SeqBittensorCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardTextSeq2SeqRequest):
                    bittensor forward request proto.
            Returns:
                bittensor.TextSeq2SeqBittensorCall (:obj:`bittensor.TextSeq2SeqBittensorCall`, `required`):
                    bittensor forward call dataclass.
        """
        # Deserialize text inputs.
        text_prompts_deserializer = bittensor.serializer( serializer_type = request_proto.text_prompt_serializer_type )
        text_prompt = text_prompts_deserializer.deserialize( request_proto.serialized_text_prompt )

        # Fill forward call.
        return bittensor.TextSeq2SeqBittensorCall(
            text_prompt = text_prompt,
            timeout = request_proto.timeout,
            topk = request_proto.topk,
            num_to_generate = request_proto.num_to_generate,
            num_beams = request_proto.num_beams,
            no_repeat_ngram_size = request_proto.no_repeat_ngram_size,
            early_stopping = request_proto.early_stopping,
            num_return_sequences = request_proto.num_return_sequences,
            do_sample = request_proto.do_sample,
            top_p = request_proto.top_p,
            temperature = request_proto.temperature,
            repetition_penalty = request_proto.repetition_penalty,
            length_penalty = request_proto.length_penalty,
            max_time = request_proto.max_time,
            num_beam_groups = request_proto.num_beam_groups,
        )
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.TextSeq2SeqBittensorCall' 
        ) -> bittensor.ForwardTextSeq2SeqResponse:
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.TextSeq2SeqBittensorCall):
                    forward call to process.
            Returns:    
                response (bittensor.ForwardTextSeq2SeqResponse):
                    serialized response proto with the results of the forward call.
        """
        # Serialize generations.
        generations_serializer = bittensor.serializer( serializer_type = forward_call.generations_serializer_type )
        serialized_generations = generations_serializer.serialize( forward_call.generations, from_type = bittensor.proto.TensorType.TORCH )

        # Set response.
        return bittensor.ForwardTextSeq2SeqResponse(
            serialized_generations = serialized_generations
        )
    

    
    