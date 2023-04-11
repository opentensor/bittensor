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
    synapse_name: str = 'text_seq2seq'
    default_blacklist_stake: float = 10

    def __init__(
            self,
            config: 'bittensor.Config' = None, 
        ):
        """ __init__: Initializes the synapse.
            Args:
                config (:obj:`bittensor.Config`, `optional`):
                    bittensor config object.
        """
        if config is None: config = self.config()
        TextSeq2SeqSynapse.check_config( config )
        super().__init__( config )
        self.config = copy.deepcopy(config)

    def _attach( self, axon: 'bittensor.axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextSeq2SeqServicer_to_server( self, axon.server )

    def __str__(self):
        return 'TextSeq2Seq'

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
        serialized_generations = generations_serializer.serialize( forward_call.outputs, from_type = bittensor.proto.TensorType.TORCH )

        # Set response.
        return bittensor.ForwardTextSeq2SeqResponse(
            version=bittensor.__version_as_int__,
            hotkey=self.axon.wallet.hotkey.ss58_address,
            serialized_generations = serialized_generations,
            message=forward_call.request_message,
            return_code=forward_call.request_code,
        )


    def pre_process_request_proto_to_backward_call(
        self, request_proto: "bittensor.BackwardRequest"
    ) -> "bittensor.BittensorCall":
        """pre_process_request_proto_to_backward_call
        ------------------------------------------
        Args:
            request_proto (bittensor.BackwardRequest):
                request_proto to process in to a backward call.
        Returns:
            bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                backward call processed from the request proto.
        """
        # TODO: Implement this.
        return bittensor.BittensorCall()
