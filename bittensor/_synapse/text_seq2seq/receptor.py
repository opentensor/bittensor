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
import asyncio
import bittensor

class TextSeq2SeqReceptor( torch.nn.Module ):
    """ bittensor receptor for text_seq2seq synapse."""

    def __init__(
            self,
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            text_prompt_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            generations_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ):
        """ Initializes the receptor
            Args:
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet object.
                endpoint (:obj:`bittensor.endpoint`, `required`):
                    bittensor endpoint object.
                text_prompt_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text prompt.
                generations_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for generations.
        """
        self.receptor = bittensor.receptor( endpoint = endpoint, wallet = wallet )
        self._text_prompt_serializer_type = text_prompt_serializer_type
        self._generations_serializer_type = generations_serializer_type

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
        num_beam_groups: int = 1,
    ) -> torch.FloatTensor:
        """
            Returns a tuple containing the prompt generations for each 

            Args:
                prompts (:obj:`torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    A tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                timeout (:type:`int`, default = bittensor.__blocktime__ `optional`):
                    Request timeout. Queries that do not respond will be replaced by zeros.
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
                max_time (:obj: float, :default: 150): 
                    The maximum time that a server can use to generate
                num_beam_groups (:obj: int, :default: 1):
                    Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. 

            Returns:
                generations (:obj:`List[str]`, `required`):
                    Generations from each endpoint.
        """
        # Run async forward.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( 
            self.async_forward ( 
                text_inputs = text_prompt, 
                timeout = timeout,
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
                max_time = max_time,
                num_beam_groups = num_beam_groups,
            ) 
        )
    
    async def async_forward( 
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
            num_beam_groups: int = 1,
        ) -> torch.FloatTensor:
        """
            Returns a tuple containing the prompt generations for each 

            Args:
                prompts (:obj:`torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    A tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                timeout (:type:`int`, default = bittensor.__blocktime__ `optional`):
                    Request timeout. Queries that do not respond will be replaced by zeros.
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
                max_time (:obj: float, :default: 150): 
                    The maximum time that a server can use to generate
                num_beam_groups (:obj: int, :default: 1):
                    Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. 

            Returns:
                generations (:obj:`List[str]`, `required`):
                    Generations from each endpoint.
        """
        # Serialize the text prompt.
        text_serializer = bittensor.bittensor.serializer_for_type( self._text_prompt_serializer_type )
        serialized_text = text_serializer.serialize( text_prompt, from_type = bittensor.proto.TensorType.TORCH )

        # Create the request.
        request = bittensor.ForwardTextSeq2SeqRequest(
            serialized_text_inputs = serialized_text,
            text_inputs_serializer_type = self._text_prompt_serializer_type,
            generations_serializer_type = self._generations_serializer_type,
            timeout = timeout,
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
            max_time = max_time,
            num_beam_groups = num_beam_groups,
        )
        # Send the request.
        asyncio_future = self.receptor.stub.ForwardTextSeq2Seq(
                request = request, 
                timeout = timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptor.sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                ))
        # Wait for the response.
        grpc_response = await asyncio.wait_for( asyncio_future, timeout = timeout )

        # Deserialize the generations.
        generations_serializer = bittensor.bittensor.serializer_for_type( self._generations_serializer_type )
        generations = generations_serializer.deserialize( grpc_response.serialized_generations, to_type = bittensor.proto.TensorType.TORCH )
        
        # Return the generations.
        return generations

    

    
    