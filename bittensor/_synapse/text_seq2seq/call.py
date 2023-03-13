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

import time
import torch
import bittensor

class TextSeq2SeqBittensorCall( bittensor.BittensorCall ):
    """ Call state for the text_seq_to_seq synapse."""

    # Name of the synapse
    name: str = 'text_seq2seq'

    def __str__( self ) -> str:
        return """
bittensor.TextSeq2SeqBittensorCall( 
    name: {},
    description: Generates text from a text prompt.
    caller: {},
    version: {},
    timeout = {}, 
    start_time = {},
    end_time = {},
    elapsed = {},
    Args:
    \ttext_prompt: torch.LongTensor = {},
    \tgenerations: torch.LongTensor = {}, 
    \ttopk:int = {}, 
    \tnum_to_generate: int = {},
    \tnum_beams: int = {},
    \tno_repeat_ngram_size: int = {},
    \tearly_stopping: bool = {},
    \tnum_return_sequences: int = {},
    \tdo_sample: bool = {},
    \ttop_p: float = {}, 
    \ttemperature: float = {},
    \trepetition_penalty: float = {},
    \tlength_penalty: float = {},
    \tmax_time: float = {},
    \tnum_beam_groups: int = {},
    \ttext_prompt_serializer_type: 'bittensor.serializer_type' = {}
    \tgenerations_serializer_type: 'bittensor.serializer_type' = {}
)
""".format(
            self.name,
            self.hotkey,
            self.version,
            self.timeout,
            self.start_time,
            self.end_time,
            time.time() - self.start_time,
            self.text_prompt,
            self.generations if self.generations != None else "To be filled by the forward call.",
            self.topk,
            self.num_to_generate,
            self.num_beams,
            self.no_repeat_ngram_size,
            self.early_stopping,
            self.num_return_sequences, 
            self.do_sample, 
            self.top_p,
            self.temperature,
            self.repetition_penalty,
            self.length_penalty,
            self.max_time,
            self.num_beam_groups,
            self.text_prompt_serializer_type,
            self.generations_serializer_type,
        )

    def __init__( 
            self, 
            text_prompt: torch.LongTensor, 
            timeout: float = bittensor.__blocktime__,
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
            text_prompt_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
            generations_serializer_type: 'bittensor.serializer_type' = bittensor.proto.Serializer.MSGPACK,
        ):
        """ Initializes the forward call object.
            Args:
                text_prompt (:obj:`torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    A tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                timeout (:type:`int`, default = bittensor.__blocktime__ `optional`):
                    Request timeout. Queries that do not respond will be replaced by zeros.
                topk (:obj:int, :default: 50):
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
                text_prompt_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for text inputs.
                generations_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                    serializer type for hidden states.
        """
        super().__init__(timeout = timeout)
        self.text_prompt = text_prompt
        self.generations = None
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
        self.max_time = max_time
        self.num_beam_groups = num_beam_groups
        self.text_prompt_serializer_type = text_prompt_serializer_type
        self.generations_serializer_type = generations_serializer_type

    def get_inputs_shape(self) -> torch.Size:
        if self.text_prompt is not None:
            return self.text_prompt.shape
        else: return None
    
    def get_outputs_shape(self) -> torch.Size:
        if self.generations is not None:
            return self.generations.shape
        else: return None