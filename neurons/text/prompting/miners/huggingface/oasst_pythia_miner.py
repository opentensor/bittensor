# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

from base import HuggingFaceMiner

class StopOnTokens( StoppingCriteria ):
    def __init__( self, stop_token_ids: List[int] = None ):
        self.stop_token_ids = stop_token_ids

    def __call__( self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class OasstPythiaMiner( HuggingFaceMiner ):
    arg_prefix = 'oasst_pythia'
    system_label = "<|system|>"
    assistant_label = "<|assistant|>"
    user_label = "<|prompter|>"

    def __init__( self ):
        super( OasstPythiaMiner, self ).__init__()
        self.stop = StopOnTokens( self.tokenizer.convert_tokens_to_ids( [ "<|endoftext|>" ] ) )

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained( self.config.oasst_pythia.model_name, torch_dtype=torch.bfloat16 )

    def load_model( self ):
        bittensor.logging.info( 'Loading ' + str( self.config.oasst_pythia.model_name ) )
        model = GPTNeoXForCausalLM.from_pretrained(
            self.config.oasst_pythia.model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        bittensor.logging.info( 'Model loaded!' )
        return model

    def forward( self, messages: List[Dict[str, str]] ):
        history = self.process_history(messages)
        prompt = history + self.assistant_label

        inputs = self.tokenizer( prompt, return_tensors="pt" )
        inputs = inputs.to( self.model.device )

        gkw = {
            **{
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": self.config.oasst_pythia.max_new_tokens,
                "temperature": self.config.oasst_pythia.temperature,
                "do_sample": self.config.oasst_pythia.do_sample,
                "top_p": self.config.oasst_pythia.top_p,
                "top_k": self.config.oasst_pythia.top_k,
                "repetition_penalty": self.config.oasst_pythia.repetition_penalty,
                "stopping_criteria": StoppingCriteriaList( [ self.stop ] ),
                "pad_token_id": self.tokenizer.eos_token_id,
            },
        }
        output = self.model.generate( **gkw )
        generation = self.tokenizer.decode( output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True )

        bittensor.logging.debug( "Message: " + str(messages ) )
        bittensor.logging.debug( "Generation: " + str(generation ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    OasstPythiaMiner().run()
