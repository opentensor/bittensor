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
from transformers import AutoTokenizer, AutoModelForCausalLM

from base import HuggingFaceMiner

class KoalaMiner( HuggingFaceMiner ):
    arg_prefix: str = 'koala'
    assistant_label: str = 'GPT:'
    user_label: str = 'USER:'
    system_label: str = 'SYSTEM:'

    def __init__( self ):
        super( KoalaMiner, self ).__init__()
        print ( self.config )

        bittensor.logging.info( 'Loading ' + str(self.config.koala.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.koala.model_name, use_fast=False )
        self.model = AutoModelForCausalLM.from_pretrained( self.config.koala.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )
        bittensor.logging.info( 'Model loaded!' )

        if self.config.koala.device != "cpu":
            self.model = self.model.to( self.config.koala.device )

    def load_tokenizer( self ):
        return AutoTokenizer.from_pretrained( self.config.koala.model_name, use_fast=False )
    
    def load_model( self ):
        return AutoModelForCausalLM.from_pretrained( self.config.koala.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )

    def forward( self, messages: List[Dict[str, str]] ) -> str:
        history = self._process_history( messages )
        prompt = history + self.system_label
        input_ids = self.tokenizer.encode( prompt, return_tensors="pt" ).to( self.config.koala.device )
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.koala.max_new_tokens,
            temperature=self.config.koala.temperature,
            do_sample=self.config.koala.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generation = self.tokenizer.decode( output[0][input_ids.shape[1]:], skip_special_tokens=True )

        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    KoalaMiner().run()