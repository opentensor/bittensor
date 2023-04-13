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
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class PythiaMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--pythia.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--pythia.max_new_tokens', type=int, help='Max tokens for model output.', default=64 ) 
        parser.add_argument( '--pythia.temperature', type=float, help='Sampling temperature of model', default=0.8 )
        parser.add_argument( '--pythia.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        
    def __init__( self ):
        super( PythiaMiner, self ).__init__()
        print ( self.config )
        
        bittensor.logging.info( 'Loading togethercomputer/Pythia-Chat-Base-7B model...' )
        self.tokenizer = AutoTokenizer.from_pretrained( "togethercomputer/Pythia-Chat-Base-7B" )
        self.model = AutoModelForCausalLM.from_pretrained( "togethercomputer/Pythia-Chat-Base-7B", torch_dtype = torch.float16 )
        bittensor.logging.info( 'Model loaded!' )

        if self.config.pythia.device == "cuda":
            self.model = self.model.to( self.config.pythia.device )

        self.pipe = pipeline( 
            "text-generation",
            self.model, 
            tokenizer = self.tokenizer,
            max_new_tokens = self.config.pythia.max_new_tokens,
            temperature = self.config.pythia.temperature,
            do_sample = self.config.pythia.do_sample,
            device = 0,
        )
    
    @staticmethod
    def _process_history(history: List[str]) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += '<human>: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += '<bot>: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += '<human>: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self._process_history(messages)
        prompt = history + "<bot>:"
        generation = self.pipe( prompt )[0]['generated_text'].replace(prompt,"").split("<human>")[0].strip()
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    PythiaMiner().run()
