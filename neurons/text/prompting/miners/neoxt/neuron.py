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
from transformers import AutoTokenizer, AutoModelForCausalLM

class NeoxtMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--neoxt.model_name', type=str, help='Name/path of model to load', default="togethercomputer/GPT-NeoXT-Chat-Base-20B" )
        parser.add_argument( '--neoxt.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--neoxt.max_new_tokens', type=int, help='Max tokens for model output.', default=64 ) 
        parser.add_argument( '--neoxt.temperature', type=float, help='Sampling temperature of model', default=0.8 )
        parser.add_argument( '--neoxt.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        
    def __init__( self ):
        super( NeoxtMiner, self ).__init__()
        print ( self.config )
        
        bittensor.logging.info( 'Loading ' + str(self.config.neoxt.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.neoxt.model_name )
        self.model = AutoModelForCausalLM.from_pretrained( self.config.neoxt.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )
        bittensor.logging.info( 'Model loaded!' )

        if self.config.neoxt.device != "cpu":
            self.model = self.model.to( self.config.neoxt.device )


    @staticmethod
    def _process_history(history: List[str]) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += '<human>: ' + message['content'].strip() + '\n'
            if message['role'] == 'assistant':
                processed_history += '<bot>: ' + message['content'].strip() + '\n'
            if message['role'] == 'user':
                processed_history += '<human>: ' + message['content'].strip() + '\n'
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self._process_history(messages)
        prompt = history + "<bot>:"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.neoxt.device)

        output = self.model.generate(
        input_ids,
        max_length=input_ids.shape[1] + self.config.neoxt.max_new_tokens,
        temperature=self.config.neoxt.temperature,
        do_sample=self.config.neoxt.do_sample,
        pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        generation = generated_text.split("<human>")[0].strip()
        
        # Logging input and generation if debugging is active
        bittensor.logging.debug("Message: " + str(messages).replace("<","-").replace(">","-"))
        bittensor.logging.debug("Generation: " + str(generation).replace("<","-").replace(">","-"))
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    NeoxtMiner().run()
