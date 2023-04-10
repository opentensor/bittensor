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

import json
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
        parser.add_argument('--pythia.device', type=str, help='Device to load model', default="cuda")
        parser.add_argument('--pythia.max_new_tokens', type=int, help='Max tokens for model output.', default=256)
        parser.add_argument('--pythia.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).')
        parser.add_argument('--pythia.temperature', type=float, help='Sampling temperature of model', default=0.8)
        
    def __init__( self ):
        super( PythiaMiner, self ).__init__()
        print ( self.config )
        model_config = {
            'model': AutoModelForCausalLM.from_pretrained("togethercomputer/Pythia-Chat-Base-7B"),
            'tokenizer': AutoTokenizer.from_pretrained("togethercomputer/Pythia-Chat-Base-7B", torch_dtype=torch.float16),
            'device':0,
        }
        pipe_config = {**model_config, **self.config.pythia}
        pipe_config['model'] = pipe_config['model'].to( self.config.pythia.device )
        self.pipe = pipeline("text-generation", **pipe_config)

    def priority( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> bool:
        return False
    
    @staticmethod
    def _process_history(history: List[str]) -> str:
        processed_history = ''
        for message in history:
            message = json.loads(message)
            if message['role'] == 'system':
                processed_history += 'system: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += 'assistant: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += 'user: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self._process_history(messages)
        return self.pipe( history )[0]['generated_text'].split(':')[-1].replace( str( history ), "") 

if __name__ == "__main__":
    bittensor.utils.version_checking()
    PythiaMiner().run()
