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

# General.
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class CerebrasMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--cerebras.device', type=str, help='Device to load model', default="cuda")
        parser.add_argument('--cerebras.max_length', type=int, default=50, help='The maximum length (in tokens) of the generated text.')
        parser.add_argument('--cerebras.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).')
        parser.add_argument('--cerebras.no_repeat_ngram_size', type=int, default=2, help='The size of the n-grams to avoid repeating in the generated text.')
        parser.add_argument('--cerebras.model_size', type=str, choices=['1.3B', '2.7B', '6.7B', '13B'], default="1.3B", help='Model size to use.')

    def __init__( self ):
        super( CerebrasMiner, self ).__init__()
        print ( self.config )

        bittensor.logging.info( "Loading Cerebras GPT {} model...".format( self.config.cerebras.model_size) )
        model = AutoModelForCausalLM.from_pretrained( "cerebras/Cerebras-GPT-{}".format( self.config.cerebras.model_size) )
        tokenizer = AutoTokenizer.from_pretrained( "cerebras/Cerebras-GPT-{}".format( self.config.cerebras.model_size)  )

        self.pipe = pipeline( 
            "text-generation", 
            model = model, 
            tokenizer = tokenizer, 
            device = 0, 
            do_sample = False, 
            max_new_tokens = self.config.cerebras.max_length,
            no_repeat_ngram_size = self.config.cerebras.no_repeat_ngram_size
        )

    @staticmethod
    def _process_history( history: List[Dict[str, str]] ) -> str:
        processed_history = ''
        for message in history:
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
    CerebrasMiner().run()
