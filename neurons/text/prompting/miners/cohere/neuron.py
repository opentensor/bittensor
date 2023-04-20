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
import json
import argparse
import bittensor
from typing import List, Dict
from langchain.llms import Cohere

class CohereMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.cohere.api_key != None, 'the miner requires passing --cohere.api_key as an argument of the config.'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--cohere.model_name', type=str, help='Name of the model.', default='command-xlarge-nightly')
        parser.add_argument('--cohere.max_tokens', type=int, help='Number of tokens to generate.', default=256)
        parser.add_argument('--cohere.temperature', type=float, help='Temperature of generation.', default=0.75)
        parser.add_argument('--cohere.k', type=int, help='Number of most likely tokens to consider at each step.', default=0)
        parser.add_argument('--cohere.p', type=int, help='Total probability mass of tokens to consider at each step.', default=1)
        parser.add_argument('--cohere.frequency_penalty', type=float, help='Penalizes repeated tokens according to frequency.', default=0.0)
        parser.add_argument('--cohere.presence_penalty', type=float, help='Penalizes repeated tokens.', default=0.0)
        parser.add_argument('--cohere.truncate', type=str, help='Specify how the client handles inputs longer than the maximum token length: Truncate from START, END or NONE', default=None)
        parser.add_argument('--cohere.stop', type=str, help='List of tokens to stop generation on.', default=None)
        parser.add_argument('--cohere.api_key', type=str, help='API key for Cohere.', required=True)

    def __init__( self ):
        super( CohereMiner, self ).__init__()
        print ( self.config )

        self.model = Cohere(
            model=self.config.cohere.model_name,
            cohere_api_key=self.config.cohere.api_key,
            max_tokens=self.config.cohere.max_tokens,
            temperature=self.config.cohere.temperature,
            k=self.config.cohere.k,
            p=self.config.cohere.p,
            frequency_penalty=self.config.cohere.frequency_penalty,
            presence_penalty=self.config.cohere.presence_penalty,
            truncate=self.config.cohere.truncate,
            stop=self.config.cohere.stop,
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
        history = self._process_history( messages )
        return self.model( history )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    CohereMiner().run()
