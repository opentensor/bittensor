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

import argparse
import bittensor
from rich import print
from typing import List, Dict

# Torch tooling.
from langchain.llms import AlephAlpha


class AlephAlphaMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.aleph.api_key != None, 'the miner requires passing --aleph.api_key as an argument of the config.'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--aleph.api_key', type=str, help='AlephAlpha API key.', required=True)
        parser.add_argument('--aleph.model', type=str, help='Model name to use.', default='luminous-base')
        parser.add_argument('--aleph.maximum_tokens', type=int, help='The maximum number of tokens to be generated.', default=64)
        parser.add_argument('--aleph.temperature', type=float, help='A non-negative float that tunes the degree of randomness in generation.', default=0.0)
        parser.add_argument('--aleph.stop_sequences', type=List[str], help='Stop tokens.', default=['user: ', 'bot: ', 'system: '])
        parser.add_argument('--aleph.top_k', type=int, help='Number of most likely tokens to consider at each step.', default=0)
        parser.add_argument('--aleph.top_p', type=float, help='Total probability mass of tokens to consider at each step.', default=0.0)
    
    def __init__( self ):
        super( AlephAlphaMiner, self ).__init__()
        print ( self.config )
        
        self.model = AlephAlpha(
            aleph_alpha_api_key = self.config.aleph.api_key,
            model = self.config.aleph.model,
            maximum_tokens = self.config.aleph.maximum_tokens,
            temperature = self.config.aleph.temperature,
            top_k = self.config.aleph.top_k,
            top_p = self.config.aleph.top_p,
            stop_sequences = self.config.aleph.stop_sequences
        )

    def priority( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> bool:
        return False

    @staticmethod
    def _process_history( history:  List[Dict[str, str]] ) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += 'system: ' + message['content'] + '\n'
            if message['role'] == 'assistant':
                processed_history += 'assistant: ' + message['content'] + '\n'
            if message['role'] == 'user':
                processed_history += 'user: ' + message['content'] + '\n'
        return processed_history

    def forward( self, messages:  List[Dict[str, str]] ) -> str:
        bittensor.logging.info('messages', str(messages))
        history = self._process_history(messages)
        bittensor.logging.info('history', str(history))
        resp = self.model(history)
        bittensor.logging.info('response', str(resp))
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    AlephAlphaMiner().run()