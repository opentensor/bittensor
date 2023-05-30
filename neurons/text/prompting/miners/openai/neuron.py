# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
import openai
import argparse
import bittensor
from typing import List, Dict

class OpenAIMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.openai.api_key != None, 'the miner requires passing --openai.api_key as an argument of the config.'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--openai.api_key', type=str, help='openai api key')
        parser.add_argument('--openai.suffix', type=str, default=None, help="The suffix that comes after a completion of inserted text.")
        parser.add_argument('--openai.max_tokens', type=int, default=256, help="The maximum number of tokens to generate in the completion.")
        parser.add_argument('--openai.temperature', type=float, default=0.7, help="Sampling temperature to use, between 0 and 2.")
        parser.add_argument('--openai.top_p', type=float, default=1, help="Nucleus sampling parameter, top_p probability mass.")
        parser.add_argument('--openai.n', type=int, default=1, help="How many completions to generate for each prompt.")
        parser.add_argument('--openai.presence_penalty', type=float, default=0, help="Penalty for tokens based on their presence in the text so far.")
        parser.add_argument('--openai.frequency_penalty', type=float, default=0, help="Penalty for tokens based on their frequency in the text so far.")
        parser.add_argument('--openai.model_name', type=str, default='gpt-3.5-turbo', help="OpenAI model to use for completion.")

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass

    def __init__( self ):
        super( OpenAIMiner, self ).__init__()
        print ( self.config )
        openai.api_key = self.config.openai.api_key

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        resp = openai.ChatCompletion.create(
            model = self.config.openai.model_name,
            messages = messages,
            temperature = self.config.openai.temperature,
            max_tokens = self.config.openai.max_tokens,
            top_p = self.config.openai.top_p,
            frequency_penalty = self.config.openai.frequency_penalty,
            presence_penalty = self.config.openai.presence_penalty,
            n = self.config.openai.n,
        )['choices'][0]['message']['content']
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    OpenAIMiner().run()