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
from typing import List, Dict, Any, Optional
from langchain.llms import GooseAI

class GooseAIMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ): pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument("--gooseai.api_key", type=str, required=True, help="GooseAI api key required.")
        parser.add_argument("--gooseai.model_name", type=str, default="gpt-neo-20b", help="Model name to use")
        parser.add_argument("--gooseai.temperature", type=float, default=0.7, help="What sampling temperature to use")
        parser.add_argument("--gooseai.max_tokens", type=int, default=256, help="The maximum number of tokens to generate in the completion")
        parser.add_argument("--gooseai.top_p", type=float, default=1, help="Total probability mass of tokens to consider at each step")
        parser.add_argument("--gooseai.min_tokens", type=int, default=1, help="The minimum number of tokens to generate in the completion")
        parser.add_argument("--gooseai.frequency_penalty", type=float, default=0, help="Penalizes repeated tokens according to frequency")
        parser.add_argument("--gooseai.presence_penalty", type=float, default=0, help="Penalizes repeated tokens")
        parser.add_argument("--gooseai.n", type=int, default=1, help="How many completions to generate for each prompt")
        parser.add_argument("--gooseai.model_kwargs", type=Dict[str, Any], default=dict(), help="Holds any model parameters valid for `create` call not explicitly specified")
        parser.add_argument("--gooseai.logit_bias", type=Optional[Dict[str, float]], default=dict(), help="Adjust the probability of specific tokens being generated")

        
    def __init__( self ):
        super( GooseAIMiner, self ).__init__()
        print ( self.config )
        model_kwargs = {
            'model': self.config.gooseai.model_name,
            'n_ctx': self.config.gooseai.max_tokens,
            'n_parts': self.config.gooseai.n,
            'temp': self.config.gooseai.temperature,
            'top_p': self.config.gooseai.top_p,
            'repeat_penalty': self.config.gooseai.frequency_penalty,
        }
        self.model = GooseAI(gooseai_api_key=self.config.gooseai.api_key, model_kwargs=model_kwargs)

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass

    @staticmethod
    def _process_history(history: List[dict]) -> str:
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
        bittensor.logging.info( 'messages', str( messages ) )
        history = self._process_history( messages )
        bittensor.logging.info( 'history', str( history ) )
        resp = self.model( history )
        bittensor.logging.info('response', str( resp ))
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    GooseAIMiner().run()
