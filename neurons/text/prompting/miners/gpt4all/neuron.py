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
from typing import List, Dict
from langchain.llms import GPT4All

class GPT4ALLMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ): pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--gpt4all.model', type=str, help='Path to pretrained gpt4all model in ggml format.', required=True)
        parser.add_argument('--gpt4all.n_ctx', type=int, default=512, help='Token context window.')
        parser.add_argument('--gpt4all.n_parts', type=int, default=-1, help='Number of parts to split the model into. If -1, the number of parts is automatically determined.')
        parser.add_argument('--gpt4all.seed', type=int, default=0, help='Seed. If -1, a random seed is used.')
        parser.add_argument('--gpt4all.f16_kv', action='store_true', default=False, help='Use half-precision for key/value cache.')
        parser.add_argument('--gpt4all.logits_all', action='store_true', default=False, help='Return logits for all tokens, not just the last token.')
        parser.add_argument('--gpt4all.vocab_only', action='store_true', default=False, help='Only load the vocabulary, no weights.')
        parser.add_argument('--gpt4all.use_mlock', action='store_true', default=False, help='Force system to keep model in RAM.')
        parser.add_argument('--gpt4all.embedding', action='store_true', default=False, help='Use embedding mode only.')
        parser.add_argument('--gpt4all.n_threads', type=int, default=4, help='Number of threads to use.')
        parser.add_argument('--gpt4all.n_predict', type=int, default=256, help='The maximum number of tokens to generate.')
        parser.add_argument('--gpt4all.temp', type=float, default=0.8, help='The temperature to use for sampling.')
        parser.add_argument('--gpt4all.top_p', type=float, default=0.95, help='The top-p value to use for sampling.')
        parser.add_argument('--gpt4all.top_k', type=int, default=40, help='The top-k value to use for sampling.')
        parser.add_argument('--gpt4all.echo', action='store_true', default=False, help='Whether to echo the prompt.')
        parser.add_argument('--gpt4all.repeat_last_n', type=int, default=64, help='Last n tokens to penalize.')
        parser.add_argument('--gpt4all.repeat_penalty', type=float, default=1.3, help='The penalty to apply to repeated tokens.')
        parser.add_argument('--gpt4all.n_batch', type=int, default=1, help='Batch size for prompt processing.')
        parser.add_argument('--gpt4all.streaming', action='store_true', default=False, help='Whether to stream the results or not.')
        
    def __init__( self ):
        super( GPT4ALLMiner, self ).__init__()
        print ( self.config )
        self.model = GPT4All(
            model=self.config.gpt4all.model,
            n_ctx=self.config.gpt4all.n_ctx,
            n_parts=self.config.gpt4all.n_parts,
            seed=self.config.gpt4all.seed,
            f16_kv=self.config.gpt4all.f16_kv,
            logits_all=self.config.gpt4all.logits_all,
            vocab_only=self.config.gpt4all.vocab_only,
            use_mlock=self.config.gpt4all.use_mlock,
            embedding=self.config.gpt4all.embedding,
            n_threads=self.config.gpt4all.n_threads,
            n_predict=self.config.gpt4all.n_predict,
            temp=self.config.gpt4all.temp,
            top_p=self.config.gpt4all.top_p,
            top_k=self.config.gpt4all.top_k,
            echo=self.config.gpt4all.echo,
            stop=['user: ', 'bot: ', 'system: '],
            repeat_last_n=self.config.gpt4all.repeat_last_n,
            repeat_penalty=self.config.gpt4all.repeat_penalty,
            n_batch=self.config.gpt4all.n_batch,
            streaming=self.config.gpt4all.streaming,
        )
    
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
    GPT4ALLMiner().run()
