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
from langchain.llms import AI21

class AI21Miner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.ai21.api_key != None, 'the miner requires passing --ai21.api_key as an argument of the config.'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--ai21.model_name', type=str, help='Name of the model.', default='j2-jumbo-instruct')
        parser.add_argument('--ai21.stop', help='Stop tokens.', default=['user: ', 'bot: ', 'system: '])
        parser.add_argument('--ai21.api_key', type=str, help='AI21 API key.', default=None)
        
    def __init__( self ):
        super( AI21Miner, self ).__init__()
        print ( self.config )
        self.model = AI21( 
            model = self.config.ai21.model_name, 
            ai21_api_key = self.config.ai21.api_key, 
            stop = self.config.ai21.stop
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

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        bittensor.logging.info('messages', str(messages))
        history = self._process_history(messages)
        resp = self.model(history)
        bittensor.logging.info('response', str(resp))
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    AI21Miner().run()
