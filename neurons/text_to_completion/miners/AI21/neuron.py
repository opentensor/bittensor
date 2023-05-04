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

from typing import List, Dict, Union, Tuple
from langchain.llms import AI21

def config():       
    parser = argparse.ArgumentParser( description = 'Text to Speech Miner' )
    parser.add_argument('--ai21.api_key', type=str, help='AI21 API key.', required=True)
    parser.add_argument('--ai21.model_name', type=str, help='Name of the model.', default='j2-jumbo-instruct')
    parser.add_argument('--ai21.stop', help='Stop tokens.', default=['user: ', 'bot: ', 'system: '])
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):

    assert config.ai21.api_key != None, 'the miner requires passing --ai21.api_key as an argument of the config.'

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( config = config )

    # --- Build A121 ---
    bittensor.logging.info( 'Loading AI21 Model...' )
    model = AI21( 
        model = config.ai21.model_name, 
        ai21_api_key = config.ai21.api_key, 
        stop = config.ai21.stop
    )
    bittensor.logging.info( 'Model loaded!' )

    # --- Build Synapse ---
    class AI21Synapse( bittensor.TextPromptingSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass
        
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
            history = self._process_history(messages)
            resp = model( history )
            return resp
        
    # --- Attach the synapse to the miner ----
    base_miner.axon.attach( AI21Synapse() )

    # --- Run Miner ----
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )






