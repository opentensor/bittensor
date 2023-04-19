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
import torch
import bittensor
from typing import List, Dict, Union, Tuple

bittensor.logging( bittensor.logging.config() )

class Synapse( bittensor.TextPromptingSynapse ):
    def priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
        return 0.0

    def blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> Union[ Tuple[bool, str], bool ]:
        return False

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str:
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:
        unravelled_message = ''
        roles = []; contents = []
        for message_dict in messages:
            message_dict = json.loads( message_dict )
            item_role = message_dict['role']
            item_content = message_dict['content']
            bittensor.logging.success(str(message_dict))
            roles.append( item_role )
            contents.append( item_content )
            if item_role == 'system': unravelled_message += 'system: ' + item_content + '\n'
            if item_role == 'assistant': unravelled_message += 'assistant: ' + item_content + '\n'
            if item_role == 'user': unravelled_message += 'user: ' + item_content + '\n'
        print('unrav', unravelled_message)
        print ('roles', roles)
        print ('contents', contents)
        return "hello im a chat bot."

# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1", metagraph = None )
dendrite = bittensor.text_prompting( axon = axon.info(), keypair = wallet.hotkey )
synapse = Synapse( axon = axon )
axon.start()

bittensor.logging.debug( "Start example")
forward_call = dendrite.forward(
    roles = ['system', 'assistant'],
    messages = ['you are chat bot', 'what is the whether'],
    timeout = 1e6
)
print ( forward_call )
backward_call = forward_call.backward( 1 )
print ( backward_call )