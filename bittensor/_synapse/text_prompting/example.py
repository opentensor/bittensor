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
from typing import Union
from typing import List, Dict
bittensor.logging( debug = True )

class Synapse( bittensor.TextPromptingSynapse ):

    def __init__( self, config: "bittensor.Config" = None ):
        super().__init__( config )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        pass

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def priority( self, request: Union[ bittensor.ForwardTextPromptingRequest, bittensor.BackwardTextPromptingRequest ] ) -> float:
        return 0.0

    def blacklist( self, request: Union[ bittensor.ForwardTextPromptingRequest, bittensor.BackwardTextPromptingRequest ] ) -> bool:
        return False

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ):
        pass

    def forward( self, messages: List[ Dict[ str, str ] ] ) -> str:
        return "hello im a chat bot."
    

config = Synapse.config()
config.axon.port = 9090
config.axon.ip = "127.0.0.1"
print ( config )
syn = Synapse( config = config )
print ( config )


wallet = bittensor.wallet().create_if_non_existent()
local_endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = "127.0.0.1",
    ip_type = 4,
    port = 9090,
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkeypub.ss58_address,
    modality = 0,
)
module = bittensor.text_prompting( endpoint = local_endpoint, wallet = wallet )
forward_response = module.forward(
    roles = ['user', 'assistant'],
    messages = [{ "user": "Human", "content": "hello"}],
    timeout = 1e6
)



