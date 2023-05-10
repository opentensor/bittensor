
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
import bittensor
from typing import Union, List, Dict
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any, Tuple

default_prompting_validator_key = '5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3'

# TODO: change this imagining to dream e.g. bt.dream( text = "hello world" )
# ASSUMING UID 13
class imagining ( torch.nn.Module ):
    _axon: 'bittensor.axon_info'
    _dendrite: 'bittensor.Dendrite'
    _subtensor: 'bittensor.Subtensor'
    _hotkey: str
    _keypair: 'bittensor.Keypair'

    def __init__(
        self,
        wallet_name: str = "default",
        hotkey: str = default_prompting_validator_key,
        subtensor_: Optional['bittensor.Subtensor'] = None,
        axon_: Optional['bittensor.axon_info'] = None,
        use_coldkey: bool = False
    ):
        super(imagining, self).__init__()
        self._hotkey = hotkey
        self._subtensor = bittensor.subtensor() if subtensor_ is None else subtensor_
        if use_coldkey:
            self._keypair = bittensor.wallet( name = wallet_name ).create_if_non_existent().coldkey
        else:
            self._keypair = bittensor.wallet( name = wallet_name ).create_if_non_existent().hotkey
        
        if axon_ is not None:
            self._axon = axon_
        else:
            self._metagraph = bittensor.metagraph( 13 )
            self._axon = self._metagraph.axons[ self._metagraph.hotkeys.index( self._hotkey ) ]
        self._text_to_image_dendrite = bittensor.text_to_image(
            keypair = self._keypair,
            axon = self._axon
        )

    def forward( 
            self,
            text: str,
            timeout: float = 24,
        ) -> Union[str, List[str]]:
        return self._text_to_image_dendrite.forward(
            text = text,
            timeout = timeout,
        ).image

__global_imagining = None
def imagine( 
        text: str, 
        wallet_name: str = "default",
        hotkey: str = default_prompting_validator_key,
        subtensor_: Optional['bittensor.subtensor'] = None,
        axon_: Optional['bittensor.axon_info'] = None,
    ) -> str:
    global __global_imagining
    if __global_imagining == None:
        __global_imagining = imagining( 
            wallet_name = wallet_name,
            hotkey = hotkey,
            subtensor_ = subtensor_,
            axon_ = axon_,
        )
    return __global_imagining( text = text )

