
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

__context_prompting_llm = None
def prompt( 
        prompt: Union[ str, Tuple[ str, str], Tuple[List[str], List[str] ], List[ str ], Dict[ str, str ], List[ Dict[ str ,str ] ] ],
        wallet_name: str = "default",
        hotkey: str = default_prompting_validator_key,
        subtensor_: Optional['bittensor.subtensor'] = None,
        axon_: Optional['bittensor.axon_info'] = None,
        return_all: bool = False,
    ) -> str:
    global __context_prompting_llm
    if __context_prompting_llm == None:
        __context_prompting_llm = prompting( 
            wallet_name = wallet_name,
            hotkey = hotkey,
            subtensor_ = subtensor_,
            axon_ = axon_,
        )
    return __context_prompting_llm( prompt = prompt, return_all = return_all )

class prompting ( torch.nn.Module ):
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
        super(prompting, self).__init__()
        self._hotkey = hotkey
        self._subtensor = bittensor.subtensor() if subtensor_ is None else subtensor_
        if use_coldkey:
            self._keypair = bittensor.wallet( name = wallet_name ).create_if_non_existent().coldkey
        else:
            self._keypair = bittensor.wallet( name = wallet_name ).create_if_non_existent().hotkey
        
        if axon_ is not None:
            self._axon = axon_
        else:
            self._metagraph = bittensor.metagraph( 1 )
            self._axon = self._metagraph.axons[ self._metagraph.hotkeys.index( self._hotkey ) ]
        self._dendrite = bittensor.text_prompting(
            keypair = self._keypair,
            axon = self._axon
        )
    def forward( 
            self,
            prompt: Union[ str, Tuple[ str, str], Tuple[List[str], List[str] ], List[ str ], Dict[ str, str ], List[ Dict[ str ,str ] ] ],
            timeout: float = 24,
            return_all: bool = False,
        ) -> Union[str, List[str]]:
        if not return_all:
            return self._dendrite.forward(
                prompt = prompt,
                timeout = timeout,
                return_call = True
            ).completion
        else:
            return self._dendrite.multi_forward(
                prompt = prompt,
                timeout = timeout,
                return_call = True
            ).multi_completions

       
    async def async_forward( 
            self,
            prompt: Union[ str, Tuple[ str, str], Tuple[List[str], List[str] ], List[ str ], Dict[ str, str ], List[ Dict[ str ,str ] ] ],
            timeout: float = 24,
            return_all: bool = False,
        ) -> Union[str, List[str]]:
        if not return_all:
            return await self._dendrite.async_forward(
                    prompt = prompt,
                    timeout = timeout,
                ).completion
        else:
            return self._dendrite.async_multi_forward(
                prompt = prompt,
                timeout = timeout,
            ).multi_completions

class BittensorLLM(LLM):
    """Wrapper around Bittensor Prompting Subnetwork. 
This Python file implements the BittensorLLM class, a wrapper around the Bittensor Prompting Subnetwork for easy integration into language models. The class provides a query method to receive responses from the subnetwork for a given user message and an implementation of the _call method to return the best response. The class can be initialized with various parameters such as the wallet name and chain endpoint.
    
    Example:
        .. code-block:: python

            from bittensor import BittensorLLM
            btllm = BittensorLLM(wallet_name="default")
    """

    wallet_name: str = 'default'
    hotkey: str = default_prompting_validator_key
    llm: prompting = None
    def __init__(self, subtensor_: Optional['bittensor.Subtensor'] = None, axon_: Optional['bittensor.axon_info'] = None, **data):
        super().__init__(**data)
        self.llm = prompting(wallet_name=self.wallet_name, hotkey=self.hotkey, subtensor_=subtensor_, axon_=axon_ )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"wallet_name": self.wallet_name, "hotkey_name": self.hotkey}

    @property
    def _llm_type(self) -> str:
        return "BittensorLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM with the given prompt and stop tokens."""
        return self.llm(prompt)
