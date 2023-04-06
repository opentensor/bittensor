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
import asyncio
import bittensor
from typing import Callable, List, Dict, Union

class TextPromptingDendritePool( torch.nn.Module ):

    def __init__(
            self, 
            metagraph: 'bittensor.metagraph', 
            wallet: 'bittensor.wallet'
        ):
        super(TextPromptingDendritePool, self).__init__()
        self.metagraph = metagraph
        self.wallet = wallet

    def backward(self,
            message: str,
            completions: List[ str ],
            rewards: Union[ List[torch.LongTensor], List[List[float]] ],
            uids: Union[torch.LongTensor, List[int]],
            prompt: str = None,
            return_call:bool = True,
            timeout: float = 12.0
        ):
        """Sends a forward query to the networks and returns rewards to miners.

        Args:
            message (:obj:`str`, `required`):
                The message used on the forward to query the networks.
            completions (:obj:`List[str]`, `required`):
                The completions from the prompt on the forward call.
            rewards (:obj:`Union[List[torch.LongTensor], List[List[float]], List[float]]`, `optional`):
                The rewards from the forward call based on the reward model.
            uids (:obj:`Union[torch.LongTensor, List[int]]`, `optional`):
                The uids from the forward call.
            prompt (:obj:`str`, `optional`, defaults to `None`):
                The prompt message used to query the network with.
            return_call (bool): 
                The result is the full forward query object.
            timeout (:obj:`float`, `optional`, defaults to `12.0`):
                The timeout for the query.
        Returns:
            None
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( 
            self.async_backward (
                message = message,
                completions = completions,
                rewards = rewards,
                uids = uids,
                prompt = prompt,
                return_call = return_call,
                timeout = timeout,
            ) 
        )

    async def async_backward(self,
            message: str,
            completions: List[ str ],
            rewards: Union[ List[torch.LongTensor], List[List[float]] ],
            uids: Union[torch.LongTensor, List[int]],
            prompt: str = None,
            return_call:bool = True,
            timeout: float = 12.0
        ):
        """Sends a forward query to the networks and returns rewards to miners.

        Args:
            message (:obj:`str`, `required`):
                The message used on the forward to query the networks.
            completions (:obj:`List[str]`, `required`):
                The completions from the prompt on the forward call.
            rewards (:obj:`Union[List[torch.FloatTensor], torch.FloatTensor List[List[float]], List[float]]`, `optional`):
                The rewards from the forward call based on the reward model.
            uids (:obj:`Union[torch.LongTensor, List[int]]`, `optional`):
                The uids from the forward call.
            prompt (:obj:`str`, `optional`, defaults to `None`):
                The prompt message used to query the network with.
            return_call (bool): 
                The result is the full forward query object.
            timeout (:obj:`float`, `optional`, defaults to `12.0`):
                The timeout for the query.
        Returns:
            None
        """
        if isinstance(uids, torch.Tensor):
            uids = [ int(el) for el in uids.tolist() ]  
        formatted_rewards = []
        if isinstance( rewards, torch.Tensor ): 
            formatted_rewards = [ float(el) for el in rewards.tolist() ]
        else:
            for element in rewards:
                # Rewards are a list of floats
                if isinstance( element, float ): 
                    formatted_rewards = rewards
                    break
                # Rewards are a list of tensors.
                if isinstance( element, torch.Tensor ): 
                    element = element.tolist()
                formatted_rewards.append( [ float(el) for el in element.tolist() ])

        # We optionally set the prompt to the message if prompt is None.
        if prompt is not None: 
            roles = ['system', 'user']
            messages = [ prompt, message ]
        else:
            roles = ['user']
            messages = [ message ]

        # The following asyncio defintion queries a single endpoint with the message
        # prompt and returns the response.
        async def call_single_uid( index:int, uid: int ) -> str:
            module = bittensor.text_prompting( endpoint = self.metagraph.endpoint_objs[ uid ], wallet = self.wallet )
            return await module.async_backward( 
                roles = roles, 
                messages = messages,
                response = completions[ index ],
                rewards = formatted_rewards[ index ],
                return_call = return_call,
                timeout = timeout 
            )
        
        # The following asyncio definition gathers the responses
        # from multiple coroutines for each uid.
        async def query():
            coroutines = [ call_single_uid( index, uid ) for index, uid in enumerate( uids ) ]                
            all_responses = await asyncio.gather(*coroutines)
            return all_responses
        
        return await query()

    def forward( 
            self, 
            message: str, 
            prompt: str = None,
            uids: Union[ torch.LongTensor, List[int] ] = None, 
            return_call:bool = True,
            timeout: float = 12 
        ) -> List['bittensor.TextPromptingForwardCall']:
        r""" Queries uids on the network for a response to the passed message.
        Args:
            message (str): The message to query the network with.
            uids (List[int]): The uids to query. If None, queries all uids.
            return_call (bool): The result is the full forward query object.
            timeout (float): The timeout for the query.
        Returns:
            responses (List['bittensor.TextPromptingForwardCall']): 
                The responses from the forward call.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( 
            self.async_forward (
                message = message,
                prompt = prompt,
                uids = uids,
                return_call = return_call,
                timeout = timeout,
            ) 
        )

    async def async_forward( 
            self, 
            message: str, 
            prompt: str = None,
            uids: Union[ torch.LongTensor, List[int] ] = None, 
            return_call:bool = True,
            timeout: float = 12 
        ) -> List['bittensor.TextPromptingForwardCall']:
        r""" Queries uids on the network for a response to the passed message.
        Args:
            message (str): The message to query the network with.
            uids (List[int]): The uids to query. If None, queries all uids.
            return_call (bool): The result is the full forward query object.
            timeout (float): The timeout for the query.
        Returns:
            responses (List['bittensor.TextPromptingForwardCall']): 
                The responses from the forward call.
        """
        # We optionally set the uids to all if uids is None.
        if uids is None: uids = range( len( self.dendrites ))
        if isinstance( uids, torch.Tensor ): uids = uids.tolist()

        # We optionally set the prompt to the message if prompt is None.
        if prompt is not None: 
            roles = ['system', 'user']
            messages = [ prompt, message ]
        else:
            roles = ['user']
            messages = [ message ]

        # The following asyncio defintion queries a single endpoint with the message
        # prompt and returns the response.
        async def call_single_uid( uid: int ) -> str:
            module = bittensor.text_prompting( endpoint = self.metagraph.endpoint_objs[ uid ], wallet = self.wallet )
            return await module.async_forward( 
                roles = roles, 
                messages = messages,
                return_call = return_call, 
                timeout = timeout 
            )
        
        # The following asyncio definition gathers the responses
        # from multiple coroutines for each uid.
        async def query():
            coroutines = [ call_single_uid( uid ) for uid in uids ]                
            all_responses = await asyncio.gather(*coroutines)
            return all_responses
        
        return await query() 

class TextPromptingDendrite(bittensor.Dendrite):
    """Dendrite for the text_prompting synapse."""

    # Dendrite name.
    name: str = "text_prompting"

    def __str__(self) -> str:
        return "TextPrompting"

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextPromptingStub(channel)

    def pre_process_forward_call_to_request_proto(
        self, forward_call: "bittensor.TextPromptingForwardCall"
    ) -> "bittensor.ForwardTextPromptingRequest":
        return bittensor.ForwardTextPromptingRequest( timeout = forward_call.timeout, messages = forward_call.messages )

    def pre_process_backward_call_to_request_proto( 
            self, backward_call: 'bittensor.TextPromptingBackwardCall' 
    ) -> 'bittensor.BackwardTextPromptingRequest':
        return bittensor.BackwardTextPromptingRequest( 
            messages = backward_call.messages, 
            response = backward_call.response,
            rewards = backward_call.rewards 
        )

    def post_process_response_proto_to_forward_call(
        self,
        forward_call: bittensor.TextPromptingForwardCall,
        response_proto: bittensor.ForwardTextPromptingResponse,
    ) -> bittensor.TextPromptingForwardCall:
        forward_call.response_code = response_proto.return_code
        forward_call.response_message = response_proto.message
        forward_call.response = response_proto.response
        return forward_call

    def forward(
            self,
            roles: List[ str ] ,
            messages: List[ str ],
            return_call:bool = True,
            timeout: float = bittensor.__blocktime__,
        ) -> "bittensor.TextPromptingForwardCall":
        forward_call=bittensor.TextPromptingForwardCall(
            messages=[json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)],
            timeout=timeout,
        )
        loop = asyncio.get_event_loop()
        response_call = loop.run_until_complete( self._async_forward( forward_call = forward_call ) )
        if return_call: return response_call
        else: return response_call.response
    
    async def async_forward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        return_call: bool = True,
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingForwardCall":
        forward_call=bittensor.TextPromptingForwardCall(
            messages = [json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)],
            timeout = timeout,
        )
        response_call = await self._async_forward( forward_call = forward_call )
        if return_call: return response_call
        else: return response_call.response

    def backward(
            self,
            roles: List[ str ],
            messages: List[ str ],
            response: str,
            rewards: Union[ List[ float], torch.FloatTensor ],
            return_call: bool = True,
            timeout: float = bittensor.__blocktime__,
        ) -> "bittensor.TextPromptingBackwardCall":
        if isinstance( rewards, torch.FloatTensor ): rewards = rewards.tolist()
        backward_call = bittensor.TextPromptingBackwardCall(
            messages = [json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)],
            response = response,
            rewards = rewards
        )
        loop = asyncio.get_event_loop()
        response_call =  loop.run_until_complete( self._async_backward( backward_call = backward_call ) )
        if return_call: return response_call
        else: return response_call.response

    async def async_backward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        response: str,        
        rewards: Union[ List[ float], torch.FloatTensor ],
        return_call: bool = True,
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingBackwardCall":
        if isinstance( rewards, torch.FloatTensor ): rewards = rewards.tolist()
        backward_call = bittensor.TextPromptingBackwardCall(
            messages = [json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)],
            response = response,
            rewards = rewards
        )
        response_call = await self._async_backward( backward_call = backward_call ) 
        if return_call: return response_call
        else: return response_call.response




