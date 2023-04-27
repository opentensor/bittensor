
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
import grpc
import json
import torch
import asyncio
import bittensor
from typing import Callable, List, Dict, Union

class TextPromptingDendritePool( torch.nn.Module ):

    def __init__(
            self, 
            keypair: Union[ 'bittensor.Wallet', 'bittensor.Keypair'],
            metagraph: 'bittensor.metagraph', 
        ):
        super(TextPromptingDendritePool, self).__init__()
        self.metagraph = metagraph
        self.keypair = keypair
        self.dendrites = [ bittensor.text_prompting( axon = axon, keypair = self.keypair ) for axon in self.metagraph.axons ]
        self.loop = asyncio.get_event_loop()
        self.priority_threadpool = bittensor.prioritythreadpool(max_workers = 1)

    def backward( self,
            forward_calls: List[ 'DendriteForwardCall' ],
            rewards: Union[ List[ float ], torch.FloatTensor ],
            timeout: float = 12.0,
            priority: int = 1,
        ):
        def _backward():
            self.loop.run_until_complete( 
                self.async_backward (
                    forward_calls = forward_calls,
                    timeout = timeout,
                ) 
            )
        future = self.priority_threadpool.submit(
            _backward,
            priority = priority
        )
        return future.result()
        

    async def async_backward(self,
            forward_calls: List[ 'DendriteForwardCall' ],
            rewards: Union[ List[ float ], torch.FloatTensor ] ,
            timeout: float = 12.0
        ):
        rewards = rewards if not isinstance( rewards, torch.Tensor ) else rewards.tolist()
        async def query():
            coroutines = [ forward_calls.async_backward( reward )for call, reward in list(zip( forward_calls, rewards )) ]                
            all_responses = await asyncio.gather( *coroutines )
            return all_responses
        await query()

    def forward( 
            self, 
            roles: Union[ str, List[str] ], 
            messages: Union[ str, List[str] ],
            uids: Union[ torch.LongTensor, List[int] ] = None, 
            return_call:bool = True,
            timeout: float = 12,
            priority: int = 1,
        ) -> List['DendriteForwardCall']:
        def _forward():
            bittensor.logging.trace( 'dendrite pool: forward: _forward: start')
            return self.loop.run_until_complete(
                self.async_forward (
                    messages = messages,
                    roles = roles,
                    uids = uids,
                    return_call = return_call,
                    timeout = timeout,
                ) 
            )
        future = self.priority_threadpool.submit(
            _forward,
            priority = priority
        )
        return future.result()

    async def async_forward( 
            self, 
            roles: Union[ str, List[str] ],
            messages: Union[ str, List[str] ],
            uids: Union[ torch.LongTensor, List[int] ] = None, 
            return_call:bool = True,
            timeout: float = 12 
        ) -> List['DendriteForwardCall']:      
        # We optionally set the uids to all if uids is None.
        if uids is None: uids = range( self.metagraph.n.item() )
        if isinstance( uids, torch.Tensor ): uids = uids.tolist()
        # The following asyncio defintion queries a single endpoint with the message
        # prompt and returns the response.
        async def call_single_uid( uid: int ) -> str:
            return await self.dendrites[uid].async_forward( 
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