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
import time
import json
import torch
import asyncio
import bittensor
from typing import Callable, List, Dict, Union

class TextPromptingDendrite:
    """Dendrite for the text_prompting dendrite."""

    def __init__(
            self,
            wallet: 'bittensor.wallet',
            endpoint: Union[ 'bittensor.Endpoint', torch.Tensor ], 
        ):
        """ Initializes the Dendrite
            Args:
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet object.
                endpoint (:obj:Union[]`bittensor.endpoint`, `required`):
                    bittensor endpoint object.
        """
        self.wallet = wallet
        if isinstance( endpoint, torch.Tensor ): 
            endpoint = bittensor.endpoint.from_tensor( endpoint )
        self.endpoint = endpoint
        self.receptor = bittensor.receptor( wallet = self.wallet, endpoint = self.endpoint )

    #################
    #### Forward ####
    #################
    def forward(
            self,
            roles: List[ str ] ,
            messages: List[ str ],
            return_dict: bool = True,
            timeout: float = bittensor.__blocktime__,
        ) -> Union[ str, dict ]:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete( 
            self._call_forward( 
                roles = roles,
                messages = messages,
                timeout = timeout,
                return_dict = return_dict
            ) 
        )
        if return_dict: return result
        else: return result.response

    async def async_forward(
        self,
        roles: List[ str ],
        messages: List[ str ],
        return_dict: bool = True,
        timeout: float = bittensor.__blocktime__,
    ) -> Union[ str, dict ]:
        result = self._call_forward( 
                roles = roles,
                messages = messages,
                timeout = timeout,
                return_dict = return_dict
            ) 
        if return_dict: return result
        else: return result.response

    async def _call_forward( 
            self, 
            roles: List[ str ] ,
            messages: List[ str ],
            timeout: float = bittensor.__blocktime__, 
        ) -> dict:
        start_time = time.time()
        packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)]
        request_proto = bittensor.ForwardTextPromptingRequest( 
            messages = packed_messages, 
            timeout = timeout, 
            hotkey = self.wallet.hotkey.ss58_address,
            version = bittensor.__version_as_int__
        )
        asyncio_future = bittensor.grpc.TextPromptingStub( self.receptor.channel ).Forward(
            request = request_proto,
            timeout = timeout,
            metadata = (
                ('rpc-auth-header','Bittensor'),
                ('bittensor-signature', self.receptor.sign() ),
                ('bittensor-version', str( bittensor.__version_as_int__ ) ),
            ))
        bittensor.logging.rpc_log ( 
            axon = False, 
            forward = True, 
            is_response = False, 
            code = bittensor.proto.ReturnCode.Success, 
            call_time = time.time() - start_time, 
            pubkey = self.endpoint.hotkey, 
            uid = self.endpoint.uid, 
            inputs = torch.Size( [len(message) for message in packed_messages ] ),
            outputs = None,
            message = "Success",
            synapse = "text_prompting"
        )
        try:
            response_proto = await asyncio.wait_for( asyncio_future, timeout = timeout )
            bittensor.logging.rpc_log(
                axon = False, 
                forward = True, 
                is_response = True, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = time.time() - start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = torch.Size( [len(message) for message in packed_messages] ), 
                outputs = torch.Size([len( response_proto.response )]),
                message = "Success",
                synapse = "text_prompting",
            )
            return {
                "response": response_proto.response,
                "hotkey": self.endpoint.hotkey,
                "uid": self.endpoint.uid,
                "start_time": start_time,
                "end_time": time.time(),
                "code": bittensor.proto.ReturnCode.Success,
            }
            
        except grpc.RpcError as rpc_error_call:
            # Request failed with GRPC code.
            code = rpc_error_call.code()
            error_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
        except asyncio.TimeoutError:
            # Catch timeout errors.
            code = bittensor.proto.ReturnCode.Timeout
            error_message = 'GRPC request timeout after: {}s'.format( timeout )
        except Exception as e:
            # Catch unknown errors.
            code = bittensor.proto.ReturnCode.UnknownException
            error_message = str( e )
        finally:
            bittensor.logging.rpc_log(
                axon = False, 
                forward = True, 
                is_response = True, 
                code = code, 
                call_time = time.time() - start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = torch.Size( [len(message) for message in self.messages] ),
                outputs = None,
                message = error_message,
                synapse = "text_prompting",
            )
            return {
                "response": response_proto.response,
                "hotkey": self.endpoint.hotkey,
                "uid": self.endpoint.uid,
                "start_time": start_time,
                "end_time": time.time(),
                "code": code,
            }
    
    
        
    #################
    #### Backward ###
    #################
    def backward(
            self, 
            roles: List[ str ],
            messages: List[ str ],
            response: str,
            rewards: Union[ List[ float], torch.FloatTensor ],
            timeout: float = bittensor.__blocktime__
        ):
        loop = asyncio.get_event_loop()
        loop.run_until_complete( 
            self._call_backward( 
                roles = roles,
                messages = messages,
                response = response,
                rewards = rewards,
                timeout = timeout,
            ) 
        )

    async def async_backward(
            self,
            roles: List[ str ],
            messages: List[ str ],
            response: str,        
            rewards: Union[ List[ float], torch.FloatTensor ],
            return_call: bool = True,
            timeout: float = bittensor.__blocktime__,
        ):
        await self._call_backward( 
            roles = roles,
            messages = messages,
            response = response,
            rewards = rewards,
            timeout = timeout,
        ) 

    async def _call_backward( 
            self, 
            roles: List[ str ],
            messages: List[ str ],
            response: str,
            rewards: Union[ List[ float ], torch.FloatTensor ],
            timeout: float = bittensor.__blocktime__
        ) -> dict:
        try:
            start_time = time.time()
            if isinstance( rewards, torch.FloatTensor ): rewards = rewards.tolist()
            packed_messages = [json.dumps({"role": role, "content": message}) for role, message in zip(roles, messages)]
            request_proto = bittensor.BackwardTextPromptingRequest( 
                messages = packed_messages, 
                response = response,
                rewards = rewards,
                hotkey = self.wallet.hotkey.ss58_address,
                version = bittensor.__version_as_int__
            )
            bittensor.grpc.TextPromptingStub( self.receptor.channel ).Backward(
                request = request_proto,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptor.sign() ),
                    ('bittensor-version',str( bittensor.__version_as_int__ )),
                )
            )
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = False, 
                is_response = False, 
                code = bittensor.proto.Success, 
                call_time = time.time() - start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = torch.Size( [ len(self.rewards) ] ),
                outputs = None,
                message = "Success",
                synapse = "text_prompting"
            )
        except grpc.RpcError as rpc_error_call:
            # Request failed with GRPC code.
            code = rpc_error_call.code()
            error_message = 'GRPC error code: {}, details: {}'.format( rpc_error_call.code(), str(rpc_error_call.details()) )
        except asyncio.TimeoutError:
            # Catch timeout errors.
            code = bittensor.proto.ReturnCode.Timeout
            error_message = 'GRPC request timeout after: {}s'.format( timeout )
        except Exception as e:
            # Catch unknown errors.
            code = bittensor.proto.ReturnCode.UnknownException
            error_message = str( e )
        finally:
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = False, 
                is_response = False, 
                code = code, 
                call_time = time.time() - start_time, 
                pubkey = self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs = torch.Size( [ len(self.rewards) ] ),
                outputs = None,
                message = error_message,
                synapse = "text_prompting"
            )
        
    


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

        # format rewards to list of lists of floats.
        def format_rewards( rewards ) -> List[List[float]]:
            ret_formatted_rewards = []
            if isinstance( rewards, torch.Tensor ): 
                ret_formatted_rewards = [ [float(el)] for el in rewards.tolist() ]
            elif isinstance( rewards, list ): 
                ret_formatted_rewards = [ [float(el)] for el in rewards ]
            else:
                # Format list of tensors and list of lists.
                for element in rewards:
                    if isinstance( element, torch.Tensor ): 
                        element = element.tolist()
                    ret_formatted_rewards.append( [ float(el) for el in element.tolist() ])
            return ret_formatted_rewards

        formatted_rewards = format_rewards( rewards )
        assert len(uids) == len(formatted_rewards), 'rewards must have same length as uids.'
        
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





