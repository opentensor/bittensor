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

import asyncio
import uuid
import time
import torch
import httpx
import bittensor as bt
from typing import Union, Optional, List

class dendrite( torch.nn.Module ):

    def __str__(self) -> str:
        return "dendrite({})".format( self.keypair.ss58_address )
    
    def __repr__(self) -> str: return self.__str__()

    def __init__( 
            self, 
            wallet: Optional[Union[ 'bt.wallet', 'bt.keypair']] = None
        ):
        """ Dendrite abstract class
            Args:
                wallet (:obj:`Union[ 'bt.wallet', 'bt.keypair']`, `required`):
                    bt wallet or keypair used for signing messages, defaults to bt.wallet()
        """
        super(dendrite, self).__init__()
        self.uuid = str(uuid.uuid1())
        self.client = httpx.AsyncClient()
        self.external_ip = bt.utils.networking.get_external_ip()
        self.keypair = (wallet.hotkey if isinstance( wallet, bt.wallet ) else wallet) or bt.wallet().hotkey

    async def forward( 
            self, 
            axons: Union[ List[ Union[ 'bt.axon_info', 'bt.axon' ] ], Union[ 'bt.axon_info', 'bt.axon' ] ],
            request: bt.BaseRequest = bt.BaseRequest(), 
            timeout: float = 12 
        ) -> bt.BaseRequest:

        # Wrap axons to list for gather op.
        if not isinstance( axons, list ): axons = [axons]

        # Build multi call.
        async def query():
            coroutines = [ self.call( axon = axon, request = request, timeout = timeout) for axon in axons ]
            all_responses = await asyncio.gather(*coroutines)
            return all_responses
        
        # Run all calls.
        responses = await query()

        # Optionally return single if only a single axon has been sent.
        if len(responses) == 1: return responses[0]
        else: return responses

    async def call( 
        self,
        axon: Union[ 'bt.AxonInfo', 'bt.axon' ],
        request: bt.BaseRequest = bt.BaseRequest(), 
        timeout: float = 12.0 
    ) -> bt.BaseRequest:
        
        # Build the endpoint str + url
        info = axon.info() if isinstance( axon, bt.axon ) else axon

        # Set dendrite side parameters.
        request.axon_ip = info.ip
        request.axon_port = info.port
        request.dendrite_ip = self.external_ip
        request.request_name = request.__class__.__name__
        request.dendrite_hotkey = self.keypair.ss58_address
        request.dendrite_nonce = time.monotonic_ns()
        request.dendrite_uuid = self.uuid
        request.dendrite_version = bt.__version_as_int__
        request.axon_hotkey = self.info.hotkey
        request.dendrite_sign()

        # Build endpoint from request.
        if request.axon_ip == request.dendrite_ip:
            endpoint = f"localhost:{str(request.axon_port)}"
        else:
            endpoint = f"{request.axon_ip}:{str(request.axon_port)}"
        url = f"http://{endpoint}/{request.request_name}"

        request.log_dendrite_outbound()
        response = await self.client.post( url, headers = request.to_dendrite_headers(), json = request.dict() )
        return response

        # # Parse response on success.
        # if response.status_code == 200:
        #     try:
        #         response_obj = request.__class__( **response.json() )
        #         # Parse the changes from the response into the request.
        #         # We skip items which are immutable.
        #         for key in request.dict().keys(): 
        #             try: setattr(request, key, getattr(response_obj, key) ) ; 
        #             except: pass
        #                         request.log_dendrite_inbound()

        #         bt.logging.debug( f"dendrite | <-- | {request_name} | {receiver_hotkey} | {info.ip}:{str(info.port)} | {response.status_code} | {response.headers['message']}")

        #     # Exception handling.
        #     except Exception as e:
        #         request.log_dendrite_inbound()

        #     finally:
        #         # Finally return request with variables fixed.
        #         request.log_dendrite_inbound()
        #         return request
            
        # request.status_code = response.status_code
        # request.axon_proccess_time = response.headers['axon_proccess_time']
        # request.message = response.headers['message']
        # return response

