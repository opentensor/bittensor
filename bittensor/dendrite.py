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
        request_name = request.__class__.__name__
        endpoint = f"localhost:{str(info.port)}" if info.ip == str(self.external_ip) else f"{info.ip}:{str(info.port)}"
        url = f"http://{endpoint}/{request_name}"

        # Build Metadata.
        request.sender_ip = str(self.external_ip)
        request.sender_nonce = f"{time.monotonic_ns()}"
        request.sender_hotkey = self.keypair.ss58_address
        request.sender_uuid = str(self.uuid)
        request.sender_timeout = timeout
        request.sender_version = bt.__version_as_int__
        request.receiver_hotkey = info.hotkey
        request.request_name = request_name
        message = f"{request.sender_nonce}.{request.sender_hotkey}.{request.receiver_hotkey}.{request.sender_uuid}"
        request.sender_signature = f"0x{self.keypair.sign(message).hex()}"
        print(request)

        try:
            request.log_dendrite_outbound( info )
            response = await self.client.post( url, headers = request.headers(), json = request.dict() )
            response = request.__class__( **response.json() )

        except Exception as e:
            # Unknown failure, set params.
            response = request.__class__( **request.dict() )
            response.return_code = bt.ReturnCode.UNKNOWN.value
            response.return_message = f"Failed to send request {str(e)}" 

        finally:
            # Finally log and exit.
            response.log_dendrite_inbound( info )
            return response