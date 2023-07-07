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
            request: bt.Request = bt.Request(), 
            timeout: float = 12 
        ) -> bt.Request:

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
        request: bt.Request = bt.Request(), 
        timeout: float = 12.0 
    ) -> bt.Request:
        
        start_time = time.time()

        # Build the endpoint str + url
        axon = axon.info() if isinstance( axon, bt.axon ) else axon
        request_name = request.__class__.__name__
        endpoint = f"localhost:{str(axon.port)}" if axon.ip == str(self.external_ip) else f"{axon.ip}:{str(axon.port)}"
        url = f"http://{endpoint}/{request_name}"

        # Build Axon headers.
        dendrite_ip = str(self.external_ip)
        dendrite_nonce = f"{time.monotonic_ns()}"
        dendrite_hotkey = self.keypair.ss58_address
        dendrite_uuid = str(self.uuid)
        dendrite_timeout = timeout
        dendrite_version = bt.__version_as_int__
        axon_hotkey = axon.hotkey
        request_name = request_name
        message = f"{dendrite_nonce}.{dendrite_hotkey}.{axon_hotkey}.{dendrite_uuid}"
        dendrite_signature = f"0x{self.keypair.sign(message).hex()}"
        headers = {
            "rpc-auth-header": "Bittensor",
            "dendrite_ip": str( dendrite_ip ),
            "dendrite_timeout": str( dendrite_timeout ),
            "dendrite_version": str( dendrite_version ),
            "dendrite_nonce": str( dendrite_nonce ),
            "dendrite_uuid": str( dendrite_uuid ),
            "dendrite_hotkey": str( dendrite_hotkey ),
            "dendrite_signature": str( dendrite_signature ),
            "axon_hotkey": str( axon_hotkey ),
            "axon_ip": str( axon.ip ),
            "axon_port": str( axon.port ),
            "request_name": str(request_name)
        }
        bt.logging.debug( f"dendrite | --> | {request_name} | {axon_hotkey} | {axon.ip}:{str(axon.port)} | 0 | Success")
        response = await self.client.post( url, headers = headers, json = request.dict() )
        # Parse response on success.
        if response.status_code == 200:
            try:
                response_obj = request.__class__( **response.json() )
                # Parse the changes from the response into the request.
                # We skip items which are immutable.
                for key in request.dict().keys(): 
                    try: setattr(request, key, getattr(response_obj, key) ) ; 
                    except: pass
                bt.logging.debug( f"dendrite | <-- | {request_name} | {axon_hotkey} | {axon.ip}:{str(axon.port)} | {response.status_code} | {response.headers['axon_status_message']}")

            # Exception handling.
            except Exception as e:
                bt.logging.debug( f"dendrite | <-- | {request_name} | {axon_hotkey} | {axon.ip}:{str(axon.port)} | 406 | Failed to parse response object with error: {str(e)}")


        bt.logging.debug( f"dendrite | <-- | {request_name} | {axon_hotkey} | {axon.ip}:{str(axon.port)} | {response.status_code} | {response.headers['axon_status_message']}")

        request.headers = bt.Headers()
        request.headers.__dict__.update( dict( response.headers ) )
        request.headers.__dict__.update( dict( headers ) )
        request.headers.dendrite_process_time = str(time.time() - start_time)
        request.headers.axon_status_code = response.status_code
        return request

