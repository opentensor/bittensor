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

        # Build Dendrite headers.
        headers = bt.Headers( **{
            "rpc-auth-header": "Bittensor",
            "dendrite_ip": str(self.external_ip),
            "dendrite_timeout": str( timeout ),
            "dendrite_version": str( bt.__version_as_int__ ),
            "dendrite_nonce": f"{time.monotonic_ns()}",
            "dendrite_uuid": str(self.uuid),
            "dendrite_hotkey": str( self.keypair.ss58_address ),
            "axon_hotkey": str( axon.hotkey ),
            "axon_ip": str( axon.ip ),
            "axon_port": str( axon.port ),
            "request_name": str(request_name)
        })
        headers.dendrite_sign( keypair = self.keypair )
        request.headers = headers

        # Make the forward call.
        bt.logging.debug( f"dendrite | --> | {headers.request_name} | {headers.axon_hotkey} | {headers.axon_ip}:{str(headers.axon_port)} | 0 | Success")
        response = await self.client.post( url, headers = headers.dict(), json = request.dict() )


        try:
            # Parse response on success.
            if response.status_code == 200:

                # Parse the changes from the response into the request.
                # We skip items which are immutable.
                response_obj = request.__class__( **response.json() )
                for key in request.dict().keys(): 
                    try: setattr(request, key, getattr(response_obj, key) ) ; 
                    except: pass

            # Now fill the dendrite and axon header values.
            # This copys the remote header values from the axon then overwrites locals.
            request.headers = bt.Headers()
            request.headers.__dict__.update( dict( response.headers ) )
            request.headers.__dict__.update( dict( headers ) )
            request.headers.dendrite_process_time = str(time.time() - start_time)

            # Log the response.
            bt.logging.debug( f"dendrite | <-- | {headers.request_name} | {headers.axon_hotkey} | {headers.axon_ip}:{str(headers.axon_port)} | {response.status_code} | {response.headers['axon_status_message']}")

        # Failed to parse response.
        except Exception as e:
            bt.logging.debug( f"dendrite | <-- | {headers.request_name} | {headers.axon_hotkey} | {headers.axon_ip}:{str(headers.axon_port)} | 406 | Failed to parse response object with error: {str(e)}")

        # Return the request.
        finally:
            return request

