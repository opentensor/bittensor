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
            synapse: bt.Synapse = bt.Synapse(), 
            timeout: float = 12 
        ) -> bt.Synapse:

        # Wrap axons to list for gather op.
        if not isinstance( axons, list ): axons = [axons]

        # Build multi call.
        async def query():
            coroutines = [ self.call( axon = axon, synapse = synapse, timeout = timeout) for axon in axons ]
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
        synapse: bt.Synapse = bt.Synapse(), 
        timeout: float = 12.0 
    ) -> bt.Request:
        
        start_time = time.time()

        # Build the endpoint str + url
        axon = axon.info() if isinstance( axon, bt.axon ) else axon
        request_name = synapse.__class__.__name__
        endpoint = f"localhost:{str(axon.port)}" if axon.ip == str(self.external_ip) else f"{axon.ip}:{str(axon.port)}"
        url = f"http://{endpoint}/{request_name}"

        # Build Dendrite + Axon headers.
        synapse.timeout = str( timeout )
        synapse.dendrite = bt.TerminalInfo(
            **{
                "ip": str(self.external_ip),
                "version": str( bt.__version_as_int__ ),
                "nonce": f"{time.monotonic_ns()}",
                "uuid": str(self.uuid),
                "hotkey": str( self.keypair.ss58_address )
            }
        )
        synapse.axon = bt.TerminalInfo(
            **{
                "ip": str( axon.ip ),
                "port": str( axon.port ),
                "hotkey": str( axon.hotkey ),
            }
        )
        # Sign the synapse request.
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}"
        synapse.dendrite.signature = f"0x{self.keypair.sign(message).hex()}"

        # Make the call.
        bt.logging.debug( f"dendrite | --> | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success")
        json_response = await self.client.post( url, headers = synapse.to_headers(), json = synapse.dict() )
    
        try:
            # Parse response on success.
            if json_response.status_code == 200:

                # Overwrite with remote axon state if allowed.
                # The protocol must have field( allow_mutation = False) to stop overwrites of input.
                response_synapse = synapse.__class__( **json_response.json() )
                for key in synapse.dict().keys(): 
                    try: setattr(synapse, key, getattr(response_synapse, key) ) ; 
                    except: pass
      
            # Overwrite None headers as set by remote.
            axon_headers = bt.Synapse.from_headers( json_response.headers )
            synapse.dendrite.__dict__.update( **(synapse.dendrite.dict(exclude_none=True) | axon_headers.dendrite.dict(exclude_none=True)) )
            synapse.axon.__dict__.update( **(synapse.axon.dict(exclude_none=True) | axon_headers.axon.dict(exclude_none=True)) )

            # Set process time and status code.
            synapse.dendrite.process_time = str(time.time() - start_time)
            synapse.dendrite.status_code = synapse.axon.status_code
            synapse.dendrite.status_message = synapse.axon.status_message

            # Log the response.
            bt.logging.debug( f"dendrite | <-- | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.axon.status_code} | {synapse.axon.status_message}")

        except Exception as e:    
            # Failed to parse response.
            synapse.dendrite.status_code = '406'
            synapse.dendrite.status_message = f"Failed to parse response object with error: {str(e)}"
            bt.logging.debug( f"dendrite | <-- | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message}")

        # Return the synapse.
        finally:
            return synapse
