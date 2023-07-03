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

import uuid
import time
import torch
import httpx
import bittensor as bt
from typing import Union, Optional

class dendrite( torch.nn.Module ):

    def __str__(self) -> str:
        return "dendrite({}, {})".format( self.keypair.ss58_address, self.endpoint_str )
    
    def __repr__(self) -> str: return self.__str__()

    def __init__(
            self,
            axon: Union[ 'bt.axon_info', 'bt.axon' ], 
            wallet: Optional[Union[ 'bt.wallet', 'bt.keypair']] = None
        ):
        """ Dendrite abstract class
            Args:
                axon (:obj:Union[`bt.axon_info`, 'bt.axon'], `required`):   
                    bt axon object or its info used to create the connection.
                wallet (:obj:`Union[ 'bt.wallet', 'bt.keypair']`, `required`):
                    bt wallet or keypair used for signing messages, defaults to bt.wallet()
        """
        super(dendrite, self).__init__()
        self.uuid = str(uuid.uuid1())
        self.client = httpx.AsyncClient()
        self.keypair = (wallet.hotkey if isinstance( wallet, bt.wallet ) else wallet) or bt.wallet().hotkey
        self.axon_info = axon.info() if isinstance( axon, bt.axon ) else axon
        self.endpoint_str = "localhost:" + str(self.axon_info.port)

    async def forward( 
            self, 
            request: bt.BaseRequest = bt.BaseRequest(), 
            timeout: float = 12 
        ) -> bt.BaseRequest:

        # Get the request name from the request type.        
        request_name = request.__class__.__name__
        url = f"http://{self.endpoint_str}/{request_name}"
    
        # Build Metadata.
        sender_nonce = f"{time.monotonic_ns()}"
        sender_hotkey = self.keypair.ss58_address
        sender_uuid = self.uuid
        sender_timeout = timeout
        sender_version = bt.__version_as_int__
        receiver_hotkey = self.axon_info.hotkey
        message = f"{sender_nonce}.{sender_hotkey}.{receiver_hotkey}.{sender_uuid}"
        sender_signature = f"0x{self.keypair.sign(message).hex()}"
        
        # Fill request metadata for middleware.
        metadata = {
            "rpc-auth-header": "Bittensor",
            "sender_timeout": str( sender_timeout ),
            "sender_version": str( sender_version ),
            "sender_nonce": str( sender_nonce ),
            "sender_uuid": str( sender_uuid ),
            "sender_hotkey": str( sender_hotkey ),
            "sender_signature": str( sender_signature ),
            "receiver_hotkey": str( receiver_hotkey ),
            "request_name": request_name,
        }

        # Fill data into request.
        request.sender_version = sender_version
        request.sender_nonce = sender_nonce
        request.sender_uuid = str(sender_uuid)
        request.sender_hotkey = str( sender_hotkey )
        request.sender_signature = str( sender_signature )
        request.sender_timeout = sender_timeout
        request.receiver_hotkey = receiver_hotkey
        request.request_name = request_name

        try:
            response = await self.client.post( url, headers = metadata, json = request.dict() )
            return request.__class__( **response.json() )
        
        except Exception as e:
            # Unknown failure, set params.
            failed_response = request.__class__( **request.dict() )
            failed_response.return_code = bt.ReturnCode.UNKNOWN.value
            failed_response.return_message = f"Failed to send request {str(e)}" 
            return failed_response