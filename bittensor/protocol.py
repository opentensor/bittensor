""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

import bittensor
from enum import Enum
from pydantic import BaseModel
from fastapi import Request
from typing import Dict, Optional, Tuple, Union, List, Callable

class ReturnCode( Enum ):
    SUCCESS = 0
    BLACKLIST = 1
    TIMEOUT = 2
    FAILEDVERIFICATION = 3
    UNKNOWN = 4

class BaseRequest( BaseModel ):
    # The call being made
    request_name: Optional[str] = None

    # Process time
    process_time: Optional[float] = None

    # Sender Signature items.
    sender_ip: Optional[ str ] = None
    sender_timeout: float = 12
    sender_version: Optional[ str ] = None
    sender_nonce: Optional[ str ] = None
    sender_uuid: Optional[ str ] = None
    sender_hotkey: Optional[ str ] = None
    sender_signature: Optional[ str ] = None 

    # Reciever Signature items.
    receiver_hotkey: Optional[ str ] = None

    # Return code and message.
    return_message: Optional[str] = 'Success'
    return_code: Optional[int] = ReturnCode.SUCCESS.value

    def log_dendrite_outbound(self, axon_info: bittensor.AxonInfo ):
        bittensor.logging.debug( f"dendrite | --> | {self.request_name} | {self.receiver_hotkey} | {axon_info.ip}:{str(axon_info.port)} | 0 | Success")

    def log_dendrite_inbound(self, axon_info: bittensor.AxonInfo ):
        bittensor.logging.debug( f"dendrite | <-- | {self.request_name} | {self.receiver_hotkey} | {axon_info.ip}:{str(axon_info.port)} | {self.return_code} | {self.return_message}")

    def log_axon_inbound(self):
        print ('log_axon_inbound')
        bittensor.logging.debug( f"axon     | <-- | {self.request_name} | {self.sender_hotkey} | {self.sender_ip}:**** | 0 | Success ")

    def log_axon_outbound(self):
        print ('log_axon_outbound')
        bittensor.logging.debug( f"axon     | --> | {self.request_name} | {self.sender_hotkey} | {self.sender_ip}:**** | {self.return_code} | {self.return_message}")

    def from_headers( request: Request ) -> 'BaseRequest':
        metadata = dict(request.headers)
        request_name = request.url.path.split("/")[1]
        base_request = BaseRequest(
            request_name = request_name,
            sender_ip = metadata.get("sender_ip") or None,
            sender_uuid = metadata.get("sender_uuid") or None,
            sender_timeout = metadata.get("sender_timeout") or None,
            sender_version = metadata.get("sender_version") or None,
            sender_nonce = metadata.get("sender_nonce") or None,
            sender_hotkey = metadata.get("sender_hotkey") or None,
            sender_signature = metadata.get("sender_signature") or None,
            receiver_hotkey = metadata.get("receiver_hotkey") or None,
        )
        return base_request
    
    def headers( self ) -> dict:
        # Fill request metadata for middleware.
        return {
            "rpc-auth-header": "Bittensor",
            "sender_ip": str(self.sender_ip),
            "sender_timeout": str( self.sender_timeout ),
            "sender_version": str( self.sender_version ),
            "sender_nonce": str( self.sender_nonce ),
            "sender_uuid": str( self.sender_uuid ),
            "sender_hotkey": str( self.sender_hotkey ),
            "sender_signature": str( self.sender_signature ),
            "receiver_hotkey": str( self.receiver_hotkey ),
            "request_name": self.request_name,
        }

