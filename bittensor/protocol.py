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

import pydantic
import bittensor
from fastapi.responses import Response
from fastapi import Request
from typing import Dict, Optional, Tuple, Union, List, Callable

class Headers( pydantic.BaseModel ):
    # Defines the http route name which is set on axon.attach( callable( request: RequestName ))
    request_name: Optional[str] = pydantic.Field(
        title = 'request_name',
        description = 'Defines the http route name which is set on axon.attach( callable( request: RequestName ))',
        examples = 'Forward',
        allow_mutation = True,
        default = "",
    )

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    axon_status_code: Optional[str] = pydantic.Field(
        title = 'axon_status_code',
        description = 'The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status',
        examples = "200",
        default = "",
        allow_mutation = True
    )

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    axon_status_message: Optional[str] = pydantic.Field(
        title = 'axon_status_message',
        description = 'The axon_status_message associated with the axon_status_code',
        examples = 'Success',
        default = "",
        allow_mutation = True
    )
        
    # Defines the the maximum amount of time the dendrite will wait before throwing a timeout error.
    dendrite_timeout: Optional[str] = pydantic.Field(
        title = 'dendrite_timeout',
        description = 'Defines the the maximum amount of time the dendrite will wait before throwing a timeout error.',
        examples = '12.0',
        default = "",
        allow_mutation = True
    )

    # Process time on axon terminal side of call. Set by the dendrite using the header axon_process_time field.
    axon_process_time: Optional[str] = pydantic.Field(
        title = 'axon_process_time',
        description = 'Process time on axon terminal side of call. Set by the dendrite using the header axon_process_time field.',
        examples = '0.1',
        default = "",
        allow_mutation = True
    )

    # Process time on dendrite terminal side of call. Set by the dendrite after the request has been processed.
    dendrite_process_time: Optional[str] = pydantic.Field(
        title = 'dendrite_process_time',
        description = 'Process time on dendrite terminal side of call. Set by the dendrite after the request has been processed.',
        examples = '0.1',
        default = "",
        allow_mutation = True
    )

    # The ip of the dendrite making the request.
    dendrite_ip: Optional[ str ] = pydantic.Field(
        title = 'dendrite_ip',
        description = 'The ip of the dendrite making the request.',
        examples = '198.123.23.1',
        default = "",
        allow_mutation = True
    )

    # The axon recieving the request.
    axon_ip: Optional[ str ] = pydantic.Field(
        title = 'axon_ip',
        description = 'The axon recieving the request.',
        examples = '198.123.23.1',
        default = "",
        allow_mutation = True
    )

    # The port sending dendrite client set on the axon side using request.client.port.
    dendrite_port: Optional[ str ] = pydantic.Field(
        title = 'dendrite_port',
        description = 'The port sending dendrite client set on the axon side using request.client.port',
        examples = '198.123.23.1',
        default = "",
        allow_mutation = True
    )

    # The host port of the sender.
    axon_port: Optional[ str ] = pydantic.Field(
        title = 'axon_port',
        description = 'The bittensor version of the sender.',
        examples = '198.123.23.1',
        default = "",
        allow_mutation = True
    )

    # The bittensor version on the dendrite as an int.
    dendrite_version: Optional[ str ] = pydantic.Field(
        title = 'dendrite_version',
        description = 'The bittensor version on the dendrite as int',
        examples = 111,
        default = "",
        allow_mutation = True
    )

    # The bittensor version on the axon as an int.
    axon_version: Optional[ str ] = pydantic.Field(
        title = 'axon_version',
        description = 'The bittensor version on the axon as int',
        examples = 111,
        default = "",
        allow_mutation = True
    )

    # A unique monotonically increasing integer nonce associate with the dendrite
    dendrite_nonce: Optional[ str ] = pydantic.Field(
        title = 'dendrite_nonce',
        description = 'A unique monotonically increasing integer nonce associate with the dendrite generated from time.monotonic_ns()',
        examples = 111111,
        default = "",
        allow_mutation = True
    )

    # A unique monotonically increasing integer nonce associate with the axon
    axon_nonce: Optional[ str ] = pydantic.Field(
        title = 'axon_nonce',
        description = 'A unique monotonically increasing integer nonce associate with the axon generated from time.monotonic_ns()',
        examples = 111111,
        default = "",
        allow_mutation = True
    )

    # A unique identifier associated with the dendrite.
    dendrite_uuid: Optional[ str ] = pydantic.Field(
        title = 'dendrite_uuid',
        description = 'A unique identifier associated with the dendrite.',
        examples = "5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
        default = "",
        allow_mutation = True
    )

    # A unique identifier associated with the axon, set on the axon side.
    axon_uuid: Optional[ str ] = pydantic.Field(
        title = 'axon_uuid',
        description = 'A unique identifier associated with the axon, set on the axon side.',
        examples = "5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
        default = "",
        allow_mutation = True
    )

    # The ss58 encoded hotkey string of the dendrite wallet.
    dendrite_hotkey: Optional[ str ] = pydantic.Field(
        title = 'dendrite_hotkey',
        description = 'The ss58 encoded hotkey string of the dendrites wallet.',
        examples = "5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
        default = "",
        allow_mutation = True
    )

    # The bittensor version on the dendrite as an int.
    axon_hotkey: Optional[ str ] = pydantic.Field(
        title = 'axon_hotkey',
        description = 'The ss58 encoded hotkey string of the axon wallet.',
        examples = "5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
        default = "",
        allow_mutation = True
    )

    # A signature verifying the tuple (dendrite_nonce, axon_hotkey, dendrite_hotkey, dendrite_uuid)
    dendrite_signature: Optional[ str ] = pydantic.Field(
        title = 'dendrite_signature',
        description = 'A signature verifying the tuple (nonce, axon_hotkey, dendrite_hotkey, dendrite_uuid)',
        examples = "0x0813029319030129u4120u10841824y0182u091u230912u",
        default = "",
        allow_mutation = True
    )
    def dendrite_sign(self, keypair):
        message = f"{self.dendrite_nonce}.{self.dendrite_hotkey}.{self.axon_hotkey}.{self.dendrite_uuid}"
        self.dendrite_signature = f"0x{keypair.sign(message).hex()}"

    # A signature verifying the tuple (axon_nonce, axon_hotkey, dendrite_hotkey, axon_uuid)
    axon_signature: Optional[ str ] = pydantic.Field(
        title = 'axon_signature',
        description = 'A signature verifying the tuple (axon_nonce, axon_hotkey, dendrite_hotkey, axon_uuid)',
        examples = "0x0813029319030129u4120u10841824y0182u091u230912u",
        default = "",
        allow_mutation = True
    )
    def axon_sign(self, keypair):
        keypair = bittensor.Keypair( ss58_address = self.axon_hotkey )
        message = f"{self.axon_nonce}.{self.axon_hotkey}.{self.dendrite_hotkey}.{self.axon_uuid}"
        self.axon_signature = f"0x{keypair.sign(message).hex()}"

class Request( pydantic.BaseModel ):

    class Config:
        validate_assignment = True

    # A signature verifying the tuple (axon_nonce, axon_hotkey, dendrite_hotkey, axon_uuid)
    headers: Optional[ Headers ] = pydantic.Field(
        title = 'headers',
        description = 'headers associated with the request.',
        examples = "bt.Headers",
        default = None,
        allow_mutation = True,
        repr=False
    )

    

    


    # def to_dendrite_headers( self ):
    #     return {
    #         'dendrite_signature': self.dendrite_signature,
    #         'dendrite_hotkey': self.dendrite_hotkey,
    #         'dendrite_uuid': self.dendrite_uuid,
    #         'dendrite_nonce': self.dendrite_nonce,
    #         'dendrite_version': self.dendrite_version,
    #         'dendrite_timeout': self.dendrite_timeout,
    #     }
    
    # def from_dendrite_headers( self, request ) -> 'BaseRequest':
    #     headers = dict( request.headers )
    #     return BaseRequest(
    #         request_name = request.url.path.split("/")[1],
    #         dendrite_ip = request.client.ip,
    #         dendrite_port = request.client.port,
    #         dendrite_signature = headers['dendrite_signature'],
    #         dendrite_hotkey = headers['dendrite_hotkey'],
    #         dendrite_uuid = headers['dendrite_uuid'],
    #         dendrite_nonce = headers['dendrite_nonce'],
    #         dendrite_version = headers['dendrite_version'],
    #         dendrite_timeout = headers['dendrite_timeout'],
    #     )
    
    # def to_axon_headers( self ):
    #     return {
    #         'dendrite_signature': self.dendrite_signature,
    #         'dendrite_hotkey': self.dendrite_hotkey,
    #         'dendrite_uuid': self.dendrite_uuid,
    #         'dendrite_nonce': self.dendrite_nonce,
    #         'dendrite_version': self.dendrite_version,
    #         'dendrite_timeout': self.dendrite_timeout,
    #     }
    
    # def to_axon_response(self, response: Response = None ) -> Response:
    #     if response == None:
    #         return JSONResponse(status_code = self.axon_status_code, headers=self.to_axon_headers(), content={})
    #     else:
    #         return 

    # def log_dendrite_outbound( self ):
    #     bittensor.logging.debug( f"dendrite | --> | {self.request_name or 'unknown'} | {self.axon_hotkey or 'unknown'} | {self.axon_ip or 'unknown'}:{str(self.axon_port) or 'unknown'} | {self.status_code or 'unknown'} | {self.status_message or 'unknown'}")

    # def log_dendrite_inbound( self ):
    #     bittensor.logging.debug( f"dendrite | <-- | {self.request_name or 'unknown'} | {self.axon_hotke or 'unknown'} | {self.axon_ip or 'unknown'}:{str(self.axon_port) or 'unknown'} | {self.status_code or 'unknown'} | {self.status_messageor or 'unknown'}" )

    # def log_axon_outbound(self):
    #     bittensor.logging.debug( f"axon     | <-- | {self.request_name or 'unknown'} | {self.dendrite_hotkey or 'unknown'} | {self.dendrite_ip or 'unknown'}:{self.dendrite_port or 'unknown'}  | {self.axon_status_code or 'unknown'} | {self.axon_status_message or 'unknown'}")
    
    # def log_axon_inbound(self):
    #     bittensor.logging.debug( f"axon     | --> | {self.request_name or 'unknown'} | {self.dendrite_hotkey or 'unknown'} | {self.dendrite_ip or 'unknown'}:{self.dendrite_port or 'unknown'}  | {self.axon_status_code or 'unknown'} | {self.axon_status_message or 'unknown'}")




    




    # # Reciever Signature items.
    # receiver_ip: Optional[ str ] = None
    # reciever_port: Optional[ str ] = None
    # receiver_hotkey: Optional[ str ] = None

    # # Return code and message.
    # status_code: Optional[int] = None
    # message: Optional[str] = None
            # bt.logging.debug( f"dendrite | --> | {request.request_name} | {request.axon_hotkey} | {request.ip}:{str(info.port)} | 0 | Success")



    # def log_dendrite_inbound(self, axon_info: bittensor.AxonInfo ):
    #     bittensor.logging.debug( f"dendrite | <-- | {self.request_name} | {self.receiver_hotkey} | {axon_info.ip}:{str(axon_info.port)} | {self.return_code} | {self.return_message}")

    # def log_axon_inbound(self):
    #     print ('log_axon_inbound')
    #     bittensor.logging.debug( f"axon     | <-- | {self.request_name} | {self.sender_hotkey} | {self.sender_ip}:**** | 0 | Success ")

    # def log_axon_outbound(self):
    #     print ('log_axon_outbound')
    #     bittensor.logging.debug( f"axon     | --> | {self.request_name} | {self.sender_hotkey} | {self.sender_ip}:**** | {self.return_code} | {self.return_message}")

    # def from_request( request: Request ) -> 'BaseRequest':
    #     metadata = dict(request.headers)
    #     request_name = request.url.path.split("/")[1]
    #     base_request = BaseRequest(
    #         request_name = request_name,
    #         sender_ip = metadata.get("sender_ip") or None,
    #         sender_uuid = metadata.get("sender_uuid") or None,
    #         sender_timeout = metadata.get("sender_timeout") or None,
    #         sender_version = metadata.get("sender_version") or None,
    #         sender_nonce = metadata.get("sender_nonce") or None,
    #         sender_hotkey = metadata.get("sender_hotkey") or None,
    #         sender_signature = metadata.get("sender_signature") or None,
    #         receiver_hotkey = metadata.get("receiver_hotkey") or None,
    #     )
    #     return base_request
    
    # def headers( self ) -> dict:
    #     # Fill request metadata for middleware.
    #     return {
    #         "rpc-auth-header": "Bittensor",
    #         "sender_ip": str(self.sender_ip),
    #         "sender_timeout": str( self.sender_timeout ),
    #         "sender_version": str( self.sender_version ),
    #         "sender_nonce": str( self.sender_nonce ),
    #         "sender_uuid": str( self.sender_uuid ),
    #         "sender_hotkey": str( self.sender_hotkey ),
    #         "sender_signature": str( self.sender_signature ),
    #         "receiver_hotkey": str( self.receiver_hotkey ),
    #         "request_name": self.request_name,
    #     }

