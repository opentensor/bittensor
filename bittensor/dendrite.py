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
import requests
import asyncio
import bittensor as bt
from typing import Union, Optional

class dendrite( torch.nn.Module ):
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
        self.keypair = (wallet.hotkey if isinstance( wallet, bt.wallet ) else wallet) or bt.wallet().hotkey
        self.axon_info = axon.info() if isinstance( axon, bt.axon ) else axon
        # if self.axon_info.ip == bt.utils.networking.get_external_ip(): 
        self.endpoint_str = "localhost:" + str(self.axon_info.port)
        # else: 
        #     self.endpoint_str = f"http://{self.axon_info.ip}:{str(self.axon_info.port)}"
        self.loop = asyncio.get_event_loop()

    def forward( self, request: bt.BaseRequest = None, timeout: float = 12 ) -> bt.BaseResponse:
        request = request or bt.BaseRequest(name = 'ping', hotkey = self.keypair.ss58_address, timeout = timeout )
        bt.logging.trace('request', request)
        url = f"http://{self.endpoint_str}/{request.name}"
        bt.logging.trace('url', url)
        self.sign(request)
        bt.logging.trace('request', request)
        response = requests.get( url, json = request.json() ) 
        bt.logging.trace('response', response.json() )
        # response = bt.BaseResponse( **response.json() ) 
        # bt.logging.trace('response', response)
        return response

    def __str__(self) -> str:
        return "dendrite({}, {})".format( self.keypair.ss58_address, self.endpoint_str )
    
    def __repr__(self) -> str: return self.__str__()

    def __exit__ ( self ): 
        self.__del__()

    def close ( self ): 
        self.__exit__()

    def __del__ ( self ):
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
                self.loop.run_until_complete ( self.channel.close() )
        except:
            pass

    def nonce ( self ): 
        return time.monotonic_ns()

    def sign(self, request: bt.BaseRequest):
        """ Creates a signature for the dendrite and returns it as a string."""
        nonce = f"{self.nonce()}"
        sender_hotkey = self.keypair.ss58_address
        receiver_hotkey = self.axon_info.hotkey
        
        message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{self.uuid}"
        signature = f"0x{self.keypair.sign(message).hex()}"

        request.nonce = nonce
        request.uuid = self.uuid
        request.sender_hotkey = sender_hotkey
        request.sender_signature = signature
        request.receiver_hotkey = receiver_hotkey
