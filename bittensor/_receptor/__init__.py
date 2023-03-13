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
import grpc
import asyncio
import bittensor
import time as clock
from grpc import _common

class receptor:
    def __init__(
            self, 
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            external_ip: 'str' = None,
            grpc_max_send_message_length: int = -1,
            grpc_max_receive_message_length: int = -1,
            grpc_keepalive_time_ms: int = 100000,
        ):
        r""" Encapsulates a grpc connection to an axon endpoint.
            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor proto.
                max_processes (:obj:`int`, `optional`, defaults to 1):
                    max number of processes to use for async calls.
                external_ip (:obj:`str`, `optional`, defaults to None):
                    external ip of the machine, if None, will use the ip from the endpoint.
                grpc_max_send_message_length (:obj:`int`, `optional`, defaults to -1):
                    max send message length for grpc.
                grpc_max_receive_message_length (:obj:`int`, `optional`, defaults to -1):
                    max receive message length for grpc.
                grpc_keepalive_time_ms (:obj:`int`, `optional`, defaults to 100000):
                    keepalive time for grpc.
        """
        if wallet == None: wallet = bittensor.wallet()
        self.wallet = wallet
        self.endpoint = endpoint
        self.external_ip = external_ip
        if endpoint.ip == external_ip:
            ip = "localhost:"
            self.endpoint_str = ip + str(endpoint.port)
        else:
            endpoint_str = endpoint.ip + ':' + str(endpoint.port)
        self.channel = grpc.aio.insecure_channel(
            endpoint_str,
            options=[('grpc.max_send_message_length', grpc_max_send_message_length),
                     ('grpc.max_receive_message_length', grpc_max_receive_message_length),
                     ('grpc.keepalive_time_ms', grpc_keepalive_time_ms)])
        self.receptor_uid = str(uuid.uuid1())
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY

    def __str__ ( self ):
        """ Returns a string representation of the receptor."""
        return "Receptor({})".format(self.endpoint) 

    def __repr__ ( self ):
        """ Returns a string representation of the receptor."""
        return self.__str__()
    
    def __exit__ ( self ):
        self.__del__()

    def close ( self ):
        self.__exit__()
    
    def __del__ ( self ):
        """ Destructor for receptor.
        """
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
                loop = asyncio.get_event_loop()
                loop.run_until_complete ( self.channel.close() )
        except:
            pass

    def sign(self) -> str:
        """ Creates a signature for the receptor and returns it as a string."""
        nonce = f"{self.nonce()}"
        sender_hotkey = self.wallet.hotkey.ss58_address
        receiver_hotkey = self.endpoint.hotkey
        message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{self.receptor_uid}"
        signature = f"0x{self.wallet.hotkey.sign(message).hex()}"
        return ".".join([nonce, sender_hotkey, signature, self.receptor_uid])

    def nonce ( self ):
        r"""creates a string representation of the time
        """
        return clock.monotonic_ns()
        
    def state ( self ):
        """ Returns the state of the receptor channel."""
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"
