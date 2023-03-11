""" Encapsulates a grpc connection to an axon endpoint as a standard auto-grad torch.nn.Module.
"""
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
import bittensor
import time as clock
from grpc import _common

class Receptor:
    def __init__(
            self, 
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            channel: 'grpc._Channel',
            stub: 'bittensor.grpc.BittensorStub',
            max_processes: int,
        ):
        r""" Initializes a receptor grpc connection.

            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor proto.
                channel (:obj:`grpc._Channel`, `required`):
                    grpc TCP channel.
                endpoint (:obj:`bittensor.grpc.BittensorStub`, `required`):
                    bittensor protocol stub created from channel.
        """
        super().__init__()
        self.wallet = wallet
        self.endpoint = endpoint
        self.channel = channel
        self.stub = stub
        self.receptor_uid = str(uuid.uuid1())
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY

    def __str__ ( self ):
        """ Returns a string representation of the receptor."""
        return "Receptor({})".format(self.endpoint) 

    def __repr__ ( self ):
        """ Returns a string representation of the receptor."""
        return self.__str__()
    
    def __del__ ( self ):
        """ Destructor for receptor.
            
        """
        print ('a')
        del self.channel
        print ('b')

    #     print ('a')
    #     try:
    #         del self.channel
    #         del self.stub
    #     except:
    #         print ('c')
    #         pass
    #     print ('b')
        # try:
        #     print ('destroy')
        #     # result = self.channel._channel.check_connectivity_state(True)
        #     # if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
        #     #     loop = asyncio.get_event_loop()
        #     #     loop.run_until_complete ( self.channel.close() )
        # except:
        #     pass
        # print ('done')

    def sign(self):
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
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"


        



        
