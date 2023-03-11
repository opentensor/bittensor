""" Implementation of Axon, services Forward and Backward requests from other neurons.
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

import grpc
import uuid
import bittensor

class Axon():
    r""" Maintains a GPRC server endpoint for processing requests from other neurons.
    """
    def __init__( 
        self, 
        wallet: 'bittensor.wallet',
        ip: str,
        port: int,
        external_ip: str,
        external_port: int,
        server: 'grpc._Server',
    ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.axon.config()
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                ip (:obj:`str`, `required`):
                    ip address of this axon.
                port (:obj:`int`, `required`):
                    port of this axon.
                external_ip (:obj:`str`, `required`):
                    external ip address of this axon.
                external_port (:obj:`int`, `required`):
                    external port of this axon.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint.
        """
        self.ip = ip
        self.port = port
        self.external_ip = external_ip
        self.external_port = external_port
        self.wallet = wallet
        self.server = server
        self.started = False        
        self._prometheus_uuid = uuid.uuid1()

    def attach( self, synapse: 'bittensor.Synapse' ) -> 'Axon':
        r""" Attaches a synapse to this axon.
        """
        synapse._attach( axon = self )
        return self

    def __str__(self) -> str:
        return "Axon({}, {}, {}, {})".format( self.ip, self.port, self.wallet.hotkey.ss58_address, "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def start(self) -> 'Axon':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
        self.server.start()
        self.started = True
        return self

    def stop(self) -> 'Axon':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
        self.started = False
        return self
