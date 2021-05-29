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

import bittensor
import argparse
import copy
import grpc
from munch import Munch

from . import receptor_impl

class receptor:

    def __new__(
            self, 
            endpoint: 'bittensor.Endpoint', 
            config: Munch = None, 
            wallet: 'bittensor.wallet' = None,
            pass_gradients: bool = None,
            timeout: int = None,
            do_backoff: bool = None,
            max_backoff:int = None
        ) -> 'bittensor.Receptor':
        r""" Initializes a receptor grpc connection.
            Args:
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor.
                config (:obj:`Munch`, `optional`): 
                    receptor.Receptor.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                pass_gradients (default=True, type=bool)
                    Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.
                timeout (default=0.5, type=float):
                    The per request RPC timeout. a.k.a the maximum request time.
                do_backoff (default=True, type=bool)
                    Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.
                max_backoff (default=100, type=int)
                    The backoff doubles until this saturation point.
        """        
        if config == None:
            config = receptor.default_config()
        config.receptor.pass_gradients = pass_gradients if pass_gradients != None else config.receptor.pass_gradients
        config.receptor.timeout = timeout if timeout != None else config.receptor.timeout
        config.receptor.do_backoff = do_backoff if do_backoff != None else config.receptor.do_backoff
        config.receptor.max_backoff = max_backoff if max_backoff != None else config.receptor.max_backoff
        receptor.check_config( config )
        config = copy.deepcopy(config) # Configuration information.

        if wallet == None:
            wallet = bittensor.wallet( self.config )

        # Loop back if the neuron is local.
        try:
            external_ip = config.axon.external_ip
        except:
            pass
        try:
            external_ip = config.axon.external_ip
        except:
            pass
        finally:
            external_ip = None
        if endpoint.ip == external_ip:
            ip = "localhost:"
            self.endpoint = ip + str(endpoint.port)
        else:
            self.endpoint = endpoint.ip + ':' + str(endpoint.port)
        
        channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        stub = bittensor.grpc.BittensorStub(self.channel)

        return receptor_impl.Receptor( 
            config = config, 
            wallet = wallet, 
            endpoint = endpoint, 
            channel = channel, 
            stub = stub
        )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        receptor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        assert config.receptor.timeout >= 0, 'timeout must be positive value, got {}'.format(config.receptor.timeout)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.add_args( parser )
        try:
            # Can be called multiple times.
            parser.add_argument('--receptor.pass_gradients', default=True, type=bool, 
                help='''Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.''')
            parser.add_argument('--receptor.timeout', default=3, type=float, 
                help='''The per request RPC timeout. a.k.a the maximum request time.''')
            parser.add_argument('--receptor.do_backoff', default=True, type=bool, 
                help='''Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.''')
            parser.add_argument('--receptor.max_backoff', default=100, type=int, 
                help='''The backoff doubles until this saturation point.''')
        except:
            pass