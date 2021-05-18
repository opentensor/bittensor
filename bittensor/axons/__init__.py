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
from munch import Munch

from . import _axon

class axon():

    def __new__(
            cls, 
            config: Munch = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            local_port: int = None,
            local_ip: str =  None,
            max_workers: int = None, 
            forward_processing_timeout:int = None,
            backward_processing_timeout:int = None,
            **kwargs
        ):
        # config for the wallet and nucleus sub-objects.
        if config == None:
            config = axon.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        config.axon.local_port = local_port if local_port != None else config.axon.local_port
        config.axon.local_ip = local_ip if local_ip != None else config.axon.local_ip
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.forward_processing_timeout = forward_processing_timeout if forward_processing_timeout != None else config.axon.forward_processing_timeout
        config.axon.backward_processing_timeout = backward_processing_timeout if backward_processing_timeout != None else config.axon.backward_processing_timeout
        axon.check_config( config )

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet( config = config )
        wallet = wallet

        return _axon.Axon( config, wallet )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        axon.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        bittensor.wallet.Wallet.add_args(parser)
        try:
            parser.add_argument('--axon.local_port', default=8091, type=int, 
                help='''The port this axon endpoint is served on. i.e. 8091''')
            parser.add_argument('--axon.local_ip', default='127.0.0.1', type=str, 
                help='''The local ip this axon binds to. ie. 0.0.0.0''')
            parser.add_argument('--axon.max_workers', default=10, type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''')
            parser.add_argument('--axon.forward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner forward process for computing and returning responses
                        back to the axon.''')
            parser.add_argument('--axon.backward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner backward process for computing and returning responses
                        back to the axon.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity and obtains the remote ip.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        assert config.axon.local_port > 1024 and config.axon.local_port < 65535, 'config.axon.local_port must be in range [1024, 65535]'

