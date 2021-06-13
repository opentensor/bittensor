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
from . import wallet_impl

class wallet:
    def __new__(
            cls, 
            config: 'bittensor.Config' = None,
            name: str = None,
            hotkey: str = None,
            path: str = None,
        ) -> 'bittensor.Wallet':
        r""" Init bittensor wallet object containing a hot and coldkey.

            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.wallet.config()
                name (required=False, default='default'):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=False, default='default'):
                    The name of hotkey used to running the miner.
                path (required=False, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """
        if config == None: config = wallet.config().wallet
        config = copy.deepcopy( config )
        config.name = name if name != None else config.name
        config.hotkey = hotkey if hotkey != None else config.hotkey
        config.path = path if path != None else config.path
        wallet.check_config( config )
        return wallet_impl.Wallet(
            name = config.name, 
            hotkey = config.hotkey, 
            path = config.path 
        )

    @staticmethod   
    def config( config: 'bittensor.Config' = None, namespace: str = 'wallet' ) -> 'bittensor.Config':
        if config == None: config = bittensor.config()
        wallet_config = bittensor.config()
        config[ namespace ] = wallet_config
        parser = argparse.ArgumentParser()
        if namespace != '': namespace += '.'
        parser.add_argument('--' + namespace + 'name', dest = 'name', required=False, default='default', help='''The name of the wallet to unlock for running bittensor''')
        parser.add_argument('--' + namespace + 'hotkey', dest = 'hotkey', required=False, default='default', help='''The name of wallet's hotkey.''')
        parser.add_argument('--' + namespace + 'path', dest = 'path', required=False, default='~/.bittensor/wallets/', help='''The path to your bittensor wallets''')
        parser.parse_known_args( namespace = wallet_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        pass

