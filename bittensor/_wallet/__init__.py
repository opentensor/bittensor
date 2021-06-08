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

from munch import Munch
from . import wallet_impl

class wallet:
    def __new__(
            cls, 
            name: str = 'default',
            hotkey: str = 'default',
            path: str = '~/.bittensor/wallets/',
        ) -> 'bittensor.Wallet':
        r""" Init bittensor wallet object containing a hot and coldkey.

            Args:
                name (required=False, default='default'):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=False, default='default'):
                    The name of hotkey used to running the miner.
                path (required=False, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """
        return wallet_impl.Wallet(
            name = name, 
            hotkey = hotkey, 
            path = path 
        )

    def extend_config( config: Munch, namespace:str = 'wallet'):
        wallet_config = Munch()
        config[namespace] = wallet_config
        if namespace != '': namespace += '.'
        parser = argparse.ArgumentParser()
        parser.add_argument('--' + namespace + 'name', dest = 'name', required=False, default='default', 
                                help='''The name of the wallet to unlock for running bittensor''')
        parser.add_argument('--' + namespace + 'hotkey', dest = 'hotkey', required=False, default='default', 
                                help='''The name of the wallet's-hotkey used to run the miner.''')
        parser.add_argument('--' + namespace + 'path', dest = 'path', required=False, default='~/.bittensor/wallets/', 
                                help='''The path to your bittensor wallets''')
        parser.parse_known_args( namespace = wallet_config )

