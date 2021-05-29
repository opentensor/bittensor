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
            config: Munch = None, 
            name: str = None,
            path: str = None,
            hotkey: str = None,
        ) -> 'bittensor.Wallet':
        r""" Init bittensor wallet object containing a hot and coldkey.

            Args:
                config (:obj:`Munch`, `optional`): 
                    bittensor.wallet.default_config()
                name (required=False, default='default):
                    The name of the wallet to unlock for running bittensor
                hotkey (required=False, default='default):
                    The name of hotkey used to running the miner.
                path (required=False, default='~/.bittensor/wallets/'):
                    The path to your bittensor wallets
        """
        if config == None:
            config = wallet.default_config()
        config.wallet.name = name if name != None else config.wallet.name
        config.wallet.path = path if path != None else config.wallet.path
        config.wallet.hotkey = hotkey if hotkey != None else config.wallet.hotkey
        config = copy.deepcopy(config)
        wallet.check_config(config)

        return wallet_impl.Wallet( config )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        wallet.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        try:
            parser.add_argument('--wallet.name', required=False, default='default', 
                                    help='''The name of the wallet to unlock for running bittensor''')
            parser.add_argument('--wallet.hotkey', required=False, default='default', 
                                    help='''The name of wallet's hotkey.''')
            parser.add_argument('--wallet.path', required=False, default='~/.bittensor/wallets/', 
                                    help='''The path to your bittensor wallets''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        pass