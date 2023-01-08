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

import argparse
import bittensor
from rich.prompt import Prompt
console = bittensor.__console__

class BecomeDelegateCommand:

    @staticmethod
    def run( cli ):
        r""" Become a delegate.
        """
        wallet = bittensor.wallet(config = cli.config)
        subtensor = bittensor.subtensor( config = cli.config )

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is already a delegate.
        if subtensor.is_hotkey_delegate( wallet.hotkey.ss58_address ):
            bittensor.__console__.print('Aborting: Hotkey {} is already a delegate.'.format(wallet.hotkey.ss58_address))
            return

        result: bool = subtensor.become_delegate( wallet )
        if not result:
            bittensor.__console__.print("Could not became a delegate on [white]{}[/white]".format(subtensor.network))
        else:
            # Check if we are a delegate.
            is_delegate: bool = subtensor.is_hotkey_delegate( wallet.hotkey.ss58_address )
            if not is_delegate:
                bittensor.__console__.print("Could not became a delegate on [white]{}[/white]".format(subtensor.network))
                return
            bittensor.__console__.print("Successfully became a delegate on [white]{}[/white]".format(subtensor.network))

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        become_delegate_parser = parser.add_parser(
            'become_delegate', 
            help='''Become a delegate on the network'''
        )
        become_delegate_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( become_delegate_parser )
        bittensor.subtensor.add_args( become_delegate_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt and not config.wallet.get('all_hotkeys') and not config.wallet.get('hotkeys'):
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)





      