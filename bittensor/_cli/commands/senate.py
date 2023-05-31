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

import sys
import argparse
import bittensor
from tqdm import tqdm
from rich.prompt import Confirm
from rich.prompt import Confirm, Prompt
from bittensor.utils.balance import Balance
from typing import List, Union, Optional, Dict, Tuple
from .utils import get_hotkey_wallets_for_wallet
console = bittensor.__console__

class SenateCommand:

    @staticmethod
    def run( cli ):
        r""" Participate in Bittensor's Senate with your senator hotkey.
        """
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )

        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
        console.print( wallet_balance )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt and not config.get('all_hotkeys') and not config.get('hotkeys'):
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        senate_parser = parser.add_parser(
            'senate',
            help='''Participate in senate motions with a senator hotkey'''
        )
        senate_parser.add_argument(
            '--no_version_checking',
            action='store_true',
            help='''Set false to stop cli version checking''',
            default = False
        )
        senate_parser.add_argument(
            '--no_prompt',
            dest='no_prompt',
            action='store_true',
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        senate_parser.add_argument(
            '--hotkeys',
            '--exclude_hotkeys',
            '--wallet.hotkeys',
            '--wallet.exclude_hotkeys',
            required=False,
            action='store',
            default=[],
            type=str,
            nargs='*',
            help='''Specify the hotkeys by name or ss58 address. (e.g. hk1 hk2 hk3)'''
        )
        senate_parser.add_argument(
            '--all_hotkeys',
            '--wallet.all_hotkeys',
            required=False,
            action='store_true',
            default=False,
            help='''To specify all hotkeys. Specifying hotkeys will exclude them from this all.'''
        )
        bittensor.wallet.add_args( senate_parser )
        bittensor.subtensor.add_args( senate_parser )