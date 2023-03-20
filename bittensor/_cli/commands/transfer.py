
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

import sys
import argparse
import bittensor
from rich.prompt import Prompt
console = bittensor.__console__

class TransferCommand:
    @staticmethod
    def run (cli):
        r""" Transfer token of amount to destination."""
        wallet = bittensor.wallet( config = cli.config )
        subtensor = bittensor.subtensor( config = cli.config )
        subtensor.transfer( wallet = wallet, dest = cli.config.dest, amount = cli.config.amount, wait_for_inclusion = True, prompt = not cli.config.no_prompt )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Get destination.
        if not config.dest:
            dest = Prompt.ask("Enter destination public key: (ss58 or ed2519)")
            if not bittensor.utils.is_valid_bittensor_address_or_public_key( dest ):
                sys.exit()
            else:
                config.dest = str(dest)

        # Get current balance and print to user.
        if not config.no_prompt:
            wallet = bittensor.wallet( config )
            subtensor = bittensor.subtensor( config )
            with bittensor.__console__.status(":satellite: Checking Balance..."):
                account_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
                bittensor.__console__.print("Balance: [green]{}[/green]".format(account_balance))
                    
        # Get amount.
        if not config.get('amount'):
            if not config.no_prompt:
                amount = Prompt.ask("Enter TAO amount to transfer")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                console.print(":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(amount))
                sys.exit(1)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        transfer_parser = parser.add_parser(
            'transfer', 
            help='''Transfer Tao between accounts.'''
        )
        transfer_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        transfer_parser.add_argument(
            '--dest', 
            dest="dest", 
            type=str, 
            required=False
        )
        transfer_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )
        transfer_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( transfer_parser )
        bittensor.subtensor.add_args( transfer_parser )