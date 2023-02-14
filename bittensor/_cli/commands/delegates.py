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
from typing import List, Optional, Dict
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
console = bittensor.__console__

# Uses rich console to pretty print a table of delegates.
def show_delegates( delegates: List['bittensor.DelegateInfo'], width: Optional[int] = None):
    delegates.sort(key=lambda delegate: delegate.total_stake, reverse=False)
    table = Table(show_footer=True, width=width, pad_edge=False, box=None)
    table.add_column("[overline white]DELEGATE",  str(len(delegates)), footer_style = "overline white", style='bold white')
    table.add_column("[overline white]TAKE", style='white')
    table.add_column("[overline white]OWNER", style='yellow')
    table.add_column("[overline white]NOMINATORS", justify='right', style='green', no_wrap=True)
    table.add_column("[overline white]TOTAL STAKE(\u03C4)", justify='right', style='green', no_wrap=True)
    for delegate in delegates:
        table.add_row(
            str(delegate.hotkey_ss58),
            str(delegate.take),
            str(delegate.owner_ss58),
            str(len(delegate.nominators)),
            str(delegate.total_stake),
        )
    bittensor.__console__.print(table)

class DelegateStakeCommand:

    @staticmethod
    def run( cli ):
        '''Delegates stake to a chain delegate.'''
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )
        subtensor.delegate( 
            wallet = wallet, 
            delegate_ss58 = config.get('delegate_ss58key'), 
            amount = config.get('amount'), 
            wait_for_inclusion = True, 
            prompt = not config.no_prompt 
        )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        delegate_stake_parser = parser.add_parser(
            'delegate', 
            help='''Delegate Stake to an account.'''
        )
        delegate_stake_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        delegate_stake_parser.add_argument(
            '--delegate_ss58key', 
            dest = "delegate_ss58key",
            type = float,  
            required = False,
            help='''The ss58 address of the choosen delegate''', 
        )
        delegate_stake_parser.add_argument(
            '--all', 
            dest="stake_all", 
            action='store_true'
        )
        delegate_stake_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )        
        delegate_stake_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( delegate_stake_parser )
        bittensor.subtensor.add_args( delegate_stake_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.get('delegate_ss58key'):
            # Check for delegates.
            with bittensor.__console__.status(":satellite: Loading delegates..."):
                subtensor = bittensor.subtensor( config = config )
                delegates: List[bittensor.DelegateInfo] = subtensor.get_delegates()

            if len(delegates) == 0:
                console.print(":cross_mark:[red]There are no delegates on {}[/red]".format(subtensor.network))
                sys.exit(1)
            
            show_delegates( delegates )
            delegate_ss58key = Prompt.ask("Enter the delegate's ss58key")
            config.delegate_ss58key = str(delegate_ss58key)
            
        # Get amount.
        if not config.get('amount') and not config.get('stake_all'):
            if not Confirm.ask("Stake all Tao from account: [bold]'{}'[/bold]?".format(config.wallet.get('name', bittensor.defaults.wallet.name))):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                config.stake_all = True

class DelegateUnstakeCommand:

    @staticmethod
    def run( cli ):
        '''Undelegates stake from a chain delegate.'''
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )
        subtensor.undelegate( 
            wallet = wallet, 
            delegate_ss58 = config.get('delegate_ss58key'), 
            amount = config.get('amount'), 
            wait_for_inclusion = True, 
            prompt = not config.no_prompt 
        )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        undelegate_stake_parser = parser.add_parser(
            'undelegate', 
            help='''Undelegate Stake from an account.'''
        )
        undelegate_stake_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        undelegate_stake_parser.add_argument(
            '--delegate_ss58key', 
            dest = "delegate_ss58key",
            type = str,  
            required = False,
            help='''The ss58 address of the choosen delegate''', 
        )
        undelegate_stake_parser.add_argument(
            '--all', 
            dest="unstake_all", 
            action='store_true'
        )
        undelegate_stake_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )        
        undelegate_stake_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( undelegate_stake_parser )
        bittensor.subtensor.add_args( undelegate_stake_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.get('delegate_ss58key'):
            # Check for delegates.
            with bittensor.__console__.status(":satellite: Loading delegates..."):
                subtensor = bittensor.subtensor( config = config )
                delegates: List[bittensor.DelegateInfo] = subtensor.get_delegates()

            if len(delegates) == 0:
                console.print(":cross_mark:[red]There are no delegates on {}[/red]".format(subtensor.network))
                sys.exit(1)
            
            show_delegates( delegates )
            delegate_ss58key = Prompt.ask("Enter the delegate's ss58key")
            config.delegate_ss58key = str(delegate_ss58key)
            
        # Get amount.
        if not config.get('amount') and not config.get('unstake_all'):
            if not Confirm.ask("Unstake all Tao to account: [bold]'{}'[/bold]?".format(config.wallet.get('name', bittensor.defaults.wallet.name))):
                amount = Prompt.ask("Enter Tao amount to unstake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                config.stake_all = True

class ListDelegatesCommand:

    @staticmethod
    def run( cli ):
        r"""
        List all delegates on the network.
        """
        subtensor = bittensor.subtensor( config = cli.config )
        with bittensor.__console__.status(":satellite: Loading delegates..."):
            delegates: bittensor.DelegateInfo = subtensor.get_delegates()
        show_delegates( delegates, width = cli.config.get('width', None) )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        list_delegates_parser = parser.add_parser(
            'list_delegates', 
            help='''List all delegates on the network'''
        )
        list_delegates_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.subtensor.add_args( list_delegates_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)



class NominateCommand:

    @staticmethod
    def run( cli ):
        r""" Nominate wallet.
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

        result: bool = subtensor.nominate( wallet )
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
        nominate_parser = parser.add_parser(
            'nominate', 
            help='''Become a delegate on the network'''
        )
        nominate_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( nominate_parser )
        bittensor.subtensor.add_args( nominate_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)





      




      