# The MIT License (MIT)
# Copyright © 2023 OpenTensor Foundation

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
from typing import List, Optional
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
from rich.console import Text
from dataclasses import dataclass
import requests

console = bittensor.__console__
@dataclass
class DelegateProfile:
    """ A delegate profile from GitHub.
    See: https://github.com/opentensor/delegate_profiles/blob/master/delegate_profiles.md
    """
    delegate_ss58: str
    displayname: str
    discord: str
    website: str
    twitter: str
    telegram: str
    description: str # 140 characters max

    @staticmethod
    def empty():
        return DelegateProfile(
            delegate_ss58='',
            displayname = '',
            discord = '',
            website = '',
            twitter = '',
            telegram = '',
            description = ''
        )
    
    @staticmethod
    def parse_line(line: str) -> 'DelegateProfile':
        """ Parses a line from a delegate profile markdown file.
        """
        # Split the line into columns.
        columns = line.split('|')
        if len(columns) < 9:
            return DelegateProfile.empty()
        
        # Parse the columns.
        delegate_ss58 = columns[1].strip()
        # Skip column 2, it's the owner ss58.
        displayname = columns[3].strip()
        discord = columns[4].strip()
        website = columns[5].strip()
        twitter = columns[6].strip()
        telegram = columns[7].strip()
        description = columns[8].strip()
        
        return DelegateProfile(
            delegate_ss58=delegate_ss58,
            displayname=displayname,
            discord=discord,
            website=website,
            twitter=twitter,
            telegram=telegram,
            description=description
        )
    
    @staticmethod
    def parse_lines(lines: List[str]) -> List['DelegateProfile']:
        """ Parses a list of lines from a delegate profile markdown file.
        """
        # Skip lines until we find the start of the table.
        while len(lines) > 0 and not '| Delegate' in lines[0].strip():
            lines.pop(0)

        # Skip the table header and separator.
        if len(lines) > 2:
            lines.pop(0) 
            lines.pop(0)
        else:
            return []
        
        # Parse the table.
        delegate_profiles: List[DelegateProfile] = []
        while len(lines) > 0 and '|' in lines[0].strip():
            delegate_profile = DelegateProfile.parse_line(lines.pop(0))
            delegate_profiles.append(delegate_profile)

        return delegate_profiles


def get_delegate_profile_readme_from_github() -> List[str]:
    """ Pulls the latest delegate profiles from GitHub.
    """
    try:
        with open('DELEGATES.md', 'r') as f:
            lines = f.readlines()
        
        return lines
    except Exception as e:
        bittensor.__console__.print(f'Failed to read local delegate profiles: {e}')
        return []

    # TODO(camfairchild): Use live github repo instead of local file.
    url: str = bittensor.__delegate_profiles_url__
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text.splitlines()
        else:
            # Some error occured.
            bittensor.__console__.print(f'Failed to pull delegate profiles from GitHub: {response.status_code}')
            return []
    except Exception as e:
        bittensor.__console__.print(f'Failed to pull delegate profiles from GitHub: {e}')
        return []


def get_delegate_profiles_from_github() -> List[DelegateProfile]:
    """ Reads delegate profiles from GitHub.
    """
    lines: List[str] = get_delegate_profile_readme_from_github()
    
    delegate_profiles = DelegateProfile.parse_lines(lines)
    
    return delegate_profiles

# Uses rich console to pretty print a table of delegates.
def show_delegates( delegates: List['bittensor.DelegateInfo'], width: Optional[int] = None):
    """ Pretty prints a table of delegates sorted by total stake.
    """
    delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
    table = Table(show_footer=True, width=width, pad_edge=False, box=None)
    table.add_column("[overline white]SS58",  str(len(delegates)), footer_style = "overline white", style='bold yellow')
    #table.add_column("[overline white]OWNER", style='yellow')
    table.add_column("[overline white]DISPLAY NAME", justify='center', style='white', no_wrap=True)
    table.add_column("[overline white]NOMS", justify='center', style='green', no_wrap=True)
    table.add_column("[overline white]OWNER STAKE(\u03C4)", justify='right', no_wrap=True)
    table.add_column("[overline white]TOTAL STAKE(\u03C4)", justify='right', style='green', no_wrap=True)
    table.add_column("[overline white]SUBNETS", justify='right', style='white', no_wrap=True)
    table.add_column("[overline white]VPERMIT", justify='right', no_wrap=True)
    #table.add_column("[overline white]TAKE", style='white', no_wrap=True)
    table.add_column("[overline white]24h/k\u03C4", style='green', justify='center')
    table.add_column("[overline white]WEBSITE", style='rgb(50,163,219)')
    #table.add_column("[overline white]DESCRIPTION", style='white')

    delegate_profiles = get_delegate_profiles_from_github()
    delegate_profiles_map = { profile.delegate_ss58: profile for profile in delegate_profiles }

    for delegate in delegates:
        owner_stake = next(
            map(lambda x: x[1], # get stake
                filter(lambda x: x[0] == delegate.owner_ss58, delegate.nominators) # filter for owner
            ),
            bittensor.Balance.from_rao(0) # default to 0 if no owner stake.
        )
        delegate_profile: Optional[DelegateProfile] = delegate_profiles_map.get(delegate.hotkey_ss58, None)
        if delegate_profile is None:
            delegate_profile = DelegateProfile.empty()
        
        table.add_row(
            f'{delegate.hotkey_ss58:8.8}...',
            #f'{delegate.owner_ss58:8.8}...',
            str(delegate_profile.displayname),
            str(len(delegate.nominators)),
            f'{owner_stake!s:13.13}',
            f'{delegate.total_stake!s:13.13}',
            str(delegate.registrations),
            str(['*' if subnet in delegate.validator_permits else '' for subnet in delegate.registrations]),
            #f'{delegate.take * 100:.1f}%',
            f'{delegate.return_per_1000!s:6.6}',
            Text(delegate_profile.website, style=f'link {delegate_profile.website}') if delegate_profile.website else '',
            #f'{delegate_profile.description:140.140}',
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
            '--delegate_ss58',
            dest = "delegate_ss58key",
            type = str,  
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
            '--delegate_ss58',
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
                config.unstake_all = True

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





      




      