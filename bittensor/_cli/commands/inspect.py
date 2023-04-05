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

import json
import argparse
import bittensor
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Prompt
from .utils import check_netuid_set, get_delegates_details, DelegatesDetails
console = bittensor.__console__

import os
import bittensor
from typing import List, Tuple, Optional, Dict

def _get_coldkey_wallets_for_path( path: str ) -> List['bittensor.wallet']:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [ bittensor.wallet( path= path, name=name ) for name in wallet_names ]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets

def _get_hotkey_wallets_for_wallet( wallet ) -> List['bittensor.wallet']:
    hotkey_wallets = []
    hotkeys_path = wallet.path + '/' + wallet.name + '/hotkeys'
    try:
        hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
    except StopIteration:
        hotkey_files = []
    for hotkey_file_name in hotkey_files:
        try:
            hotkey_for_name = bittensor.wallet( path = wallet.path, name = wallet.name, hotkey = hotkey_file_name )
            if hotkey_for_name.hotkey_file.exists_on_device() and not hotkey_for_name.hotkey_file.is_encrypted():
                hotkey_wallets.append( hotkey_for_name )
        except Exception:
            pass
    return hotkey_wallets

class InspectCommand:
    @staticmethod
    def run (cli):
        r""" Inspect a cold, hot pair.
        """
        if cli.config.get('all', d=False) == True:
            wallets = _get_coldkey_wallets_for_path( cli.config.wallet.path )
        else:
            wallets = [bittensor.wallet( config = cli.config )]
        subtensor = bittensor.subtensor( config = cli.config )

        netuids = subtensor.get_all_subnet_netuids()

        registered_delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(url = bittensor.__delegates_details_url__)
        if registered_delegate_info is None:
            bittensor.__console__.print( ':warning:[yellow]Could not get delegate info from chain.[/yellow]')
            registered_delegate_info = {}

        neuron_state_dict = {}
        for netuid in tqdm( netuids ):
            neuron_state_dict[netuid] = subtensor.neurons_lite( netuid )

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column("[overline white]Coldkey", footer_style = "overline white", style='bold white')
        table.add_column("[overline white]Balance", footer_style = "overline white", style='green')
        table.add_column("[overline white]Delegate", footer_style = "overline white", style='blue')
        table.add_column("[overline white]Stake", footer_style = "overline white", style='green')
        table.add_column("[overline white]Emission", footer_style = "overline white", style='green')
        table.add_column("[overline white]Netuid", footer_style = "overline white", style='bold white')
        table.add_column("[overline white]Hotkey", footer_style = "overline white", style='yellow')
        table.add_column("[overline white]Stake", footer_style = "overline white", style='green')
        table.add_column("[overline white]Emission", footer_style = "overline white", style='green')
        for wallet in tqdm( wallets ):
            delegates: List[Tuple(bittensor.DelegateInfo, bittensor.Balance)] = subtensor.get_delegated( coldkey_ss58=wallet.coldkeypub.ss58_address )
            if not wallet.coldkeypub_file.exists_on_device(): continue
            cold_balance = wallet.get_balance( subtensor = subtensor )
            table.add_row(
                wallet.name,
                str(cold_balance),
                '',
                '',
                '',
                '',
                '',
                '',
                '',
            )
            for dele, staked in delegates:
                if dele.hotkey_ss58 in registered_delegate_info:
                    delegate_name = registered_delegate_info[dele.hotkey_ss58].name
                else:
                    delegate_name = dele.hotkey_ss58
                table.add_row(
                    '',
                    '',
                    str(delegate_name),
                    str(staked),
                    str(dele.total_daily_return.tao * (staked.tao/dele.total_stake.tao)),
                    '',
                    '',
                    '',
                    ''
                )

            hotkeys = _get_hotkey_wallets_for_wallet( wallet )
            for netuid in netuids:
                for neuron in neuron_state_dict[netuid]:
                    if neuron.coldkey == wallet.coldkeypub.ss58_address:
                        table.add_row(
                            '',
                            '',
                            '',
                            '',
                            '',
                            str( netuid ),
                            str( neuron.hotkey ),
                            str( neuron.stake ),
                            str( bittensor.Balance.from_tao(neuron.emission) )
                        )
               
        bittensor.__console__.print(table)
            
                

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if not config.get( 'all', d=None ) and config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        inspect_parser = parser.add_parser(
            'inspect', 
            help='''Inspect a wallet (cold, hot) pair'''
        )
        inspect_parser.add_argument( 
            '--all', 
            action='store_true', 
            help='''Check all coldkey wallets.''', 
            default = False 
        )
        inspect_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        inspect_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.wallet.add_args( inspect_parser )
        bittensor.subtensor.add_args( inspect_parser )