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

class StakeCommand:

    @staticmethod
    def run( cli ):
        r""" Stake token of amount to hotkey(s).
        """
        config = cli.config.copy()
        wallet = bittensor.wallet( config = config )
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )
        
        # Get the hotkey_names (if any) and the hotkey_ss58s.
        hotkeys_to_stake_to: List[Tuple[Optional[str], str]] = []
        if config.get('all_hotkeys'):
            # Stake to all hotkeys.
            all_hotkeys: List[bittensor.wallet] = get_hotkey_wallets_for_wallet( wallet = wallet )
            # Get the hotkeys to exclude. (d)efault to no exclusions.
            hotkeys_to_exclude: List[str] = cli.config.get('hotkeys', d=[])
            # Exclude hotkeys that are specified.
            hotkeys_to_stake_to = [
                (wallet.hotkey_str, wallet.hotkey.ss58_address) for wallet in all_hotkeys
                    if wallet.hotkey_str not in hotkeys_to_exclude
            ] # definitely wallets

        elif config.get('hotkeys'):
            # Stake to specific hotkeys.
            for hotkey_ss58_or_hotkey_name in config.get('hotkeys'):
                if bittensor.utils.is_valid_ss58_address( hotkey_ss58_or_hotkey_name ):
                    # If the hotkey is a valid ss58 address, we add it to the list.
                    hotkeys_to_stake_to.append( (None, hotkey_ss58_or_hotkey_name ) )
                else:
                    # If the hotkey is not a valid ss58 address, we assume it is a hotkey name.
                    #  We then get the hotkey from the wallet and add it to the list.
                    wallet_ = bittensor.wallet( config = config, hotkey = hotkey_ss58_or_hotkey_name )
                    hotkeys_to_stake_to.append( (wallet_.hotkey_str, wallet_.hotkey.ss58_address ) )
        elif config.wallet.get('hotkey'):
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            hotkey_ss58_or_name = config.wallet.get('hotkey')
            if bittensor.utils.is_valid_ss58_address( hotkey_ss58_or_name ):
                hotkeys_to_stake_to = [ (None, hotkey_ss58_or_name) ]
            else:
                # Hotkey is not a valid ss58 address, so we assume it is a hotkey name.
                wallet_ = bittensor.wallet( config = config, hotkey = hotkey_ss58_or_name )
                hotkeys_to_stake_to = [ (wallet_.hotkey_str, wallet_.hotkey.ss58_address ) ]
        else:
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            assert config.wallet.hotkey is not None
            hotkeys_to_stake_to = [ (None, bittensor.wallet( config = config ).hotkey.ss58_address) ]
        
        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
        final_hotkeys: List[Tuple[str, str]] = [] 
        final_amounts: List[Union[float, Balance]] = []
        for hotkey in tqdm(hotkeys_to_stake_to):
            hotkey: Tuple[Optional[str], str] # (hotkey_name (or None), hotkey_ss58)
            if not subtensor.is_hotkey_registered_any( hotkey_ss58 = hotkey[1] ):
                # Hotkey is not registered.
                if (len(hotkeys_to_stake_to) == 1):
                    # Only one hotkey, error
                    bittensor.__console__.print(f"[red]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Aborting.[/red]")
                    return None
                else:
                    # Otherwise, print warning and skip
                    bittensor.__console__.print(f"[yellow]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Skipping.[/yellow]")
                    continue


            stake_amount_tao: float = config.get('amount')
            if config.get('max_stake'):
                # Get the current stake of the hotkey from this coldkey.
                hotkey_stake: Balance = subtensor.get_stake_for_coldkey_and_hotkey( hotkey_ss58 = hotkey[1], coldkey_ss58 = wallet.coldkeypub.ss58_address )
                stake_amount_tao: float = config.get('max_stake') - hotkey_stake.tao

                # If the max_stake is greater than the current wallet balance, stake the entire balance.
                stake_amount_tao: float = min(stake_amount_tao, wallet_balance.tao)
                if stake_amount_tao <= 0.00001: # Threshold because of fees, might create a loop otherwise
                    # Skip hotkey if max_stake is less than current stake.
                    continue
                wallet_balance = Balance.from_tao(wallet_balance.tao - stake_amount_tao)
            
                if wallet_balance.tao < 0:
                    # No more balance to stake.
                    break

            final_amounts.append(stake_amount_tao)
            final_hotkeys.append(hotkey) # add both the name and the ss58 address.

        if len(final_hotkeys) == 0:
            # No hotkeys to stake to.
            bittensor.__console__.print("Not enough balance to stake to any hotkeys or max_stake is less than current stake.")
            return None

        # Ask to stake
        if not config.no_prompt:
            if not Confirm.ask(f"Do you want to stake to the following keys from {wallet.name}:\n" + \
                    "".join([
                        f"    [bold white]- {hotkey[0] + ':' if hotkey[0] else ''}{hotkey[1]}: {f'{amount} {bittensor.__tao_symbol__}' if amount else 'All'}[/bold white]\n" for hotkey, amount in zip(final_hotkeys, final_amounts)
                    ])
                ):
                return None
        
        if len(final_hotkeys) == 1:
            # do regular stake
            return subtensor.add_stake( wallet=wallet, hotkey_ss58 = final_hotkeys[0][1], amount = None if config.get('stake_all') else final_amounts[0], wait_for_inclusion = True, prompt = not config.no_prompt )

        subtensor.add_stake_multiple( wallet = wallet, hotkey_ss58s=[hotkey_ss58 for _, hotkey_ss58 in final_hotkeys], amounts =  None if config.get('stake_all') else final_amounts, wait_for_inclusion = True, prompt = False )


    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt and not config.get('all_hotkeys') and not config.get('hotkeys'):
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
                    
        # Get amount.
        if not config.get('amount') and not config.get('stake_all') and not config.get('max_stake'):
            if not Confirm.ask("Stake all Tao from account: [bold]'{}'[/bold]?".format(config.wallet.get('name', bittensor.defaults.wallet.name))):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                config.stake_all = True

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        stake_parser = parser.add_parser(
            'stake', 
            help='''Stake to your hotkey accounts.'''
        )
        stake_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        stake_parser.add_argument(
            '--all', 
            dest="stake_all", 
            action='store_true'
        )
        stake_parser.add_argument(
            '--uid', 
            dest="uid", 
            type=int, 
            required=False
        )
        stake_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )        
        stake_parser.add_argument(
            '--max_stake', 
            dest="max_stake",
            type=float,
            required=False,
            action='store',
            default=None,
            help='''Specify the maximum amount of Tao to have staked in each hotkey.'''
        )
        stake_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        stake_parser.add_argument(
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
        stake_parser.add_argument(
            '--all_hotkeys',
            '--wallet.all_hotkeys',
            required=False,
            action='store_true',
            default=False,
            help='''To specify all hotkeys. Specifying hotkeys will exclude them from this all.'''
        )
        bittensor.wallet.add_args( stake_parser )
        bittensor.subtensor.add_args( stake_parser )