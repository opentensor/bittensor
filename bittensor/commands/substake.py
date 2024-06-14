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
from rich.prompt import Confirm, Prompt
from bittensor.utils.balance import Balance
from typing import List, Union, Optional, Dict, Tuple
from .utils import get_hotkey_wallets_for_wallet
from bittensor.utils.user_io import (
    user_input_float,
    user_input_int,
    user_input_str,
    user_input_confirmation,
    print_summary_header,
    print_summary_footer,
    print_summary_item,
)
from . import defaults

console = bittensor.__console__

class SubStakeCommand:
    """
    Adds stake to a specific hotkey account on a specific subnet, specified by `netuid`.

    Usage:
        Users can specify the amount to stake, the hotkeys to stake to (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the staking process.

    Optional arguments:
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--netuid`` (int): The subnet to stake to.
        - ``--max_stake`` (float): Sets the maximum amount of TAO to have staked in each hotkey.
        - ``--hotkey`` (str): Specifies hotkey by name or SS58 address to stake to.

    The command prompts for confirmation before executing the staking operation.

    Example usage (args):

        btcli substake add --amount <tao amt to stake> --wallet.name <wallet to pull tao from> --hotkey <ss58_address stake destination>

        btcli substake add --amount 100 --netuid 1 --wallet.name default --hotkey 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

    Example usgage (prompt):

        btcli substake add

        Enter netuid (0): 1

        Enter wallet name (default):

        Enter hotkey name or ss58_address to stake to (default):

        Stake all Tao from account: 'default'? [y/n]: n

        Enter Tao amount to stake: 100

        Do you want to stake to the following hotkey on netuid 1:
          - from   default:5GeYLB44QY9wcqJmFZvJW8D3EYPDaJGSgGfkbJVxUbkVcU7C
          - to     default:5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH
          - amount 100.0 τ
          - for:
        [y/n]: y
        Enter password to unlock key:
        ✅ Finalized

        Balance:
          τ100 ➡ τ0
        Stake:
          τ1,234 ➡ τ1,334

    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Stake token of amount to hotkey on subnet of given netuid."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            SubStakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Stake token of amount to hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)
        dynamic_info = subtensor.get_dynamic_info_for_netuid(config.netuid)
        hotkey_tup: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)

        assert bittensor.utils.is_valid_ss58_address(config.get("hotkey"))
        hotkey_tup = (config.get("hotkey_name"), config.get("hotkey"))

        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        if not subtensor.is_hotkey_registered_any(hotkey_ss58=hotkey_tup[1]):
            # Hotkey is not registered.
            bittensor.__console__.print(
                f"[red]Hotkey [bold]{hotkey_tup[1]}[/bold] is not registered. Aborting.[/red]"
            )
            return None

        # Amount to stake
        if config.get("amount"):
            stake_amount_tao = bittensor.Balance.from_tao(config.get("amount"))

        elif config.get("stake_all"):
            old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
            stake_amount_tao = bittensor.Balance.from_tao(old_balance.tao)

        else:
            stake_amount_tao = bittensor.Balance.from_tao(0.0)

        # Print summary
        print_summary_header("Add Subnet Stake")
        print_summary_item("wallet", wallet.name)
        print_summary_item("netuid", config.netuid)
        print_summary_item("price", dynamic_info.price)
        amount = "stake all"
        if not config.get("stake_all"):
            amount = stake_amount_tao.__str__()
        hotkey_str = hotkey_tup[0] if hotkey_tup[0] != None else hotkey_tup[1]
        print_summary_item("hotkey", hotkey_str)
        print_summary_item("amount to stake", amount)
        print_summary_item("amount received", dynamic_info.tao_to_alpha_with_slippage(stake_amount_tao)[0])
        print_summary_footer()

        # Ask to continue
        if not config.no_prompt:
            if not user_input_confirmation("continue"):
                return None

        return subtensor.add_substake(
            wallet=wallet,
            hotkey_ss58=hotkey_tup[1],
            netuid=config.netuid,
            amount=stake_amount_tao,
            wait_for_inclusion=True,
            prompt=not config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Retrieve hotkey string from cli parameters
        if config.get("hotkey"):
            hotkey_str = config.get("hotkey")
        elif config.get("wallet.hotkey"):
            hotkey_str = config.get("wallet.hotkey")
        elif config.wallet.get("hotkey"):
            hotkey_str = config.wallet.get("hotkey")
        elif not config.no_prompt:
            hotkey_str = Prompt.ask(
                "Enter hotkey name or ss58_address to stake to",
                default=defaults.wallet.hotkey,
            )
        else:
            print("ERROR: Hotkey is needed to proceed")
            sys.exit(1)

        # parse hotkey string into config.hotkey and config.hotkey_name if available
        if bittensor.utils.is_valid_ss58_address(hotkey_str):
            config.hotkey = str(hotkey_str)
        else:
            config.hotkey_name = hotkey_str
            wallet_delegate = bittensor.wallet(name=hotkey_str)
            config.hotkey = wallet_delegate.hotkey.ss58_address

        if not config.is_set("netuid") and not config.no_prompt:
            netuid = Prompt.ask("Enter netuid", default="0")
            config.netuid = int(netuid)
        if not config.netuid:
            print("ERROR: netuid is needed to proceed")
            sys.exit(1)

        # Get amount.
        if not config.get("amount") and not config.get("max_stake"):
            if not Confirm.ask(
                "Stake all Tao from account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.stake_all = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "add",
            help="""Add stake to a specific hotkey on subnet `netuid` from your coldkey.""",
        )
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--hotkey",
            "--wallet.hotkey",
            required=False,
            type=str,
            help="""Specify the hotkey by name or ss58 address.""",
        )
        bittensor.wallet.add_args(stake_parser)
        bittensor.subtensor.add_args(stake_parser)


class RemoveSubStakeCommand:
    """
    Removes stake to a specific hotkey account on a specific subnet, specified by `netuid`.

    Usage:
        Users can specify the amount to unstake, the hotkey to unstake from (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the unstaking process.

    Optional arguments:
        - ``--wallet.name (str)``: The wallet return TAO to.
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--netuid`` (int): The subnet to stake to.
        - ``--hotkey`` (str): Specifies hotkey by name or SS58 address to stake to.
        - ``--all`` (bool): Unstake all TAO from the specified hotkey.

    The command prompts for confirmation before executing the staking operation.

    Example usage (args):

        btcli substake remove --amount <tao amt to unstake> --wallet.name <wallet coldkey to return tao to> --hotkey <ss58_address to unstake from>

        btcli substake remove --amount 111 --wallet.name default --netuid 1 --hotkey 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

    Example usage (prompt):

        btcli substake remove

        Enter netuid (0): 1

        Enter wallet name (default):

        Enter hotkey name or ss58_address to unstake from (default): 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

        Unstake all τao
        from account: 'default'
        and hotkey  : '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'
        from subnet : '1'
        [y/n]: n

        Enter Tao amount to unstake: 1337

        Enter password to unlock key:

        Do you want to unstake:
            amount: τ500
            from  : default
            netuid: 1
        [y/n]: y

        ✅ Finalized
        Balance:
            τ5000 ➡ τ6337
        Unstaked:
            τ16,961 ➡ τ15,623

    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Unstake token of amount to hotkey on subnet of given netuid."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RemoveSubStakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Unstake token amount from hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)
        dynamic_info = subtensor.get_dynamic_info_for_netuid(config.netuid)

        hotkey_tup: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)

        assert bittensor.utils.is_valid_ss58_address(config.get("hotkey"))
        hotkey_tup = (config.get("hotkey_name"), config.get("hotkey"))

        # Get coldkey balance
        if not subtensor.is_hotkey_registered_any(hotkey_ss58=hotkey_tup[1]):
            # Hotkey is not registered.
            bittensor.__console__.print(
                f"[red]Hotkey [bold]{hotkey_tup[1]}[/bold] is not registered. Aborting.[/red]"
            )
            return None

        # Get the current stake of the hotkey from this coldkey.
        hotkey_subnet_balance: Balance = (
            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                hotkey_ss58=hotkey_tup[1],
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                netuid=config.netuid,
            )
        )

        # Get amount to unstake
        if config.get("amount"):
            unstake_amount_alpha = bittensor.Balance.from_tao(config.get("amount"))

        elif config.get("unstake_all"):
            unstake_amount_alpha = hotkey_subnet_balance

        else:
            unstake_amount_alpha = bittensor.Balance.from_tao(0.0)

        if hotkey_subnet_balance < unstake_amount_alpha:
            bittensor.__console__.print(
                f"Unstake amount [green][bold]{unstake_amount_alpha}[/bold][/green] is greater than current stake for hotkey [bold]{hotkey_tup[1]}[/bold]. Unstaking all."
            )
            unstake_amount_alpha = hotkey_subnet_balance

        # Get currently staked on hotkey provided
        currently_staked = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
            netuid=config.netuid,
            hotkey_ss58=hotkey_tup[1],
            coldkey_ss58=wallet.coldkeypub.ss58_address,
        )

        # Print summary
        print_summary_header("Remove Subnet Stake")
        print_summary_item("wallet", wallet.name)
        print_summary_item("netuid", config.netuid)
        print_summary_item("price", dynamic_info.price)
        amount = "unstake all"
        if not config.get("unstake_all"):
            amount = unstake_amount_alpha.__str__()
        hotkey_str = hotkey_tup[0] if hotkey_tup[0] != None else hotkey_tup[1]
        print_summary_item("hotkey", hotkey_str)
        print_summary_item("amount to unstake", amount)
        print_summary_item("amount received", dynamic_info.alpha_to_tao_with_slippage(unstake_amount_alpha)[0])
        print_summary_item("currently staked", currently_staked)
        print_summary_footer()

        # Ask to continue
        if not config.no_prompt:
            if not user_input_confirmation("continue"):
                return None

        return subtensor.remove_substake(
            wallet=wallet,
            hotkey_ss58=hotkey_tup[1],
            netuid=config.netuid,
            amount=unstake_amount_alpha,
            wait_for_inclusion=True,
            prompt=not config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.is_set("hotkey")
            and not config.is_set("wallet.hotkey")
            and not config.wallet.get("hotkey")
            and not config.no_prompt
        ):
            hotkey = Prompt.ask(
                "Enter hotkey name or ss58_address to unstake from",
                default=defaults.wallet.hotkey,
            )
            if bittensor.utils.is_valid_ss58_address(hotkey):
                config.hotkey = str(hotkey)
            else:
                config.hotkey_name = hotkey
                wallet_delegate = bittensor.wallet(name=hotkey)
                config.hotkey = wallet_delegate.hotkey.ss58_address

        if not config.is_set("netuid") and not config.no_prompt:
            netuid = Prompt.ask("Enter netuid", default="0")
            config.netuid = int(netuid)

        # Get amount.
        if not config.get("amount") and not config.get("unstake_all"):
            if not Confirm.ask("Unstake all Alpha"):
                amount = Prompt.ask("Enter Alpha amount to unstake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red]Invalid Alpha amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.unstake_all = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "remove",
            help="""Remove stake from a specific hotkey on subnet `netuid` from your coldkey.""",
        )
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--all", dest="unstake_all", action="store_true")
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--hotkey",
            "--wallet.hotkey",
            required=False,
            type=str,
            help="""Specify the hotkey by name or ss58 address.""",
        )
        bittensor.wallet.add_args(stake_parser)
        bittensor.subtensor.add_args(stake_parser)
