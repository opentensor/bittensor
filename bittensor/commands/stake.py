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
import os
import sys
import re
from typing import List, Union, Optional, Dict, Tuple

from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.console import Console
from rich.text import Text
from tqdm import tqdm

import bittensor
from bittensor.utils.balance import Balance
from .utils import (
    get_hotkey_wallets_for_wallet,
    get_delegates_details,
    DelegatesDetails,
)
from . import defaults  # type: ignore
from ..utils import wallet_utils
from ..utils.formatting import u64_to_float, u16_to_float

console = bittensor.__console__


class StakeCommand:
    """
    Executes the ``add`` command to stake tokens to one or more hotkeys from a user's coldkey on the Bittensor network.

    This command is used to allocate tokens to different hotkeys, securing their position and influence on the network.

    Usage:
        Users can specify the amount to stake, the hotkeys to stake to (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the staking process.

    Optional arguments:
        - ``--all`` (bool): When set, stakes all available tokens from the coldkey.
        - ``--uid`` (int): The unique identifier of the neuron to which the stake is to be added.
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--max_stake`` (float): Sets the maximum amount of TAO to have staked in each hotkey.
        - ``--hotkeys`` (list): Specifies hotkeys by name or SS58 address to stake to.
        - ``--all_hotkeys`` (bool): When set, stakes to all hotkeys associated with the wallet, excluding any specified in --hotkeys.

    The command prompts for confirmation before executing the staking operation.

    Example usage::

        btcli stake add --amount 100 --wallet.name <my_wallet> --wallet.hotkey <my_hotkey>

    Note:
        This command is critical for users who wish to distribute their stakes among different neurons (hotkeys) on the network.
        It allows for a strategic allocation of tokens to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Stake token of amount to hotkey(s)."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            StakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Stake token of amount to hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        # Get the hotkey_names (if any) and the hotkey_ss58s.
        hotkeys_to_stake_to: List[Tuple[Optional[str], str]] = []
        if config.get("all_hotkeys"):
            # Stake to all hotkeys.
            all_hotkeys: List[bittensor.wallet] = get_hotkey_wallets_for_wallet(
                wallet=wallet
            )
            # Get the hotkeys to exclude. (d)efault to no exclusions.
            hotkeys_to_exclude: List[str] = cli.config.get("hotkeys", d=[])
            # Exclude hotkeys that are specified.
            hotkeys_to_stake_to = [
                (wallet.hotkey_str, wallet.hotkey.ss58_address)
                for wallet in all_hotkeys
                if wallet.hotkey_str not in hotkeys_to_exclude
            ]  # definitely wallets

        elif config.get("hotkeys"):
            # Stake to specific hotkeys.
            for hotkey_ss58_or_hotkey_name in config.get("hotkeys"):
                if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_hotkey_name):
                    # If the hotkey is a valid ss58 address, we add it to the list.
                    hotkeys_to_stake_to.append((None, hotkey_ss58_or_hotkey_name))
                else:
                    # If the hotkey is not a valid ss58 address, we assume it is a hotkey name.
                    #  We then get the hotkey from the wallet and add it to the list.
                    wallet_ = bittensor.wallet(
                        config=config, hotkey=hotkey_ss58_or_hotkey_name
                    )
                    hotkeys_to_stake_to.append(
                        (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                    )
        elif config.wallet.get("hotkey"):
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            hotkey_ss58_or_name = config.wallet.get("hotkey")
            if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_name):
                hotkeys_to_stake_to = [(None, hotkey_ss58_or_name)]
            else:
                # Hotkey is not a valid ss58 address, so we assume it is a hotkey name.
                wallet_ = bittensor.wallet(config=config, hotkey=hotkey_ss58_or_name)
                hotkeys_to_stake_to = [
                    (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                ]
        else:
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            assert config.wallet.hotkey is not None
            hotkeys_to_stake_to = [
                (None, bittensor.wallet(config=config).hotkey.ss58_address)
            ]

        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        final_hotkeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for hotkey in tqdm(hotkeys_to_stake_to):
            hotkey: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)
            if not subtensor.is_hotkey_registered_any(hotkey_ss58=hotkey[1]):
                # Hotkey is not registered.
                if len(hotkeys_to_stake_to) == 1:
                    # Only one hotkey, error
                    bittensor.__console__.print(
                        f"[red]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Aborting.[/red]"
                    )
                    return None
                else:
                    # Otherwise, print warning and skip
                    bittensor.__console__.print(
                        f"[yellow]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Skipping.[/yellow]"
                    )
                    continue

            stake_amount_tao: float = config.get("amount")
            if config.get("max_stake"):
                # Get the current stake of the hotkey from this coldkey.
                hotkey_stake: Balance = subtensor.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=hotkey[1], coldkey_ss58=wallet.coldkeypub.ss58_address
                )
                stake_amount_tao: float = config.get("max_stake") - hotkey_stake.tao

                # If the max_stake is greater than the current wallet balance, stake the entire balance.
                stake_amount_tao: float = min(stake_amount_tao, wallet_balance.tao)
                if (
                        stake_amount_tao <= 0.00001
                ):  # Threshold because of fees, might create a loop otherwise
                    # Skip hotkey if max_stake is less than current stake.
                    continue
                wallet_balance = Balance.from_tao(wallet_balance.tao - stake_amount_tao)

                if wallet_balance.tao < 0:
                    # No more balance to stake.
                    break

            final_amounts.append(stake_amount_tao)
            final_hotkeys.append(hotkey)  # add both the name and the ss58 address.

        if len(final_hotkeys) == 0:
            # No hotkeys to stake to.
            bittensor.__console__.print(
                "Not enough balance to stake to any hotkeys or max_stake is less than current stake."
            )
            return None

        # Ask to stake
        if not config.no_prompt:
            if not Confirm.ask(
                    f"Do you want to stake to the following keys from {wallet.name}:\n"
                    + "".join(
                        [
                            f"    [bold white]- {hotkey[0] + ':' if hotkey[0] else ''}{hotkey[1]}: {f'{amount} {bittensor.__tao_symbol__}' if amount else 'All'}[/bold white]\n"
                            for hotkey, amount in zip(final_hotkeys, final_amounts)
                        ]
                    )
            ):
                return None

        if len(final_hotkeys) == 1:
            # do regular stake
            return subtensor.add_stake(
                wallet=wallet,
                hotkey_ss58=final_hotkeys[0][1],
                amount=None if config.get("stake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )

        subtensor.add_stake_multiple(
            wallet=wallet,
            hotkey_ss58s=[hotkey_ss58 for _, hotkey_ss58 in final_hotkeys],
            amounts=None if config.get("stake_all") else final_amounts,
            wait_for_inclusion=True,
            prompt=False,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
                not config.is_set("wallet.hotkey")
                and not config.no_prompt
                and not config.wallet.get("all_hotkeys")
                and not config.wallet.get("hotkeys")
        ):
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        # Get amount.
        if (
                not config.get("amount")
                and not config.get("stake_all")
                and not config.get("max_stake")
        ):
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
            "add", help="""Add stake to your hotkey accounts from your coldkey."""
        )
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument("--uid", dest="uid", type=int, required=False)
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--max_stake",
            dest="max_stake",
            type=float,
            required=False,
            action="store",
            default=None,
            help="""Specify the maximum amount of Tao to have staked in each hotkey.""",
        )
        stake_parser.add_argument(
            "--hotkeys",
            "--exclude_hotkeys",
            "--wallet.hotkeys",
            "--wallet.exclude_hotkeys",
            required=False,
            action="store",
            default=[],
            type=str,
            nargs="*",
            help="""Specify the hotkeys by name or ss58 address. (e.g. hk1 hk2 hk3)""",
        )
        stake_parser.add_argument(
            "--all_hotkeys",
            "--wallet.all_hotkeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all hotkeys. Specifying hotkeys will exclude them from this all.""",
        )
        bittensor.wallet.add_args(stake_parser)
        bittensor.subtensor.add_args(stake_parser)


def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


def _get_hotkey_wallets_for_wallet(wallet) -> List["bittensor.wallet"]:
    hotkey_wallets = []
    hotkeys_path = wallet.path + "/" + wallet.name + "/hotkeys"
    try:
        hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
    except StopIteration:
        hotkey_files = []
    for hotkey_file_name in hotkey_files:
        try:
            hotkey_for_name = bittensor.wallet(
                path=wallet.path, name=wallet.name, hotkey=hotkey_file_name
            )
            if (
                    hotkey_for_name.hotkey_file.exists_on_device()
                    and not hotkey_for_name.hotkey_file.is_encrypted()
            ):
                hotkey_wallets.append(hotkey_for_name)
        except Exception:
            pass
    return hotkey_wallets


class StakeShow:
    """
    Executes the ``show`` command to list all stake accounts associated with a user's wallet on the Bittensor network.

    This command provides a comprehensive view of the stakes associated with both hotkeys and delegates linked to the user's coldkey.

    Usage:
        The command lists all stake accounts for a specified wallet or all wallets in the user's configuration directory.
        It displays the coldkey, balance, account details (hotkey/delegate name), stake amount, and the rate of return.

    Optional arguments:
        - ``--all`` (bool): When set, the command checks all coldkey wallets instead of just the specified wallet.

    The command compiles a table showing:

    - Coldkey: The coldkey associated with the wallet.
    - Balance: The balance of the coldkey.
    - Account: The name of the hotkey or delegate.
    - Stake: The amount of TAO staked to the hotkey or delegate.
    - Rate: The rate of return on the stake, typically shown in TAO per day.

    Example usage::

        btcli stake show --all

    Note:
        This command is essential for users who wish to monitor their stake distribution and returns across various accounts on the Bittensor network.
        It provides a clear and detailed overview of the user's staking activities.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Show all stake accounts."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            StakeShow._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Show all stake accounts."""
        if cli.config.get("all", d=False) == True:
            wallets = _get_coldkey_wallets_for_path(cli.config.wallet.path)
        else:
            wallets = [bittensor.wallet(config=cli.config)]
        registered_delegate_info: Optional[Dict[str, DelegatesDetails]] = (
            get_delegates_details(url=bittensor.__delegates_details_url__)
        )

        def get_stake_accounts(
                wallet, subtensor
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Get stake account details for the given wallet.

            Args:
                wallet: The wallet object to fetch the stake account details for.

            Returns:
                A dictionary mapping SS58 addresses to their respective stake account details.
            """

            wallet_stake_accounts = {}

            # Get this wallet's coldkey balance.
            cold_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

            # Populate the stake accounts with local hotkeys data.
            wallet_stake_accounts.update(get_stakes_from_hotkeys(subtensor, wallet))

            # Populate the stake accounts with delegations data.
            wallet_stake_accounts.update(get_stakes_from_delegates(subtensor, wallet))

            return {
                "name": wallet.name,
                "balance": cold_balance,
                "accounts": wallet_stake_accounts,
            }

        def get_stakes_from_hotkeys(
                subtensor, wallet
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Fetch stakes from hotkeys for the provided wallet.

            Args:
                wallet: The wallet object to fetch the stakes for.

            Returns:
                A dictionary of stakes related to hotkeys.
            """
            hotkeys = get_hotkey_wallets_for_wallet(wallet)
            stakes = {}
            for hot in hotkeys:
                emission = sum(
                    [
                        n.emission
                        for n in subtensor.get_all_neurons_for_pubkey(
                        hot.hotkey.ss58_address
                    )
                    ]
                )
                hotkey_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=hot.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                )
                stakes[hot.hotkey.ss58_address] = {
                    "name": hot.hotkey_str,
                    "stake": hotkey_stake,
                    "rate": emission,
                }
            return stakes

        def get_stakes_from_delegates(
                subtensor, wallet
        ) -> Dict[str, Dict[str, Union[str, Balance]]]:
            """Fetch stakes from delegates for the provided wallet.

            Args:
                wallet: The wallet object to fetch the stakes for.

            Returns:
                A dictionary of stakes related to delegates.
            """
            delegates = subtensor.get_delegated(
                coldkey_ss58=wallet.coldkeypub.ss58_address
            )
            stakes = {}
            for dele, staked in delegates:
                for nom in dele.nominators:
                    if nom[0] == wallet.coldkeypub.ss58_address:
                        delegate_name = (
                            registered_delegate_info[dele.hotkey_ss58].name
                            if dele.hotkey_ss58 in registered_delegate_info
                            else dele.hotkey_ss58
                        )
                        stakes[dele.hotkey_ss58] = {
                            "name": delegate_name,
                            "stake": nom[1],
                            "rate": dele.total_daily_return.tao
                                    * (nom[1] / dele.total_stake.tao),
                        }
            return stakes

        def get_all_wallet_accounts(
                wallets,
                subtensor,
        ) -> List[Dict[str, Dict[str, Union[str, Balance]]]]:
            """Fetch stake accounts for all provided wallets using a ThreadPool.

            Args:
                wallets: List of wallets to fetch the stake accounts for.

            Returns:
                A list of dictionaries, each dictionary containing stake account details for each wallet.
            """

            accounts = []
            # Create a progress bar using tqdm
            with tqdm(total=len(wallets), desc="Fetching accounts", ncols=100) as pbar:
                for wallet in wallets:
                    accounts.append(get_stake_accounts(wallet, subtensor))
                    pbar.update()
            return accounts

        accounts = get_all_wallet_accounts(wallets, subtensor)

        total_stake = 0
        total_balance = 0
        total_rate = 0
        for acc in accounts:
            total_balance += acc["balance"].tao
            for key, value in acc["accounts"].items():
                total_stake += value["stake"].tao
                total_rate += float(value["rate"])
        table = Table(show_footer=True, pad_edge=False, box=None, expand=False)
        table.add_column(
            "[overline white]Coldkey", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]Balance",
            "\u03c4{:.5f}".format(total_balance),
            footer_style="overline white",
            style="green",
        )
        table.add_column(
            "[overline white]Account", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]Stake",
            "\u03c4{:.5f}".format(total_stake),
            footer_style="overline white",
            style="green",
        )
        table.add_column(
            "[overline white]Rate",
            "\u03c4{:.5f}/d".format(total_rate),
            footer_style="overline white",
            style="green",
        )
        for acc in accounts:
            table.add_row(acc["name"], acc["balance"], "", "")
            for key, value in acc["accounts"].items():
                table.add_row(
                    "", "", value["name"], value["stake"], str(value["rate"]) + "/d"
                )
        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
                not config.get("all", d=None)
                and not config.is_set("wallet.name")
                and not config.no_prompt
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "show", help="""List all stake accounts for wallet."""
        )
        list_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all coldkey wallets.""",
            default=False,
        )

        bittensor.wallet.add_args(list_parser)
        bittensor.subtensor.add_args(list_parser)


class SetChildKeyTakeCommand:
    """
    Executes the ``set_childkey_take`` command to modify your childkey take on a specified subnet on the Bittensor network to the caller.

    This command is used to modify your childkey take on a specified subnet on the Bittensor network. 

    Usage:
        Users can specify the amount or 'take' for their child hotkeys (``SS58`` address),
        the user needs to have access to the ss58 hotkey this call, and the take must be between 0 and 18%.

    The command prompts for confirmation before executing the set_childkey_take operation.

    Example usage::

        btcli stake set_childkey_take --hotkey <childkey> --netuid 1 --take 0.18
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Set childkey take."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetChildKeyTakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        console = Console()
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        netuid = cli.config.netuid
        total_subnets = subtensor.get_total_subnets()
        if total_subnets is not None and not (0 <= netuid <= total_subnets):
            console.print("Netuid is outside the current subnet range")
            return

        # get parent hotkey
        if wallet and wallet.hotkey:
            hotkey = wallet.hotkey.ss58_address
            console.print(f"Hotkey is {hotkey}")
        elif cli.config.is_set("hotkey"):
            hotkey = cli.config.hotkey
        else:
            hotkey = Prompt.ask("Enter child hotkey (ss58)")

        if not wallet_utils.is_valid_ss58_address(hotkey):
            console.print(
                f":cross_mark:[red] Invalid SS58 address: {hotkey}[/red]"
            )
            return

        if not cli.config.is_set("take"):
            cli.config.take = Prompt.ask(
                "Enter the percentage of take for your child hotkey (between 0 and 0.18 representing 0-18%)"
            )

        # extract take from cli input
        try:
            take = float(cli.config.take)
        except ValueError:
            print(":cross_mark:[red]Take must be a float value using characters between 0 and 9.[/red]")
            return

        if take < 0 or take > 0.18:
            console.print(
                f":cross_mark:[red]Invalid take: Childkey Take must be between 0 and 0.18 (representing 0% to 18%). Proposed take is {take}.[/red]")
            return

        success, message = subtensor.set_childkey_take(
            wallet=wallet,
            netuid=netuid,
            hotkey=hotkey,
            take=take,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            console.print(
                ":white_heavy_check_mark: [green]Set childkey take.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to set childkey take.[/red] {message}"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_childkey_take_parser = parser.add_parser(
            "set_childkey_take", help="""Set childkey take."""
        )
        set_childkey_take_parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False
        )
        set_childkey_take_parser.add_argument(
            "--hotkey", dest="hotkey", type=str, required=False
        )
        set_childkey_take_parser.add_argument(
            "--take", dest="take", type=float, required=False
        )
        set_childkey_take_parser.add_argument(
            "--wait_for_inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=True,
            help="""Wait for the transaction to be included in a block.""",
        )
        set_childkey_take_parser.add_argument(
            "--wait_for_finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
            help="""Wait for the transaction to be finalized.""",
        )
        set_childkey_take_parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
            help="""Prompt for confirmation before proceeding.""",
        )
        bittensor.wallet.add_args(set_childkey_take_parser)
        bittensor.subtensor.add_args(set_childkey_take_parser)


class GetChildKeyTakeCommand:
    """
    Executes the ``get_childkey_take`` command to get your childkey take on a specified subnet on the Bittensor network to the caller.

    This command is used to get your childkey take on a specified subnet on the Bittensor network. 

    Usage:
        Users can get the amount or 'take' for their child hotkeys (``SS58`` address)

    Example usage::

        btcli stake get_childkey_take --hotkey <childkey> --netuid 1
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Get childkey take."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            GetChildKeyTakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        console = Console()
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        netuid = cli.config.netuid
        total_subnets = subtensor.get_total_subnets()
        if total_subnets is not None and not (0 <= netuid <= total_subnets):
            console.print("Netuid is outside the current subnet range")
            return

        # get parent hotkey
        if wallet and wallet.hotkey:
            hotkey = wallet.hotkey.ss58_address
            console.print(f"Hotkey is {hotkey}")
        elif cli.config.is_set("hotkey"):
            hotkey = cli.config.hotkey
        else:
            hotkey = Prompt.ask("Enter child hotkey (ss58)")

        if not wallet_utils.is_valid_ss58_address(hotkey):
            console.print(
                f":cross_mark:[red] Invalid SS58 address: {hotkey}[/red]"
            )
            return

        take_u16 = subtensor.get_childkey_take(
            netuid=netuid,
            hotkey=hotkey,
        )

        # Result
        if take_u16:
            take = u16_to_float(take_u16)
            console.print(
                f"The childkey take for {hotkey} is {take * 100}%."
            )
        else:
            console.print(
                ":cross_mark:[red] Unable to get childkey take.[/red]"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        get_childkey_take_parser = parser.add_parser(
            "get_childkey_take", help="""Get childkey take."""
        )
        get_childkey_take_parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False
        )
        get_childkey_take_parser.add_argument(
            "--hotkey", dest="hotkey", type=str, required=False
        )
        bittensor.wallet.add_args(get_childkey_take_parser)
        bittensor.subtensor.add_args(get_childkey_take_parser)

    @staticmethod
    def get_take(subtensor, hotkey, netuid) -> float:
        """
        Get the take value for a given subtensor, hotkey, and netuid.

        @param subtensor: The subtensor object.
        @param hotkey: The hotkey to retrieve the take value for.
        @param netuid: The netuid to retrieve the take value for.

        @return: The take value as a float. If the take value is not available, it returns 0.

        """
        take_u16 = subtensor.get_childkey_take(
            netuid=netuid,
            hotkey=hotkey,
        )
        if take_u16:
            return u16_to_float(take_u16)
        else:
            return 0


class SetChildrenCommand:
    """
    Executes the ``set_children`` command to add children hotkeys on a specified subnet on the Bittensor network to the caller.

    This command is used to delegate authority to different hotkeys, securing their position and influence on the subnet.

    Usage:
        Users can specify the amount or 'proportion' to delegate to child hotkeys (``SS58`` address),
        the user needs to have sufficient authority to make this call, and the sum of proportions must equal 1,
        representing 100% of the proportion allocation.

    The command prompts for confirmation before executing the set_children operation.

    Example usage::

        btcli stake set_children --children <child_hotkey>,<child_hotkey> --hotkey <parent_hotkey> --netuid 1 --proportions 0.4,0.6

    Note:
        This command is critical for users who wish to delegate children hotkeys among different neurons (hotkeys) on the network.
        It allows for a strategic allocation of authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Set children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        console = Console()
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        netuid = cli.config.netuid
        total_subnets = subtensor.get_total_subnets()
        if total_subnets is not None and not (0 <= netuid < total_subnets):
            console.print("Netuid is outside the current subnet range")
            return

        # get parent hotkey
        if wallet and wallet.hotkey:
            hotkey = wallet.hotkey.ss58_address
        elif cli.config.is_set("hotkey"):
            hotkey = cli.config.hotkey
        else:
            hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        if not wallet_utils.is_valid_ss58_address(hotkey):
            console.print(
                f":cross_mark:[red] Invalid SS58 address: {hotkey}[/red]"
            )
            return

        # get current children
        curr_children = GetChildrenCommand.retrieve_children(
            subtensor=subtensor,
            hotkey=hotkey,
            netuid=netuid,
            render_table=False,
        )

        if curr_children:
            # print the table of current children
            hotkey_stake = subtensor.get_total_stake_for_hotkey(hotkey)
            GetChildrenCommand.render_table(
                subtensor=subtensor,
                hotkey=hotkey,
                hotkey_stake=hotkey_stake,
                children=curr_children,
                netuid=netuid,
                prompt=False,
            )

        # get new children
        if not cli.config.is_set("children"):
            cli.config.children = Prompt.ask(
                "Enter child hotkeys (ss58) as comma-separated values"
            )
        proposed_children = [str(x) for x in re.split(r"[ ,]+", cli.config.children)]

        # Set max 5 children
        if len(proposed_children) > 5:
            console.print(
                ":cross_mark:[red] Too many children. Maximum 5 children per hotkey[/red]"
            )
            return

        # Validate children SS58 addresses
        for child in proposed_children:
            if not wallet_utils.is_valid_ss58_address(child):
                console.print(f":cross_mark:[red] Invalid SS58 address: {child}[/red]")
                return

        # get proportions for new children
        if not cli.config.is_set("proportions"):
            cli.config.proportions = Prompt.ask(
                "Enter the percentage of proportion for each child as comma-separated values (total from all children must be less than or equal to 1)"
            )

        # extract proportions and child addresses from cli input
        proportions = [float(x) for x in re.split(r"[ ,]+", str(cli.config.proportions))]
        total_proposed = sum(proportions)
        if total_proposed > 1:
            console.print(
                f":cross_mark:[red]Invalid proportion: The sum of all proportions must be less or equal to than 1 (representing 100% of the allocation). Proposed sum addition is proportions is {total_proposed}.[/red]")
            return

        if len(proportions) != len(proposed_children):
            console.print(
                ":cross_mark:[red]Invalid proportion and children length: The count of children and number of proportion values entered do not match.[/red]")
            return

        # combine proposed and current children
        children_with_proportions = list(zip(proportions, proposed_children))

        SetChildrenCommand.print_current_stake(subtensor=subtensor, children=proposed_children,
                                               hotkey=hotkey)

        success, message = subtensor.set_children(
            wallet=wallet,
            netuid=netuid,
            hotkey=hotkey,
            children_with_proportions=children_with_proportions,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            if cli.config.wait_for_finalization and cli.config.wait_for_inclusion:
                console.print("New Status:")
                GetChildrenCommand.retrieve_children(
                    subtensor=subtensor,
                    hotkey=hotkey,
                    netuid=netuid,
                    render_table=True,
                )
            console.print(
                ":white_heavy_check_mark: [green]Set children hotkeys.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to set children hotkeys.[/red] {message}"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_children_parser = parser.add_parser(
            "set_children", help="""Set multiple children hotkeys."""
        )
        set_children_parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False
        )
        set_children_parser.add_argument(
            "--children", dest="children", type=str, required=False
        )
        set_children_parser.add_argument(
            "--hotkey", dest="hotkey", type=str, required=False
        )
        set_children_parser.add_argument(
            "--proportions", dest="proportions", type=str, required=False
        )
        set_children_parser.add_argument(
            "--wait_for_inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=True,
            help="""Wait for the transaction to be included in a block.""",
        )
        set_children_parser.add_argument(
            "--wait_for_finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
            help="""Wait for the transaction to be finalized.""",
        )
        set_children_parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=True,
            help="""Prompt for confirmation before proceeding.""",
        )
        bittensor.wallet.add_args(set_children_parser)
        bittensor.subtensor.add_args(set_children_parser)

    @staticmethod
    def print_current_stake(subtensor, children, hotkey):
        console = Console()
        parent_stake = subtensor.get_total_stake_for_hotkey(
            ss58_address=hotkey
        )
        console.print("Current Status:")
        console.print(
            f"Parent HotKey: {hotkey}  |  ", style="cyan", end="", no_wrap=True
        )
        console.print(f"Total Parent Stake: {parent_stake}τ")
        for child in children:
            child_stake = subtensor.get_total_stake_for_hotkey(child)
            console.print(f"Child Hotkey:  {child}  | Current Child Stake: {child_stake}τ")


class GetChildrenCommand:
    """
    Executes the ``get_children_info`` command to get all child hotkeys on a specified subnet on the Bittensor network.

    This command is used to view delegated authority to different hotkeys on the subnet.

    Usage:
        Users can specify the subnet and see the children and the proportion that is given to them.

        The command compiles a table showing:

    - ChildHotkey: The hotkey associated with the child.
    - ParentHotKey: The hotkey associated with the parent.
    - Proportion: The proportion that is assigned to them.
    - Expiration: The expiration of the hotkey.

    Example usage::

        btcli stake get_children --netuid 1

    Note:
        This command is for users who wish to see child hotkeys among different neurons (hotkeys) on the network.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Get children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            return GetChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        console = Console()
        wallet = bittensor.wallet(config=cli.config)

        # set netuid
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))
        netuid = cli.config.netuid
        total_subnets = subtensor.get_total_subnets()
        if total_subnets is not None and not (0 <= netuid < total_subnets):
            console.print("Netuid is outside the current subnet range")
            return

        # get parent hotkey
        if wallet and wallet.hotkey:
            hotkey = wallet.hotkey.ss58_address
            console.print(f"Hotkey is {hotkey}")
        elif cli.config.is_set("hotkey"):
            hotkey = cli.config.hotkey
        else:
            hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        if not wallet_utils.is_valid_ss58_address(hotkey):
            console.print(
                f":cross_mark:[red] Invalid SS58 address: {hotkey}[/red]"
            )
            return

        children = subtensor.get_children(hotkey, netuid)
        hotkey_stake = subtensor.get_total_stake_for_hotkey(hotkey)

        GetChildrenCommand.render_table(
            subtensor, hotkey, hotkey_stake, children, netuid, True
        )

        return children

    @staticmethod
    def retrieve_children(
            subtensor: "bittensor.subtensor", hotkey: str, netuid: int, render_table: bool
    ) -> list[tuple[int, str]]:
        """
    
        Static method to retrieve children for a given subtensor.
    
        Args:
            subtensor (bittensor.subtensor): The subtensor object used to interact with the Bittensor network.
            hotkey (str): The hotkey of the parent.
            netuid (int): The network unique identifier of the subtensor.
            render_table (bool): Flag indicating whether to render the retrieved children in a table.
    
        Returns:
            List[str]: A list of children hotkeys.
    
        """
        children = subtensor.get_children(hotkey, netuid)
        if render_table:
            hotkey_stake = subtensor.get_total_stake_for_hotkey(hotkey)
            GetChildrenCommand.render_table(
                subtensor, hotkey, hotkey_stake, children, netuid, False
            )
        return children

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "get_children", help="""Get child hotkeys on subnet."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--hotkey", dest="hotkey", type=str, required=False)

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def render_table(
            subtensor: "bittensor.subtensor",
            hotkey: str,
            hotkey_stake: "Balance",
            children: list[Tuple[int, str]],
            netuid: int,
            prompt: bool,
    ):
        """

        Render a table displaying information about child hotkeys on a particular subnet.

        Parameters:
        - subtensor: An instance of the "bittensor.subtensor" class.
        - hotkey: The hotkey of the parent node.
        - children: A list of tuples containing information about child hotkeys. Each tuple should contain:
            - The proportion of the child's stake relative to the total stake.
            - The hotkey of the child node.
        - netuid: The ID of the subnet.
        - prompt: A boolean indicating whether to display a prompt for adding a child hotkey.

        Returns:
        None

        Example Usage:
            subtensor = bittensor.subtensor_instance
            hotkey = "parent_hotkey"
            children = [(0.5, "child1_hotkey"), (0.3, "child2_hotkey"), (0.2, "child3_hotkey")]
            netuid = 1234
            prompt = True
            render_table(subtensor, hotkey, children, netuid, prompt)

        """
        console = Console()

        # Initialize Rich table for pretty printing
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="green",
            style="green",
        )

        # Add columns to the table with specific styles
        table.add_column("Index", style="cyan", no_wrap=True, justify="right")
        table.add_column("ChildHotkey", style="cyan", no_wrap=True)
        table.add_column("Proportion", style="cyan", no_wrap=True, justify="right")
        table.add_column("Childkey Take", style="cyan", no_wrap=True, justify="right")
        table.add_column(
            "New Stake Weight", style="cyan", no_wrap=True, justify="right"
        )

        if not children:
            console.print(table)
            console.print(
                f"There are currently no child hotkeys on subnet {netuid} with Parent HotKey {hotkey}."
            )
            if prompt:
                command = f"btcli stake set_children --children <child_hotkey> --hotkey <parent_hotkey> --netuid {netuid} --proportion <float>"
                console.print(
                    f"To add a child hotkey you can run the command: [white]{command}[/white]"
                )
            return

        # calculate totals
        total_proportion = 0
        total_stake = 0
        total_stake_weight = 0
        avg_take = 0

        children_info = []
        for child in children:
            proportion = child[0]
            child_hotkey = child[1]
            child_stake = subtensor.get_total_stake_for_hotkey(
                ss58_address=child_hotkey
            ) or Balance(0)

            child_take = subtensor.get_childkey_take(child_hotkey, netuid)
            child_take = u16_to_float(child_take)

            # add to totals
            total_stake += child_stake.tao
            avg_take += child_take

            proportion = u64_to_float(proportion)

            children_info.append((proportion, child_hotkey, child_stake, child_take))

        children_info.sort(
            key=lambda x: x[0], reverse=True
        )  # sorting by proportion (highest first)

        # add the children info to the table
        for i, (proportion, hotkey, stake, child_take) in enumerate(children_info, 1):
            proportion_percent = proportion * 100  # Proportion in percent
            proportion_tao = hotkey_stake.tao * proportion  # Proportion in TAO

            total_proportion += proportion_percent

            # Conditionally format text
            proportion_str = f"{proportion_percent:.3f}% ({proportion_tao:.3f}τ)"
            stake_weight = stake.tao + proportion_tao
            total_stake_weight += stake_weight
            take_str = f"{child_take * 100:.3f}%"

            hotkey = Text(hotkey, style="red" if proportion == 0 else "")
            table.add_row(
                str(i),
                hotkey,
                proportion_str,
                take_str,
                str(f"{stake_weight:.3f}"),
            )

        avg_take = avg_take / len(children_info)

        # add totals row
        table.add_row(
            "",
            "Total",
            f"{total_proportion:.3f}%",
            f"(avg) {avg_take * 100:.3f}%",
            f"{total_stake_weight:.3f}τ",
        )
        console.print(table)
