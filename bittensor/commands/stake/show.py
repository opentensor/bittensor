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
from typing import List, Union, Optional, Dict

from rich.prompt import Prompt
from rich.table import Table
from tqdm import tqdm

import bittensor
from bittensor.utils.balance import Balance
from .. import defaults  # type: ignore
from ..utils import (
    get_hotkey_wallets_for_wallet,
    get_delegates_details,
    DelegatesDetails,
)

console = bittensor.__console__


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
        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=bittensor.__delegates_details_url__)

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
