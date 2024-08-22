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

import bittensor
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import Dict, List, Tuple, TYPE_CHECKING
from .. import defaults
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor.subtensor import Subtensor


def _get_coldkey_ss58_addresses_for_path(path: str) -> Tuple[List[str], List[str]]:
    """Get all coldkey ss58 addresses from path."""

    def list_coldkeypub_files(dir_path):
        abspath = os.path.abspath(os.path.expanduser(dir_path))
        coldkey_files = []
        wallet_names = []

        for potential_wallet_name in os.listdir(abspath):
            coldkey_path = os.path.join(
                abspath, potential_wallet_name, "coldkeypub.txt"
            )
            if os.path.isdir(
                os.path.join(abspath, potential_wallet_name)
            ) and os.path.exists(coldkey_path):
                coldkey_files.append(coldkey_path)
                wallet_names.append(potential_wallet_name)
            else:
                bittensor.logging.warning(
                    f"{coldkey_path} does not exist. Excluding..."
                )
        return coldkey_files, wallet_names

    coldkey_files, wallet_names = list_coldkeypub_files(path)
    addresses = [
        bittensor.keyfile(coldkey_path).keypair.ss58_address
        for coldkey_path in coldkey_files
    ]
    return addresses, wallet_names


def _get_total_stake_for_coldkey(
    subtensor: "Subtensor", ss58_address: str
) -> "Balance":
    """Gets all staked balance for the coldkey address."""
    sub_stakes = subtensor.get_stake_info_for_coldkeys(
        coldkey_ss58_list=[ss58_address]
    )[ss58_address]

    # Token pricing info.
    dynamic_info = subtensor.get_all_subnet_dynamic_info()

    all_staked_tao = 0

    # Iterate over sub-stakes and aggregate them by hotkey.
    for substake in sub_stakes:
        if substake.stake.rao == 0:
            continue
        netuid = substake.netuid
        pool = dynamic_info[netuid]
        alpha_value = bittensor.Balance.from_rao(int(substake.stake.rao)).set_unit(
            netuid
        )
        tao_locked = (
            pool.tao_in
            if pool.is_dynamic
            else pool.tao_in
        )
        issuance = pool.alpha_out if pool.is_dynamic else tao_locked
        tao_ownership = 0

        if alpha_value.tao > 0.00009 and issuance.tao != 0:
            tao_ownership = bittensor.Balance.from_tao(
                (alpha_value.tao / issuance.tao) * tao_locked.tao
            )

        all_staked_tao += tao_ownership.rao

    return Balance.from_rao(all_staked_tao)


def _get_balances(
    subtensor: "Subtensor", coldkeys, wallet_names
) -> Tuple[Dict["str", tuple[str, "Balance", "Balance"]], int, int]:
    free_balances = [subtensor.get_balance(coldkeys[i]) for i in range(len(coldkeys))]

    staked_balances = [
        _get_total_stake_for_coldkey(subtensor, coldkeys[i])
        for i in range(len(coldkeys))
    ]

    total_free_balance = sum(free_balances)
    total_staked_balance = sum(staked_balances)

    balances = {
        name: (coldkey, free, staked)
        for name, coldkey, free, staked in sorted(
            zip(wallet_names, coldkeys, free_balances, staked_balances)
        )
    }
    return balances, total_free_balance, total_staked_balance


class WalletBalanceCommand:
    """
    Executes the ``balance`` command to check the balance of the wallet on the Bittensor network.

    This command provides a detailed view of the wallet's coldkey balances, including free and staked balances.

    Usage:
        The command lists the balances of all wallets in the user's configuration directory, showing the wallet name, coldkey address, and the respective free and staked balances.

    Optional arguments:
        None. The command uses the wallet and subtensor configurations to fetch balance data.

    Example usages:

        - To display the balance of a single wallet, use the command with the `--wallet.name` argument to specify the wallet name:

        ```
        btcli w balance --wallet.name WALLET
        ```

        - Alternatively, you can invoke the command without specifying a wallet name, which will prompt you to enter the wallets path:

        ```
        btcli w balance
        ```

        - To display the balances of all wallets, use the `--all` argument:

        ```
        btcli w balance --all
        ```

    Note:
        When using `btcli`, `w` is used interchangeably with `wallet`. You may use either based on your preference for brevity or clarity.
        This command is essential for users to monitor their financial status on the Bittensor network.
        It helps in keeping track of assets and ensuring the wallet's financial health.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Check the balance of the wallet."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            WalletBalanceCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        total_staked_balance = 0
        balances = {}

        if cli.config.get("all", d=None):
            coldkeys, wallet_names = _get_coldkey_ss58_addresses_for_path(
                cli.config.wallet.path
            )

            balances, total_free_balance, total_staked_balance = _get_balances(
                subtensor=subtensor, coldkeys=coldkeys, wallet_names=wallet_names
            )
        else:
            coldkey_wallet = bittensor.wallet(config=cli.config)
            if (
                coldkey_wallet.coldkeypub_file.exists_on_device()
                and not coldkey_wallet.coldkeypub_file.is_encrypted()
            ):
                coldkeys = [coldkey_wallet.coldkeypub.ss58_address]
                wallet_names = [coldkey_wallet.name]

                balances, total_free_balance, total_staked_balance = _get_balances(
                    subtensor=subtensor, coldkeys=coldkeys, wallet_names=wallet_names
                )

            if not coldkey_wallet.coldkeypub_file.exists_on_device():
                bittensor.__console__.print("[bold red]No wallets found.")
                return

        table = Table(
            title="[white]Wallet Coldkey Balances",
            width=bittensor.__console__.width - 5,
            safe_box=True,
            padding=(0, 1),
            collapse_padding=False,
            pad_edge=True,
            expand=True,
            show_header=True,
            show_footer=True,
            show_edge=False,
            show_lines=False,
            leading=0,
            style="none",
            row_styles=None,
            header_style="bold",
            footer_style="bold",
            border_style="rgb(7,54,66)",
            title_style="bold magenta",
            title_justify="center",
            highlight=False,
        )
        table.add_column(
            "[bold magenta]Wallet Name",
            header_style="overline bold magenta",
            footer_style="overline bold magenta",
            style="rgb(211,54,130)",  # Solarized magenta
            no_wrap=True,
        )

        table.add_column(
            "[bold yellow]Coldkey Address",
            header_style="overline bold yellow",
            footer_style="overline bold yellow",
            style="rgb(181,137,0)",  # Solarized yellow
            no_wrap=True,
        )

        for type_str in ["Free", "Stake", "Total"]:
            table.add_column(
                f"[bold cyan]{type_str} Balance",
                header_style="overline bold cyan",
                footer_style="overline bold cyan",
                justify="right",
                style="rgb(42,161,152)",  # Solarized cyan
                no_wrap=True,
            )

        for name, (coldkey, free, staked) in balances.items():
            table.add_row(
                name,
                coldkey,
                str(free),
                str(staked),
                str(free + staked),
            )
        table.add_row()
        table.add_row(
            "Total Balance Across All Coldkeys",
            "",
            str(total_free_balance),
            str(total_staked_balance),
            str(total_free_balance + total_staked_balance),
        )
        table.show_footer = True
        bittensor.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        balance_parser = parser.add_parser(
            "balance", help="""Checks the balance of the wallet."""
        )
        balance_parser.add_argument(
            "--all",
            dest="all",
            action="store_true",
            help="""View balance for all wallets.""",
            default=False,
        )

        bittensor.wallet.add_args(balance_parser)
        bittensor.subtensor.add_args(balance_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.is_set("wallet.path")
            and not config.no_prompt
            and not config.get("all", d=None)
        ):
            path = Prompt.ask("Enter wallets path", default=defaults.wallet.path)
            config.wallet.path = str(path)

            if (
                not config.is_set("wallet.name")
                and not config.no_prompt
                and not config.get("all", d=None)
            ):
                wallet_name = Prompt.ask(
                    "Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name
                )
                config.wallet.name = str(wallet_name)

        if not config.is_set("subtensor.network") and not config.no_prompt:
            network = Prompt.ask(
                "Enter network",
                default=defaults.subtensor.network,
                choices=bittensor.__networks__,
            )
            config.subtensor.network = str(network)
            (
                _,
                config.subtensor.chain_endpoint,
            ) = bittensor.subtensor.determine_chain_endpoint_and_network(str(network))