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
import bittensor
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Prompt
from .utils import (
    get_delegates_details,
    DelegatesDetails,
    get_hotkey_wallets_for_wallet,
    get_all_wallets_for_path,
    filter_netuids_by_registered_hotkeys,
)
from . import defaults

console = bittensor.__console__

import os
import bittensor
from typing import List, Tuple, Optional, Dict


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


class InspectCommand:
    """
    Executes the ``inspect`` command, which compiles and displays a detailed report of a user's wallet pairs (coldkey, hotkey) on the Bittensor network.

    This report includes balance and
    staking information for both the coldkey and hotkey associated with the wallet.

    Optional arguments:
        - ``all``: If set to ``True``, the command will inspect all wallets located within the specified path. If set to ``False``, the command will inspect only the wallet specified by the user.

    The command gathers data on:

    - Coldkey balance and delegated stakes.
    - Hotkey stake and emissions per neuron on the network.
    - Delegate names and details fetched from the network.

    The resulting table includes columns for:

    - **Coldkey**: The coldkey associated with the user's wallet.
    - **Balance**: The balance of the coldkey.
    - **Delegate**: The name of the delegate to which the coldkey has staked funds.
    - **Stake**: The amount of stake held by both the coldkey and hotkey.
    - **Emission**: The emission or rewards earned from staking.
    - **Netuid**: The network unique identifier of the subnet where the hotkey is active.
    - **Hotkey**: The hotkey associated with the neuron on the network.

    Usage:
        This command can be used to inspect a single wallet or all wallets located within a
        specified path. It is useful for a comprehensive overview of a user's participation
        and performance in the Bittensor network.

    Example usage::

            btcli wallet inspect
            btcli wallet inspect --all

    Note:
        The ``inspect`` command is for displaying information only and does not perform any
        transactions or state changes on the Bittensor network. It is intended to be used as
        part of the Bittensor CLI and not as a standalone function within user code.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Inspect a cold, hot pair."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            InspectCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        if cli.config.get("all", d=False) == True:
            wallets = _get_coldkey_wallets_for_path(cli.config.wallet.path)
            all_hotkeys = get_all_wallets_for_path(cli.config.wallet.path)
        else:
            wallets = [bittensor.wallet(config=cli.config)]
            all_hotkeys = get_hotkey_wallets_for_wallet(wallets[0])

        netuids = subtensor.get_all_subnet_netuids()
        netuids = filter_netuids_by_registered_hotkeys(
            cli, subtensor, netuids, all_hotkeys
        )
        bittensor.logging.debug(f"Netuids to check: {netuids}")

        registered_delegate_info: Optional[Dict[str, DelegatesDetails]] = (
            get_delegates_details(url=bittensor.__delegates_details_url__)
        )
        if registered_delegate_info is None:
            bittensor.__console__.print(
                ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
            )
            registered_delegate_info = {}

        neuron_state_dict = {}
        for netuid in tqdm(netuids):
            neurons = subtensor.neurons_lite(netuid)
            neuron_state_dict[netuid] = neurons if neurons != None else []

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column(
            "[overline white]Coldkey", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]Balance", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Delegate", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]Stake", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Emission", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Netuid", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]Hotkey", footer_style="overline white", style="yellow"
        )
        table.add_column(
            "[overline white]Stake", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]Emission", footer_style="overline white", style="green"
        )
        for wallet in tqdm(wallets):
            delegates: List[Tuple[bittensor.DelegateInfo, bittensor.Balance]] = (
                subtensor.get_delegated(coldkey_ss58=wallet.coldkeypub.ss58_address)
            )
            if not wallet.coldkeypub_file.exists_on_device():
                continue
            cold_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
            table.add_row(wallet.name, str(cold_balance), "", "", "", "", "", "", "")
            for dele, staked in delegates:
                if dele.hotkey_ss58 in registered_delegate_info:
                    delegate_name = registered_delegate_info[dele.hotkey_ss58].name
                else:
                    delegate_name = dele.hotkey_ss58
                table.add_row(
                    "",
                    "",
                    str(delegate_name),
                    str(staked),
                    str(
                        dele.total_daily_return.tao
                        * (staked.tao / dele.total_stake.tao)
                    ),
                    "",
                    "",
                    "",
                    "",
                )

            hotkeys = _get_hotkey_wallets_for_wallet(wallet)
            for netuid in netuids:
                for neuron in neuron_state_dict[netuid]:
                    if neuron.coldkey == wallet.coldkeypub.ss58_address:
                        hotkey_name: str = ""

                        hotkey_names: List[str] = [
                            wallet.hotkey_str
                            for wallet in filter(
                                lambda hotkey: hotkey.hotkey.ss58_address
                                == neuron.hotkey,
                                hotkeys,
                            )
                        ]
                        if len(hotkey_names) > 0:
                            hotkey_name = f"{hotkey_names[0]}-"

                        table.add_row(
                            "",
                            "",
                            "",
                            "",
                            "",
                            str(netuid),
                            f"{hotkey_name}{neuron.hotkey}",
                            str(neuron.stake),
                            str(bittensor.Balance.from_tao(neuron.emission)),
                        )

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.is_set("wallet.name")
            and not config.no_prompt
            and not config.get("all", d=None)
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.netuids != [] and config.netuids != None:
            if not isinstance(config.netuids, list):
                config.netuids = [int(config.netuids)]
            else:
                config.netuids = [int(netuid) for netuid in config.netuids]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        inspect_parser = parser.add_parser(
            "inspect", help="""Inspect a wallet (cold, hot) pair"""
        )
        inspect_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all coldkey wallets.""",
            default=False,
        )
        inspect_parser.add_argument(
            "--netuids",
            dest="netuids",
            type=int,
            nargs="*",
            help="""Set the netuid(s) to filter by.""",
            default=None,
        )

        bittensor.wallet.add_args(inspect_parser)
        bittensor.subtensor.add_args(inspect_parser)
