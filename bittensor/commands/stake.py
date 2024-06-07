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
import os
import re
import sys
import rich
import torch
import argparse
from tqdm import tqdm
from rich.table import Table
from rich.prompt import Confirm, Prompt
from typing import Dict, List, Optional, Tuple, Union

import bittensor
from . import defaults
from .delegates import show_delegates
from bittensor.utils.balance import Balance
from .utils import (
    get_delegates_details,
    get_hotkey_wallets_for_wallet,
    DelegatesDetails,
)
from substrateinterface.exceptions import SubstrateRequestException

console = bittensor.__console__


class StakeWeightsCommand:
    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set weights for root network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            StakeWeightsCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set weights for root network."""
        wallet = bittensor.wallet(config=cli.config)
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        # Get values if not set.
        if not cli.config.is_set("netuids"):
            example = (
                ", ".join(map(str, [subnet.netuid for subnet in subnets][:3])) + " ..."
            )
            cli.config.netuids = Prompt.ask(f"Enter netuids (e.g. {example})")

        if not cli.config.is_set("weights"):
            example = (
                ", ".join(
                    map(
                        str,
                        [
                            "{:.2f}".format(float(1 / len(subnets)))
                            for subnet in subnets
                        ][:3],
                    )
                )
                + " ..."
            )
            cli.config.weights = Prompt.ask(f"Enter weights (e.g. {example})")

        # Parse from string
        netuids = torch.tensor(
            list(map(int, re.split(r"[ ,]+", cli.config.netuids))), dtype=torch.long
        )
        print(cli.config.weights)
        print(re.split(r"[ ,]+", cli.config.weights))
        weights = torch.tensor(
            list(map(float, re.split(r"[ ,]+", cli.config.weights))),
            dtype=torch.float32,
        )

        # Run the set weights operation.
        subtensor.stake_set_weights(
            wallet=wallet,
            hotkey=cli.config.delegate_ss58key,
            netuids=netuids,
            weights=weights,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "weights",
            help="""Distribute delegated stake across subnets based on weights.""",
        )
        parser.add_argument(
            "--delegate_ss58key",
            "--delegate_ss58",
            dest="delegate_ss58key",
            type=str,
            required=False,
            help="""The ss58 address of the chosen delegate""",
        )
        # TODO fix does not work: --netuids 1,2,3 --weights 0.33, 0.33, 0.33
        parser.add_argument("--netuids", dest="netuids", type=str, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)
        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.get("delegate_ss58key"):
            # Check for delegates.
            with bittensor.__console__.status(":satellite: Loading delegates..."):
                subtensor = bittensor.subtensor(config=config, log_verbose=False)
                delegates: List[bittensor.DelegateInfo] = subtensor.get_delegates()
                try:
                    prev_delegates = subtensor.get_delegates(
                        max(0, subtensor.block - 1200)
                    )
                except SubstrateRequestException:
                    prev_delegates = None

            if prev_delegates is None:
                bittensor.__console__.print(
                    ":warning: [yellow]Could not fetch delegates history[/yellow]"
                )

            if len(delegates) == 0:
                console.print(
                    ":cross_mark: [red]There are no delegates on {}[/red]".format(
                        subtensor.network
                    )
                )
                sys.exit(1)

            delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
            show_delegates(delegates, prev_delegates=prev_delegates)
            delegate_index = Prompt.ask("Enter delegate index")
            config.delegate_ss58key = str(delegates[int(delegate_index)].hotkey_ss58)
            console.print(
                "Selected: [yellow]{}[/yellow]".format(config.delegate_ss58key)
            )


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


class StakeList:
    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Show all stake accounts."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            StakeList._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)
        substakes = subtensor.get_substake_for_coldkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address
        )
        netuids = subtensor.get_all_subnet_netuids()

        # Get registered delegates details.
        registered_delegate_info: Optional[DelegatesDetails] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        # Token pricing info.
        dynamic_info = subtensor.get_dynamic_info()

        # Build map of hotkeys to netuids to stake
        hot_tao_totals = {}
        hot_alpha_totals = {}
        netuid_totals = {}
        hot_netuid_pairs = {}
        for substake in substakes:
            netuid = substake["netuid"]
            if substake["hotkey"] not in hot_netuid_pairs:
                hot_netuid_pairs[substake["hotkey"]] = {}
                hot_tao_totals[substake["hotkey"]] = 0.0
                hot_alpha_totals[substake["hotkey"]] = 0.0
            if netuid not in netuid_totals:
                netuid_totals[netuid] = 0.0
            hot_netuid_pairs[substake["hotkey"]][netuid] = substake["stake"]
            hot_tao_totals[substake["hotkey"]] += (
                substake["stake"] * dynamic_info[netuid]["price"]
            )
            hot_alpha_totals[substake["hotkey"]] += substake["stake"]
            netuid_totals[netuid] += substake["stake"] * dynamic_info[netuid]["price"]

        table = Table(show_footer=True, pad_edge=False, box=None, expand=False)
        table.add_column(
            "[white]Hotkey", footer_style="overline white", style="dark_slate_gray3"
        )
        table.add_column(
            "[white]Subnet", footer_style="overline white", style="dark_slate_gray3"
        )
        table.add_column(f"[white]Stake", footer_style="overline white", style="blue")
        table.add_column(f"[white]TAO", footer_style="overline white", style="blue")
        table.add_column(f"[white]Price", footer_style="overline white", style="white")

        # Fill rows
        for hotkey in hot_netuid_pairs.keys():
            # Switch on named hotkeys
            if hotkey in registered_delegate_info:
                row_name = registered_delegate_info[hotkey].name
            else:
                row_name = hotkey[:10]
            for netuid in netuids:
                # Hotkey and subnet
                row = [
                    row_name,
                    "{} ({})".format(bittensor.Balance.get_unit(netuid), netuid),
                ]
                # Stake
                row.append(
                    "[green]{:,.4f}[/green]".format(
                        bittensor.Balance.from_rao(
                            int(hot_netuid_pairs[hotkey].get(netuid, 0))
                        ).tao
                    )
                )
                # Stake in TAO
                row.append(
                    f"[blue]{hot_netuid_pairs[hotkey].get(netuid, 0) * dynamic_info[ netuid ]['price'] }[/blue]"
                )
                # Price
                row.append(f"[white]{dynamic_info[ netuid ]['price']:.4f}[/white]")

                table.add_row(*row)

        table.add_row("")
        row_totals = ["TOTAL", ""]
        # Stake
        row_totals.append("[green]{:,.4f}[/green]".format(hot_alpha_totals[hotkey].tao))
        # Stake in TAO
        row_totals.append("[blue]{:,.4f}[/blue]".format(hot_tao_totals[hotkey].tao))
        table.add_row(*row_totals)

        table.box = None
        table.pad_edge = False
        table.width = None
        column_descriptions_table = Table(
            title="Stake List",
            box=rich.box.HEAVY_HEAD,
            safe_box=False,
            padding=(0, 1),
            collapse_padding=False,
            pad_edge=False,
            expand=False,
            show_header=True,
            show_footer=False,
            show_edge=False,
            show_lines=False,
            leading=0,
            style="none",
            row_styles=None,
            header_style="table.header",
            footer_style="table.footer",
            border_style=None,
            title_style=None,
            caption_style=None,
            title_justify="center",
            caption_justify="center",
            highlight=False,
        )
        column_descriptions_table.add_column("No.", justify="left", style="bold")
        column_descriptions_table.add_column("Column", justify="left")
        column_descriptions_table.add_column("Description", justify="left")

        column_descriptions = [
            (
                "[bold white]1.[/bold white]",
                "[bold white]Hotkey[/bold white]",
                "The staking account's associated hotkey or delegate name.",
            ),
            (
                "[bold white]2.[/bold white]",
                "[bold white]Subnet[/bold white]",
                "The Subnet ID and symbol.",
            ),
            (
                "[bold white]3.[/bold white]",
                "[bold white]Stake[/bold white]",
                "The [green]Alpha(\u03B1)[/green] currently staked in the subnet.",
            ),
            (
                "[bold white]4.[/bold white]",
                "[bold white]TAO[/bold white]",
                "Corresponding stake value in [blue]TAO[/blue] based on the current Alpha price.",
            ),
            (
                "[bold white]5.[/bold white]",
                "[bold white]Price[/bold white]",
                "Current Alpha price for reference.",
            ),
        ]

        for no, name, description in column_descriptions:
            column_descriptions_table.add_row(no, name, description)
        bittensor.__console__.print(column_descriptions_table)
        bittensor.__console__.print("\n")
        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "list", help="""List all stake accounts for wallet."""
        )
        bittensor.wallet.add_args(list_parser)
        bittensor.subtensor.add_args(list_parser)


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
            "\u03C4{:.5f}".format(total_balance),
            footer_style="overline white",
            style="green",
        )
        table.add_column(
            "[overline white]Account", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]Stake",
            "\u03C4{:.5f}".format(total_stake),
            footer_style="overline white",
            style="green",
        )
        table.add_column(
            "[overline white]Rate",
            "\u03C4{:.5f}/d".format(total_rate),
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
