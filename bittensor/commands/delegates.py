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

import argparse
import os
import sys
from typing import List, Dict, Optional

from rich.console import Text
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich.table import Table
from substrateinterface.exceptions import SubstrateRequestException
from tqdm import tqdm

import bittensor
from . import defaults
from .identity import SetIdentityCommand
from .utils import get_delegates_details, DelegatesDetails


def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


console = bittensor.__console__


def show_delegates_lite(
    delegates_lite: List["bittensor.DelegateInfoLite"], width: Optional[int] = None
):
    """
    This method is a lite version of the :func:`show_delegates`. This method displays a formatted table of Bittensor network delegates with detailed statistics to the console.

    The table is sorted by total stake in descending order and provides
    a snapshot of delegate performance and status, helping users make informed decisions for staking or nominating.

    This helper function is not intended to be used directly in user code unless specifically required.

    Args:
        delegates_lite (List[bittensor.DelegateInfoLite]): A list of delegate information objects to be displayed.
        width (Optional[int]): The width of the console output table. Defaults to ``None``, which will make the table expand to the maximum width of the console.

    The output table contains the following columns. To display more columns, use the :func:`show_delegates` function.

    - INDEX: The numerical index of the delegate.
    - DELEGATE: The name of the delegate.
    - SS58: The truncated SS58 address of the delegate.
    - NOMINATORS: The number of nominators supporting the delegate.
    - VPERMIT: Validator permits held by the delegate for the subnets.
    - TAKE: The percentage of the delegate's earnings taken by the network.
    - DELEGATE/(24h): The earnings of the delegate in the last 24 hours.
    - Desc: A brief description provided by the delegate.

    Usage:
        This function is typically used within the Bittensor CLI to show current delegate options to users who are considering where to stake their tokens.

    Example usage::

        show_delegates_lite(delegates_lite, width=80)

    Note:
        This function is primarily for display purposes within a command-line interface and does not return any values. It relies on the `rich <https://github.com/Textualize/rich>`_ Python library to render
        the table in the console.
    """

    registered_delegate_info: Optional[Dict[str, DelegatesDetails]] = (
        get_delegates_details(url=bittensor.__delegates_details_url__)
    )
    if registered_delegate_info is None:
        bittensor.__console__.print(
            ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
        )
        registered_delegate_info = {}

    table = Table(show_footer=True, width=width, pad_edge=False, box=None, expand=True)
    table.add_column(
        "[overline white]INDEX",
        str(len(delegates_lite)),
        footer_style="overline white",
        style="bold white",
    )
    table.add_column(
        "[overline white]DELEGATE",
        style="rgb(50,163,219)",
        no_wrap=True,
        justify="left",
    )
    table.add_column(
        "[overline white]SS58",
        str(len(delegates_lite)),
        footer_style="overline white",
        style="bold yellow",
    )
    table.add_column(
        "[overline white]NOMINATORS", justify="center", style="green", no_wrap=True
    )
    table.add_column("[overline white]VPERMIT", justify="right", no_wrap=False)
    table.add_column("[overline white]TAKE", style="white", no_wrap=True)
    table.add_column("[overline white]DELEGATE/(24h)", style="green", justify="center")
    table.add_column("[overline white]Desc", style="rgb(50,163,219)")

    for i, d in enumerate(delegates_lite):
        if d.delegate_ss58 in registered_delegate_info:
            delegate_name = registered_delegate_info[d.delegate_ss58].name
            delegate_url = registered_delegate_info[d.delegate_ss58].url
            delegate_description = registered_delegate_info[d.delegate_ss58].description
        else:
            delegate_name = ""
            delegate_url = ""
            delegate_description = ""

        table.add_row(
            # `INDEX` column
            str(i),
            # `DELEGATE` column
            Text(delegate_name, style=f"link {delegate_url}"),
            # `SS58` column
            f"{d.delegate_ss58:8.8}...",
            # `NOMINATORS` column
            str(d.nominators),
            # `VPERMIT` column
            str(d.registrations),
            # `TAKE` column
            f"{d.take * 100:.1f}%",
            # `DELEGATE/(24h)` column
            f"τ{bittensor.Balance.from_tao(d.total_daily_return * 0.18) !s:6.6}",
            # `Desc` column
            str(delegate_description),
            end_section=True,
        )
    bittensor.__console__.print(table)


# Uses rich console to pretty print a table of delegates.
def show_delegates(
    delegates: List["bittensor.DelegateInfo"],
    prev_delegates: Optional[List["bittensor.DelegateInfo"]],
    width: Optional[int] = None,
):
    """
    Displays a formatted table of Bittensor network delegates with detailed statistics to the console.

    The table is sorted by total stake in descending order and provides
    a snapshot of delegate performance and status, helping users make informed decisions for staking or nominating.

    This is a helper function that is called by the :func:`list_delegates` and :func:`my_delegates`, and is not intended
    to be used directly in user code unless specifically required.

    Args:
        delegates (List[bittensor.DelegateInfo]): A list of delegate information objects to be displayed.
        prev_delegates (Optional[List[bittensor.DelegateInfo]]): A list of delegate information objects from a previous state, used to calculate changes in stake. Defaults to ``None``.
        width (Optional[int]): The width of the console output table. Defaults to ``None``, which will make the table expand to the maximum width of the console.

    The output table contains the following columns:

    - INDEX: The numerical index of the delegate.
    - DELEGATE: The name of the delegate.
    - SS58: The truncated SS58 address of the delegate.
    - NOMINATORS: The number of nominators supporting the delegate.
    - DELEGATE STAKE(τ): The stake that is directly delegated to the delegate.
    - TOTAL STAKE(τ): The total stake held by the delegate, including nominators' stake.
    - CHANGE/(4h): The percentage change in the delegate's stake over the past 4 hours.
    - VPERMIT: Validator permits held by the delegate for the subnets.
    - TAKE: The percentage of the delegate's earnings taken by the network.
    - NOMINATOR/(24h)/kτ: The earnings per 1000 τ staked by nominators in the last 24 hours.
    - DELEGATE/(24h): The earnings of the delegate in the last 24 hours.
    - Desc: A brief description provided by the delegate.

    Usage:
        This function is typically used within the Bittensor CLI to show current delegate options to users who are considering where to stake their tokens.

    Example usage::

        show_delegates(current_delegates, previous_delegates, width=80)

    Note:
        This function is primarily for display purposes within a command-line interface and does
        not return any values. It relies on the `rich <https://github.com/Textualize/rich>`_ Python library to render
        the table in the
        console.
    """

    delegates.sort(key=lambda delegate: delegate.total_stake, reverse=True)
    prev_delegates_dict = {}
    if prev_delegates is not None:
        for prev_delegate in prev_delegates:
            prev_delegates_dict[prev_delegate.hotkey_ss58] = prev_delegate

    registered_delegate_info: Optional[Dict[str, DelegatesDetails]] = (
        get_delegates_details(url=bittensor.__delegates_details_url__)
    )
    if registered_delegate_info is None:
        bittensor.__console__.print(
            ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
        )
        registered_delegate_info = {}

    table = Table(show_footer=True, width=width, pad_edge=False, box=None, expand=True)
    table.add_column(
        "[overline white]INDEX",
        str(len(delegates)),
        footer_style="overline white",
        style="bold white",
    )
    table.add_column(
        "[overline white]DELEGATE",
        style="rgb(50,163,219)",
        no_wrap=True,
        justify="left",
    )
    table.add_column(
        "[overline white]SS58",
        str(len(delegates)),
        footer_style="overline white",
        style="bold yellow",
    )
    table.add_column(
        "[overline white]NOMINATORS", justify="center", style="green", no_wrap=True
    )
    table.add_column(
        "[overline white]DELEGATE STAKE(\u03c4)", justify="right", no_wrap=True
    )
    table.add_column(
        "[overline white]TOTAL STAKE(\u03c4)",
        justify="right",
        style="green",
        no_wrap=True,
    )
    table.add_column("[overline white]CHANGE/(4h)", style="grey0", justify="center")
    table.add_column("[overline white]VPERMIT", justify="right", no_wrap=False)
    table.add_column("[overline white]TAKE", style="white", no_wrap=True)
    table.add_column(
        "[overline white]NOMINATOR/(24h)/k\u03c4", style="green", justify="center"
    )
    table.add_column("[overline white]DELEGATE/(24h)", style="green", justify="center")
    table.add_column("[overline white]Desc", style="rgb(50,163,219)")

    for i, delegate in enumerate(delegates):
        owner_stake = next(
            map(
                lambda x: x[1],  # get stake
                filter(
                    lambda x: x[0] == delegate.owner_ss58, delegate.nominators
                ),  # filter for owner
            ),
            bittensor.Balance.from_rao(0),  # default to 0 if no owner stake.
        )
        if delegate.hotkey_ss58 in registered_delegate_info:
            delegate_name = registered_delegate_info[delegate.hotkey_ss58].name
            delegate_url = registered_delegate_info[delegate.hotkey_ss58].url
            delegate_description = registered_delegate_info[
                delegate.hotkey_ss58
            ].description
        else:
            delegate_name = ""
            delegate_url = ""
            delegate_description = ""

        if delegate.hotkey_ss58 in prev_delegates_dict:
            prev_stake = prev_delegates_dict[delegate.hotkey_ss58].total_stake
            if prev_stake == 0:
                rate_change_in_stake_str = "[green]100%[/green]"
            else:
                rate_change_in_stake = (
                    100
                    * (float(delegate.total_stake) - float(prev_stake))
                    / float(prev_stake)
                )
                if rate_change_in_stake > 0:
                    rate_change_in_stake_str = "[green]{:.2f}%[/green]".format(
                        rate_change_in_stake
                    )
                elif rate_change_in_stake < 0:
                    rate_change_in_stake_str = "[red]{:.2f}%[/red]".format(
                        rate_change_in_stake
                    )
                else:
                    rate_change_in_stake_str = "[grey0]0%[/grey0]"
        else:
            rate_change_in_stake_str = "[grey0]NA[/grey0]"

        table.add_row(
            # INDEX
            str(i),
            # DELEGATE
            Text(delegate_name, style=f"link {delegate_url}"),
            # SS58
            f"{delegate.hotkey_ss58:8.8}...",
            # NOMINATORS
            str(len([nom for nom in delegate.nominators if nom[1].rao > 0])),
            # DELEGATE STAKE
            f"{owner_stake!s:13.13}",
            # TOTAL STAKE
            f"{delegate.total_stake!s:13.13}",
            # CHANGE/(4h)
            rate_change_in_stake_str,
            # VPERMIT
            str(delegate.registrations),
            # TAKE
            f"{delegate.take * 100:.1f}%",
            # NOMINATOR/(24h)/k
            f"{bittensor.Balance.from_tao( delegate.total_daily_return.tao * (1000/ (0.001 + delegate.total_stake.tao)))!s:6.6}",
            # DELEGATE/(24h)
            f"{bittensor.Balance.from_tao(delegate.total_daily_return.tao * 0.18) !s:6.6}",
            # Desc
            str(delegate_description),
            end_section=True,
        )
    bittensor.__console__.print(table)


class DelegateStakeCommand:
    """
    Executes the ``delegate`` command, which stakes Tao to a specified delegate on the Bittensor network.

    This action allocates the user's Tao to support a delegate, potentially earning staking rewards in return.

    Optional Arguments:
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``delegate_ss58key``: The ``SS58`` address of the delegate to stake to.
        - ``amount``: The amount of Tao to stake.
        - ``all``: If specified, the command stakes all available Tao.

    The command interacts with the user to determine the delegate and the amount of Tao to be staked. If the ``--all``
    flag is used, it delegates the entire available balance.

    Usage:
        The user must specify the delegate's SS58 address and the amount of Tao to stake. The function sends a
        transaction to the subtensor network to delegate the specified amount to the chosen delegate. These values are
        prompted if not provided.

    Example usage::

        btcli delegate --delegate_ss58key <SS58_ADDRESS> --amount <AMOUNT>
        btcli delegate --delegate_ss58key <SS58_ADDRESS> --all

    Note:
        This command modifies the blockchain state and may incur transaction fees. It requires user confirmation and
        interaction, and is designed to be used within the Bittensor CLI environment. The user should ensure the
        delegate's address and the amount to be staked are correct before executing the command.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Delegates stake to a chain delegate."""
        try:
            config = cli.config.copy()
            wallet = bittensor.wallet(config=config)
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            subtensor.delegate(
                wallet=wallet,
                delegate_ss58=config.get("delegate_ss58key"),
                amount=config.get("amount"),
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        delegate_stake_parser = parser.add_parser(
            "delegate", help="""Delegate Stake to an account."""
        )
        delegate_stake_parser.add_argument(
            "--delegate_ss58key",
            "--delegate_ss58",
            dest="delegate_ss58key",
            type=str,
            required=False,
            help="""The ss58 address of the choosen delegate""",
        )
        delegate_stake_parser.add_argument(
            "--all", dest="stake_all", action="store_true"
        )
        delegate_stake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        bittensor.wallet.add_args(delegate_stake_parser)
        bittensor.subtensor.add_args(delegate_stake_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
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

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Get amount.
        if not config.get("amount") and not config.get("stake_all"):
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
                        ":cross_mark: [red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.stake_all = True


class DelegateUnstakeCommand:
    """
    Executes the ``undelegate`` command, allowing users to withdraw their staked Tao from a delegate on the Bittensor
    network.

    This process is known as "undelegating" and it reverses the delegation process, freeing up the staked tokens.

    Optional Arguments:
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``delegate_ss58key``: The ``SS58`` address of the delegate to undelegate from.
        - ``amount``: The amount of Tao to undelegate.
        - ``all``: If specified, the command undelegates all staked Tao from the delegate.

    The command prompts the user for the amount of Tao to undelegate and the ``SS58`` address of the delegate from which
    to undelegate. If the ``--all`` flag is used, it will attempt to undelegate the entire staked amount from the
    specified delegate.

    Usage:
        The user must provide the delegate's SS58 address and the amount of Tao to undelegate. The function will then
        send a transaction to the Bittensor network to process the undelegation.

    Example usage::

        btcli undelegate --delegate_ss58key <SS58_ADDRESS> --amount <AMOUNT>
        btcli undelegate --delegate_ss58key <SS58_ADDRESS> --all

    Note:
        This command can result in a change to the blockchain state and may incur transaction fees. It is interactive
        and requires confirmation from the user before proceeding. It should be used with care as undelegating can
        affect the delegate's total stake and
        potentially the user's staking rewards.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Undelegates stake from a chain delegate."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            DelegateUnstakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    def _run(self: "bittensor.cli", subtensor: "bittensor.subtensor"):
        """Undelegates stake from a chain delegate."""
        config = self.config.copy()
        wallet = bittensor.wallet(config=config)
        subtensor.undelegate(
            wallet=wallet,
            delegate_ss58=config.get("delegate_ss58key"),
            amount=config.get("amount"),
            wait_for_inclusion=True,
            prompt=not config.no_prompt,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        undelegate_stake_parser = parser.add_parser(
            "undelegate", help="""Undelegate Stake from an account."""
        )
        undelegate_stake_parser.add_argument(
            "--delegate_ss58key",
            "--delegate_ss58",
            dest="delegate_ss58key",
            type=str,
            required=False,
            help="""The ss58 address of the choosen delegate""",
        )
        undelegate_stake_parser.add_argument(
            "--all", dest="unstake_all", action="store_true"
        )
        undelegate_stake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        bittensor.wallet.add_args(undelegate_stake_parser)
        bittensor.subtensor.add_args(undelegate_stake_parser)

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

        # Get amount.
        if not config.get("amount") and not config.get("unstake_all"):
            if not Confirm.ask(
                "Unstake all Tao to account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Tao amount to unstake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark: [red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.unstake_all = True


class ListDelegatesCommand:
    """
    Displays a formatted table of Bittensor network delegates, providing a comprehensive overview of delegate statistics and information.

    This table helps users make informed decisions on which delegates to allocate their TAO stake.

    Optional Arguments:
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``subtensor.network``: The name of the network to use for the command.

    The table columns include:

    - INDEX: The delegate's index in the sorted list.
    - DELEGATE: The name of the delegate.
    - SS58: The delegate's unique SS58 address (truncated for display).
    - NOMINATORS: The count of nominators backing the delegate.
    - DELEGATE STAKE(τ): The amount of delegate's own stake (not the TAO delegated from any nominators).
    - TOTAL STAKE(τ): The delegate's cumulative stake, including self-staked and nominators' stakes.
    - CHANGE/(4h): The percentage change in the delegate's stake over the last four hours.
    - SUBNETS: The subnets to which the delegate is registered.
    - VPERMIT: Indicates the subnets for which the delegate has validator permits.
    - NOMINATOR/(24h)/kτ: The earnings per 1000 τ staked by nominators in the last 24 hours.
    - DELEGATE/(24h): The total earnings of the delegate in the last 24 hours.
    - DESCRIPTION: A brief description of the delegate's purpose and operations.

    Sorting is done based on the ``TOTAL STAKE`` column in descending order. Changes in stake are highlighted:
    increases in green and decreases in red. Entries with no previous data are marked with ``NA``. Each delegate's name
    is a hyperlink to their respective URL, if available.

    Example usage::

        btcli root list_delegates
        btcli root list_delegates --wallet.name my_wallet
        btcli root list_delegates --subtensor.network finney # can also be `test` or `local`

    Note:
        This function is part of the Bittensor CLI tools and is intended for use within a console application. It prints
        directly to the console and does not return any value.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""
        List all delegates on the network.
        """
        try:
            cli.config.subtensor.network = "archive"
            cli.config.subtensor.chain_endpoint = (
                "wss://archive.chain.opentensor.ai:443"
            )
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            ListDelegatesCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""
        List all delegates on the network.
        """
        with bittensor.__console__.status(":satellite: Loading delegates..."):
            delegates: list[bittensor.DelegateInfo] = subtensor.get_delegates()

            try:
                prev_delegates = subtensor.get_delegates(max(0, subtensor.block - 1200))
            except SubstrateRequestException:
                prev_delegates = None

        if prev_delegates is None:
            bittensor.__console__.print(
                ":warning: [yellow]Could not fetch delegates history[/yellow]"
            )

        show_delegates(
            delegates,
            prev_delegates=prev_delegates,
            width=cli.config.get("width", None),
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_delegates_parser = parser.add_parser(
            "list_delegates", help="""List all delegates on the network"""
        )
        bittensor.subtensor.add_args(list_delegates_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass


class NominateCommand:
    """
    Executes the ``nominate`` command, which facilitates a wallet to become a delegate on the Bittensor network.

    This command handles the nomination process, including wallet unlocking and verification of the hotkey's current
    delegate status.

    The command performs several checks:

    - Verifies that the hotkey is not already a delegate to prevent redundant nominations.
    - Tries to nominate the wallet and reports success or failure.

    Upon success, the wallet's hotkey is registered as a delegate on the network.

    Optional Arguments:
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``wallet.hotkey``: The name of the hotkey to use for the command.

    Usage:
        To run the command, the user must have a configured wallet with both hotkey and coldkey. If the wallet is not
        already nominated, this command will initiate the process.

    Example usage::

        btcli root nominate
        btcli root nominate --wallet.name my_wallet --wallet.hotkey my_hotkey

    Note:
        This function is intended to be used as a CLI command. It prints the outcome directly to the console and does
        not return any value. It should not be called programmatically in user code due to its interactive nature and
        side effects on the network state.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Nominate wallet."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            NominateCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Nominate wallet."""
        wallet = bittensor.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        try:
            wallet.coldkey
        except bittensor.KeyFileError:
            bittensor.__console__.print(
                ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
            )
            return

        # Check if the hotkey is already a delegate.
        if subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            bittensor.__console__.print(
                "Aborting: Hotkey {} is already a delegate.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        result: bool = subtensor.nominate(wallet)
        if not result:
            bittensor.__console__.print(
                "Could not became a delegate on [white]{}[/white]".format(
                    subtensor.network
                )
            )
        else:
            # Check if we are a delegate.
            is_delegate: bool = subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address)
            if not is_delegate:
                bittensor.__console__.print(
                    "Could not became a delegate on [white]{}[/white]".format(
                        subtensor.network
                    )
                )
                return
            bittensor.__console__.print(
                "Successfully became a delegate on [white]{}[/white]".format(
                    subtensor.network
                )
            )

            # Prompt use to set identity on chain.
            if not cli.config.no_prompt:
                do_set_identity = Prompt.ask(
                    f"Subnetwork registered successfully. Would you like to set your identity? [y/n]",
                    choices=["y", "n"],
                )

                if do_set_identity.lower() == "y":
                    subtensor.close()
                    config = cli.config.copy()
                    SetIdentityCommand.check_config(config)
                    cli.config = config
                    SetIdentityCommand.run(cli)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        nominate_parser = parser.add_parser(
            "nominate", help="""Become a delegate on the network"""
        )
        bittensor.wallet.add_args(nominate_parser)
        bittensor.subtensor.add_args(nominate_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class MyDelegatesCommand:
    """
    Executes the ``my_delegates`` command within the Bittensor CLI, which retrieves and displays a table of delegated
    stakes from a user's wallet(s) to various delegates on the Bittensor network.

    The command provides detailed insights into the user's
    staking activities and the performance of their chosen delegates.

    Optional Arguments:
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``all``: If specified, the command aggregates information across all wallets.

    The table output includes the following columns:

    - Wallet: The name of the user's wallet.
    - OWNER: The name of the delegate's owner.
    - SS58: The truncated SS58 address of the delegate.
    - Delegation: The amount of Tao staked by the user to the delegate.
    - τ/24h: The earnings from the delegate to the user over the past 24 hours.
    - NOMS: The number of nominators for the delegate.
    - OWNER STAKE(τ): The stake amount owned by the delegate.
    - TOTAL STAKE(τ): The total stake amount held by the delegate.
    - SUBNETS: The list of subnets the delegate is a part of.
    - VPERMIT: Validator permits held by the delegate for various subnets.
    - 24h/kτ: Earnings per 1000 Tao staked over the last 24 hours.
    - Desc: A description of the delegate.

    The command also sums and prints the total amount of Tao delegated across all wallets.

    Usage:
        The command can be run as part of the Bittensor CLI suite of tools and requires no parameters if a single wallet
        is used. If multiple wallets are present, the ``--all`` flag can be specified to aggregate information across
        all wallets.

    Example usage::

        btcli my_delegates
        btcli my_delegates --all
        btcli my_delegates --wallet.name my_wallet

    Note:
        This function is typically called by the CLI parser and is not intended to be used directly in user code.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Delegates stake to a chain delegate."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            MyDelegatesCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        """Delegates stake to a chain delegate."""
        config = cli.config.copy()
        if config.get("all", d=None):
            wallets = _get_coldkey_wallets_for_path(config.wallet.path)
        else:
            wallets = [bittensor.wallet(config=config)]

        table = Table(show_footer=True, pad_edge=False, box=None, expand=True)
        table.add_column(
            "[overline white]Wallet", footer_style="overline white", style="bold white"
        )
        table.add_column(
            "[overline white]OWNER",
            style="rgb(50,163,219)",
            no_wrap=True,
            justify="left",
        )
        table.add_column(
            "[overline white]SS58", footer_style="overline white", style="bold yellow"
        )
        table.add_column(
            "[overline green]Delegation",
            footer_style="overline green",
            style="bold green",
        )
        table.add_column(
            "[overline green]\u03c4/24h",
            footer_style="overline green",
            style="bold green",
        )
        table.add_column(
            "[overline white]NOMS", justify="center", style="green", no_wrap=True
        )
        table.add_column(
            "[overline white]OWNER STAKE(\u03c4)", justify="right", no_wrap=True
        )
        table.add_column(
            "[overline white]TOTAL STAKE(\u03c4)",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]SUBNETS", justify="right", style="white", no_wrap=True
        )
        table.add_column("[overline white]VPERMIT", justify="right", no_wrap=True)
        table.add_column("[overline white]24h/k\u03c4", style="green", justify="center")
        table.add_column("[overline white]Desc", style="rgb(50,163,219)")
        total_delegated = 0

        for wallet in tqdm(wallets):
            if not wallet.coldkeypub_file.exists_on_device():
                continue
            delegates = subtensor.get_delegated(
                coldkey_ss58=wallet.coldkeypub.ss58_address
            )

            my_delegates = {}  # hotkey, amount
            for delegate in delegates:
                for coldkey_addr, staked in delegate[0].nominators:
                    if (
                        coldkey_addr == wallet.coldkeypub.ss58_address
                        and staked.tao > 0
                    ):
                        my_delegates[delegate[0].hotkey_ss58] = staked

            delegates.sort(key=lambda delegate: delegate[0].total_stake, reverse=True)
            total_delegated += sum(my_delegates.values())

            registered_delegate_info: Optional[DelegatesDetails] = (
                get_delegates_details(url=bittensor.__delegates_details_url__)
            )
            if registered_delegate_info is None:
                bittensor.__console__.print(
                    ":warning:[yellow]Could not get delegate info from chain.[/yellow]"
                )
                registered_delegate_info = {}

            for i, delegate in enumerate(delegates):
                owner_stake = next(
                    map(
                        lambda x: x[1],  # get stake
                        filter(
                            lambda x: x[0] == delegate[0].owner_ss58,
                            delegate[0].nominators,
                        ),  # filter for owner
                    ),
                    bittensor.Balance.from_rao(0),  # default to 0 if no owner stake.
                )
                if delegate[0].hotkey_ss58 in registered_delegate_info:
                    delegate_name = registered_delegate_info[
                        delegate[0].hotkey_ss58
                    ].name
                    delegate_url = registered_delegate_info[delegate[0].hotkey_ss58].url
                    delegate_description = registered_delegate_info[
                        delegate[0].hotkey_ss58
                    ].description
                else:
                    delegate_name = ""
                    delegate_url = ""
                    delegate_description = ""

                if delegate[0].hotkey_ss58 in my_delegates:
                    table.add_row(
                        wallet.name,
                        Text(delegate_name, style=f"link {delegate_url}"),
                        f"{delegate[0].hotkey_ss58:8.8}...",
                        f"{my_delegates[delegate[0].hotkey_ss58]!s:13.13}",
                        f"{delegate[0].total_daily_return.tao * (my_delegates[delegate[0].hotkey_ss58]/delegate[0].total_stake.tao)!s:6.6}",
                        str(len(delegate[0].nominators)),
                        f"{owner_stake!s:13.13}",
                        f"{delegate[0].total_stake!s:13.13}",
                        str(delegate[0].registrations),
                        str(
                            [
                                "*" if subnet in delegate[0].validator_permits else ""
                                for subnet in delegate[0].registrations
                            ]
                        ),
                        # f'{delegate.take * 100:.1f}%',s
                        f"{ delegate[0].total_daily_return.tao * ( 1000 / ( 0.001 + delegate[0].total_stake.tao ) )!s:6.6}",
                        str(delegate_description),
                        # f'{delegate_profile.description:140.140}',
                    )

        bittensor.__console__.print(table)
        bittensor.__console__.print("Total delegated Tao: {}".format(total_delegated))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        delegate_stake_parser = parser.add_parser(
            "my_delegates",
            help="""Show all delegates where I am delegating a positive amount of stake""",
        )
        delegate_stake_parser.add_argument(
            "--all",
            action="store_true",
            help="""Check all coldkey wallets.""",
            default=False,
        )
        bittensor.wallet.add_args(delegate_stake_parser)
        bittensor.subtensor.add_args(delegate_stake_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.get("all", d=None)
            and not config.is_set("wallet.name")
            and not config.no_prompt
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)


class SetTakeCommand:
    """
    Executes the ``set_take`` command, which sets the delegate take.

    The command performs several checks:

        1. Hotkey is already a delegate
        2. New take value is within 0-18% range

    Optional Arguments:
        - ``take``: The new take value
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``wallet.hotkey``: The name of the hotkey to use for the command.

    Usage:
        To run the command, the user must have a configured wallet with both hotkey and coldkey. Also, the hotkey should already be a delegate.

    Example usage::
        btcli root set_take --wallet.name my_wallet --wallet.hotkey my_hotkey

    Note:
        This function can be used to update the takes individually for every subnet
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set delegate take."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetTakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set delegate take."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is not a delegate.
        if not subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            bittensor.__console__.print(
                "Aborting: Hotkey {} is NOT a delegate.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        # Prompt user for take value.
        new_take_str = config.get("take")
        if new_take_str == None:
            new_take = FloatPrompt.ask(f"Enter take value (0.18 for 18%)")
        else:
            new_take = float(new_take_str)

        if new_take > 0.18:
            bittensor.__console__.print("ERROR: Take value should not exceed 18%")
            return

        result: bool = subtensor.set_take(
            wallet=wallet,
            delegate_ss58=wallet.hotkey.ss58_address,
            take=new_take,
        )
        if not result:
            bittensor.__console__.print("Could not set the take")
        else:
            # Check if we are a delegate.
            is_delegate: bool = subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address)
            if not is_delegate:
                bittensor.__console__.print(
                    "Could not set the take [white]{}[/white]".format(subtensor.network)
                )
                return
            bittensor.__console__.print(
                "Successfully set the take on [white]{}[/white]".format(
                    subtensor.network
                )
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_take_parser = parser.add_parser(
            "set_take", help="""Set take for delegate"""
        )
        set_take_parser.add_argument(
            "--take",
            dest="take",
            type=float,
            required=False,
            help="""Take as a float number""",
        )
        bittensor.wallet.add_args(set_take_parser)
        bittensor.subtensor.add_args(set_take_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
