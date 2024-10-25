# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import re
from typing import List, Tuple

from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

import bittensor
from bittensor.utils.balance import Balance
from .. import defaults
from ...utils import wallet_utils

console = bittensor.__console__


class SetChildrenCommand:
    """
    Executes the ``set_children`` command to add children hotkeys on a specified subnet on the Bittensor network.

    This command is used to delegate authority to different hotkeys, securing their position and influence on the subnet.

    Usage:
        Users can specify the amount or 'proportion' to delegate to child hotkeys (``SS58`` address),
        the user needs to have sufficient authority to make this call, and the sum of proportions cannot be greater than 1.

    The command prompts for confirmation before executing the set_children operation.

    Example usage::

        btcli stake set_children --children <child_hotkey>,<child_hotkey> --hotkey <parent_hotkey> --netuid 1 --proportions 0.3,0.3

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
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # display children
        GetChildrenCommand.retrieve_children(
            subtensor=subtensor,
            hotkey=cli.config.hotkey,
            netuid=cli.config.netuid,
            render_table=True,
        )

        if not cli.config.is_set("children"):
            cli.config.children = Prompt.ask(
                "Enter children hotkey (ss58) as comma-separated values"
            )

        if not cli.config.is_set("proportions"):
            cli.config.proportions = Prompt.ask(
                "Enter proportions for children as comma-separated values (sum less than 1)"
            )

        # Parse from strings
        netuid = cli.config.netuid

        # extract proportions and child addresses from cli input
        proportions = [float(x) for x in re.split(r"[ ,]+", cli.config.proportions)]
        children = [str(x) for x in re.split(r"[ ,]+", cli.config.children)]

        # Validate children SS58 addresses
        for child in children:
            if not wallet_utils.is_valid_ss58_address(child):
                console.print(f":cross_mark:[red] Invalid SS58 address: {child}[/red]")
                return

        total_proposed = sum(proportions)
        if total_proposed > 1:
            raise ValueError(
                f"Invalid proportion: The sum of all proportions cannot be greater than 1. Proposed sum of proportions is {total_proposed}."
            )

        children_with_proportions = list(zip(proportions, children))

        success, message = subtensor.set_children(
            wallet=wallet,
            netuid=netuid,
            hotkey=cli.config.hotkey,
            children_with_proportions=children_with_proportions,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            GetChildrenCommand.retrieve_children(
                subtensor=subtensor,
                hotkey=cli.config.hotkey,
                netuid=cli.config.netuid,
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
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter [light_salmon3]hotkey[/light_salmon3] name",
                default=defaults.wallet.hotkey,
            )
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
            default=False,
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
            default=False,
            help="""Prompt for confirmation before proceeding.""",
        )
        bittensor.wallet.add_args(set_children_parser)
        bittensor.subtensor.add_args(set_children_parser)


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
        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        # Get values if not set.
        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # Parse from strings
        netuid = cli.config.netuid
        hotkey = cli.config.hotkey

        children = subtensor.get_children(hotkey, netuid)

        GetChildrenCommand.render_table(subtensor, hotkey, children, netuid, True)

        return children

    @staticmethod
    def retrieve_children(
        subtensor: "bittensor.subtensor", hotkey: str, netuid: int, render_table: bool
    ):
        """

        Static method to retrieve children for a given subtensor.

        Args:
            subtensor (bittensor.subtensor): The subtensor object used to interact with the Bittensor network.
            hotkey (str): The hotkey of the tensor owner.
            netuid (int): The network unique identifier of the subtensor.
            render_table (bool): Flag indicating whether to render the retrieved children in a table.

        Returns:
            List[str]: A list of children hotkeys.

        """
        children = subtensor.get_children(hotkey, netuid)
        if render_table:
            GetChildrenCommand.render_table(subtensor, hotkey, children, netuid, False)
        return children

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter [light_salmon3]hotkey[/light_salmon3] name",
                default=defaults.wallet.hotkey,
            )
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
        table.add_column("Total Stake", style="cyan", no_wrap=True, justify="right")

        if not children:
            console.print(table)
            console.print(
                f"There are currently no child hotkeys on subnet {netuid} with ParentHotKey {hotkey}."
            )
            if prompt:
                command = f"btcli stake set_children --children <child_hotkey> --hotkey <parent_hotkey> --netuid {netuid} --proportion <float>"
                console.print(
                    f"To add a child hotkey you can run the command: [white]{command}[/white]"
                )
            return

        console.print("ParentHotKey:", style="cyan", no_wrap=True)
        console.print(hotkey)

        # calculate totals
        total_proportion = 0
        total_stake = 0

        children_info = []
        for child in children:
            proportion = child[0]
            child_hotkey = child[1]
            child_stake = subtensor.get_total_stake_for_hotkey(
                ss58_address=child_hotkey
            ) or Balance(0)

            # add to totals
            total_proportion += proportion
            total_stake += child_stake

            children_info.append((proportion, child_hotkey, child_stake))

        children_info.sort(
            key=lambda x: x[0], reverse=True
        )  # sorting by proportion (highest first)

        # add the children info to the table
        for i, (proportion, hotkey, stake) in enumerate(children_info, 1):
            proportion_str = Text(
                str(proportion), style="red" if proportion == 0 else ""
            )
            hotkey = Text(hotkey, style="red" if proportion == 0 else "")
            table.add_row(
                str(i),
                hotkey,
                proportion_str,
                str(stake),
            )

        # add totals row
        table.add_row("", "Total", str(total_proportion), str(total_stake), "")
        console.print(table)


class RevokeChildrenCommand:
    """
    Executes the ``revoke_children`` command to remove all children hotkeys on a specified subnet on the Bittensor network.

    This command is used to remove delegated authority from all child hotkeys, removing their position and influence on the subnet.

    Usage:
        Users need to specify the parent hotkey and the subnet ID (netuid).
        The user needs to have sufficient authority to make this call.

    The command prompts for confirmation before executing the revoke_children operation.

    Example usage::

        btcli stake revoke_children --hotkey <parent_hotkey> --netuid 1

    Note:
        This command is critical for users who wish to remove children hotkeys on the network.
        It allows for a complete removal of delegated authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Revokes all children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RevokeChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # Get and display current children information
        current_children = GetChildrenCommand.retrieve_children(
            subtensor=subtensor,
            hotkey=cli.config.hotkey,
            netuid=cli.config.netuid,
            render_table=False,
        )

        # Parse from strings
        netuid = cli.config.netuid

        # Prepare children with zero proportions
        children_with_zero_proportions = [(0.0, child[1]) for child in current_children]

        success, message = subtensor.set_children(
            wallet=wallet,
            netuid=netuid,
            children_with_proportions=children_with_zero_proportions,
            hotkey=cli.config.hotkey,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            if cli.config.wait_for_finalization and cli.config.wait_for_inclusion:
                GetChildrenCommand.retrieve_children(
                    subtensor=subtensor,
                    hotkey=cli.config.hotkey,
                    netuid=cli.config.netuid,
                    render_table=True,
                )
            console.print(
                ":white_heavy_check_mark: [green]Revoked all children hotkeys.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to revoke children hotkeys.[/red] {message}"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter [light_salmon3]hotkey[/light_salmon3] name",
                default=defaults.wallet.hotkey,
            )
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "revoke_children", help="""Revoke all children hotkeys."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--hotkey", dest="hotkey", type=str, required=False)
        parser.add_argument(
            "--wait_for_inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
            help="""Wait for the transaction to be included in a block.""",
        )
        parser.add_argument(
            "--wait_for_finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=False,
            help="""Wait for the transaction to be finalized.""",
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
            help="""Prompt for confirmation before proceeding.""",
        )
        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)
