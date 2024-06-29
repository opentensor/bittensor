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

from rich.table import Table
from rich.prompt import Prompt
from rich.console import Console
from rich.text import Text

import bittensor
from .. import defaults  # type: ignore
from ... import ChildInfo
from ...utils.formatting import u16_to_float

console = bittensor.__console__


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
        r"""Set child hotkey."""
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
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        # Parse from strings
        netuid = cli.config.netuid

        children = subtensor.get_children_info(
            netuid=netuid,
        )

        if cli.config.interactive:
            # interactive flag for new experience
            GetChildrenCommand.render_table_interactive(children, netuid)
        else:
            GetChildrenCommand.render_table(children, netuid)

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

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def render_table(children: list[ChildInfo], netuid: int):
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
        table.add_column("ParentHotKeys", style="cyan", no_wrap=True)
        table.add_column("Proportion", style="cyan", no_wrap=True, justify="right")
        table.add_column("Total Stake", style="cyan", no_wrap=True, justify="right")
        table.add_column("Emissions/Day", style="cyan", no_wrap=True, justify="right")
        table.add_column(
            "Return per 1000 TAO", style="cyan", no_wrap=True, justify="right"
        )
        table.add_column("Take", style="cyan", no_wrap=True, justify="right")

        sum_proportion = 0.0
        sum_total_stake = 0.0
        sum_emissions_per_day = 0.0
        sum_return_per_1000 = 0.0
        sum_take = 0.0

        child_hotkeys_set = set()
        parent_hotkeys_set = set()

        if not children:
            console.print(table)
            # Summary Row
            summary = Text(
                "Total (0) | Total (0) | 0.000000 | 0.0000 | 0.0000 | 0.0000 | 0.000000",
                style="dim",
            )
            console.print(summary)

            command = f"btcli stake set_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid {netuid} --proportion <float that is less than 1 >"
            console.print(f"There are currently no child hotkeys on subnet {netuid}.")
            console.print(
                f"To add a child hotkey you can run the command: [white]{command}[/white]"
            )
            return

        # Sort children by proportion (highest first)
        sorted_children = sorted(
            children.items(), key=lambda item: item[1][0].proportion, reverse=True
        )

        # Populate table with children data
        index = 1
        for child_hotkey, child_infos in sorted_children:
            for child_info in child_infos:
                table.add_row(
                    str(index),
                    child_info.child_ss58[:5] + "..." + child_info.child_ss58[-5:],
                    str(len(child_info.parents)),
                    str(u16_to_float(child_info.proportion)),
                    str(child_info.total_stake),
                    str(child_info.emissions_per_day),
                    str(child_info.return_per_1000),
                    str(child_info.take),
                )

                # Update totals and sets
                child_hotkeys_set.add(child_info.child_ss58)
                parent_hotkeys_set.update(p[1] for p in child_info.parents)
                sum_proportion += child_info.proportion
                sum_total_stake += float(child_info.total_stake)
                sum_emissions_per_day += float(child_info.emissions_per_day)
                sum_return_per_1000 += float(child_info.return_per_1000)
                sum_take += float(child_info.take)

        # Calculate averages
        total_child_hotkeys = len(child_hotkeys_set)
        total_parent_hotkeys = len(parent_hotkeys_set)
        avg_emissions_per_day = (
            sum_emissions_per_day / total_child_hotkeys if total_child_hotkeys else 0
        )
        avg_return_per_1000 = (
            sum_return_per_1000 / total_child_hotkeys if total_child_hotkeys else 0
        )

        # Print table to console
        console.print(table)

        # Add a summary row with fixed-width fields
        summary = Text(
            f"Total ({total_child_hotkeys:3}) | Total ({total_parent_hotkeys:3}) | "
            f"Total ({u16_to_float(sum_proportion):10.6f}) | Total ({sum_total_stake:10.4f}) | "
            f"Avg ({avg_emissions_per_day:10.4f}) | Avg ({avg_return_per_1000:10.4f}) | "
            f"Total ({sum_take:10.6f})",
            style="dim",
        )
        console.print(summary)

    @staticmethod
    def render_help_details():
        console = Console()
        console.print("\nColumn Information:")
        console.print(
            "[cyan]ChildHotkey:[/cyan] Truncated child hotkey associated with the child"
        )
        console.print(
            "[cyan]ParentHotKeys:[/cyan] The number of parent hotkeys associated with the child"
        )
        console.print(
            "[cyan]Proportion:[/cyan] Proportion of stake allocated to this child"
        )
        console.print("[cyan]Total Stake:[/cyan] Total stake of the child")
        console.print("[cyan]Emissions/Day:[/cyan] Emissions per day for this child")
        console.print(
            "[cyan]Return per 1000 TAO:[/cyan] Return per 1000 TAO staked for this child"
        )
        console.print("[cyan]Take:[/cyan] Commission taken by the child")

    @staticmethod
    def render_child_details(child_info: ChildInfo):
        console = Console()
        console.print("In child hotkey")

        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="green",
            style="green",
        )

        table.add_column("ParentHotKey", style="cyan", no_wrap=True)
        table.add_column("Proportion", style="cyan", no_wrap=True, justify="right")

        for parent in child_info.parents:
            table.add_row(str(parent[0]), str(parent[1]))

        console.print(table)
        command = Prompt.ask("To revoke child hotkey press r (b for back, q for quit)")
        if command == "r":
            console.print("Revoking child key coming soon")
        elif command == "b":
            console.print("going back coming soon")
        elif command == "q":
            console.clear()
            return

    @staticmethod
    def render_table_interactive(children: list[ChildInfo], netuid: int):
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
        table.add_column("ParentHotKeys", style="cyan", no_wrap=True)
        table.add_column("Proportion", style="cyan", no_wrap=True, justify="right")
        table.add_column("Total Stake", style="cyan", no_wrap=True, justify="right")
        table.add_column("Emissions/Day", style="cyan", no_wrap=True, justify="right")
        table.add_column(
            "Return per 1000 TAO", style="cyan", no_wrap=True, justify="right"
        )
        table.add_column("Take", style="cyan", no_wrap=True, justify="right")

        sum_proportion = 0.0
        sum_total_stake = 0.0
        sum_emissions_per_day = 0.0
        sum_return_per_1000 = 0.0
        sum_take = 0.0

        child_hotkeys_set = set()
        parent_hotkeys_set = set()

        if not children:
            console.print(table)
            # Summary Row
            summary = Text(
                "Total (0) | Total (0) | 0.000000 | 0.0000 | 0.0000 | 0.0000 | 0.000000",
                style="dim",
            )
            console.print(summary)

            # command = f"btcli stake set_child --child <child_hotkey> --hotkey <parent_hotkey> --netuid {netuid} --proportion <float that is less than 1 >"
            console.print(f"There are currently no child hotkeys on subnet {netuid}.")
            # console.print(
            #     f"To add a child hotkey you can run the command: [white]{command}[/white]"
            # )
            command = Prompt.ask(
                "To add a child hotkey press s (h for help, q for quit)"
            )
            if command == "s":
                console.print("Adding child key cmming soon")
            elif command == "h":
                GetChildrenCommand.render_help_details()
                GetChildrenCommand.render_table_interactive(children, netuid)
            else:
                return

        # Sort children by proportion (highest first)
        sorted_children = sorted(
            children.items(), key=lambda item: item[1][0].proportion, reverse=True
        )

        # Populate table with children data
        index = 1
        for child_hotkey, child_infos in sorted_children:
            for child_info in child_infos:
                table.add_row(
                    str(index),
                    child_info.child_ss58[:5] + "..." + child_info.child_ss58[-5:],
                    str(len(child_info.parents)),
                    str(u16_to_float(child_info.proportion)),
                    str(child_info.total_stake),
                    str(child_info.emissions_per_day),
                    str(child_info.return_per_1000),
                    str(child_info.take),
                )

                # Update totals and sets
                child_hotkeys_set.add(child_info.child_ss58)
                parent_hotkeys_set.update(p[1] for p in child_info.parents)
                sum_proportion += child_info.proportion
                sum_total_stake += float(child_info.total_stake)
                sum_emissions_per_day += float(child_info.emissions_per_day)
                sum_return_per_1000 += float(child_info.return_per_1000)
                sum_take += float(child_info.take)

        # Calculate averages
        total_child_hotkeys = len(child_hotkeys_set)
        total_parent_hotkeys = len(parent_hotkeys_set)
        avg_emissions_per_day = (
            sum_emissions_per_day / total_child_hotkeys if total_child_hotkeys else 0
        )
        avg_return_per_1000 = (
            sum_return_per_1000 / total_child_hotkeys if total_child_hotkeys else 0
        )

        # Print table to console
        console.print(table)

        # Add a summary row with fixed-width fields
        summary = Text(
            f"Total ({total_child_hotkeys:3}) | Total ({total_parent_hotkeys:3}) | "
            f"Total ({u16_to_float(sum_proportion):10.6f}) | Total ({sum_total_stake:10.4f}) | "
            f"Avg ({avg_emissions_per_day:10.4f}) | Avg ({avg_return_per_1000:10.4f}) | "
            f"Total ({sum_take:10.6f})",
            style="dim",
        )
        console.print(summary)

        command = Prompt.ask(
            "To see more information enter child index (s for adding child hotkey, h for help, q for quit)"
        )
        if command == "s":
            console.print("Adding child key coming soon")
        elif command == "h":
            GetChildrenCommand.render_help_details()
            GetChildrenCommand.render_table_interactive(children, netuid)
        elif command == "q":
            return
        else:
            try:
                selected_index = int(command) - 1
                if 0 <= selected_index < len(children):
                    console.print(children)
                    selected_child = children[selected_index]
                    GetChildrenCommand.render_child_details(selected_child)
                else:
                    console.print(
                        f"Invalid index: {command}. Please select a valid index."
                    )
                    GetChildrenCommand.render_table_interactive(children, netuid)
            except ValueError:
                console.print(
                    "Invalid input. Please enter a valid index or 'h' for help."
                )
                GetChildrenCommand.render_table_interactive(children, netuid)
