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
import bittensor as bt
from rich.table import Table

class ListSubnetsCommand:
    @staticmethod
    def run(cli: "bt.cli"):
        r"""List all subnet netuids in the network."""
        try:
            subtensor: "bt.subtensor" = bt.subtensor(
                config=cli.config, log_verbose=False
            )
            ListSubnetsCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bt.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bt.cli", subtensor: "bt.subtensor"):
        r"""List all subnet netuids in the network."""
        
        # Initialize variables to store aggregated data
        rows = []
        subnets = subtensor.get_all_subnet_dynamic_info()
        for subnet in subnets:
            rows.append(
                (
                    str(subnet.netuid),
                    f"[light_goldenrod1]{subnet.symbol}[light_goldenrod1]",
                    f"τ {subnet.emission.tao:.4f}",
                    # f"P( τ {subnet.tao_in.tao:,.4f},",
                    f"τ {subnet.tao_in.tao:,.4f}",
                    # f"{subnet.alpha_in.tao:,.4f} {subnet.symbol} )",
                    f"{subnet.alpha_out.tao:,.4f} {subnet.symbol}",
                    f"{subnet.price.tao:.4f} τ/{subnet.symbol}",
                    str(subnet.blocks_since_last_step) + "/" + str(subnet.tempo),
                    # f"{subnet.owner_locked}" + "/" + f"{subnet.total_locked}",
                    # f"{subnet.owner[:3]}...{subnet.owner[-3:]}",
                )
            )

        # Define table properties
        console_width = bt.__console__.width - 5
        table = Table(
            title="Subnet Info",
            width=console_width,
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
        table.title = f"[white]Subnets - {subtensor.network}\n"

        # Add columns to the table
        # price_total = f"τ{total_price.tao:.2f}/{bt.Balance.from_rao(dynamic_emission).tao:.2f}"
        # above_price_threshold = total_price.tao > bt.Balance.from_rao(dynamic_emission).tao

        table.add_column("Netuid", style="rgb(253,246,227)", no_wrap=True, justify="center")
        table.add_column("Symbol", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column(f"Emission ({bt.Balance.get_unit(0)})", style="rgb(38,139,210)", no_wrap=True, justify="right")
        table.add_column(f"TAO({bt.Balance.get_unit(0)})", style="medium_purple", no_wrap=True, justify="right")
        # table.add_column(f"{bt.Balance.get_unit(1)})", style="rgb(42,161,152)", no_wrap=True, justify="left")
        table.add_column(f"Stake({bt.Balance.get_unit(1)})", style="green", no_wrap=True, justify="right")
        table.add_column(f"Rate ({bt.Balance.get_unit(1)}/{bt.Balance.get_unit(0)})", style="light_goldenrod2", no_wrap=True, justify="center")
        table.add_column("Tempo (k/n)", style="light_salmon3", no_wrap=True, justify="center")
        # table.add_column(f"Locked ({bt.Balance.get_unit(1)})", style="rgb(38,139,210)", no_wrap=True, justify="center")
        # table.add_column("Owner", style="rgb(38,139,210)", no_wrap=True, justify="center")

        # Sort rows by subnet.emission.tao, keeping the first subnet in the first position
        sorted_rows = [rows[0]] + sorted(rows[1:], key=lambda x: x[2], reverse=True)

        # Add rows to the table
        for row in sorted_rows:
            table.add_row(*row)

        # Print the table
        bt.__console__.print(table)
        bt.__console__.print(
            """
[bold white]Description[/bold white]:
    The table displays relevant information about each subnet on the network. 
    The columns are as follows:
        - [bold white]Netuid[/bold white]: The unique identifier for the subnet (its index).
        - [bold white]Symbol[/bold white]: The symbol representing the subnet's stake.
        - [bold white]Emission[/bold white]: The amount of TAO added to the subnet every block. Calculated by dividing the TAO (t) column values by the sum of the TAO (t) column.
        - [bold white]TAO[/bold white]: The TAO staked into the subnet ( which dynamically changes during stake, unstake and emission events ).
        - [bold white]Stake[/bold white]: The outstanding supply of stake across all staking accounts on this subnet.
        - [bold white]Rate[/bold white]: The rate of conversion between TAO and the subnet's staking unit.
        - [bold white]Tempo[/bold white]: The number of blocks between epochs. Represented as (k/n) where k is the blocks since the last epoch and n is the total blocks in the epoch.
"""
)


    @staticmethod
    def check_config(config: "bt.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list", help="""List all subnets on the network"""
        )
        bt.subtensor.add_args(list_subnets_parser)