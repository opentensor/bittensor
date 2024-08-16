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
from typing import List
from tqdm import tqdm


class ListSubnetsCommand:
    @staticmethod
    def run(cli: "bt.cli"):
        """List all subnet netuids in the network."""
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
        """List all subnet netuids in the network."""
        # Fetch all subnet information
        subnets: List[int] = subtensor.get_subnets()

        # Initialize variables to store aggregated data
        rows = []
        total_price = 0
        total_emission = 0
        dynamic_emission = 0
        # Process each subnet and collect relevant data
        for netuid in tqdm(subnets):
            type = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="SubnetMechanism",
                params=[netuid]
            ).value
            emission = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="EmissionValues",
                params=[netuid]
            ).value/10**9
            tao_in = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="SubnetTAO",
                params=[netuid]
            ).value/10**9
            alpha_in = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="SubnetAlphaIn",
                params=[netuid]
            ).value/10**9
            alpha_out = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="SubnetAlphaOut",
                params=[netuid]
            ).value/10**9
            tempo = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="Tempo",
                params=[netuid]
            ).value
            price = float(tao_in) / float(alpha_in) if alpha_in > 0 else 1.0
            total_price += price
            total_emission += emission
            sn_symbol = f"{bt.Balance.get_unit(netuid)}\u200E"

            # Append row data for the table
            rows.append(
                (
                    str(netuid),
                    f"[light_goldenrod1]{sn_symbol}[light_goldenrod1]",
                    f"τ{bt.Balance.from_tao(emission).tao:.4f}",
                    f"P( τ{tao_in:,.4f},",
                    f"{alpha_in:,.4f}{sn_symbol} )",
                    f"{alpha_out:,.4f}{sn_symbol}",
                    f"{price:.4f}τ/{sn_symbol}",
                    str(tempo),
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

        table.add_column("Index", style="rgb(253,246,227)", no_wrap=True, justify="center")
        table.add_column("Symbol", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Emission", style="rgb(38,139,210)", no_wrap=True, justify="center")
        table.add_column(f"P({bt.Balance.unit},", style="rgb(108,113,196)", no_wrap=True, justify="right")
        table.add_column(f"{bt.Balance.get_unit(1)})", style="rgb(42,161,152)", no_wrap=True, justify="left")
        table.add_column(f"{bt.Balance.get_unit(1)}", style="rgb(133,153,0)", no_wrap=True, justify="center")
        table.add_column("Price", style="rgb(181,137,0)", no_wrap=True, justify="center")
        table.add_column("Tempo", style="rgb(38,139,210)", no_wrap=True, justify="center")

        # Add rows to the table
        for row in rows:
            table.add_row(*row)

        # Print the table
        bt.__console__.print(table)

    @staticmethod
    def check_config(config: "bt.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list", help="""List all subnets on the network"""
        )
        bt.subtensor.add_args(list_subnets_parser)
