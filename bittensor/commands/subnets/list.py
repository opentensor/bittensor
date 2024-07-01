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
from typing import Optional, List, Dict
from ..utils import get_delegates_details 

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
        # Fetch all subnet information
        subnets: List[bt.SubnetInfoV2] = subtensor.get_all_subnets_info_v2()

        # Initialize variables to store aggregated data
        rows = []
        total_neurons = 0
        total_registered = 0
        total_price = 0
        total_emission = 0
        dynamic_emission = 0
        n_dtao = 0
        n_stao = 0
        total_tao_locked = 0

        # Fetch delegate information
        delegate_info = get_delegates_details(url=bt.__delegates_details_url__)

        # Process each subnet and collect relevant data
        for subnet in subnets:
            pool = subnet.dynamic_pool
            total_neurons += subnet.max_n
            total_registered += subnet.subnetwork_n
            total_price += pool.price if pool.is_dynamic else bt.Balance.from_rao(0)
            total_emission += subnet.emission_value
            dynamic_emission += subnet.emission_value if pool.is_dynamic else 0
            tao_locked = subnet.tao_locked
            total_tao_locked += tao_locked

            sn_symbol = f"{bt.Balance.get_unit(subnet.netuid)}\u200E"
            alpha_out_str = (
                f"{sn_symbol}{pool.alpha_outstanding.tao:,.4f}"
                if pool.is_dynamic
                else f"τ{tao_locked.tao:,.4f}"
            )
            if pool.is_dynamic:
                n_dtao += 1
            else:
                n_stao += 1

            # Append row data for the table
            rows.append(
                (
                    str(subnet.netuid),
                    f"[light_goldenrod1]{sn_symbol}[light_goldenrod1]",
                    f"{subnet.subnetwork_n}/{subnet.max_n}",
                    "[indian_red]dynamic[/indian_red]" if pool.is_dynamic else "[light_sky_blue3]stable[/light_sky_blue3]",
                    f"τ{bt.Balance.from_rao(subnet.emission_value).tao:.4f}",
                    f"τ{tao_locked.tao:,.4f}",
                    f"P({pool.tao_reserve},",
                    f"{pool.alpha_reserve.tao:.4f}{sn_symbol})",
                    alpha_out_str,
                    f"{pool.price.tao:.4f}τ/{sn_symbol}" if pool.is_dynamic else "[grey0]NA[/grey0]",
                    str(subnet.hyperparameters["tempo"]),
                    f"{subnet.burn!s:8.8}",
                    f"{delegate_info[subnet.owner_ss58].name if subnet.owner_ss58 in delegate_info else subnet.owner_ss58[:5] + '...' + subnet.owner_ss58[-5:]}",
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
        price_total = f"τ{total_price.tao:.2f}/{bt.Balance.from_rao(dynamic_emission).tao:.2f}"
        above_price_threshold = total_price.tao > bt.Balance.from_rao(dynamic_emission).tao

        table.add_column("Index", style="rgb(253,246,227)", no_wrap=True, justify="center")
        table.add_column("Symbol", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("n", style="rgb(108,113,196)", no_wrap=True, justify="center")
        table.add_column("Type", style="rgb(181,137,0)", no_wrap=True, justify="center")
        table.add_column("Emission", style="rgb(38,139,210)", no_wrap=True, justify="center")
        table.add_column(f"{bt.Balance.unit}", style="rgb(220,50,47)", no_wrap=True, justify="right")
        table.add_column(f"P({bt.Balance.unit},", style="rgb(108,113,196)", no_wrap=True, justify="right")
        table.add_column(f"{bt.Balance.get_unit(1)})", style="rgb(42,161,152)", no_wrap=True, justify="left")
        table.add_column(f"{bt.Balance.get_unit(1)}", style="rgb(133,153,0)", no_wrap=True, justify="right")
        table.add_column("Price", style="rgb(181,137,0)", no_wrap=True, justify="center", footer=f"[red]↓ {price_total}[/red]" if above_price_threshold else f"[green]↑ {price_total}[/green]")
        table.add_column("Tempo", style="rgb(38,139,210)", no_wrap=True, justify="center")
        table.add_column("Burn", style="rgb(220,50,47)", no_wrap=True, justify="center")
        table.add_column("Owner", style="rgb(108,113,196)", no_wrap=True)

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