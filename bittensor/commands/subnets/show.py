# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
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
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from bittensor.chain_data import SubnetState
from bittensor.config import config as Config
from bittensor.subtensor import Subtensor

if TYPE_CHECKING:
    from bittensor.cli import cli as Cli


class ShowSubnet:

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser("show", help="""Show Subnet Stake.""")
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--no_prompt", "--y", "-y", dest='no_prompt', required=False, action='store_true', help="""Specify this flag to delegate stake""")
        Subtensor.add_args(stake_parser)
        
    @staticmethod
    def check_config(config: "Config"): pass
    
    @staticmethod
    def run(cli: "Cli"):
        console = Console()
        config = cli.config.copy()
        subtensor = Subtensor(config=config, log_verbose=False)
        
        subnet_state: "SubnetState" = SubnetState.from_vec_u8(
            subtensor.substrate.rpc_request(method="subnetInfo_getSubnetState", params=[config.netuid, None])['result']
        )
        # Define table properties
        console_width = console.width - 5

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
        table.title = f"[white]Subnet State for subnet #{config.netuid}."

        # Add columns to the table
        table.add_column("Index", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Hotkeys", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Coldkeys", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Active", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Validator Permit", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Pruning Score", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Last Update", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Emission", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Dividends", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Incentives", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Consensus", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Trust", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Rank", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Block at registration", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Local Stake", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Global Stake", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column("Stake Weight", style="rgb(211,54,130)", no_wrap=True, justify="center")

        for idx, hk in enumerate(subnet_state.hotkeys):
            table.add_row(
                subnet_state.hotkeys[idx],
                subnet_state.hotkeys[idx],
                subnet_state.coldkeys[idx],
                "Yes" if subnet_state.active[idx] else "No",
                "Yes" if subnet_state.validator_permit[idx] else "No",
                str(subnet_state.pruning_score[idx]),
                str(subnet_state.last_update[idx]),
                str(subnet_state.emission[idx]),
                str(subnet_state.dividends[idx]),
                str(subnet_state.incentives[idx]),
                str(subnet_state.consensus[idx]),
                str(subnet_state.trust[idx]),
                str(subnet_state.rank[idx]),
                str(subnet_state.block_at_registration[idx]),
                str(subnet_state.local_stake[idx]),
                str(subnet_state.global_stake[idx]),
                str(subnet_state.stake_weight[idx]),
            )

        # Print the table
        config.print(table)
