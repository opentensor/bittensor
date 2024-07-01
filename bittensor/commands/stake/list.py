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
import typing
import argparse
from rich.table import Table
from rich.prompt import Prompt
from typing import Optional

import bittensor
from .. import defaults
from ..utils import (
    get_delegates_details,
    DelegatesDetails,
)
from substrateinterface.exceptions import SubstrateRequestException

console = bittensor.__console__

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
        
        # Iterate over substakes and aggregate them by hotkey.
        hotkeys_to_substakes: typing.Dict[str, typing.List[typing.Dict]] = {}
        for substake in substakes:
            hotkey = substake["hotkey"]
            if substake["stake"].rao == 0: continue
            if hotkey not in hotkeys_to_substakes:
                hotkeys_to_substakes[hotkey] = []
            hotkeys_to_substakes[hotkey].append( substake )
            
            
        def table_substakes( hotkey:str, substakes: typing.List[typing.Dict] ):
            # Create table structure.
            name = registered_delegate_info[hotkey].name + f" ({hotkey})" if hotkey in registered_delegate_info else hotkey
            table = Table(show_footer=True, pad_edge=False, box=None, expand=False, title=f"{name}")
            table.add_column(f"[white]", footer_style="overline white", style="grey89")
            table.add_column(f"[white]", footer_style="white", style="light_goldenrod1", justify="center", width=5, no_wrap=True)
            table.add_column(f"[white]({bittensor.Balance.unit}/{bittensor.Balance.get_unit(1)})", footer_style="white", style="light_goldenrod2", justify="center" )
            table.add_column(f"[white]{bittensor.Balance.get_unit(1)}", footer_style="overline white", style="green",  justify="center" )
            table.add_column(f"[white]{bittensor.Balance.unit}", footer_style="overline white", style="blue", justify="center" )
            table.add_column(f"[white]Swap({bittensor.Balance.get_unit(1)}) -> {bittensor.Balance.unit}", footer_style="overline white", style="blue", justify="center" )
            table.add_column(f"[white]Slippage (%)", footer_style="overline white", style="blue", justify="center" )
            # table.add_column(f"[white]Subnet TAO{bittensor.Balance.unit}", footer_style="white", style="blue", justify="center" )
            # table.add_column(f"[white]P({bittensor.Balance.unit},", style="cornflower_blue", justify="right")
            # table.add_column(f"[white]{bittensor.Balance.get_unit(1)})", style="green", justify="left")
            # table.add_column(f"[white]Issuance({bittensor.Balance.get_unit(1)})", style="aquamarine3", justify="center")
            table.add_column(f"[white]Ownership({bittensor.Balance.get_unit(1)})", style="aquamarine3", justify="center")
            table.add_column(f"[white]GDT({bittensor.Balance.unit})", style="aquamarine3", justify="center")
            for substake in substakes:
                netuid = substake['netuid']
                pool = dynamic_info[netuid]
                symbol = f"{bittensor.Balance.get_unit(netuid)}\u200E"
                price = "{:.4f}{}".format( pool.price.__float__(), f"τ/{bittensor.Balance.get_unit(netuid)}\u200E") if pool.is_dynamic else f"{1.0}τ/{symbol}"
                alpha_value = bittensor.Balance.from_rao( int(substake['stake']) ).set_unit(netuid)
                tao_value = pool.alpha_to_tao(alpha_value)
                swapped_tao_value, slippage = pool.alpha_to_tao_with_slippage( substake['stake'] )
                if pool.is_dynamic:
                    slippage_percentage = 100 * float(slippage) / float(slippage + swapped_tao_value) if slippage + swapped_tao_value != 0 else 0
                    slippage_percentage = f"{slippage_percentage:.4f}%"
                else:
                    slippage_percentage = 'N/A'                
                tao_locked = pool.tao_reserve if pool.is_dynamic else subtensor.get_total_subnet_stake(netuid).set_unit(netuid)
                issuance = pool.alpha_outstanding if pool.is_dynamic else tao_locked
                if alpha_value.tao > 0.00009:
                    if issuance.tao != 0:
                        alpha_ownership = "{:.4f}".format((alpha_value.tao / issuance.tao) * 100)
                        tao_ownership = bittensor.Balance.from_tao((alpha_value.tao / issuance.tao) * tao_locked.tao)
                    else:
                        alpha_ownership = "0.0000"
                        tao_ownership = "0.0000"
                    row = [
                        str(netuid), # Number
                        symbol, # Symbol
                        price, # Price
                        f"[dark_sea_green]{ alpha_value }", # Alpha value
                        f"[light_slate_blue]{ tao_value }[/light_slate_blue]", # Tao equiv
                        f"[cadet_blue]{ swapped_tao_value }[/cadet_blue]", # Swap amount.
                        f"[indian_red]{ slippage_percentage }[/indian_red]", # Slippage.
                        # str( bittensor.Balance.from_tao( tao_locked.tao ) ), # Tao on network
                        # "P(" + str( pool.tao_reserve ) + ",", # Pool tao
                        # str( pool.alpha_reserve ) + ")", # Pool alpha
                        # str( pool.alpha_outstanding if pool.is_dynamic else tao_locked ), # Pool alpha Outstanding.
                        f"[light_salmon3]{alpha_ownership}%[/light_salmon3]", # Ownership.
                        f"[medium_purple]{tao_ownership}[/medium_purple]" # Tao ownership.
                    ]
                    table.add_row(*row)
            bittensor.__console__.print(table)

        # Print help table
        htable = Table(show_footer=False, pad_edge=False, box=None, expand=False, title="Help")
        htable.add_column("Column")
        htable.add_column("Details")
        htable.add_row(*[
            f"[white]({bittensor.Balance.unit}/{bittensor.Balance.get_unit(1)})",
            "Subnet token current price"
        ])
        htable.add_row(*[
            f"[white]{bittensor.Balance.get_unit(1)}",
            "Subnet token balance"
        ])
        htable.add_row(*[
            f"[white]{bittensor.Balance.unit}",
            f"Subnet token balance converted to {bittensor.Balance.unit} using current price"
        ])
        htable.add_row(*[
            f"[white]Swap({bittensor.Balance.get_unit(1)}) -> {bittensor.Balance.unit}",
            f"{bittensor.Balance.unit} that will be received if full balance is unstaked for this subnet (with slippage)"
        ])
        htable.add_row(*[
            f"[white]Slippage (%)",
            f"Slippage percentage (if full balance is unstaked)"
        ])
        htable.add_row(*[
            f"[white]Ownership({bittensor.Balance.get_unit(1)})",
            f"Percentage of total Alpha owned in this subnet"
        ])
        htable.add_row(*[
            f"[white]GDT({bittensor.Balance.unit})",
            f"Global Dynamic Tao owned in this subnet"
        ])
        bittensor.__console__.print("")
        bittensor.__console__.print(htable)
        bittensor.__console__.print("")

        # Iterate over each hotkey and make a table
        for hotkey in hotkeys_to_substakes.keys():
            table_substakes( hotkey, hotkeys_to_substakes[hotkey] )
        


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
