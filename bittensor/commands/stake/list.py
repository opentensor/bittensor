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
from rich.console import Console

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
        substakes = subtensor.get_stake_info_for_coldkeys(
            coldkey_ss58_list=[wallet.coldkeypub.ss58_address]
        )[wallet.coldkeypub.ss58_address]
        netuids: typing.List[int] = subtensor.get_subnets()

        # Get registered delegates details.
        registered_delegate_info: Optional[DelegatesDetails] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        # Token pricing info.
        dynamic_info = subtensor.get_all_subnet_dynamic_info()
        emission_drain_tempo = int(subtensor.query_module("SubtensorModule", "HotkeyEmissionTempo").value)
        balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
    
        # Iterate over substakes and aggregate them by hotkey.
        hotkeys_to_substakes: typing.Dict[str, typing.List[typing.Dict]] = {}
        for substake in substakes:
            hotkey = substake.hotkey_ss58
            if substake.stake.rao == 0: continue
            if hotkey not in hotkeys_to_substakes:
                hotkeys_to_substakes[hotkey] = []
            hotkeys_to_substakes[hotkey].append( substake )
            
        def table_substakes( hotkey:str, substakes: typing.List[typing.Dict] ):
            # Create table structure.
            name = registered_delegate_info[hotkey].name + f" ({hotkey})" if hotkey in registered_delegate_info else hotkey
            rows = []
            total_global_tao = bittensor.Balance(0)
            for substake in substakes:
                netuid = substake.netuid
                pool = dynamic_info[netuid]
                symbol = f"{bittensor.Balance.get_unit(netuid)}\u200E"
                price = "{:.4f}{}".format( pool.price.__float__(), f"τ/{bittensor.Balance.get_unit(netuid)}\u200E") if pool.is_dynamic else f"{1.0}τ/{symbol}"
                alpha_value = bittensor.Balance.from_rao( int(substake.stake.rao) ).set_unit(netuid)
                locked_value = bittensor.Balance.from_rao( int(substake.locked.rao) ).set_unit(netuid)
                tao_value = pool.alpha_to_tao(alpha_value)
                swapped_tao_value, slippage = pool.alpha_to_tao_with_slippage( substake.stake )
                if pool.is_dynamic:
                    slippage_percentage = 100 * float(slippage) / float(slippage + swapped_tao_value) if slippage + swapped_tao_value != 0 else 0
                    slippage_percentage = f"{slippage_percentage:.4f}%"
                else:
                    slippage_percentage = 'N/A'                
                tao_locked = pool.tao_in if pool.is_dynamic else subtensor.get_total_subnet_stake(netuid).set_unit(netuid)
                issuance = pool.alpha_out if pool.is_dynamic else tao_locked
                per_block_emission = substake.emission.tao / ( ( emission_drain_tempo / pool.tempo) * pool.tempo )
                if alpha_value.tao > 0.00009:
                    if issuance.tao != 0:
                        alpha_ownership = "{:.4f}".format((alpha_value.tao / issuance.tao) * 100)
                        tao_ownership = bittensor.Balance.from_tao((alpha_value.tao / issuance.tao) * tao_locked.tao)
                        total_global_tao += tao_ownership
                    else:
                        alpha_ownership = "0.0000"
                        tao_ownership = "0.0000"
                    rows.append([
                        str(netuid), # Number
                        symbol, # Symbol
                        f"[medium_purple]{tao_ownership}[/medium_purple]", # Tao ownership.
                        price, # Price
                        f"[dark_sea_green]{ alpha_value }", # Alpha value
                        f"[light_slate_blue]{ locked_value }[/light_slate_blue]", # Locked value
                        f"[light_slate_blue]{ tao_value }[/light_slate_blue]", # Tao equiv
                        f"[cadet_blue]{ swapped_tao_value }[/cadet_blue]", # Swap amount.
                        f"[light_salmon3]{alpha_ownership}%[/light_salmon3]", # Ownership.
                        str(bittensor.Balance.from_tao(per_block_emission).set_unit(netuid)), # emission per block.
                    ])
            # table = Table(show_footer=True, pad_edge=False, box=None, expand=False, title=f"{name}")
            table = Table(
                title=f"[white]hotkey: {name}[/white]",
                width=bittensor.__console__.width - 5,
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
            table.add_column(f"[white]", footer_style="overline white", style="grey89")
            table.add_column(f"[white]", footer_style="white", style="light_goldenrod1", justify="center", width=5, no_wrap=True)
            table.add_column(f"[white]Global({bittensor.Balance.unit})", style="aquamarine3", justify="center", footer=f"{total_global_tao}")
            table.add_column(f"[white]({bittensor.Balance.unit}/{bittensor.Balance.get_unit(1)})", footer_style="white", style="light_goldenrod2", justify="center" )
            table.add_column(f"[white]Local({bittensor.Balance.get_unit(1)})", footer_style="overline white", style="green",  justify="center" )
            table.add_column(f"[white]Locked({bittensor.Balance.get_unit(1)})", footer_style="overline white", style="green",  justify="center" )
            table.add_column(f"[white]Value({bittensor.Balance.unit})", footer_style="overline white", style="blue", justify="center" )
            table.add_column(f"[white]Swaped({bittensor.Balance.get_unit(1)}) -> {bittensor.Balance.unit}", footer_style="overline white", style="blue", justify="center" )
            table.add_column(f"[white]Control({bittensor.Balance.get_unit(1)})", style="aquamarine3", justify="center")
            table.add_column(f"[white]Emission({bittensor.Balance.get_unit(1)})", style="aquamarine3", justify="center")
            for row in rows:
                table.add_row(*row)
            bittensor.__console__.print(table)

        # Iterate over each hotkey and make a table
        for hotkey in hotkeys_to_substakes.keys():
            table_substakes( hotkey, hotkeys_to_substakes[hotkey] )
            
        bittensor.__console__.print("\n\n")
        bittensor.__console__.print(f"Wallet: {wallet.name} [{wallet.coldkeypub.ss58_address}], Balance: {balance}")
        bittensor.__console__.print("\n\n")


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
