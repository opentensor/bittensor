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
        substakes = subtensor.get_stake_info_for_coldkeys(
            coldkey_ss58_list=[cli.config.coldkey_address]
        )[cli.config.coldkey_address]

        # Get registered delegates details.
        registered_delegate_info: Optional[DelegatesDetails] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        # Token pricing info.
        dynamic_info = subtensor.get_all_subnet_dynamic_info()
        emission_drain_tempo = int(subtensor.query_module("SubtensorModule", "HotkeyEmissionTempo").value)
        balance = subtensor.get_balance( cli.config.coldkey_address )
    
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
            total_tao_value = bittensor.Balance(0)
            for substake in substakes:
                netuid = substake.netuid
                pool = dynamic_info[netuid]
                symbol = f"{bittensor.Balance.get_unit(netuid)}\u200E"
                price = "{:.4f}{}".format( pool.price.__float__(), f" τ/{bittensor.Balance.get_unit(netuid)}\u200E") if pool.is_dynamic else f" {1.0000} τ/{symbol} "
                alpha_value = bittensor.Balance.from_rao( int(substake.stake.rao) ).set_unit(netuid)
                locked_value = bittensor.Balance.from_rao( int(substake.locked.rao) ).set_unit(netuid)
                tao_value = pool.alpha_to_tao(alpha_value)
                total_tao_value += tao_value
                swapped_tao_value, slippage = pool.alpha_to_tao_with_slippage( substake.stake )
                if pool.is_dynamic:
                    slippage_percentage = 100 * float(slippage) / float(slippage + swapped_tao_value) if slippage + swapped_tao_value != 0 else 0
                    slippage_percentage = f"[dark_red]{slippage_percentage:.3f}%[/dark_red]"
                else:
                    slippage_percentage = '0.000%'                
                tao_locked = pool.tao_in 
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
                        # f"[medium_purple]{tao_ownership}[/medium_purple] ([light_salmon3]{ alpha_ownership }[/light_salmon3][white]%[/white])", # Tao ownership.
                        f"[medium_purple]{tao_ownership}[/medium_purple]", # Tao ownership.
                        # f"[dark_sea_green]{ alpha_value }", # Alpha value
                        f"{substake.stake.tao:,.4f} {pool.symbol}",
                        f"{pool.price.tao:.4f} τ/{pool.symbol}",
                        f"[light_slate_blue]{ tao_value }[/light_slate_blue]", # Tao equiv
                        f"[cadet_blue]{ swapped_tao_value }[/cadet_blue] ({slippage_percentage})", # Swap amount.
                        # f"[light_salmon3]{ alpha_ownership }%[/light_salmon3]", # Ownership.
                        f"[bold cadet_blue]YES[/bold cadet_blue]" if substake.is_registered else f"[dark_red]NO[/dark_red]", # Registered.
                        str(bittensor.Balance.from_tao(per_block_emission).set_unit(netuid)) if substake.is_registered else "[dark_red]N/A[/dark_red]", # emission per block.
                        f"[light_slate_blue]{ locked_value }[/light_slate_blue]", # Locked value
                    ])
            # table = Table(show_footer=True, pad_edge=False, box=None, expand=False, title=f"{name}")
            table = Table(
                title=f"[white]hotkey:[/white] [light_salmon3]{name}[/light_salmon3]\n",
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
            table.add_column(f"[white]Netuid", footer_style="overline white", style="grey89")
            table.add_column(f"[white]Symbol", footer_style="white", style="light_goldenrod1", justify="right", width=5, no_wrap=True)
            table.add_column(f"[white]TAO({bittensor.Balance.unit})", style="aquamarine3", justify="right", footer=f"{total_global_tao}")
            table.add_column(f"[white]Stake({bittensor.Balance.get_unit(1)})", footer_style="overline white", style="green",  justify="right" )
            table.add_column(f"[white]Rate({bittensor.Balance.unit}/{bittensor.Balance.get_unit(1)})", footer_style="white", style="light_goldenrod2", justify="center" )
            table.add_column(f"[white]Value({bittensor.Balance.get_unit(1)} x {bittensor.Balance.unit}/{bittensor.Balance.get_unit(1)})", footer_style="overline white", style="blue", justify="right", footer=f"{total_tao_value}")
            table.add_column(f"[white]Swap({bittensor.Balance.get_unit(1)}) -> {bittensor.Balance.unit}", footer_style="overline white", style="white", justify="right" )
            # table.add_column(f"[white]Control({bittensor.Balance.get_unit(1)})", style="aquamarine3", justify="right")
            table.add_column(f"[white]Registered", style="red", justify="right")
            table.add_column(f"[white]Emission({bittensor.Balance.get_unit(1)}/block)", style="aquamarine3", justify="right")
            table.add_column(f"[white]Locked({bittensor.Balance.get_unit(1)})", footer_style="overline white", style="green",  justify="right" )
            for row in rows:
                table.add_row(*row)
            bittensor.__console__.print(table)
            return total_global_tao,total_tao_value

        # Iterate over each hotkey and make a table
        all_hotkeys_total_global_tao = bittensor.Balance(0)
        all_hotkeys_total_tao_value = bittensor.Balance(0)
        for hotkey in hotkeys_to_substakes.keys():
            stake, value = table_substakes( hotkey, hotkeys_to_substakes[hotkey] )
            all_hotkeys_total_global_tao += stake
            all_hotkeys_total_tao_value += value

        bittensor.__console__.print("\n\n")
        bittensor.__console__.print(f"Wallet:\n  Coldkey SS58: [bold dark_green]{cli.config.coldkey_address}[/bold dark_green]\n  Free Balance: [aquamarine3]{balance}[/aquamarine3]\n  Total TAO ({bittensor.Balance.unit}): [aquamarine3]{all_hotkeys_total_global_tao}[/aquamarine3]\n  Total Value ({bittensor.Balance.unit}): [aquamarine3]{all_hotkeys_total_tao_value}[/aquamarine3]")
        bittensor.__console__.print(
            """
[bold white]Description[/bold white]:
    Each table displays information about your coldkey's staking accounts with a hotkey. 
    The header of the table displays the hotkey and the footer displays the total stake and total value of all your staking accounts. 
    The columns of the table are as follows:
        - [bold white]Netuid[/bold white]: The unique identifier for the subnet (its index).
        - [bold white]Symbol[/bold white]: The symbol representing the subnet stake's unit.
        - [bold white]TAO[/bold white]: The hotkey's TAO balance on this subnet. This is this hotkey's proportion of total TAO staked into the subnet divided by the hotkey's share of outstanding stake.
        - [bold white]Stake[/bold white]: The hotkey's stake balance in subnets staking unit.
        - [bold white]Rate[/bold white]: The rate of exchange between the subnet's staking unit and the subnet's TAO.
        - [bold white]Value[/bold white]: The price of the hotkey's stake in TAO computed via the exchange rate.
        - [bold white]Swap[/bold white]: The amount of TAO recieved when unstaking all of the hotkey's stake (with slippage).
        - [bold white]Registered[/bold white]: Whether the hotkey is registered on this subnet.
        - [bold white]Emission[/bold white]: If registered, the emission (in stake) attained by this hotkey on this subnet per block.
        - [bold white]Locked[/bold white]: The total amount of stake locked (not able to be unstaked).
"""
)


    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name or [bold green]ss58_address[/bold green]", default=defaults.wallet.name)
            if bittensor.utils.is_valid_ss58_address(wallet_name):
                config.coldkey_address = str(wallet_name)
            else:
                wallet = bittensor.wallet( name = wallet_name, config = config)
                config.coldkey_address = wallet.coldkeypub.ss58_address
        else:
            wallet = bittensor.wallet( name = config.wallet.name, config = config)
            config.coldkey_address = wallet.coldkeypub.ss58_address

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser(
            "list", help="""List all stake accounts for wallet."""
        )
        bittensor.wallet.add_args(list_parser)
        bittensor.subtensor.add_args(list_parser)
