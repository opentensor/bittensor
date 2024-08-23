
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

import sys
import argparse
import bittensor as bt
from . import select_delegate
from rich.table import Table
from rich.prompt import Confirm, Prompt
from bittensor.utils.slippage import (Operation, show_slippage_warning_if_needed)

class AddStakeCommand:
    """
    Command to add stake to a specific hotkey on a subnet `netuid` from your coldkey.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds arguments to the parser for the add stake command.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        """
        stake_parser = parser.add_parser("add", help="""Add stake to a specific hotkey on subnet `netuid` from your coldkey.""")
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument("--hotkey_ss58", required=False, type=str, help="""Specify the hotkey by name or ss58 address.""")
        stake_parser.add_argument("--delegate", required=False, action='store_true', help="""Specify this flag to delegate stake""" )
        stake_parser.add_argument("--no_prompt", "--y", "-y", dest = 'no_prompt', required=False, action='store_true', help="""Specify this flag to delegate stake""" )
        bt.wallet.add_args(stake_parser)
        bt.subtensor.add_args(stake_parser)
        
    @staticmethod
    def check_config(config: "bt.config"):
        """
        Checks the configuration for the add stake command.

        Args:
            config (bt.config): The configuration to be checked.
        """
        pass
    
    @staticmethod
    def run(cli: "bt.cli"):
        """
        Executes the add stake command.

        Args:
            cli (bt.cli): The command line interface object containing the configuration and other necessary data.
        """
        
        # Get config and subtensor connection.
        config = cli.config.copy()
        subtensor = bt.subtensor(config=config, log_verbose=False)

        # Get netuid
        netuids = [config.get('netuid')]
        if config.is_set("netuid"):
            netuids = [config.get('netuid')]
        elif not config.no_prompt:
            netuid_or_all = Prompt.ask("Enter netuid (\"[blue]all[/blue]\" for all subnets)", default="0")
            if netuid_or_all.lower() == "all":
                netuids = subtensor.get_subnets()
            else:
                netuids = [int(netuid_or_all)]
        else:
            bt.logging.error("netuid is needed to proceed")
            sys.exit(1)
                        
        # Get wallet.
        wallet = bt.wallet( config = config )
        if config.is_set("wallet.name"):
            wallet = bt.wallet( config = config )
        elif not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=bt.defaults.wallet.name)
            config.wallet.name = str(wallet_name)
            wallet = bt.wallet( config = config )
        else:
            bt.logging.error("--wallet.name is needed to proceed")
            sys.exit(1)
            
        # check coldkey
        if not wallet.coldkey_file.exists_on_device(): 
            bt.__console__.print(f"\n:cross_mark: [red]Failed[/red]: your coldkey: {wallet.name} does not exist on this device. To create it run:\n\n\tbtcli w new_coldkey --wallet.name {wallet.name}\n")
            sys.exit(1)

        # Get which hotkey we are staking to.
        staking_address_ss58 = config.get('hotkey_ss58')
        staking_address_name = staking_address_ss58

        # If no hotkey is specified, and no prompt is set, delegate to the selected delegate.
        if config.is_set("wallet.hotkey"):
            wallet = bt.wallet(name=wallet.name, hotkey=config.wallet.hotkey, path=wallet.path)
            staking_address_name = wallet.hotkey_str
            staking_address_ss58 = wallet.hotkey.ss58_address
        elif config.is_set("hotkey_ss58"):
            staking_address_name = config.get('hotkey_ss58')
        elif not staking_address_ss58 and not config.no_prompt:
            hotkey_str = Prompt.ask("Enter staking [light_salmon3]hotkey[/light_salmon3] name or ss58_address", default=bt.defaults.wallet.hotkey)
            if bt.utils.is_valid_ss58_address(hotkey_str):
                staking_address_ss58 = str(hotkey_str)
                staking_address_name = hotkey_str
            else:
                wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_str, path=config.wallet.path)
                staking_address_ss58 = wallet.hotkey.ss58_address
                staking_address_name = hotkey_str
        elif not staking_address_ss58:
            bt.logging.error("--hotkey_ss58 must be specified when using --no_prompt or --y")
            sys.exit(1)

        # Init the table.
        table = Table(
            title=f"[white]Staking operation from Coldkey SS58[/white]: [bold dark_green]{wallet.coldkeypub.ss58_address}[/bold dark_green]\n",
            width=bt.__console__.width - 5,
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
         
        # Determine the amount we are staking.
        rows = []
        stake_amount_balance = []
        current_stake_balances = []
        current_wallet_balance: bt.Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address).set_unit(0)
        remaining_wallet_balance = current_wallet_balance
        max_slippage = 0
        for netuid in netuids:
            
            # Check that the subnet exists.
            dynamic_info = subtensor.get_subnet_dynamic_info( netuid )
            if not dynamic_info:
                bt.logging.error(f"Subnet with netuid: {netuid} does not exist.")
                continue
            
            # Get old staking balance.
            current_stake_balance: bt.Balance = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=staking_address_ss58,
                netuid=netuid,
            ).set_unit(netuid)
            current_stake_balances.append( current_stake_balance )

            # Get the amount.
            amount_to_stake_as_balance = None
            if config.get("amount"):
                amount_to_stake_as_balance = bt.Balance.from_tao(config.amount)
            elif config.get("stake_all"):
                amount_to_stake_as_balance = current_wallet_balance/len(netuids)
            elif not config.get("amount") and not config.get("max_stake"):
                if Confirm.ask(f"Stake all: [bold]{remaining_wallet_balance}[/bold]?"):
                    amount_to_stake_as_balance = remaining_wallet_balance
                else:
                    try:
                        amount = float(Prompt.ask(f"Enter amount to stake in {bt.Balance.get_unit(0)} to subnet: {netuid}"))
                        amount_to_stake_as_balance = bt.Balance.from_tao(amount)
                    except ValueError:
                        bt.__console__.print(f":cross_mark:[red]Invalid amount: {amount}[/red]")
                        sys.exit(1)
            stake_amount_balance.append( amount_to_stake_as_balance )

            # Check enough to stake.
            amount_to_stake_as_balance.set_unit(0)
            if amount_to_stake_as_balance > remaining_wallet_balance:
                bt.__console__.print(f"[red]Not enough stake[/red]:[bold white]\n wallet balance:{remaining_wallet_balance} < staking amount: {amount_to_stake_as_balance}[/bold white]")
                sys.exit(1)
            remaining_wallet_balance -= amount_to_stake_as_balance

            # Slippage warning
            received_amount, slippage = dynamic_info.tao_to_alpha_with_slippage( amount_to_stake_as_balance )
            if dynamic_info.is_dynamic:
                slippage_pct_float = 100 * float(slippage) / float(slippage + received_amount) if slippage + received_amount != 0 else 0
                slippage_pct = f"{slippage_pct_float:.4f} %"
            else:
                slippage_pct_float = 0
                slippage_pct = 'N/A'
            max_slippage = max(slippage_pct_float, max_slippage)
            rows.append(
                (
                    str(netuid),
                    # f"{staking_address_ss58[:3]}...{staking_address_ss58[-3:]}",
                    f"{staking_address_ss58}",
                    str(amount_to_stake_as_balance),
                    str(1/float(dynamic_info.price)) + f" {bt.Balance.get_unit(netuid)}/{bt.Balance.get_unit(0)} ",
                    str(received_amount.set_unit(netuid)),
                    str(slippage_pct),
                )
            )
           
        table.add_column("Netuid", justify="center", style="grey89")
        table.add_column("Hotkey", justify="center", style="light_salmon3")
        table.add_column(f"Amount ({bt.Balance.get_unit(0)})", justify="center", style="dark_sea_green")
        table.add_column(f"Rate ({bt.Balance.get_unit(netuid)}/{bt.Balance.get_unit(0)})", justify="center", style="light_goldenrod2")
        table.add_column(f"Recieved ({bt.Balance.get_unit(netuid)})", justify="center", style="light_slate_blue")
        table.add_column("Slippage", justify="center", style="rgb(220,50,47)")
        for row in rows:
            table.add_row(*row)
        bt.__console__.print(table)
        message = ""
        if max_slippage > 5:
            message += f"-------------------------------------------------------------------------------------------------------------------\n"
            message += f"[bold][yellow]WARNING:[/yellow]\tThe slippage on one of your operations is high: [bold red]{max_slippage} %[/bold red], this may result in a loss of funds.[/bold] \n"
            message += f"-------------------------------------------------------------------------------------------------------------------\n"
            bt.__console__.print(message)
        bt.__console__.print(
            """
[bold white]Description[/bold white]:
    The table displays information about the stake operation you are about to perform.
    The columns are as follows:
        - [bold white]Netuid[/bold white]: The netuid of the subnet you are staking to.
        - [bold white]Hotkey[/bold white]: The ss58 address of the hotkey you are staking to. 
        - [bold white]Amount[/bold white]: The TAO you are staking into this subnet onto this hotkey.
        - [bold white]Rate[/bold white]: The rate of exchange between your TAO and the subnet's stake.
        - [bold white]Received[/bold white]: The amount of stake you will receive on this subnet after slippage.
        - [bold white]Slippage[/bold white]: The slippage percentage of the stake operation. (0% if the subnet is not dynamic i.e. root).
""")
        if not config.no_prompt:
            if not Confirm.ask("Would you like to continue?"):
                sys.exit(1)
        
        # Perform staking operation.
        wallet.coldkey
        with bt.__console__.status(f"\n:satellite: Staking {amount_to_stake_as_balance} to {staking_address_name} on netuid: {netuid} ..."):
            for netuid_i, amount, current in list(zip(netuids, stake_amount_balance, current_stake_balances)):
                call = subtensor.substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="add_stake",
                    call_params={
                        "hotkey": staking_address_ss58,
                        "netuid": netuid_i,
                        "amount_staked": amount.rao,
                    },
                )
                extrinsic = subtensor.substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
                response = subtensor.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True, wait_for_finalization=False)
                if config.no_prompt:
                    bt.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                else:
                    response.process_events()
                    if not response.is_success:
                        bt.__console__.print(f":cross_mark: [red]Failed[/red] with error: {response.error_message}")
                    else:
                        new_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)
                        new_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                            coldkey_ss58=wallet.coldkeypub.ss58_address,
                            hotkey_ss58=staking_address_ss58,
                            netuid=netuid_i,
                        ).set_unit(netuid_i)
                        bt.__console__.print(f"Balance:\n  [blue]{current_wallet_balance}[/blue] :arrow_right: [green]{new_balance}[/green]")
                        bt.__console__.print(f"Subnet: {netuid_i} Stake:\n  [blue]{current}[/blue] :arrow_right: [green]{new_stake}[/green]")
                        
