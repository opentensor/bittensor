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
from rich.table import Table
from rich.prompt import Confirm, Prompt


class RemoveStakeCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        unstake_parser = parser.add_parser(
            "remove",
            help="""Remove stake from a specific hotkey on subnet `netuid` from your coldkey.""",
        )
        unstake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        unstake_parser.add_argument("--all", dest="un_stake_all", action="store_true")
        unstake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        unstake_parser.add_argument(
            "--hotkey_ss58",
            required=False,
            type=str,
            help="""Specify the hotkey by name or ss58 address.""",
        )
        unstake_parser.add_argument(
            "--delegate",
            required=False,
            action="store_true",
            help="""Specify this flag to delegate stake""",
        )
        unstake_parser.add_argument(
            "--no_prompt",
            "--y",
            "-y",
            dest="no_prompt",
            required=False,
            action="store_true",
            help="""Specify this flag to delegate stake""",
        )
        bt.wallet.add_args(unstake_parser)
        bt.subtensor.add_args(unstake_parser)

    @staticmethod
    def check_config(config: "bt.config"):
        pass

    @staticmethod
    def run(cli: "bt.cli"):
        # Get config and subtensor connection.
        config = cli.config.copy()
        subtensor = bt.subtensor(config=config, log_verbose=False)

        # Get netuid
        netuids = [config.get("netuid")]
        if config.is_set("netuid"):
            netuid = [config.get("netuid")]
        elif not config.no_prompt:
            netuid_or_all = Prompt.ask(
                'Enter netuid ("[blue]all[/blue]" for all subnets)', default="0"
            )
            if netuid_or_all.lower() == "all":
                netuids = subtensor.get_subnets()
            else:
                netuids = [int(netuid_or_all)]
        else:
            bt.logging.error("netuid is needed to proceed")
            sys.exit(1)

        # Get wallet.
        wallet = bt.wallet(config=config)
        if config.is_set("wallet.name"):
            wallet = bt.wallet(config=config)
        elif not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=bt.defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
            wallet = bt.wallet(config=config)
        else:
            bt.logging.error("--wallet.name is needed to proceed")
            sys.exit(1)

        # check coldkey
        if not wallet.coldkey_file.exists_on_device():
            bt.__console__.print(
                f"\n:cross_mark: [red]Failed[/red]: your coldkey: {wallet.name} does not exist on this device. To create it run:\n\n\tbtcli w new_coldkey --wallet.name {wallet.name}\n"
            )
            sys.exit(1)

        # Get which hotkey we are staking to.
        staking_address_ss58 = config.get("hotkey_ss58")
        staking_address_name = staking_address_ss58

        # If no hotkey is specified, and no prompt is set, delegate to the selected delegate.
        if config.is_set("wallet.hotkey"):
            wallet = bt.wallet(
                name=wallet.name, hotkey=config.wallet.hotkey, path=wallet.path
            )
            staking_address_name = wallet.hotkey_str
            staking_address_ss58 = wallet.hotkey.ss58_address
        elif config.is_set("hotkey_ss58"):
            staking_address_name = config.get("hotkey_ss58")
            staking_address_ss58 = config.get("hotkey_ss58")
        elif not staking_address_ss58 and not config.no_prompt:
            hotkey_str = Prompt.ask(
                "Enter staking [light_salmon3]hotkey[/light_salmon3] name or ss58_address",
                default=bt.defaults.wallet.hotkey,
            )
            if bt.utils.is_valid_ss58_address(hotkey_str):
                staking_address_ss58 = str(hotkey_str)
                staking_address_name = hotkey_str
            else:
                wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_str)
                staking_address_ss58 = wallet.hotkey.ss58_address
                staking_address_name = hotkey_str
        elif not staking_address_ss58:
            bt.logging.error(
                "--hotkey_ss58 must be specified when using --no_prompt or --y"
            )
            sys.exit(1)

        # Check to see if the passed hotkey is valid.
        # if not subtensor.is_hotkey_registered_any(hotkey_ss58=staking_address_ss58):
        #     bt.__console__.print(f"[red]Hotkey [bold]{staking_address_ss58}[/bold] is not registered on any subnets. Aborting.[/red]")
        #     sys.exit(1)

        # Get old staking balance.
        table = Table(
            title=f"[white]Unstake operation to Coldkey SS58: [bold dark_green]{wallet.coldkeypub.ss58_address}[/bold dark_green]\n",
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
        rows = []
        unstake_amount_balance = []
        current_stake_balances = []
        total_received_amount = bt.Balance.from_tao(0)
        current_wallet_balance: bt.Balance = subtensor.get_balance(
            wallet.coldkeypub.ss58_address
        )
        max_float_slippage = 0
        non_zero_netuids = []
        for netuid in netuids:
            # Check that the subnet exists.
            dynamic_info = subtensor.get_subnet_dynamic_info(netuid)
            if dynamic_info is None:
                bt.__console__.print(f"[red]Subnet: {netuid} does not exist.[/red]")
                sys.exit(1)

            current_stake_balance: bt.Balance = (
                subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=staking_address_ss58,
                    netuid=netuid,
                )
            )
            if current_stake_balance.tao == 0:
                continue
            non_zero_netuids.append(netuid)
            current_stake_balances.append(current_stake_balance)

            # Determine the amount we are staking.
            amount_to_unstake_as_balance = None
            if config.get("amount"):
                amount_to_unstake_as_balance = bt.Balance.from_tao(config.amount)
            elif config.get("un_stake_all"):
                amount_to_unstake_as_balance = current_stake_balance
            elif not config.get("amount") and not config.get("max_stake"):
                if Confirm.ask(
                    f"Unstake all: [bold]{current_stake_balance}[/bold] from [bold]{staking_address_name}[/bold] on netuid: {netuid}?"
                ):
                    amount_to_unstake_as_balance = current_stake_balance
                else:
                    try:
                        # TODO add unit.
                        amount = float(
                            Prompt.ask(
                                f"Enter amount to unstake in {bt.Balance.get_unit(netuid)} from subnet: {netuid}"
                            )
                        )
                        amount_to_unstake_as_balance = bt.Balance.from_tao(amount)
                    except ValueError:
                        bt.__console__.print(
                            f":cross_mark:[red]Invalid amount Please use `--amount` with `--no_prompt`.[/red]"
                        )
                        sys.exit(1)
            unstake_amount_balance.append(amount_to_unstake_as_balance)

            # Check enough to stake.
            amount_to_unstake_as_balance.set_unit(netuid)
            if amount_to_unstake_as_balance > current_stake_balance:
                bt.__console__.print(
                    f"[red]Not enough stake to remove[/red]:[bold white]\n stake balance:{current_stake_balance} < unstaking amount: {amount_to_unstake_as_balance}[/bold white]"
                )
                sys.exit(1)

            received_amount, slippage = dynamic_info.alpha_to_tao_with_slippage(
                amount_to_unstake_as_balance
            )
            total_received_amount += received_amount
            if dynamic_info.is_dynamic:
                slippage_pct_float = (
                    100 * float(slippage) / float(slippage + received_amount)
                    if slippage + received_amount != 0
                    else 0
                )
                slippage_pct = f"{slippage_pct_float:.4f} %"
            else:
                slippage_pct_float = 0
                slippage_pct = f"{slippage_pct_float}%"
            max_float_slippage = max(max_float_slippage, slippage_pct_float)

            rows.append(
                (
                    str(netuid),
                    # f"{staking_address_ss58[:3]}...{staking_address_ss58[-3:]}",
                    f"{staking_address_ss58}",
                    str(amount_to_unstake_as_balance),
                    str(float(dynamic_info.price))
                    + f"({bt.Balance.get_unit(0)}/{bt.Balance.get_unit(netuid)})",
                    str(received_amount),
                    str(slippage_pct),
                )
            )

        table.add_column("Netuid", justify="center", style="grey89")
        table.add_column("Hotkey", justify="center", style="light_salmon3")
        table.add_column(
            f"Amount ({bt.Balance.get_unit(1)})",
            justify="center",
            style="dark_sea_green",
        )
        table.add_column(
            f"Rate ({bt.Balance.get_unit(0)}/{bt.Balance.get_unit(1)})",
            justify="center",
            style="light_goldenrod2",
        )
        table.add_column(
            f"Recieved ({bt.Balance.get_unit(0)})",
            justify="center",
            style="light_slate_blue",
            footer=f"{total_received_amount}",
        )
        table.add_column("Slippage", justify="center", style="rgb(220,50,47)")
        for row in rows:
            table.add_row(*row)
        bt.__console__.print(table)
        message = ""
        if max_float_slippage > 5:
            message += f"-------------------------------------------------------------------------------------------------------------------\n"
            message += f"[bold][yellow]WARNING:[/yellow]\tThe slippage on one of your operations is high: [bold red]{max_float_slippage} %[/bold red], this may result in a loss of funds.[/bold] \n"
            message += f"-------------------------------------------------------------------------------------------------------------------\n"
            bt.__console__.print(message)
        if not config.no_prompt:
            if not Confirm.ask("Would you like to continue?"):
                sys.exit(1)
        bt.__console__.print(
            """
[bold white]Description[/bold white]:
    The table displays information about the stake remove operation you are about to perform.
    The columns are as follows:
        - [bold white]Netuid[/bold white]: The netuid of the subnet you are unstaking from.
        - [bold white]Hotkey[/bold white]: The ss58 address of the hotkey you are unstaking from. 
        - [bold white]Amount[/bold white]: The stake amount you are removing from this key.
        - [bold white]Rate[/bold white]: The rate of exchange between TAO and the subnet's stake.
        - [bold white]Received[/bold white]: The amount of free balance TAO you will receive on this subnet after slippage.
        - [bold white]Slippage[/bold white]: The slippage percentage of the unstake operation. (0% if the subnet is not dynamic i.e. root).
"""
        )

        # Perform staking operation.
        wallet.coldkey  # decrypt key.
        with bt.__console__.status(
            f"\n:satellite: Unstaking {amount_to_unstake_as_balance} from {staking_address_name} on netuid: {netuid} ..."
        ):
            for netuid_i, amount, current in list(
                zip(non_zero_netuids, unstake_amount_balance, current_stake_balances)
            ):
                call = subtensor.substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="remove_stake",
                    call_params={
                        "hotkey": staking_address_ss58,
                        "netuid": netuid_i,
                        "amount_unstaked": amount.rao,
                    },
                )
                extrinsic = subtensor.substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = subtensor.substrate.submit_extrinsic(
                    extrinsic, wait_for_inclusion=True, wait_for_finalization=False
                )
                if config.no_prompt:
                    bt.__console__.print(":white_heavy_check_mark: [green]Sent[/green]")
                else:
                    response.process_events()
                    if not response.is_success:
                        bt.__console__.print(
                            f":cross_mark: [red]Failed[/red] with error: {response.error_message}"
                        )
                    else:
                        new_balance = subtensor.get_balance(
                            address=wallet.coldkeypub.ss58_address
                        )
                        new_stake = (
                            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                                coldkey_ss58=wallet.coldkeypub.ss58_address,
                                hotkey_ss58=staking_address_ss58,
                                netuid=netuid_i,
                            ).set_unit(netuid_i)
                        )
                        bt.__console__.print(
                            f"Balance:\n  [blue]{current_wallet_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                        )
                        bt.__console__.print(
                            f"Subnet: {netuid_i} Stake:\n  [blue]{current}[/blue] :arrow_right: [green]{new_stake}[/green]"
                        )
