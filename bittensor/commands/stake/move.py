import argparse
import sys

from rich.prompt import Confirm, Prompt
from rich.table import Table

import bittensor as bt
from bittensor.utils import format_error_message


class MoveStakeCommand:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "move",
            help="""Move stake from one hotkey to another on different subnets.""",
        )
        stake_parser.add_argument(
            "--origin_netuid", dest="origin_netuid", type=int, required=False
        )
        stake_parser.add_argument(
            "--destination_netuid", dest="destination_netuid", type=int, required=False
        )
        stake_parser.add_argument(
            "--origin_hotkey",
            required=False,
            type=str,
            help="""Specify the origin hotkey by name or ss58 address.""",
        )
        stake_parser.add_argument(
            "--destination_hotkey",
            required=False,
            type=str,
            help="""Specify the destination hotkey by name or ss58 address.""",
        )
        # stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument(
            "--no_prompt",
            "--y",
            "-y",
            dest="no_prompt",
            required=False,
            action="store_true",
            help="""Specify this flag to move stake without prompt""",
        )
        bt.wallet.add_args(stake_parser)
        bt.subtensor.add_args(stake_parser)

    @staticmethod
    def check_config(config: "bt.config"):
        pass

    @staticmethod
    def run(cli: "bt.cli"):
        # Get config and subtensor connection.
        config = cli.config.copy()
        subtensor = bt.subtensor(config=config, log_verbose=False)

        # Get netuids
        origin_netuid = config.get("origin_netuid")
        destination_netuid = config.get("destination_netuid")
        if not origin_netuid:
            if not config.no_prompt:
                origin_netuid = int(Prompt.ask("Enter origin netuid", default="0"))
            else:
                bt.logging.error("origin_netuid is needed to proceed")
                sys.exit(1)

        if not destination_netuid:
            if not config.no_prompt:
                destination_netuid = int(
                    Prompt.ask("Enter destination netuid", default="0")
                )
            else:
                bt.logging.error("destination_netuid is needed to proceed")
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

        # Get hotkeys.
        origin_hotkey_ss58 = config.get("origin_hotkey")
        if not origin_hotkey_ss58:
            if not config.no_prompt:
                hotkey_str = Prompt.ask(
                    "Enter origin hotkey [bold blue]name[/bold blue] or [bold green]ss58_address[/bold green]",
                    default=bt.defaults.wallet.hotkey,
                )
                if bt.utils.is_valid_ss58_address(hotkey_str):
                    origin_hotkey_ss58 = str(hotkey_str)
                else:
                    wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_str)
                    origin_hotkey_ss58 = wallet.hotkey.ss58_address
            else:
                bt.logging.error("origin_hotkey is needed to proceed")
                sys.exit(1)

        destination_hotkey_ss58 = config.get("destination_hotkey")
        if not destination_hotkey_ss58:
            if not config.no_prompt:
                hotkey_str = Prompt.ask(
                    "Enter destination hotkey [bold blue]name[/bold blue] or [bold green]ss58_address[/bold green]",
                    default=bt.defaults.wallet.hotkey,
                )
                if bt.utils.is_valid_ss58_address(hotkey_str):
                    destination_hotkey_ss58 = str(hotkey_str)
                else:
                    wallet = bt.wallet(name=config.wallet.name, hotkey=hotkey_str)
                    destination_hotkey_ss58 = wallet.hotkey.ss58_address
            else:
                bt.logging.error("destination_hotkey is needed to proceed")
                sys.exit(1)

        # Get the wallet stake balances.
        origin_stake_balance: bt.Balance = (
            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=origin_hotkey_ss58,
                netuid=origin_netuid,
            ).set_unit(origin_netuid)
        )
        amount_to_move_as_balance = origin_stake_balance

        destination_stake_balance: bt.Balance = (
            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=destination_hotkey_ss58,
                netuid=destination_netuid,
            ).set_unit(destination_netuid)
        )

        bt.__console__.print(
            ":warning: [yellow]This command moves all of your stake.[/yellow]"
        )

        # TODO: Re-add In-case amount to stake is handled from subtensor
        # # Determine the amount we are moving.
        # amount_to_move_as_balance = None
        # if config.get("amount"):
        #     amount_to_move_as_balance = bt.Balance.from_tao(config.amount)
        # elif config.get("stake_all"):
        #     amount_to_move_as_balance = origin_stake_balance
        # elif not config.get("amount") and not config.get("max_stake"):
        #     if Confirm.ask(f"Move all: [bold]{origin_stake_balance}[/bold]?"):
        #         amount_to_move_as_balance = origin_stake_balance
        #     else:
        #         try:
        #             amount = float(
        #                 Prompt.ask(
        #                     f"Enter amount to move in {bt.Balance.get_unit(origin_netuid)}"
        #                 )
        #             )
        #             amount_to_move_as_balance = bt.Balance.from_tao(amount)
        #         except ValueError:
        #             bt.__console__.print(
        #                 f":cross_mark:[red]Invalid amount: {amount}[/red]"
        #             )
        #             sys.exit(1)
        # try:
        #     amount_to_move_as_balance = bt.Balance.from_tao(origin_stake_balance)
        #     amount_to_move_as_balance.set_unit(origin_netuid)
        # except ValueError:
        #     bt.__console__.print(f":cross_mark:[red]Invalid amount: {origin_stake_balance}[/red]")
        #     sys.exit(1)
        # # Check enough to move.
        # amount_to_move_as_balance.set_unit(origin_netuid)
        # if amount_to_move_as_balance > origin_stake_balance:
        #     bt.__console__.print(
        #         f"[red]Not enough stake[/red]:[bold white]\n stake balance:{origin_stake_balance} < moving amount: {amount_to_move_as_balance}[/bold white]"
        #     )
        #     sys.exit(1)

        # Slippage warning
        if not config.no_prompt:
            if origin_netuid == destination_netuid:
                received_amount_destination = amount_to_move_as_balance
                slippage_pct_float = 0
                slippage_pct = f"{slippage_pct_float}%"
                price = bt.Balance.from_tao(1).set_unit(origin_netuid)
                price_str = (
                    str(float(price.tao))
                    + f"{bt.Balance.get_unit(origin_netuid)}/{bt.Balance.get_unit(origin_netuid)}"
                )
            else:
                dynamic_origin = subtensor.get_subnet_dynamic_info(origin_netuid)
                dynamic_destination = subtensor.get_subnet_dynamic_info(
                    destination_netuid
                )
                price = (
                    float(dynamic_origin.price) * 1 / float(dynamic_destination.price)
                )
                received_amount_tao, slippage = (
                    dynamic_origin.alpha_to_tao_with_slippage(amount_to_move_as_balance)
                )
                received_amount_destination, slippage = (
                    dynamic_destination.tao_to_alpha_with_slippage(received_amount_tao)
                )
                received_amount_destination.set_unit(destination_netuid)
                slippage_pct_float = (
                    100
                    * float(slippage)
                    / float(slippage + received_amount_destination)
                    if slippage + received_amount_destination != 0
                    else 0
                )
                slippage_pct = f"{slippage_pct_float:.4f} %"
                price_str = (
                    str(float(price))
                    + f"{bt.Balance.get_unit( destination_netuid )}/{bt.Balance.get_unit( origin_netuid )}"
                )

            table = Table(
                title="[white]Move Stake",
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
            table.add_column("origin netuid", justify="center", style="rgb(133,153,0)")
            table.add_column("origin hotkey", justify="center", style="rgb(38,139,210)")
            table.add_column("dest netuid", justify="center", style="rgb(133,153,0)")
            table.add_column("dest hotkey", justify="center", style="rgb(38,139,210)")
            table.add_column(
                f"amount ({bt.Balance.get_unit(origin_netuid)})",
                justify="center",
                style="rgb(38,139,210)",
            )
            table.add_column(
                f"rate ({bt.Balance.get_unit(destination_netuid)}/{bt.Balance.get_unit(origin_netuid)})",
                justify="center",
                style="rgb(42,161,152)",
            )
            table.add_column(
                f"received ({bt.Balance.get_unit(destination_netuid)})",
                justify="center",
                style="rgb(220,50,47)",
            )
            table.add_column("slippage", justify="center", style="rgb(181,137,0)")

            table.add_row(
                bt.Balance.get_unit(origin_netuid) + "(" + str(origin_netuid) + ")",
                f"{origin_hotkey_ss58[:3]}...{origin_hotkey_ss58[-3:]}",
                bt.Balance.get_unit(destination_netuid)
                + "("
                + str(destination_netuid)
                + ")",
                f"{destination_hotkey_ss58[:3]}...{destination_hotkey_ss58[-3:]}",
                str(amount_to_move_as_balance),
                price_str,
                str(received_amount_destination.set_unit(destination_netuid)),
                str(slippage_pct),
            )

            bt.__console__.print(table)
            message = ""
            if slippage_pct_float > 5:
                message += f"\t-------------------------------------------------------------------------------------------------------------------\n"
                message += f"\t[bold][yellow]WARNING:[/yellow]\tSlippage is high: [bold red]{slippage_pct}[/bold red], this may result in a loss of funds.[/bold] \n"
                message += f"\t-------------------------------------------------------------------------------------------------------------------\n"
                bt.__console__.print(message)
            if not Confirm.ask("Would you like to continue?"):
                sys.exit(1)

        # Perform staking operation.
        wallet.coldkey
        with bt.__console__.status(
            f"\n:satellite: Moving {amount_to_move_as_balance} from {origin_hotkey_ss58} on netuid: {origin_netuid} to {destination_hotkey_ss58} on netuid: {destination_netuid} ..."
        ):
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="move_stake",
                call_params={
                    "origin_hotkey": origin_hotkey_ss58,
                    "origin_netuid": origin_netuid,
                    "destination_hotkey": destination_hotkey_ss58,
                    "destination_netuid": destination_netuid,
                    # "amount_moved": amount_to_move_as_balance.rao,
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
                return
            else:
                response.process_events()
                if not response.is_success:
                    bt.__console__.print(
                        f":cross_mark: [red]Failed[/red] with error: {format_error_message(response.error_message)}"
                    )
                    return
                else:
                    new_origin_stake_balance: bt.Balance = (
                        subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                            coldkey_ss58=wallet.coldkeypub.ss58_address,
                            hotkey_ss58=origin_hotkey_ss58,
                            netuid=origin_netuid,
                        ).set_unit(origin_netuid)
                    )
                    new_destination_stake_balance: bt.Balance = (
                        subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                            coldkey_ss58=wallet.coldkeypub.ss58_address,
                            hotkey_ss58=destination_hotkey_ss58,
                            netuid=destination_netuid,
                        ).set_unit(destination_netuid)
                    )
                    bt.__console__.print(
                        f"Origin Stake:\n  [blue]{origin_stake_balance}[/blue] :arrow_right: [green]{new_origin_stake_balance}[/green]"
                    )
                    bt.__console__.print(
                        f"Destination Stake:\n  [blue]{destination_stake_balance}[/blue] :arrow_right: [green]{new_destination_stake_balance}[/green]"
                    )
                    return
