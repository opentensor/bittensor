import sys
import argparse
import bittensor as bt
from . import select_delegate
from rich.table import Table
from rich.prompt import Confirm, Prompt
from bittensor.utils.slippage import Operation, show_slippage_warning_if_needed


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
        netuid = config.get("netuid")
        if config.is_set("netuid"):
            netuid = config.get("netuid")
        elif not config.no_prompt:
            netuid = int(Prompt.ask("Enter netuid", default="0"))
        else:
            bt.logging.error("netuid is needed to proceed")
            sys.exit(1)

        # Get wallet.
        wallet = bt.wallet(config=config)
        if config.is_set("--wallet.name"):
            wallet = bt.wallet(config=config)
        elif not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter wallet name", default=bt.defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)
            wallet = bt.wallet(config=config)
        else:
            bt.logging.error("--wallet.name is needed to proceed")
            sys.exit(1)

        # Get which hotkey we are staking to.
        staking_address_ss58 = config.get("hotkey_ss58")
        staking_address_name = staking_address_ss58

        # If no hotkey is specified, and no prompt is set, delegate to the selected delegate.
        if not staking_address_ss58 and not config.no_prompt:
            hotkey_str = Prompt.ask(
                "Enter staking hotkey [bold blue]name[/bold blue] or [bold green]ss58_address[/bold green]",
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
        if not subtensor.is_hotkey_registered_any(hotkey_ss58=staking_address_ss58):
            bt.__console__.print(
                f"[red]Hotkey [bold]{staking_address_ss58}[/bold] is not registered on any subnets. Aborting.[/red]"
            )
            sys.exit(1)

        # Get old staking balance.
        current_stake_balance: bt.Balance = (
            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=staking_address_ss58,
                netuid=netuid,
            )
        )
        # Get current wallet balance
        current_wallet_balance: bt.Balance = subtensor.get_balance(
            wallet.coldkeypub.ss58_address
        )

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
                            f"Enter amount to unstake in {bt.Balance.get_unit(netuid)}"
                        )
                    )
                    amount_to_unstake_as_balance = bt.Balance.from_tao(amount)
                except ValueError:
                    bt.__console__.print(
                        f":cross_mark:[red]Invalid amount: {amount}[/red]"
                    )
                    sys.exit(1)

        # Check enough to stake.
        amount_to_unstake_as_balance.set_unit(netuid)
        if amount_to_unstake_as_balance > current_stake_balance:
            bt.__console__.print(
                f"[red]Not enough stake to remove[/red]:[bold white]\n stake balance:{current_stake_balance} < unstaking amount: {amount_to_unstake_as_balance}[/bold white]"
            )
            sys.exit(1)

        # Slippage warning
        if not config.no_prompt:
            dynamic_info = subtensor.get_subnet_dynamic_info(netuid)
            received_amount, slippage = dynamic_info.alpha_to_tao_with_slippage(
                amount_to_unstake_as_balance
            )
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

            table = Table(
                title=f"[white]Remove Stake: {wallet.coldkeypub.ss58_address}",
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
            table.add_column("netuid", justify="center", style="rgb(133,153,0)")
            table.add_column("hotkey", justify="center", style="rgb(42,161,152)")
            table.add_column(
                f"amount ({bt.Balance.get_unit(netuid)})",
                justify="center",
                style="rgb(220,50,47)",
            )
            table.add_column(
                f"rate ({bt.Balance.get_unit(0)}/{bt.Balance.get_unit(netuid)})",
                justify="center",
                style="rgb(42,161,152)",
            )
            table.add_column(
                f"received ({bt.Balance.get_unit(0)})",
                justify="center",
                style="rgb(42,161,152)",
            )
            table.add_column("slippage", justify="center", style="rgb(220,50,47)")
            table.add_row(
                str(netuid),
                f"{staking_address_ss58[:3]}...{staking_address_ss58[-3:]}",
                str(amount_to_unstake_as_balance),
                str(float(dynamic_info.price))
                + f"({bt.Balance.get_unit(0)}/{bt.Balance.get_unit(netuid)})",
                str(received_amount),
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
        wallet.coldkey  # decrypt key.
        with bt.__console__.status(
            f"\n:satellite: Unstaking {amount_to_unstake_as_balance} from {staking_address_name} on netuid: {netuid} ..."
        ):
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="remove_stake",
                call_params={
                    "hotkey": staking_address_ss58,
                    "netuid": netuid,
                    "amount_unstaked": amount_to_unstake_as_balance.rao,
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
                        f":cross_mark: [red]Failed[/red] with error: {response.error_message}"
                    )
                    return
                else:
                    new_balance = subtensor.get_balance(
                        address=wallet.coldkeypub.ss58_address
                    )
                    new_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=staking_address_ss58,
                        netuid=netuid,
                    ).set_unit(netuid)
                    bt.__console__.print(
                        f"Balance:\n  [blue]{current_wallet_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                    )
                    bt.__console__.print(
                        f"Stake:\n  [blue]{current_stake_balance}[/blue] :arrow_right: [green]{new_stake}[/green]"
                    )
                    return
