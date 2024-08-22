from .. import defaults
from ..wallet.set_identity import SetIdentityCommand
import sys
import argparse
import bittensor
from bittensor.utils import format_error_message
from rich.prompt import Confirm, Prompt
from rich.table import Table

class RegisterSubnetworkCommand:

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Register a subnetwork"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RegisterSubnetworkCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register a subnetwork"""
        
        # Get your wallet.
        wallet = bittensor.wallet(config=cli.config)
        
        # check coldkey
        if not wallet.coldkeypub_file.exists_on_device(): 
            bittensor.__console__.print(f"\n:cross_mark: [red]Failed[/red]: your coldkey: {wallet.name} does not exist on this device. To create it run:\n\n\tbtcli w new_coldkey --wallet.name {wallet.name}\n")
            sys.exit(1)

        # check hotkey
        if not wallet.hotkey_file.exists_on_device(): 
            bittensor.__console__.print(f"\n:cross_mark: [red]Failed[/red]: your hotkey: {wallet.hotkey_str} does not exist on this device. To create it run:\n\n\tbtcli w new_hotkey --wallet.name {wallet.name} --wallet.hotkey {wallet.hotkey_str}\n")
            sys.exit(1)
        
        # Get your balance.
        your_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        
        # Get the burn cost and check your balance.
        burn_cost = bittensor.utils.balance.Balance(subtensor.get_subnet_burn_cost())     
        if burn_cost > your_balance:
            bittensor.__console__.print(
                f"Your balance of: [green]{your_balance}[/green] is not enough to pay the subnet lock cost of: [green]{burn_cost}[/green]"
            )
            return False
        
        # Show creation table.
        console_width = bittensor.__console__.width - 5
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
        table.title = f"[white]Create - {subtensor.network}\n"
        subnets = subtensor.get_all_subnet_dynamic_info()
        netuid = len(subnets)
        unit = bittensor.Balance.get_unit(netuid)
        
        table.add_column("Netuid", style="rgb(253,246,227)", no_wrap=True, justify="center")
        table.add_column("Symbol", style="rgb(211,54,130)", no_wrap=True, justify="center")
        table.add_column(f"Cost ({bittensor.Balance.get_unit(0)})", style="rgb(38,139,210)", no_wrap=True, justify="right")
        table.add_column(f"Recieved({unit})", style="green", no_wrap=True, justify="right")
        table.add_column(f"Locked ({unit})", style="light_goldenrod2", no_wrap=True, justify="center")
        table.add_column("Coldkey", style="bold dark_green", no_wrap=True, justify="center")
        table.add_column("Hotkey", style="light_salmon3", no_wrap=True, justify="center")
        table.add_row(
            str(netuid),
            f"[light_goldenrod1]{unit}[light_goldenrod1]",
            f"τ {burn_cost.tao:.4f}",
            f"{burn_cost.tao:,.4f} {unit}",
            f"{0.0:,.4f} {unit}",
            f"{wallet.coldkeypub.ss58_address}",
            f"{wallet.hotkey.ss58_address}",
        )
        bittensor.__console__.print(table)
        if not Confirm.ask(
            f"Do you want to register subnet: {netuid} for [green]{ burn_cost }[/green]?"
        ):
            return False

        wallet.coldkey  # unlock coldkey

        with bittensor.__console__.status(f":satellite: Registering subnet {netuid}..."):
            with subtensor.substrate as substrate:
                # create extrinsic call
                call = substrate.compose_call(
                    call_module="SubtensorModule",
                    call_function="register_network",
                    call_params={
                        "hotkey": wallet.hotkey.ss58_address,
                        "mechid": 1,
                    },
                )
                extrinsic = substrate.create_signed_extrinsic(
                    call=call, keypair=wallet.coldkey
                )
                response = substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )

                # process if registration successful
                response.process_events()
                if not response.is_success:
                    bittensor.__console__.print(
                        f":cross_mark: [red]Failed[/red]: {format_error_message(response.error_message)}"
                    )
                # Successful registration, final check for membership
                else:
                    bittensor.__console__.print(
                        f":white_heavy_check_mark: [green]Registered subnetwork with netuid: {netuid} [/green]"
                    )
                    return True

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
            
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            wallet_hotkey = Prompt.ask("Enter wallet [light_salmon3]hotkey[/light_salmon3] (to register on the network at creation)", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(wallet_hotkey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "create",
            help="""Create a new bittensor subnetwork on this chain.""",
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)
