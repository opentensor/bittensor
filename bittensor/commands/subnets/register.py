from .. import defaults
from ..utils import check_netuid_set

import argparse
import bittensor
from bittensor.utils import format_error_message
from rich.prompt import Prompt, Confirm
from rich.table import Table
import sys


class RegisterCommand:
    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register neuron by recycling some TAO."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RegisterCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register neuron by recycling some TAO."""

        # Create wallet.
        wallet = bittensor.wallet(config=cli.config)

        # check coldkey
        if not wallet.coldkeypub_file.exists_on_device():
            bittensor.__console__.print(
                f"\n:cross_mark: [red]Failed[/red]: your coldkey: {wallet.name} does not exist on this device. To create it run:\n\n\tbtcli w new_coldkey --wallet.name {wallet.name}\n"
            )
            sys.exit(1)

        # check hotkey
        if not wallet.hotkey_file.exists_on_device():
            bittensor.__console__.print(
                f"\n:cross_mark: [red]Failed[/red]: your hotkey: {wallet.hotkey_str} does not exist on this device. To create it run:\n\n\tbtcli w new_hotkey --wallet.name {wallet.name} --wallet.hotkey {wallet.hotkey_str}\n"
            )
            sys.exit(1)

        # Get netuid
        netuid = cli.config.netuid

        # Check if subnet exists.
        if not subtensor.subnet_exists(netuid):
            bittensor.__console__.print(
                "\n:cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.\n".format(
                    netuid
                )
            )
            return False

        # Check if we are already registered.
        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid
        )
        if not neuron.is_null:
            bittensor.__console__.print(
                f":white_heavy_check_mark: [green]Your hotkey is already registered on subnet: {netuid}[/green]:\n"
                "uid: [bold white]{}[/bold white]\n"
                "netuid: [bold white]{}[/bold white]\n"
                "hotkey: [bold white]{}[/bold white]\n"
                "coldkey: [bold white]{}[/bold white]".format(
                    neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey
                )
            )
            return True

        # Get my current balance.
        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        # Get the recycle amount.
        burn_cost = subtensor.recycle(netuid=netuid)

        # Check enough balance.
        if old_balance < burn_cost:
            bittensor.__console__.print(
                f"[red]Insufficient balance {old_balance} to register neuron. Current burn cost of {burn_cost}[/red]"
            )
            sys.exit(1)

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
        table.title = f"[white]Register - {subtensor.network}\n"
        table.add_column(
            "Netuid", style="rgb(253,246,227)", no_wrap=True, justify="center"
        )
        table.add_column(
            "Symbol", style="rgb(211,54,130)", no_wrap=True, justify="center"
        )
        table.add_column(
            f"Cost ({bittensor.Balance.get_unit(0)})",
            style="rgb(38,139,210)",
            no_wrap=True,
            justify="right",
        )
        table.add_column(
            "Hotkey", style="light_salmon3", no_wrap=True, justify="center"
        )
        table.add_column(
            "Coldkey", style="bold dark_green", no_wrap=True, justify="center"
        )

        table.add_row(
            str(netuid),
            f"[light_goldenrod1]{bittensor.Balance.get_unit(netuid)}[light_goldenrod1]",
            f"τ {burn_cost.tao:.4f}",
            f"{wallet.hotkey.ss58_address}",
            f"{wallet.coldkeypub.ss58_address}",
        )
        bittensor.__console__.print(table)
        if not cli.config.no_prompt:
            if not Confirm.ask(
                f"Do you want to register on subnet: {netuid} for [green]{ burn_cost }[/green]?"
            ):
                sys.exit(1)

        wallet.coldkey  # unlock coldkey
        with bittensor.__console__.status(
            f":satellite: Registering hotkey on [bold]subnet:{netuid}[/bold]..."
        ):
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="burned_register",
                call_params={
                    "netuid": netuid,
                    "hotkey": wallet.hotkey.ss58_address,
                },
            )
            extrinsic = subtensor.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=not cli.config.no_prompt,
                wait_for_finalization=not cli.config.no_prompt,
            )
            if cli.config.no_prompt:
                bittensor.__console__.print(
                    "Extrinsic submitted. Not waiting for inclusion."
                )
                return
            response.process_events()
            if not response.is_success:
                return False, format_error_message(response.error_message)
            else:
                # Get neuron if exists.
                registerd_neuron = subtensor.get_neuron_for_pubkey_and_subnet(
                    wallet.hotkey.ss58_address, netuid=netuid
                )
                if not registerd_neuron.is_null:
                    bittensor.__console__.print(
                        ":white_heavy_check_mark: [green]Registered[/green]:\n"
                        "uid: [bold white]{}[/bold white]\n"
                        "netuid: [bold white]{}[/bold white]\n"
                        "hotkey: [bold white]{}[/bold white]\n"
                        "coldkey: [bold white]{}[/bold white]".format(
                            registerd_neuron.uid,
                            registerd_neuron.netuid,
                            registerd_neuron.hotkey,
                            registerd_neuron.coldkey,
                        )
                    )
                else:
                    # neuron not found, try again
                    bittensor.__console__.print(
                        ":cross_mark: [red]Unknown error. Key did not register.[/red]"
                    )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        register_parser = parser.add_parser(
            "register", help="""Register a wallet to a network."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )
        register_parser.add_argument(
            "--no_prompt",
            "--y",
            "-y",
            dest="no_prompt",
            required=False,
            action="store_true",
            help="""Specify this flag to delegate stake""",
        )
        bittensor.wallet.add_args(register_parser)
        bittensor.subtensor.add_args(register_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        # DEPRECATED
        # if (
        #     not config.is_set("subtensor.network")
        #     and not config.is_set("subtensor.chain_endpoint")
        #     and not config.no_prompt
        # ):
        #     config.subtensor.network = Prompt.ask(
        #         "Enter subtensor network",
        #         choices=bittensor.__networks__,
        #         default=defaults.subtensor.network,
        #     )
        #     _, endpoint = bittensor.subtensor.determine_chain_endpoint_and_network(
        #         config.subtensor.network
        #     )
        #     config.subtensor.chain_endpoint = endpoint

        check_netuid_set(
            config, subtensor=bittensor.subtensor(config=config, log_verbose=False)
        )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter [light_salmon3]hotkey[/light_salmon3] name",
                default=defaults.wallet.hotkey,
            )
            config.wallet.hotkey = str(hotkey)
