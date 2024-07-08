import argparse
import sys

from rich.prompt import Confirm, Prompt

import bittensor
from . import defaults

console = bittensor.__console__


class ScheduleColdKeySwapCommand:
    """
    Executes the ``schedule_coldkey_swap`` command to schedule a coldkey swap on the Bittensor network.

    This command is used to schedule a swap of the user's coldkey to a new coldkey.

    Usage:
        Users need to specify the new coldkey address. The command checks for the validity of the new coldkey and prompts for confirmation before proceeding with the scheduling process.

    Optional arguments:
        - ``--new_coldkey`` (str): The SS58 address of the new coldkey.

    The command prompts for confirmation before executing the scheduling operation.

    Example usage::

        btcli wallet schedule_coldkey_swap --new_coldkey <new_coldkey_ss58_address>

    Note:
        This command is important for users who wish to change their coldkey on the network.
    """

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        """
        Checks and prompts for necessary configuration settings.

        Args:
            config (bittensor.config): The configuration object.

        Prompts the user for wallet name and new coldkey SS58 address if not set in the config.
        """
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name: str = Prompt.ask(
                "Enter wallet name", default=defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

        if not config.get("new_coldkey") and not config.no_prompt:
            new_coldkey: str = Prompt.ask("Enter new coldkey SS58 address")
            config.new_coldkey = str(new_coldkey)

    @staticmethod
    def add_args(command_parser: argparse.ArgumentParser):
        """
        Adds arguments to the command parser.

        Args:
            command_parser (argparse.ArgumentParser): The command parser to add arguments to.
        """
        swap_parser = command_parser.add_parser(
            "schedule_coldkey_swap",
            help="""Schedule a swap of the coldkey on the Bittensor network. There is a 72-hour delay on this. 
            If there is another call to schedule_coldkey_swap , this key goes into arbitration to determine 
            on which key the swap will occur. This is a free transaction. Coldkeys require a balance of at least τ1 to 
            initiate a coldkey swap.""",
        )
        swap_parser.add_argument(
            "--new_coldkey",
            dest="new_coldkey",
            type=str,
            required=False,  # Make this argument optional
            help="""Specify the new coldkey SS58 address.""",
        )
        bittensor.wallet.add_args(swap_parser)
        bittensor.subtensor.add_args(swap_parser)

    @staticmethod
    def run(cli: "bittensor.cli"):
        """
        Runs the schedule coldkey swap command.

        Args:
            cli (bittensor.cli): The CLI object containing configuration and command-line interface utilities.
        """
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            ScheduleColdKeySwapCommand._run(cli, subtensor)
        except Exception as e:
            print("Oh no! ", e)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        """
        Internal method to execute the coldkey swap scheduling.

        Args:
            cli (bittensor.cli): The CLI object containing configuration and command-line interface utilities.
            subtensor (bittensor.subtensor): The subtensor object for blockchain interactions.
        """
        bittensor.__console__.print(
            ":warning:[yellow]If you call this on the same key multiple times, the key will enter arbitration.[/yellow]"
        )
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)
        
        arbitration_check = subtensor.check_in_arbitration(wallet.coldkey.ss58_address)
        if arbitration_check == 0:
            bittensor.__console__.print(
                "Good news. There has been no previous key swap initiated for your coldkey swap."
            )
        if arbitration_check == 1:
            bittensor.__console__.print(
                ":warning:[yellow]There has been a swap request made for this key previously."
                " By proceeding, you understand this will initiate arbitration for your key.[/yellow]")
        if arbitration_check > 1:
            bittensor.__console__.print(
                ":warning:[yellow]This key is currently in arbitration. You can submit an additional swap request"
                " for you coldkey, but you understand that the key is already in arbitration.[/yellow]"
            )

        new_coldkey_ss58: str = config.get("new_coldkey")
        # Prompt for new_coldkey if not provided
        if not new_coldkey_ss58:
            new_coldkey_ss58 = Prompt.ask("Enter new coldkey SS58 address")
            config.new_coldkey = str(new_coldkey_ss58)
        # Validate the new coldkey SS58 address
        if not bittensor.utils.is_valid_ss58_address(new_coldkey_ss58):
            bittensor.__console__.print(
                f":cross_mark:[red] Invalid new coldkey SS58 address[/red] [bold white]{new_coldkey_ss58}[/bold white]"
            )
            sys.exit()
        # Prompt for confirmation if no_prompt is not set
        if not cli.config.no_prompt:
            if not Confirm.ask(
                f"Do you want to schedule a coldkey swap to: [bold white]{new_coldkey_ss58}[/bold white]?"
            ):
                return None
        # Schedule the coldkey swap
        subtensor.schedule_coldkey_swap(
            wallet=wallet,
            new_coldkey=new_coldkey_ss58,
            wait_for_inclusion=True,
            prompt=not cli.config.no_prompt,
        )
