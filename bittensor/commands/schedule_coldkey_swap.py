import argparse

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
            bittensor.logging.warning(f"failed to call cold_key_swap: {e}")
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
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        bittensor.__console__.print(
            "[yellow]If you call this on the same key multiple times, the key will enter arbitration.[/yellow]"
        )

        ScheduleColdKeySwapCommand.check_arbitration_status(subtensor, wallet)

        # Get the values for the command
        if not cli.config.is_set("new_coldkey"):
            cli.config.new_coldkey = Prompt.ask("Enter new coldkey SS58 address")

        # Validate the new coldkey SS58 address
        if not bittensor.utils.is_valid_ss58_address(cli.config.new_coldkey):
            raise ValueError(
                f":cross_mark:[red] Invalid new coldkey SS58 address[/red] [bold white]{cli.config.new_coldkey}[/bold white]"
            )

        # Prompt for confirmation if no_prompt is not set
        if not cli.config.no_prompt:
            if not Confirm.ask(
                f"Do you want to schedule a coldkey swap to: [bold white]{cli.config.new_coldkey}[/bold white]?"
            ):
                return None

        success, message = subtensor.schedule_coldkey_swap(
            wallet=wallet,
            new_coldkey=cli.config.new_coldkey,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        if success:
            bittensor.__console__.print("Scheduled Cold Key Swap Successfully.")
        else:
            bittensor.__console__.print(f"Failed to Scheduled Cold Key Swap: {message}")

    @staticmethod
    def check_arbitration_status(subtensor, wallet):
        arbitration_check = subtensor.check_in_arbitration(wallet.coldkey.ss58_address)
        if arbitration_check == 0:
            bittensor.__console__.print(
                "[green]Good news. There has been no previous key swap initiated for your coldkey swap.[/green]"
            )
        if arbitration_check == 1:
            bittensor.__console__.print(
                "[yellow]A previous swap request has been made for this key."
                " Proceeding will initiate the arbitration process for your key.[/yellow]"
            )
        if arbitration_check > 1:
            bittensor.__console__.print(
                "[red]This key is currently undergoing arbitration due to multiple swap requests. You can submit an additional swap request,"
                " but be aware it won't cancel the ongoing arbitration process.[/red]"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        """
        Checks and prompts for necessary configuration settings.

        Args:
            config (bittensor.config): The configuration object.

        Prompts the user for wallet name if not set in the config.
        """
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name: str = Prompt.ask(
                "Enter wallet name", default=defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(command_parser: argparse.ArgumentParser):
        """
        Adds arguments to the command parser.

        Args:
            command_parser (argparse.ArgumentParser): The command parser to add arguments to.
        """
        schedule_coldkey_swap_parser = command_parser.add_parser(
            "schedule_coldkey_swap",
            help="""Schedule a swap of the coldkey on the Bittensor network. There is a 72-hour delay on this. 
            If there is another call to schedule_coldkey_swap , this key goes into arbitration to determine 
            on which key the swap will occur. This is a free transaction. Coldkeys require a balance of at least τ0.5 to 
            initiate a coldkey swap.""",
        )
        schedule_coldkey_swap_parser.add_argument(
            "--new_coldkey",
            dest="new_coldkey",
            type=str,
            required=False,  # Make this argument optional
            help="""Specify the new coldkey SS58 address.""",
        )

        schedule_coldkey_swap_parser.add_argument(
            "--wait-for-inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=True,
        )
        schedule_coldkey_swap_parser.add_argument(
            "--wait-for-finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
        )
        schedule_coldkey_swap_parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=True,
        )

        bittensor.wallet.add_args(schedule_coldkey_swap_parser)
        bittensor.subtensor.add_args(schedule_coldkey_swap_parser)


class CheckColdKeySwapCommand:
    """
    Executes the ``check_coldkey_swap`` command to check swap status of a coldkey in the Bittensor network.

    Usage:
        Users need to specify the wallet they want to check the swap status of.

    Example usage::

        btcli wallet check_coldkey_swap

    Note:
        This command is important for users who wish check if swap requests were made against their coldkey.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """
        Runs the check coldkey swap command.

        Args:
            cli (bittensor.cli): The CLI object containing configuration and command-line interface utilities.
        """
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            CheckColdKeySwapCommand._run(cli, subtensor)
        except Exception as e:
            bittensor.logging.warning(f"Failed to get swap status: {e}")
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        """
        Internal method to check coldkey swap status.

        Args:
            cli (bittensor.cli): The CLI object containing configuration and command-line interface utilities.
            subtensor (bittensor.subtensor): The subtensor object for blockchain interactions.
        """
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        CheckColdKeySwapCommand.fetch_arbitration_stats(subtensor, wallet)

    @staticmethod
    def fetch_arbitration_stats(subtensor, wallet):
        arbitration_check = subtensor.check_in_arbitration(wallet.coldkey.ss58_address)
        if arbitration_check == 0:
            bittensor.__console__.print(
                "[green]There has been no previous key swap initiated for your coldkey.[/green]"
            )
        if arbitration_check == 1:
            bittensor.__console__.print(
                "[yellow]There has been 1 swap request made for this coldkey already."
                " By adding another swap request, the key will enter arbitration.[/yellow]"
            )
        if arbitration_check > 1:
            bittensor.__console__.print(
                f"[red]This coldkey is currently in arbitration with a total swaps of {arbitration_check}.[/red]"
            )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        """
        Checks and prompts for necessary configuration settings.

        Args:
            config (bittensor.config): The configuration object.

        Prompts the user for wallet name if not set in the config.
        """
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name: str = Prompt.ask(
                "Enter wallet name", default=defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

    @staticmethod
    def add_args(command_parser: argparse.ArgumentParser):
        """
        Adds arguments to the command parser.

        Args:
            command_parser (argparse.ArgumentParser): The command parser to add arguments to.
        """
        swap_parser = command_parser.add_parser(
            "check_coldkey_swap",
            help="""Check the status of swap requests for a coldkey on the Bittensor network.
            Adding more than one swap request will make the key go into arbitration mode.""",
        )
        bittensor.wallet.add_args(swap_parser)
        bittensor.subtensor.add_args(swap_parser)
