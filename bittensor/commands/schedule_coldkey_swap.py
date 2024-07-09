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

import argparse

from rich.prompt import Prompt

import bittensor
from bittensor.utils.formatting import convert_blocks_to_time
from .utils import check_for_cuda_config
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
        config = cli.config.copy()
        subtensor: "bittensor.subtensor" = bittensor.subtensor(
            config=config, log_verbose=False
        )
        ScheduleColdKeySwapCommand._run(cli, subtensor)
        try:
            pass
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

        ScheduleColdKeySwapCommand.fetch_arbitration_stats(subtensor, wallet)

        # Get the values for the command
        if not cli.config.is_set("new_coldkey"):
            cli.config.new_coldkey = Prompt.ask("Enter new coldkey SS58 address")

        # Validate the new coldkey SS58 address
        if not bittensor.utils.is_valid_ss58_address(cli.config.new_coldkey):
            raise ValueError(
                f":cross_mark:[red] Invalid new coldkey SS58 address[/red] [bold white]{cli.config.new_coldkey}[/bold white]"
            )

        if not config.no_prompt:
            check_for_cuda_config(config, config.cuda)

        success, message = subtensor.schedule_coldkey_swap(
            wallet=wallet,
            new_coldkey=cli.config.new_coldkey,
            tpb=cli.config.cuda.get("tpb", None),
            update_interval=cli.config.get("update_interval", None),
            num_processes=cli.config.get("num_processes", None),
            cuda=cli.config.cuda.get("use_cuda", defaults.pow_register.cuda.use_cuda),
            dev_id=cli.config.cuda.get("dev_id", None),
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=not cli.config.no_prompt,
        )

        if success:
            bittensor.__console__.print("Scheduled Cold Key Swap Successfully.")
        else:
            bittensor.__console__.print(f"Failed to Scheduled Cold Key Swap: {message}")

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
            required=False,  # Make this argument optional
            default=True,
        )

        ## CUDA acceleration args.
        schedule_coldkey_swap_parser.add_argument(
            "--swap.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="cuda.use_cuda",
            default=defaults.pow_register.cuda.use_cuda,
            help="""Set flag to use CUDA to pow_register.""",
            action="store_true",
            required=False,
        )
        schedule_coldkey_swap_parser.add_argument(
            "--swap.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="cuda.use_cuda",
            default=not defaults.pow_register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )
        schedule_coldkey_swap_parser.add_argument(
            "--swap.cuda.dev_id",
            "--cuda.dev_id",
            dest="cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.pow_register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        schedule_coldkey_swap_parser.add_argument(
            "--swap.cuda.tpb",
            "--cuda.tpb",
            dest="cuda.tpb",
            type=int,
            default=defaults.pow_register.cuda.tpb,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )
        bittensor.wallet.add_args(schedule_coldkey_swap_parser)
        bittensor.subtensor.add_args(schedule_coldkey_swap_parser)

    @staticmethod
    def fetch_arbitration_stats(subtensor, wallet):
        arbitration_check = len(
            subtensor.check_in_arbitration(wallet.coldkey.ss58_address)
        )
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
        arbitration_check = len(
            subtensor.check_in_arbitration(wallet.coldkey.ss58_address)
        )
        if arbitration_check == 0:
            bittensor.__console__.print(
                "[green]There has been no previous key swap initiated for your coldkey.[/green]"
            )
        if arbitration_check == 1:
            bittensor.__console__.print(
                "[yellow]There has been 1 swap request made for this coldkey already."
                " By adding another swap request, the key will enter arbitration."
                f" Your key swap is scheduled for {hours} hours, {minutes} minutes, {seconds} seconds"
                f" from now.[/yellow]"
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
