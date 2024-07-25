# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
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

import argparse

from rich.prompt import Prompt

import bittensor
from bittensor.utils.formatting import convert_blocks_to_time
from . import defaults

console = bittensor.__console__


def fetch_arbitration_stats(subtensor, wallet):
    """
    Performs a check of the current arbitration data (if any), and displays it through the bittensor console.
    """
    arbitration_check = len(
        subtensor.check_in_arbitration(wallet.coldkeypub.ss58_address)
    )
    if arbitration_check == 0:
        bittensor.__console__.print(
            "[green]There has been no previous key swap initiated for your coldkey.[/green]"
        )
    if arbitration_check == 1:
        arbitration_remaining = subtensor.get_remaining_arbitration_period(
            wallet.coldkeypub.ss58_address
        )
        hours, minutes, seconds = convert_blocks_to_time(arbitration_remaining)
        bittensor.__console__.print(
            "[yellow]There has been 1 swap request made for this coldkey already."
            " By adding another swap request, the key will enter arbitration."
            f" Your key swap is scheduled for {hours} hours, {minutes} minutes, {seconds} seconds"
            " from now.[/yellow]"
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

        fetch_arbitration_stats(subtensor, wallet)

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
