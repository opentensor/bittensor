# The MIT License (MIT)
# Copyright © 2023 OpenTensor Foundation

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
import bittensor
import re
import numpy as np
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.console import Text
from substrateinterface.exceptions import SubstrateRequestException
from .. import defaults


class SetTakeCommand:
    """
    Executes the ``set_take`` command, which sets the delegate take for a specified subnet.

    The command performs several checks:

        1. Hotkey is already a delegate
        2. netid matches one of the existing subnets
        3. New take value is within 0-18% range

    Optional Arguments:
        - ``netuid``: The ID of subnet to update the take for
        - ``take``: The new take value
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``wallet.hotkey``: The name of the hotkey to use for the command.

    Usage:
        To run the command, the user must have a configured wallet with both hotkey and coldkey. Also, the hotkey should already be a delegate.

    Example usage::
        btcli root set_take --wallet.name my_wallet --wallet.hotkey my_hotkey

    Note:
        This function can be used to update the takes individually for every subnet
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set take for a subnet."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetTakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set take for a subnet."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)

        # Get available netuids
        netuids = subtensor.get_all_subnet_netuids()

        # Prompt user for netuid and take value.
        netuid = config.get("netuid")
        if netuid == None:
            netuid = IntPrompt.ask(
                f"Enter subnet ID ({netuids[0]}..{netuids[len(netuids)-1]})"
            )
        else:
            netuid = int(netuid)
        # Check if netuid exists
        if not netuid in netuids:
            bittensor.__console__.print(
                "ERROR: This netuid ({}) doesn't exist on the network".format(netuid)
            )
            return

        # Print current take
        current_take = subtensor.get_delegate_take(wallet.hotkey.ss58_address, netuid)
        bittensor.__console__.print(f"Current take: {current_take * 100.:.2f} %")

        # Prompt user for take value.
        new_take_str = config.get("take")
        if new_take_str == None:
            new_take = FloatPrompt.ask(f"Enter new take value (0.18 for 18%)")
        else:
            new_take = float(new_take_str)

        if new_take > 0.18:
            bittensor.__console__.print("ERROR: Take value should not exceed 18%")
            return

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        result: bool = subtensor.set_take(
            wallet=wallet,
            delegate_ss58=wallet.hotkey.ss58_address,
            netuid=netuid,
            take=new_take,
        )
        if not result:
            bittensor.__console__.print("Could not set the take")
        else:
            bittensor.__console__.print(
                "Successfully set the take on [white]{}[/white]".format(
                    subtensor.network
                )
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_take_parser = parser.add_parser(
            "set_take", help="""Set take for delegate on a subnet"""
        )
        set_take_parser.add_argument(
            "--netuid",
            dest="netuid",
            type=int,
            required=False,
            help="""Id of subnet to set take for""",
        )
        set_take_parser.add_argument(
            "--take",
            dest="take",
            type=float,
            required=False,
            help="""Take as a float number""",
        )
        bittensor.wallet.add_args(set_take_parser)
        bittensor.subtensor.add_args(set_take_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
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
