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

import os
import argparse
import bittensor
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.console import Text
from tqdm import tqdm
from substrateinterface.exceptions import SubstrateRequestException
from .utils import get_delegates_details, DelegatesDetails 
from .identity import SetIdentityCommand
from . import defaults

def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets

console = bittensor.__console__

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

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is not a delegate.
        if not subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            bittensor.__console__.print(
                "Aborting: Hotkey {} is NOT a delegate.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        # Get available netuids
        netuids = subtensor.get_all_subnet_netuids()

        # Prompt user for netuid and take value.
        netuid = config.get("netuid")
        if netuid == None:
            netuid = IntPrompt.ask(f"Enter subnet ID")
        else:
            netuid = int(netuid)
        # Check if netuid exists
        if not netuid in netuids:
            bittensor.__console__.print(
                "ERROR: This netuid ({}) doesn't exist on the network".format(netuid)
            )
            return

        # Prompt user for take value.
        new_take_str = config.get("take")
        if new_take_str == None:
            new_take = FloatPrompt.ask(f"Enter take value (0.18 for 18%)")
        else:
            new_take = float(new_take_str)

        if new_take > 0.18:
            bittensor.__console__.print("ERROR: Take value should not exceed 18%")
            return

        result: bool = subtensor.set_take(
            wallet=wallet,
            delegate_ss58=wallet.hotkey.ss58_address,
            netuid=netuid,
            take=new_take,
        )
        if not result:
            bittensor.__console__.print("Could not set the take")
        else:
            # Check if we are a delegate.
            is_delegate: bool = subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address)
            if not is_delegate:
                bittensor.__console__.print(
                    "Could not set the take [white]{}[/white]".format(subtensor.network)
                )
                return
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
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class SetDelegateTakesCommand:
    """
    Executes the ``set_delegate_takes`` command, which sets the delegate takes for multiple subnets.

    The command performs several checks:

        1. Hotkey is already a delegate
        2. Each netid matches one of the existing subnets
        3. New take values are within the 0-18% range

    Optional Arguments:
        - ``takes``: A list of tuples where each tuple contains a subnet ID and the new take value
        - ``wallet.name``: The name of the wallet to use for the command.
        - ``wallet.hotkey``: The name of the hotkey to use for the command.

    Usage:
        To run the command, the user must have a configured wallet with both hotkey and coldkey. Also, the hotkey should already be a delegate.

    Example usage::
        btcli root set_delegate_takes --wallet.name my_wallet --wallet.hotkey my_hotkey --netuids [1,2] --takes [0.15,0.10)]

    Note:
        This function can be used to update the takes for multiple subnets in a single command.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set takes for multiple subnets."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetDelegateTakesCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set takes for multiple subnets."""
        wallet = bittensor.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is not a delegate.
        if not subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            console.print(
                f"Aborting: Hotkey {wallet.hotkey.ss58_address} is NOT a delegate.",
                style="bold red",
            )
            return

        # Get available netuids
        netuids = subtensor.get_all_subnet_netuids()

        # Get values if not set.
        if not cli.config.is_set("netuids"):
            example = ", ".join(map(str, netuids[:3])) + " ..."
            cli.config.netuids = Prompt.ask(f"Enter netuids (e.g. {example})")

        if not cli.config.is_set("takes"):
            example = (
                ", ".join(
                    map(
                        str,
                        ["{:.2f}".format(0.18) for _ in range(min(len(netuids), 3))],
                    )
                )
                + " ..."
            )
            cli.config.takes = Prompt.ask(f"Enter takes (e.g. {example})")

        # Parse from string
        netuids_input = np.array(
            list(map(int, re.split(r"[ ,]+", cli.config.netuids))), dtype=np.int64
        )
        takes_input = np.array(
            list(map(float, re.split(r"[ ,]+", cli.config.takes))), dtype=np.float32
        )

        # Validate and collect takes
        takes = []
        for netuid, take in zip(netuids_input, takes_input):
            if netuid not in netuids:
                console.print(
                    f"ERROR: This netuid ({netuid}) doesn't exist on the network",
                    style="bold red",
                )
                continue
            if take > 0.18:
                console.print(
                    "ERROR: Take value should be in the range of 0 to 18%",
                    style="bold red",
                )
                continue
            takes.append((netuid, take))

        if not takes:
            console.print(
                "No valid takes were provided or all provided takes were invalid.",
                style="bold red",
            )
            return

        result: bool = subtensor.set_delegates_takes(
            wallet=wallet, takes=takes, delegate_ss58=wallet.hotkey.ss58_address
        )
        if not result:
            console.print("Could not set the takes", style="bold red")
        else:
            console.print(
                f"Successfully set the takes on [white]{subtensor.network}[/white]",
                style="bold green",
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        set_takes_parser = parser.add_parser(
            "set_delegate_takes",
            help="""Set takes for multiple delegates on different subnets""",
        )
        set_takes_parser.add_argument(
            "--netuids",
            dest="netuids",
            type=str,
            required=False,
            help="""The netuids of the subnets to set takes for""",
        )
        set_takes_parser.add_argument(
            "--takes",
            dest="takes",
            type=str,
            required=False,
            help="""The takes values for each subnet (0.18 for 18%)""",
        )
        bittensor.wallet.add_args(set_takes_parser)
        bittensor.subtensor.add_args(set_takes_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
