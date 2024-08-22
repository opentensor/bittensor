# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

"""Module that encapsulates the CommitWeightCommand and the RevealWeightCommand. Used to commit and reveal weights
for a specific subnet on the Bittensor Network."""

import argparse
import os
import re

import numpy as np
from rich.prompt import Prompt, Confirm

import bittensor
import bittensor.utils.weight_utils as weight_utils
from bittensor.commands import defaults  # type: ignore


class CommitWeightCommand:
    """
    Executes the ``commit`` command to commit weights for specific subnet on the Bittensor network.

    Usage:
        The command allows committing weights for a specific subnet. Users need to specify the netuid (network unique identifier), corresponding UIDs, and weights they wish to commit.

    Optional arguments:
        - ``--netuid`` (int): The netuid of the subnet for which weights are to be commited.
        - ``--uids`` (str): Corresponding UIDs for the specified netuid, in comma-separated format.
        - ``--weights`` (str): Corresponding weights for the specified UIDs, in comma-separated format.

    Example usage:
        $ btcli wt commit --netuid 1 --uids 1,2,3,4 --weights 0.1,0.2,0.3,0.4

    Note:
        This command is used to commit weights for a specific subnet and requires the user to have the necessary permissions.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Commit weights for a specific subnet."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            CommitWeightCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Commit weights for a specific subnet"""
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("uids"):
            cli.config.uids = Prompt.ask("Enter UIDs (comma-separated)")

        if not cli.config.is_set("weights"):
            cli.config.weights = Prompt.ask("Enter weights (comma-separated)")

        # Parse from string
        netuid = cli.config.netuid
        uids = np.array(
            [int(x) for x in re.split(r"[ ,]+", cli.config.uids)], dtype=np.int64
        )
        weights = np.array(
            [float(x) for x in re.split(r"[ ,]+", cli.config.weights)], dtype=np.float32
        )
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        if not cli.config.is_set("salt"):
            # Generate random salt
            salt_length = 8
            salt = list(os.urandom(salt_length))

            if not Confirm.ask(
                f"Have you recorded the [red]salt[/red]: [bold white]'{salt}'[/bold white]? It will be "
                f"required to reveal weights."
            ):
                return False, "User cancelled the operation."
        else:
            salt = np.array(
                [int(x) for x in re.split(r"[ ,]+", cli.config.salt)],
                dtype=np.int64,
            ).tolist()

        # Run the commit weights operation
        success, message = subtensor.commit_weights(
            wallet=wallet,
            netuid=netuid,
            uids=weight_uids,
            weights=weight_vals,
            salt=salt,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            bittensor.__console__.print("Weights committed successfully")
        else:
            bittensor.__console__.print(f"Failed to commit weights: {message}")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "commit", help="""Commit weights for a specific subnet."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--uids", dest="uids", type=str, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)
        parser.add_argument("--salt", dest="salt", type=str, required=False)
        parser.add_argument(
            "--wait-for-inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--wait-for-finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.no_prompt and not config.is_set("wallet.name"):
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.no_prompt and not config.is_set("wallet.hotkey"):
            hotkey = Prompt.ask("Enter [light_salmon3]hotkey[/light_salmon3] name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class RevealWeightCommand:
    """
    Executes the ``reveal`` command to reveal weights for a specific subnet on the Bittensor network.
    Usage:
        The command allows revealing weights for a specific subnet. Users need to specify the netuid (network unique identifier), corresponding UIDs, and weights they wish to reveal.
    Optional arguments:
        - ``--netuid`` (int): The netuid of the subnet for which weights are to be revealed.
        - ``--uids`` (str): Corresponding UIDs for the specified netuid, in comma-separated format.
        - ``--weights`` (str): Corresponding weights for the specified UIDs, in comma-separated format.
        - ``--salt`` (str): Corresponding salt for the hash function, integers in comma-separated format.
    Example usage::
        $ btcli wt reveal --netuid 1 --uids 1,2,3,4 --weights 0.1,0.2,0.3,0.4 --salt 163,241,217,11,161,142,147,189
    Note:
        This command is used to reveal weights for a specific subnet and requires the user to have the necessary permissions.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Reveal weights for a specific subnet."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RevealWeightCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Reveal weights for a specific subnet."""
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("uids"):
            cli.config.uids = Prompt.ask("Enter UIDs (comma-separated)")

        if not cli.config.is_set("weights"):
            cli.config.weights = Prompt.ask("Enter weights (comma-separated)")

        if not cli.config.is_set("salt"):
            cli.config.salt = Prompt.ask("Enter salt (comma-separated)")

        # Parse from string
        netuid = cli.config.netuid
        version = bittensor.__version_as_int__
        uids = np.array(
            [int(x) for x in re.split(r"[ ,]+", cli.config.uids)],
            dtype=np.int64,
        )
        weights = np.array(
            [float(x) for x in re.split(r"[ ,]+", cli.config.weights)],
            dtype=np.float32,
        )
        salt = np.array(
            [int(x) for x in re.split(r"[ ,]+", cli.config.salt)],
            dtype=np.int64,
        )
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids=uids, weights=weights
        )

        # Run the reveal weights operation.
        success, message = subtensor.reveal_weights(
            wallet=wallet,
            netuid=netuid,
            uids=weight_uids,
            weights=weight_vals,
            salt=salt,
            version_key=version,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        if success:
            bittensor.__console__.print("Weights revealed successfully")
        else:
            bittensor.__console__.print(f"Failed to reveal weights: {message}")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "reveal", help="""Reveal weights for a specific subnet."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--uids", dest="uids", type=str, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)
        parser.add_argument("--salt", dest="salt", type=str, required=False)
        parser.add_argument(
            "--wait-for-inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--wait-for-finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter [light_salmon3]hotkey[/light_salmon3] name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
