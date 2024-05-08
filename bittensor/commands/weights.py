import re
import torch
import typing
import argparse
import numpy as np
import bittensor
from typing import List, Optional, Dict
from rich.prompt import Prompt
from rich.table import Table
from . import defaults


class CommitWeightCommand:
    """
    Executes the ``commit`` command to commit weights for a specific subnet on the Bittensor network.

    Usage:
        The command allows committing weights for a specific subnet. Users need to specify the netuid (network unique identifier) and corresponding weights they wish to commit.

    Optional arguments:
        - ``--netuid`` (int): The netuid of the subnet for which weights are to be committed.
        - ``--weights`` (str): Corresponding weights for the specified netuid, in comma-separated format.

    Example usage::

        $ btcli wt commit --netuid 1 --weights 0.1,0.2,0.3,0.4

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
        r"""Commit weights for a specific subnet."""
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask(f"Enter netuid"))

        if not cli.config.is_set("weights"):
            cli.config.weights = Prompt.ask(f"Enter weights (comma-separated)")

        # Parse from string
        netuid = cli.config.netuid
        weights = torch.tensor(
            list(map(float, re.split(r"[ ,]+", cli.config.weights))),
            dtype=torch.float32,
        )

        # Run the commit weights operation.
        success, message = subtensor.commit_weights(
            wallet=wallet,
            netuid=netuid,
            weights=weights,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        if success:
            bittensor.__console__.print(f"✅ Weights committed successfully")
        else:
            bittensor.__console__.print(f"❌ Failed to commit weights: {message}")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "commit", help="""Commit weights for a specific subnet."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)
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
            default=True,
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
