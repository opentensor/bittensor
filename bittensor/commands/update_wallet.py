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

import sys
import os
import json
import argparse
import bittensor
from typing import List, Optional
from rich.table import Table
from rich.prompt import Prompt
from rich.prompt import Confirm
from rich.console import Text
from tqdm import tqdm

import os
import bittensor
from typing import List


def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    """Get all coldkey wallet names from path."""
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


console = bittensor.__console__


class UpdateWalletCommand:
    @staticmethod
    def run(cli):
        """Check if any of the wallets needs an update."""
        config = cli.config.copy()
        if config.get("all", d=False) == True:
            wallets = _get_coldkey_wallets_for_path(config.wallet.path)
        else:
            wallets = [bittensor.wallet(config=config)]

        for wallet in wallets:
            print("\n===== ", wallet, " =====")
            wallet.coldkey_file.check_and_update_encryption()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        update_wallet_parser = parser.add_parser(
            "update_wallet", help="""Delegate Stake to an account."""
        )
        update_wallet_parser.add_argument("--all", action="store_true")
        update_wallet_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.wallet.add_args(update_wallet_parser)
        bittensor.subtensor.add_args(update_wallet_parser)

    @staticmethod
    def check_config(config: "bittensor.Config"):
        # Ask the user to specify the wallet if the wallet name is not clear.
        if (
            config.get("all", d=False) == False
            and config.wallet.get("name") == bittensor.defaults.wallet.name
            and not config.no_prompt
        ):
            wallet_name = Prompt.ask(
                "Enter wallet name", default=bittensor.defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)
