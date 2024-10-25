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
import bittensor
import os
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import List
from .. import defaults


def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    """Get all coldkey wallet names from path."""
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


class UpdateWalletCommand:
    """
    Executes the ``update`` command to check and potentially update the security of the wallets in the Bittensor network.

    This command is used to enhance wallet security using modern encryption standards.

    Usage:
        The command checks if any of the wallets need an update in their security protocols.
        It supports updating all legacy wallets or a specific one based on the user's choice.

    Optional arguments:
        - ``--all`` (bool): When set, updates all legacy wallets.
        - ``--no_prompt`` (bool): Disables user prompting during the update process.

    Example usage::

        btcli wallet update --all

    Note:
        This command is important for maintaining the highest security standards for users' wallets.
        It is recommended to run this command periodically to ensure wallets are up-to-date with the latest security practices.
    """

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
            "update",
            help="""Updates the wallet security using NaCL instead of ansible vault.""",
        )
        update_wallet_parser.add_argument("--all", action="store_true")
        bittensor.wallet.add_args(update_wallet_parser)
        bittensor.subtensor.add_args(update_wallet_parser)

    @staticmethod
    def check_config(config: "bittensor.Config"):
        if config.get("all", d=False) == False:
            if not config.no_prompt:
                if Confirm.ask("Do you want to update all legacy wallets?"):
                    config["all"] = True

        # Ask the user to specify the wallet if the wallet name is not clear.
        if (
            config.get("all", d=False) == False
            and config.wallet.get("name") == bittensor.defaults.wallet.name
            and not config.no_prompt
        ):
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=bittensor.defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
