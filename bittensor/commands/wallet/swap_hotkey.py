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
from rich.prompt import Prompt
from copy import deepcopy

from .. import defaults


class SwapHotkeyCommand:
    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Swap your hotkey for all registered axons on the network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SwapHotkeyCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Swap your hotkey for all registered axons on the network."""
        wallet = bittensor.wallet(config=cli.config)

        # This creates an unnecessary amount of extra data, but simplifies implementation.
        new_config = deepcopy(cli.config)
        new_config.wallet.hotkey = new_config.wallet.hotkey_b
        new_wallet = bittensor.wallet(config=new_config)

        subtensor.swap_hotkey(
            wallet=wallet,
            new_wallet=new_wallet,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            prompt=False,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        swap_hotkey_parser = parser.add_parser(
            "swap_hotkey", help="""Swap your associated hotkey."""
        )

        swap_hotkey_parser.add_argument(
            "--wallet.hotkey_b",
            type=str,
            default=defaults.wallet.hotkey,
            help="""Name of the new hotkey""",
            required=False,
        )

        bittensor.wallet.add_args(swap_hotkey_parser)
        bittensor.subtensor.add_args(swap_hotkey_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.is_set("subtensor.network")
            and not config.is_set("subtensor.chain_endpoint")
            and not config.no_prompt
        ):
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                choices=bittensor.__networks__,
                default=defaults.subtensor.network,
            )
            _, endpoint = bittensor.subtensor.determine_chain_endpoint_and_network(
                config.subtensor.network
            )
            config.subtensor.chain_endpoint = endpoint

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter old hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.is_set("wallet.hotkey_b") and not config.no_prompt:
            hotkey = Prompt.ask("Enter new hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey_b = str(hotkey)
