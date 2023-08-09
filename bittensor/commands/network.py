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

import time
import argparse
import bittensor
from . import defaults
from rich.prompt import Prompt

console = bittensor.__console__

class RegisterSubnetworkCommand:
    @staticmethod
    def run(cli):
        r"""Register a subnetwork"""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        # Call register command.
        subtensor.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "register_subnet",
            help="""Register a new bittensor subnetwork on this chain.""",
        )
        parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)


class GetSubnetBurnCostCommand:
    @staticmethod
    def run(cli):
        r"""Register a subnetwork"""
        config = cli.config.copy()
        subtensor: bittensor.subtensor = bittensor.subtensor( config = config )
        while True:
            try: 
                bittensor.__console__.print(f"Subnet burn cost: [green]{bittensor.utils.balance.Balance( subtensor.get_subnet_burn_cost() )}[/green]")
                time.sleep( bittensor.__blocktime__ )
            except KeyboardInterrupt:
                break

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "get_subnet_burn_cost",
            help="""Return the price to register a subnet""",
        )
        parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.subtensor.add_args(parser)
