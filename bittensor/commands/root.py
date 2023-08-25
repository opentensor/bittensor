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

import sys
import re
import torch
import argparse
import bittensor
from typing import List
from rich.prompt import Prompt, Confirm

from . import defaults

console = bittensor.__console__

class RootRegisterCommand:
    @staticmethod
    def run(cli):
        r"""Register to root network."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)

        subtensor.root_register(
            wallet=wallet, prompt=not cli.config.no_prompt
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "register", help="""Register a wallet to the root network."""
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("subtensor.network") and not config.is_set("subtensor.chain_endpoint") and not config.no_prompt:
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                choices=bittensor.__networks__,
                default=defaults.subtensor.network,
            )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class RootList:
    @staticmethod
    def run(cli):
        r"""List the root network"""
        subtensor = bittensor.subtensor(config=cli.config)
        root_neurons: typing.List[bittensor.NeuronInfoLite] = subtensor.neurons_lite( netuid=0 )
        print (root_neurons)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "list", help="""List the root network"""
        )
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("subtensor.network") and not config.is_set("subtensor.chain_endpoint") and not config.no_prompt:
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                choices=bittensor.__networks__,
                default=defaults.subtensor.network,
            )

class RootSetWeightsCommand:
    @staticmethod
    def run(cli):
        r"""Set weights for root network."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        # Get values if not set.
        if not cli.config.is_set("netuids"):
            example = ", ".join(map(str, [subnet.netuid for subnet in subnets][:3]))  + ' ...'
            cli.config.netuids = Prompt.ask(f"Enter netuids (e.g. {example})")

        if not cli.config.is_set("weights"):
            example = ", ".join(map(str, [ "{:.2f}".format( float(1/len(subnets)) ) for subnet in subnets][:3])) + ' ...'
            cli.config.weights = Prompt.ask(f"Enter weights (e.g. {example})")

        # Parse from string
        netuids = torch.tensor( list(map(int, re.split(r'[ ,]+', cli.config.netuids))), dtype = torch.long )
        weights = torch.tensor( list(map(float, re.split(r'[ ,]+', cli.config.weights))), dtype = torch.float32 )
      
        # Run the set weights operation.
        subtensor.root_set_weights(
            wallet = wallet,
            netuids = netuids,
            weights = weights,
            version_key = 0,
            prompt = not cli.config.no_prompt,
            wait_for_finalization = True,
            wait_for_inclusion = True
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "weights", help="""Set weights for root network."""
        )
        parser.add_argument(
            "--netuids", dest="netuids", type=str, required=False
        )
        parser.add_argument(
            "--weights", dest="weights", type=str, required=False
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("subtensor.network") and not config.is_set("subtensor.chain_endpoint") and not config.no_prompt:
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                choices=bittensor.__networks__,
                default=defaults.subtensor.network,
            )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)