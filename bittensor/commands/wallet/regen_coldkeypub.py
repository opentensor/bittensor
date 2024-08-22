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
import sys
from rich.prompt import Prompt, Confirm
from rich.table import Table
from .. import defaults


class RegenColdkeypubCommand:
    """
    Executes the ``regen_coldkeypub`` command to regenerate the public part of a coldkey (coldkeypub) for a wallet on the Bittensor network.

    This command is used when a user needs to recreate their coldkeypub from an existing public key or SS58 address.

    Usage:
        The command requires either a public key in hexadecimal format or an ``SS58`` address to regenerate the coldkeypub. It optionally allows overwriting an existing coldkeypub file.

    Optional arguments:
        - ``--public_key_hex`` (str): The public key in hex format.
        - ``--ss58_address`` (str): The SS58 address of the coldkey.
        - ``--overwrite_coldkeypub`` (bool): Overwrites the existing coldkeypub file with the new one.

    Example usage::

        btcli wallet regen_coldkeypub --ss58_address 5DkQ4...

    Note:
        This command is particularly useful for users who need to regenerate their coldkeypub, perhaps due to file corruption or loss.
        It is a recovery-focused utility that ensures continued access to wallet functionalities.
    """

    def run(cli):
        r"""Creates a new coldkeypub under this wallet."""
        wallet = bittensor.wallet(config=cli.config)
        wallet.regenerate_coldkeypub(
            ss58_address=cli.config.get("ss58_address"),
            public_key=cli.config.get("public_key_hex"),
            overwrite=cli.config.overwrite_coldkeypub,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if config.ss58_address == None and config.public_key_hex == None:
            prompt_answer = Prompt.ask(
                "Enter the ss58_address or the public key in hex"
            )
            if prompt_answer.startswith("0x"):
                config.public_key_hex = prompt_answer
            else:
                config.ss58_address = prompt_answer
        if not bittensor.utils.is_valid_bittensor_address_or_public_key(
            address=(
                config.ss58_address if config.ss58_address else config.public_key_hex
            )
        ):
            sys.exit(1)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        regen_coldkeypub_parser = parser.add_parser(
            "regen_coldkeypub",
            help="""Regenerates a coldkeypub from the public part of the coldkey.""",
        )
        regen_coldkeypub_parser.add_argument(
            "--public_key",
            "--pubkey",
            dest="public_key_hex",
            required=False,
            default=None,
            type=str,
            help="The public key (in hex) of the coldkey to regen e.g. 0x1234 ...",
        )
        regen_coldkeypub_parser.add_argument(
            "--ss58_address",
            "--addr",
            "--ss58",
            dest="ss58_address",
            required=False,
            default=None,
            type=str,
            help="The ss58 address of the coldkey to regen e.g. 5ABCD ...",
        )
        regen_coldkeypub_parser.add_argument(
            "--overwrite_coldkeypub",
            default=False,
            action="store_true",
            help="""Overwrite the old coldkeypub file with the newly generated coldkeypub""",
        )
        bittensor.wallet.add_args(regen_coldkeypub_parser)
        bittensor.subtensor.add_args(regen_coldkeypub_parser)
