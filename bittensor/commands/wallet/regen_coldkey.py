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
from typing import Optional
import bittensor
import os
from rich.prompt import Prompt, Confirm
from rich.table import Table
from .. import defaults


class RegenColdkeyCommand:
    """
    Executes the ``regen_coldkey`` command to regenerate a coldkey for a wallet on the Bittensor network.

    This command is used to create a new coldkey from an existing mnemonic, seed, or JSON file.

    Usage:
        Users can specify a mnemonic, a seed string, or a JSON file path to regenerate a coldkey.
        The command supports optional password protection for the generated key and can overwrite an existing coldkey.

    Optional arguments:
        - ``--mnemonic`` (str): A mnemonic phrase used to regenerate the key.
        - ``--seed`` (str): A seed hex string used for key regeneration.
        - ``--json`` (str): Path to a JSON file containing an encrypted key backup.
        - ``--json_password`` (str): Password to decrypt the JSON file.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_coldkey`` (bool): Overwrites the existing coldkey with the new one.

    Example usage::

        btcli wallet regen_coldkey --mnemonic "word1 word2 ... word12"

    Note:
        This command is critical for users who need to regenerate their coldkey, possibly for recovery or security reasons.
        It should be used with caution to avoid overwriting existing keys unintentionally.
    """

    def run(cli):
        r"""Creates a new coldkey under this wallet."""
        wallet = bittensor.wallet(config=cli.config)

        json_str: Optional[str] = None
        json_password: Optional[str] = None
        if cli.config.get("json"):
            file_name: str = cli.config.get("json")
            if not os.path.exists(file_name) or not os.path.isfile(file_name):
                raise ValueError("File {} does not exist".format(file_name))
            with open(cli.config.get("json"), "r") as f:
                json_str = f.read()
            # Password can be "", assume if None
            json_password = cli.config.get("json_password", "")
        wallet.regenerate_coldkey(
            mnemonic=cli.config.mnemonic,
            seed=cli.config.seed,
            json=(json_str, json_password),
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_coldkey,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if (
            config.mnemonic == None
            and config.get("seed", d=None) == None
            and config.get("json", d=None) == None
        ):
            prompt_answer = Prompt.ask("Enter mnemonic, seed, or json file location")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            elif len(prompt_answer.split(" ")) > 1:
                config.mnemonic = prompt_answer
            else:
                config.json = prompt_answer

        if config.get("json", d=None) and config.get("json_password", d=None) == None:
            config.json_password = Prompt.ask(
                "Enter json backup password", password=True
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        regen_coldkey_parser = parser.add_parser(
            "regen_coldkey", help="""Regenerates a coldkey from a passed value"""
        )
        regen_coldkey_parser.add_argument(
            "--mnemonic",
            required=False,
            nargs="+",
            help="Mnemonic used to regen your key i.e. horse cart dog ...",
        )
        regen_coldkey_parser.add_argument(
            "--seed",
            required=False,
            default=None,
            help="Seed hex string used to regen your key i.e. 0x1234...",
        )
        regen_coldkey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help="""Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)""",
        )
        regen_coldkey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help="""Password to decrypt the json file.""",
        )
        regen_coldkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated bittensor key with a password.""",
            default=True,
        )
        regen_coldkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated bittensor key with a password.""",
        )
        regen_coldkey_parser.add_argument(
            "--overwrite_coldkey",
            default=False,
            action="store_true",
            help="""Overwrite the old coldkey with the newly generated coldkey""",
        )
        bittensor.wallet.add_args(regen_coldkey_parser)
        bittensor.subtensor.add_args(regen_coldkey_parser)
