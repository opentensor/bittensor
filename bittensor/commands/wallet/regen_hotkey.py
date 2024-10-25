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
from typing import Optional
from .. import defaults


class RegenHotkeyCommand:
    """
    Executes the ``regen_hotkey`` command to regenerate a hotkey for a wallet on the Bittensor network.

    Similar to regenerating a coldkey, this command creates a new hotkey from a mnemonic, seed, or JSON file.

    Usage:
        Users can provide a mnemonic, seed string, or a JSON file to regenerate the hotkey.
        The command supports optional password protection and can overwrite an existing hotkey.

    Optional arguments:
        - ``--mnemonic`` (str): A mnemonic phrase used to regenerate the key.
        - ``--seed`` (str): A seed hex string used for key regeneration.
        - ``--json`` (str): Path to a JSON file containing an encrypted key backup.
        - ``--json_password`` (str): Password to decrypt the JSON file.
        - ``--use_password`` (bool): Enables password protection for the generated key.
        - ``--overwrite_hotkey`` (bool): Overwrites the existing hotkey with the new one.

    Example usage::

        btcli wallet regen_hotkey
        btcli wallet regen_hotkey --seed 0x1234...

    Note:
        This command is essential for users who need to regenerate their hotkey, possibly for security upgrades or key recovery.
        It should be used cautiously to avoid accidental overwrites of existing keys.
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

        wallet.regenerate_hotkey(
            mnemonic=cli.config.mnemonic,
            seed=cli.config.seed,
            json=(json_str, json_password),
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_hotkey,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter [light_salmon3]hotkey[/light_salmon3] name",
                default=defaults.wallet.hotkey,
            )
            config.wallet.hotkey = str(hotkey)
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
        regen_hotkey_parser = parser.add_parser(
            "regen_hotkey", help="""Regenerates a hotkey from a passed mnemonic"""
        )
        regen_hotkey_parser.add_argument(
            "--mnemonic",
            required=False,
            nargs="+",
            help="Mnemonic used to regen your key i.e. horse cart dog ...",
        )
        regen_hotkey_parser.add_argument(
            "--seed",
            required=False,
            default=None,
            help="Seed hex string used to regen your key i.e. 0x1234...",
        )
        regen_hotkey_parser.add_argument(
            "--json",
            required=False,
            default=None,
            help="""Path to a json file containing the encrypted key backup. (e.g. from PolkadotJS)""",
        )
        regen_hotkey_parser.add_argument(
            "--json_password",
            required=False,
            default=None,
            help="""Password to decrypt the json file.""",
        )
        regen_hotkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated bittensor key with a password.""",
            default=False,
        )
        regen_hotkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated bittensor key with a password.""",
        )
        regen_hotkey_parser.add_argument(
            "--overwrite_hotkey",
            dest="overwrite_hotkey",
            action="store_true",
            default=False,
            help="""Overwrite the old hotkey with the newly generated hotkey""",
        )
        bittensor.wallet.add_args(regen_hotkey_parser)
        bittensor.subtensor.add_args(regen_hotkey_parser)
