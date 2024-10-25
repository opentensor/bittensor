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
from rich.table import Table
from .. import defaults


class WalletCreateCommand:
    """
    Executes the ``create`` command to generate both a new coldkey and hotkey under a specified wallet on the Bittensor network.

    This command is a comprehensive utility for creating a complete wallet setup with both cold and hotkeys.

    Usage:
        The command facilitates the creation of a new coldkey and hotkey with an optional word count for the mnemonics.
        It supports password protection for the coldkey and allows overwriting of existing keys.

    Optional arguments:
        - ``--n_words`` (int): The number of words in the mnemonic phrase for both keys.
        - ``--use_password`` (bool): Enables password protection for the coldkey.
        - ``--overwrite_coldkey`` (bool): Overwrites the existing coldkey with the new one.
        - ``--overwrite_hotkey`` (bool): Overwrites the existing hotkey with the new one.

    Example usage::

        btcli wallet create --n_words 21

    Note:
        This command is ideal for new users setting up their wallet for the first time or for those who wish to completely renew their wallet keys.
        It ensures a fresh start with new keys for secure and effective participation in the network.
    """

    def run(cli):
        r"""Creates a new coldkey and hotkey under this wallet."""
        wallet = bittensor.wallet(config=cli.config)
        wallet.create_new_coldkey(
            n_words=cli.config.n_words,
            use_password=cli.config.use_password,
            overwrite=cli.config.overwrite_coldkey,
        )
        wallet.create_new_hotkey(
            n_words=cli.config.n_words,
            use_password=False,
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

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_coldkey_parser = parser.add_parser(
            "create",
            help="""Creates a new coldkey (for containing balance) under the specified path. """,
        )
        new_coldkey_parser.add_argument(
            "--n_words",
            type=int,
            choices=[12, 15, 18, 21, 24],
            default=12,
            help="""The number of words representing the mnemonic. i.e. horse cart dog ... x 24""",
        )
        new_coldkey_parser.add_argument(
            "--use_password",
            dest="use_password",
            action="store_true",
            help="""Set true to protect the generated bittensor key with a password.""",
            default=True,
        )
        new_coldkey_parser.add_argument(
            "--no_password",
            dest="use_password",
            action="store_false",
            help="""Set off protects the generated bittensor key with a password.""",
        )
        new_coldkey_parser.add_argument(
            "--overwrite_coldkey",
            action="store_true",
            default=False,
            help="""Overwrite the old coldkey with the newly generated coldkey""",
        )
        new_coldkey_parser.add_argument(
            "--overwrite_hotkey",
            action="store_true",
            default=False,
            help="""Overwrite the old hotkey with the newly generated hotkey""",
        )
        bittensor.wallet.add_args(new_coldkey_parser)
        bittensor.subtensor.add_args(new_coldkey_parser)
