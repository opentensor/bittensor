# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import argparse
import bittensor
from rich.prompt import Prompt
from . import defaults

console = bittensor.__console__


class TransferCommand:
    """
    Executes the ``transfer`` command to transfer TAO tokens from one account to another on the Bittensor network.

    This command is used for transactions between different accounts, enabling users to send tokens to other participants on the network.

    Usage:
        The command requires specifying the destination address (public key) and the amount of TAO to be transferred.
        It checks for sufficient balance and prompts for confirmation before proceeding with the transaction.

    Optional arguments:
        - ``--dest`` (str): The destination address for the transfer. This can be in the form of an SS58 or ed2519 public key.
        - ``--amount`` (float): The amount of TAO tokens to transfer.

    The command displays the user's current balance before prompting for the amount to transfer, ensuring transparency and accuracy in the transaction.

    Example usage::

        btcli wallet transfer --dest 5Dp8... --amount 100

    Note:
        This command is crucial for executing token transfers within the Bittensor network. Users should verify the destination address and amount before confirming the transaction to avoid errors or loss of funds.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Transfer token of amount to destination."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            TransferCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Transfer token of amount to destination."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor.transfer(
            wallet=wallet,
            dest=cli.config.dest,
            amount=cli.config.amount,
            wait_for_inclusion=True,
            prompt=not cli.config.no_prompt,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Get destination.
        if not config.dest and not config.no_prompt:
            dest = Prompt.ask("Enter destination public key: (ss58 or ed2519)")
            if not bittensor.utils.is_valid_bittensor_address_or_public_key(dest):
                sys.exit()
            else:
                config.dest = str(dest)

        # Get current balance and print to user.
        if not config.no_prompt:
            wallet = bittensor.wallet(config=config)
            subtensor = bittensor.subtensor(config=config, log_verbose=False)
            with bittensor.__console__.status(":satellite: Checking Balance..."):
                account_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
                bittensor.__console__.print(
                    "Balance: [green]{}[/green]".format(account_balance)
                )

        # Get amount.
        if not config.get("amount"):
            if not config.no_prompt:
                amount = Prompt.ask("Enter TAO amount to transfer")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                console.print(
                    ":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(
                        amount
                    )
                )
                sys.exit(1)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        transfer_parser = parser.add_parser(
            "transfer", help="""Transfer Tao between accounts."""
        )
        transfer_parser.add_argument("--dest", dest="dest", type=str, required=False)
        transfer_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )

        bittensor.wallet.add_args(transfer_parser)
        bittensor.subtensor.add_args(transfer_parser)
