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
import argparse
import bittensor
from rich.prompt import Prompt
from ..utils import check_for_cuda_reg_config

from .. import defaults


class RunFaucetCommand:
    """
    Executes the ``faucet`` command to obtain test TAO tokens by performing Proof of Work (PoW).

    IMPORTANT:
        **THIS COMMAND IS CURRENTLY DISABLED.**

    This command is particularly useful for users who need test tokens for operations on the Bittensor testnet.

    Usage:
        The command uses the PoW mechanism to validate the user's effort and rewards them with test TAO tokens. It is typically used in testnet environments where real value transactions are not necessary.

    Optional arguments:
        - ``--faucet.num_processes`` (int): Specifies the number of processors to use for the PoW operation. A higher number of processors may increase the chances of successful computation.
        - ``--faucet.update_interval`` (int): Sets the frequency of nonce processing before checking for the next block, which impacts the PoW operation's responsiveness.
        - ``--faucet.no_output_in_place`` (bool): When set, it disables in-place output of registration statistics for cleaner log visibility.
        - ``--faucet.verbose`` (bool): Enables verbose output for detailed statistical information during the PoW process.
        - ``--faucet.cuda.use_cuda`` (bool): Activates the use of CUDA for GPU acceleration in the PoW process, suitable for CUDA-compatible GPUs.
        - ``--faucet.cuda.no_cuda`` (bool): Disables the use of CUDA, opting for CPU-based calculations.
        - ``--faucet.cuda.dev_id`` (int[]): Allows selection of specific CUDA device IDs for the operation, useful in multi-GPU setups.
        - ``--faucet.cuda.tpb`` (int): Determines the number of Threads Per Block for CUDA operations, affecting GPU calculation efficiency.

    These options provide flexibility in configuring the PoW process according to the user's hardware capabilities and preferences.

    Example usage::

        btcli wallet faucet --faucet.num_processes 4 --faucet.cuda.use_cuda

    Note:
        This command is meant for use in testnet environments where users can experiment with the network without using real TAO tokens.
        It's important for users to have the necessary hardware setup, especially when opting for CUDA-based GPU calculations.

    **THIS COMMAND IS CURRENTLY DISABLED.**
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register neuron."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RunFaucetCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register neuron."""
        wallet = bittensor.wallet(config=cli.config)
        success = subtensor.run_faucet(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
            tpb=cli.config.pow_register.cuda.get("tpb", None),
            update_interval=cli.config.pow_register.get("update_interval", None),
            num_processes=cli.config.pow_register.get("num_processes", None),
            cuda=cli.config.pow_register.cuda.get(
                "use_cuda", defaults.pow_register.cuda.use_cuda
            ),
            dev_id=cli.config.pow_register.cuda.get("dev_id", None),
            output_in_place=cli.config.pow_register.get(
                "output_in_place", defaults.pow_register.output_in_place
            ),
            log_verbose=cli.config.pow_register.get(
                "verbose", defaults.pow_register.verbose
            ),
        )
        if not success:
            bittensor.logging.error("Faucet run failed.")
            sys.exit(1)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        run_faucet_parser = parser.add_parser(
            "faucet", help="""Perform PoW to receieve test TAO in your wallet."""
        )
        run_faucet_parser.add_argument(
            "--faucet.num_processes",
            "-n",
            dest="pow_register.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.pow_register.num_processes,
        )
        run_faucet_parser.add_argument(
            "--faucet.update_interval",
            "--faucet.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.pow_register.update_interval,
        )
        run_faucet_parser.add_argument(
            "--faucet.no_output_in_place",
            "--no_output_in_place",
            dest="pow_register.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.pow_register.output_in_place,
        )
        run_faucet_parser.add_argument(
            "--faucet.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.pow_register.verbose,
        )

        ## Registration args for CUDA registration.
        run_faucet_parser.add_argument(
            "--faucet.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="pow_register.cuda.use_cuda",
            default=defaults.pow_register.cuda.use_cuda,
            help="""Set flag to use CUDA to pow_register.""",
            action="store_true",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="pow_register.cuda.use_cuda",
            default=not defaults.pow_register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.pow_register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--faucet.cuda.tpb",
            "--cuda.tpb",
            type=int,
            default=defaults.pow_register.cuda.tpb,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )
        bittensor.wallet.add_args(run_faucet_parser)
        bittensor.subtensor.add_args(run_faucet_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)
        if not config.no_prompt:
            check_for_cuda_reg_config(config)
