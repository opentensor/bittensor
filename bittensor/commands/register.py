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
from rich.prompt import Prompt, Confirm
from .utils import check_netuid_set, check_for_cuda_reg_config

from . import defaults

console = bittensor.__console__


class RegisterCommand:
    @staticmethod
    def run(cli):
        r"""Register neuron."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)

        # Verify subnet exists
        if not subtensor.subnet_exists(netuid=cli.config.netuid):
            bittensor.__console__.print(
                f"[red]Subnet {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        subtensor.register(
            wallet=wallet,
            netuid=cli.config.netuid,
            prompt=not cli.config.no_prompt,
            TPB=cli.config.register.cuda.get("TPB", None),
            update_interval=cli.config.register.get("update_interval", None),
            num_processes=cli.config.register.get("num_processes", None),
            cuda=cli.config.register.cuda.get(
                "use_cuda", defaults.register.cuda.use_cuda
            ),
            dev_id=cli.config.register.cuda.get("dev_id", None),
            output_in_place=cli.config.register.get(
                "output_in_place", defaults.register.output_in_place
            ),
            log_verbose=cli.config.register.get("verbose", defaults.register.verbose),
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        register_parser = parser.add_parser(
            "register", help="""Register a wallet to a network."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )
        register_parser.add_argument(
            "--register.num_processes",
            "-n",
            dest="register.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.register.num_processes,
        )
        register_parser.add_argument(
            "--register.update_interval",
            "--register.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.register.update_interval,
        )
        register_parser.add_argument(
            "--register.no_output_in_place",
            "--no_output_in_place",
            dest="register.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.register.output_in_place,
        )
        register_parser.add_argument(
            "--register.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.register.verbose,
        )

        ## Registration args for CUDA registration.
        register_parser.add_argument(
            "--register.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="register.cuda.use_cuda",
            default=defaults.register.cuda.use_cuda,
            help="""Set flag to use CUDA to register.""",
            action="store_true",
            required=False,
        )
        register_parser.add_argument(
            "--register.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="register.cuda.use_cuda",
            default=not defaults.register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )

        register_parser.add_argument(
            "--register.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        register_parser.add_argument(
            "--register.cuda.TPB",
            "--cuda.TPB",
            type=int,
            default=defaults.register.cuda.TPB,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )

        bittensor.wallet.add_args(register_parser)
        bittensor.subtensor.add_args(register_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        check_netuid_set(config, subtensor=bittensor.subtensor(config=config))

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.no_prompt:
            check_for_cuda_reg_config(config)


class RecycleRegisterCommand:
    @staticmethod
    def run(cli):
        r"""Register neuron by recycling some TAO."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)

        # Verify subnet exists
        if not subtensor.subnet_exists(netuid=cli.config.netuid):
            bittensor.__console__.print(
                f"[red]Subnet {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        # Check current recycle amount
        current_recycle = subtensor.burn(netuid=cli.config.netuid)
        balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)

        # Check balance is sufficient
        if balance < current_recycle:
            bittensor.__console__.print(
                f"[red]Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO[/red]"
            )
            sys.exit(1)

        if not cli.config.no_prompt:
            if (
                Confirm.ask(
                    f"Your balance is: [bold green]{balance}[/bold green]\nThe cost to register by recycle is [bold red]{current_recycle}[/bold red]\nDo you want to continue?",
                    default=False,
                )
                == False
            ):
                sys.exit(1)

        subtensor.burned_register(
            wallet=wallet, netuid=cli.config.netuid, prompt=not cli.config.no_prompt
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        recycle_register_parser = parser.add_parser(
            "recycle_register", help="""Register a wallet to a network."""
        )
        recycle_register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )

        bittensor.wallet.add_args(recycle_register_parser)
        bittensor.subtensor.add_args(recycle_register_parser)

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

        check_netuid_set(config, subtensor=bittensor.subtensor(config=config))

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class RunFaucetCommand:
    @staticmethod
    def run(cli):
        r"""Register neuron."""
        wallet = bittensor.wallet(config=cli.config)
        subtensor = bittensor.subtensor(config=cli.config)
        subtensor.run_faucet(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
            TPB=cli.config.register.cuda.get("TPB", None),
            update_interval=cli.config.register.get("update_interval", None),
            num_processes=cli.config.register.get("num_processes", None),
            cuda=cli.config.register.cuda.get(
                "use_cuda", defaults.register.cuda.use_cuda
            ),
            dev_id=cli.config.register.cuda.get("dev_id", None),
            output_in_place=cli.config.register.get(
                "output_in_place", defaults.register.output_in_place
            ),
            log_verbose=cli.config.register.get("verbose", defaults.register.verbose),
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        run_faucet_parser = parser.add_parser(
            "faucet", help="""Perform PoW to receieve test TAO in your wallet."""
        )
        run_faucet_parser.add_argument(
            "--register.num_processes",
            "-n",
            dest="register.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.register.num_processes,
        )
        run_faucet_parser.add_argument(
            "--register.update_interval",
            "--register.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.register.update_interval,
        )
        run_faucet_parser.add_argument(
            "--register.no_output_in_place",
            "--no_output_in_place",
            dest="register.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.register.output_in_place,
        )
        run_faucet_parser.add_argument(
            "--register.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.register.verbose,
        )

        ## Registration args for CUDA registration.
        run_faucet_parser.add_argument(
            "--register.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="register.cuda.use_cuda",
            default=defaults.register.cuda.use_cuda,
            help="""Set flag to use CUDA to register.""",
            action="store_true",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--register.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="register.cuda.use_cuda",
            default=not defaults.register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--register.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        run_faucet_parser.add_argument(
            "--register.cuda.TPB",
            "--cuda.TPB",
            type=int,
            default=defaults.register.cuda.TPB,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )
        bittensor.wallet.add_args(run_faucet_parser)
        bittensor.subtensor.add_args(run_faucet_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.no_prompt:
            check_for_cuda_reg_config(config)
