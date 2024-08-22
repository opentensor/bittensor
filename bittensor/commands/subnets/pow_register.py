from .. import defaults
from ..utils import check_netuid_set, check_for_cuda_reg_config

import argparse
import bittensor
from rich.prompt import Prompt
import sys


class PowRegisterCommand:
    """
    Executes the ``pow_register`` command to register a neuron on the Bittensor network using Proof of Work (PoW).

    This method is an alternative registration process that leverages computational work for securing a neuron's place on the network.

    Usage:
        The command starts by verifying the existence of the specified subnet. If the subnet does not exist, it terminates with an error message.
        On successful verification, the PoW registration process is initiated, which requires solving computational puzzles.

    Optional arguments:
        - ``--netuid`` (int): The netuid for the subnet on which to serve the neuron. Mandatory for specifying the target subnet.
        - ``--pow_register.num_processes`` (int): The number of processors to use for PoW registration. Defaults to the system's default setting.
        - ``--pow_register.update_interval`` (int): The number of nonces to process before checking for the next block during registration. Affects the frequency of update checks.
        - ``--pow_register.no_output_in_place`` (bool): When set, disables the output of registration statistics in place. Useful for cleaner logs.
        - ``--pow_register.verbose`` (bool): Enables verbose output of registration statistics for detailed information.
        - ``--pow_register.cuda.use_cuda`` (bool): Enables the use of CUDA for GPU-accelerated PoW calculations. Requires a CUDA-compatible GPU.
        - ``--pow_register.cuda.no_cuda`` (bool): Disables the use of CUDA, defaulting to CPU-based calculations.
        - ``--pow_register.cuda.dev_id`` (int): Specifies the CUDA device ID, useful for systems with multiple CUDA-compatible GPUs.
        - ``--pow_register.cuda.tpb`` (int): Sets the number of Threads Per Block for CUDA operations, affecting the GPU calculation dynamics.

    The command also supports additional wallet and subtensor arguments, enabling further customization of the registration process.

    Example usage::

        btcli pow_register --netuid 1 --pow_register.num_processes 4 --cuda.use_cuda

    Note:
        This command is suited for users with adequate computational resources to participate in PoW registration. It requires a sound understanding
        of the network's operations and PoW mechanics. Users should ensure their systems meet the necessary hardware and software requirements,
        particularly when opting for CUDA-based GPU acceleration.

    This command may be disabled according on the subnet owner's directive. For example, on netuid 1 this is permanently disabled.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register neuron."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            PowRegisterCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register neuron."""
        wallet = bittensor.wallet(config=cli.config)

        # Verify subnet exists
        if not subtensor.subnet_exists(netuid=cli.config.netuid):
            bittensor.__console__.print(
                f"[red]Subnet {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        registered = subtensor.register(
            wallet=wallet,
            netuid=cli.config.netuid,
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
        if not registered:
            sys.exit(1)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        register_parser = parser.add_parser(
            "pow_register", help="""Register a wallet to a network using PoW."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )
        register_parser.add_argument(
            "--pow_register.num_processes",
            "-n",
            dest="pow_register.num_processes",
            help="Number of processors to use for POW registration",
            type=int,
            default=defaults.pow_register.num_processes,
        )
        register_parser.add_argument(
            "--pow_register.update_interval",
            "--pow_register.cuda.update_interval",
            "--cuda.update_interval",
            "-u",
            help="The number of nonces to process before checking for next block during registration",
            type=int,
            default=defaults.pow_register.update_interval,
        )
        register_parser.add_argument(
            "--pow_register.no_output_in_place",
            "--no_output_in_place",
            dest="pow_register.output_in_place",
            help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.",
            action="store_false",
            required=False,
            default=defaults.pow_register.output_in_place,
        )
        register_parser.add_argument(
            "--pow_register.verbose",
            help="Whether to ouput the registration statistics verbosely.",
            action="store_true",
            required=False,
            default=defaults.pow_register.verbose,
        )

        ## Registration args for CUDA registration.
        register_parser.add_argument(
            "--pow_register.cuda.use_cuda",
            "--cuda",
            "--cuda.use_cuda",
            dest="pow_register.cuda.use_cuda",
            default=defaults.pow_register.cuda.use_cuda,
            help="""Set flag to use CUDA to register.""",
            action="store_true",
            required=False,
        )
        register_parser.add_argument(
            "--pow_register.cuda.no_cuda",
            "--no_cuda",
            "--cuda.no_cuda",
            dest="pow_register.cuda.use_cuda",
            default=not defaults.pow_register.cuda.use_cuda,
            help="""Set flag to not use CUDA for registration""",
            action="store_false",
            required=False,
        )

        register_parser.add_argument(
            "--pow_register.cuda.dev_id",
            "--cuda.dev_id",
            type=int,
            nargs="+",
            default=defaults.pow_register.cuda.dev_id,
            help="""Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).""",
            required=False,
        )
        register_parser.add_argument(
            "--pow_register.cuda.tpb",
            "--cuda.tpb",
            type=int,
            default=defaults.pow_register.cuda.tpb,
            help="""Set the number of Threads Per Block for CUDA.""",
            required=False,
        )

        bittensor.wallet.add_args(register_parser)
        bittensor.subtensor.add_args(register_parser)

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

        check_netuid_set(
            config, subtensor=bittensor.subtensor(config=config, log_verbose=False)
        )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter [light_salmon3]hotkey[/light_salmon3] name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.no_prompt:
            check_for_cuda_reg_config(config)
