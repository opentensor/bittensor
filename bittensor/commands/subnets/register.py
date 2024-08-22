from .. import defaults
from ..utils import check_netuid_set

import argparse
import bittensor
from rich.prompt import Prompt, Confirm
import sys


class RegisterCommand:
    """
    Executes the ``register`` command to register a neuron on the Bittensor network by recycling some TAO (the network's native token).

    This command is used to add a new neuron to a specified subnet within the network, contributing to the decentralization and robustness of Bittensor.

    Usage:
        Before registering, the command checks if the specified subnet exists and whether the user's balance is sufficient to cover the registration cost.

        The registration cost is determined by the current recycle amount for the specified subnet. If the balance is insufficient or the subnet does not exist, the command will exit with an appropriate error message.

        If the preconditions are met, and the user confirms the transaction (if ``no_prompt`` is not set), the command proceeds to register the neuron by recycling the required amount of TAO.

    The command structure includes:

    - Verification of subnet existence.
    - Checking the user's balance against the current recycle amount for the subnet.
    - User confirmation prompt for proceeding with registration.
    - Execution of the registration process.

    Columns Displayed in the confirmation prompt:

    - Balance: The current balance of the user's wallet in TAO.
    - Cost to Register: The required amount of TAO needed to register on the specified subnet.

    Example usage::

        btcli subnets register --netuid 1

    Note:
        This command is critical for users who wish to contribute a new neuron to the network. It requires careful consideration of the subnet selection and an understanding of the registration costs. Users should ensure their wallet is sufficiently funded before attempting to register a neuron.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register neuron by recycling some TAO."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RegisterCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register neuron by recycling some TAO."""
        wallet = bittensor.wallet(config=cli.config)

        # Verify subnet exists
        if not subtensor.subnet_exists(netuid=cli.config.netuid):
            bittensor.__console__.print(
                f"[red]Subnet {cli.config.netuid} does not exist[/red]"
            )
            sys.exit(1)

        # Check current recycle amount
        current_recycle = subtensor.recycle(netuid=cli.config.netuid)
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
        register_parser = parser.add_parser(
            "register", help="""Register a wallet to a network."""
        )
        register_parser.add_argument(
            "--netuid",
            type=int,
            help="netuid for subnet to serve this neuron on",
            default=argparse.SUPPRESS,
        )

        bittensor.wallet.add_args(register_parser)
        bittensor.subtensor.add_args(register_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        # DEPRECATED
        # if (
        #     not config.is_set("subtensor.network")
        #     and not config.is_set("subtensor.chain_endpoint")
        #     and not config.no_prompt
        # ):
        #     config.subtensor.network = Prompt.ask(
        #         "Enter subtensor network",
        #         choices=bittensor.__networks__,
        #         default=defaults.subtensor.network,
        #     )
        #     _, endpoint = bittensor.subtensor.determine_chain_endpoint_and_network(
        #         config.subtensor.network
        #     )
        #     config.subtensor.chain_endpoint = endpoint

        check_netuid_set(
            config, subtensor=bittensor.subtensor(config=config, log_verbose=False)
        )

        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter [light_salmon3]hotkey[/light_salmon3] name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
