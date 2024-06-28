from .. import defaults
from ..wallet.set_identity import SetIdentityCommand
import argparse
import bittensor
from rich.prompt import Prompt

class RegisterSubnetworkCommand:
    """
    Executes the ``register_subnetwork`` command to register a new subnetwork on the Bittensor network.

    This command facilitates the creation and registration of a subnetwork, which involves interaction with the user's wallet and the Bittensor subtensor. It ensures that the user has the necessary credentials and configurations to successfully register a new subnetwork.

    Usage:
        Upon invocation, the command performs several key steps to register a subnetwork:

        1. It copies the user's current configuration settings.
        2. It accesses the user's wallet using the provided configuration.
        3. It initializes the Bittensor subtensor object with the user's configuration.
        4. It then calls the ``register_subnetwork`` function of the subtensor object, passing the user's wallet and a prompt setting based on the user's configuration.

    If the user's configuration does not specify a wallet name and ``no_prompt`` is not set, the command will prompt the user to enter a wallet name. This name is then used in the registration process.

    The command structure includes:

    - Copying the user's configuration.
    - Accessing and preparing the user's wallet.
    - Initializing the Bittensor subtensor.
    - Registering the subnetwork with the necessary credentials.

    Example usage::

        btcli subnets create

    Note:
        This command is intended for advanced users of the Bittensor network who wish to contribute by adding new subnetworks. It requires a clear understanding of the network's functioning and the roles of subnetworks. Users should ensure that they have secured their wallet and are aware of the implications of adding a new subnetwork to the Bittensor ecosystem.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register a subnetwork"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RegisterSubnetworkCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register a subnetwork"""
        wallet = bittensor.wallet(config=cli.config)

        # Call register command.
        success = subtensor.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )
        if success and not cli.config.no_prompt:
            # Prompt for user to set identity.
            do_set_identity = Prompt.ask(
                f"Subnetwork registered successfully. Would you like to set your identity? [y/n]",
                choices=["y", "n"],
            )

            if do_set_identity.lower() == "y":
                subtensor.close()
                config = cli.config.copy()
                SetIdentityCommand.check_config(config)
                cli.config = config
                SetIdentityCommand.run(cli)

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "create",
            help="""Create a new bittensor subnetwork on this chain.""",
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)
