import argparse
from rich.table import Table
from rich.prompt import Prompt

import bittensor


class GetIdentityCommand:
    """
    Executes the :func:`get_identity` command, which retrieves and displays the identity details of a user's coldkey or hotkey associated with the Bittensor network. This function
    queries the subtensor chain for information such as the stake, rank, and trust associated
    with the provided key.

    Optional Arguments:
        - ``key``: The ``ss58`` address of the coldkey or hotkey to query.

    The command performs the following actions:

    - Connects to the subtensor network and retrieves the identity information.
    - Displays the information in a structured table format.

    The displayed table includes:

    - **Address**: The ``ss58`` address of the queried key.
    - **Item**: Various attributes of the identity such as stake, rank, and trust.
    - **Value**: The corresponding values of the attributes.

    Usage:
        The user must provide an ``ss58`` address as input to the command. If the address is not
        provided in the configuration, the user is prompted to enter one.

    Example usage::

        btcli wallet get_identity --key <s58_address>

    Note:
        This function is designed for CLI use and should be executed in a terminal. It is
        primarily used for informational purposes and has no side effects on the network state.
    """

    def run(cli: "bittensor.cli"):
        r"""Queries the subtensor chain for user identity."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            GetIdentityCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        console = bittensor.__console__

        with console.status(":satellite: [bold green]Querying chain identity..."):
            identity = subtensor.query_identity(cli.config.key)

        table = Table(title="[bold white italic]On-Chain Identity")
        table.add_column("Item", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Address", cli.config.key)
        for key, value in identity.items():
            table.add_row(key, str(value) if value is not None else "None")

        console.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("key") and not config.no_prompt:
            config.key = Prompt.ask(
                "Enter coldkey or hotkey ss58 address", default=None
            )
            if config.key is None:
                raise ValueError("key must be set")
        if not config.is_set("subtensor.network") and not config.no_prompt:
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                default=bittensor.defaults.subtensor.network,
                choices=bittensor.__networks__,
            )
            (
                _,
                config.subtensor.chain_endpoint,
            ) = bittensor.subtensor.determine_chain_endpoint_and_network(
                config.subtensor.network
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_coldkey_parser = parser.add_parser(
            "get_identity",
            help="""Creates a new coldkey (for containing balance) under the specified path. """,
        )
        new_coldkey_parser.add_argument(
            "--key",
            type=str,
            default=None,
            help="""The coldkey or hotkey ss58 address to query.""",
        )
        bittensor.wallet.add_args(new_coldkey_parser)
        bittensor.subtensor.add_args(new_coldkey_parser)
