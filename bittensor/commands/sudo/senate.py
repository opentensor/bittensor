import argparse
import bittensor
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import Optional, Dict
from ..utils import get_delegates_details, DelegatesDetails
from .. import defaults

console = bittensor.__console__


class SenateCommand:
    """
    Executes the ``senate`` command to view the members of Bittensor's governance protocol, known as the Senate.

    This command lists the delegates involved in the decision-making process of the Bittensor network.

    Usage:
        The command retrieves and displays a list of Senate members, showing their names and wallet addresses.
        This information is crucial for understanding who holds governance roles within the network.

    Example usage::

        btcli root senate

    Note:
        This command is particularly useful for users interested in the governance structure and participants of the Bittensor network. It provides transparency into the network's decision-making body.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""View Bittensor's governance protocol proposals"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            SenateCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""View Bittensor's governance protocol proposals"""
        console = bittensor.__console__
        console.print(
            ":satellite: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.subtensor.network
            )
        )

        senate_members = subtensor.get_senate_members()
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        table = Table(show_footer=False)
        table.title = "[white]Senate"
        table.add_column(
            "[overline white]NAME",
            footer_style="overline white",
            style="rgb(50,163,219)",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]ADDRESS",
            footer_style="overline white",
            style="yellow",
            no_wrap=True,
        )
        table.show_footer = True

        for ss58_address in senate_members:
            table.add_row(
                (
                    delegate_info[ss58_address].name
                    if ss58_address in delegate_info
                    else ""
                ),
                ss58_address,
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_parser = parser.add_parser(
            "senate", help="""View senate and it's members"""
        )

        bittensor.wallet.add_args(senate_parser)
        bittensor.subtensor.add_args(senate_parser)
