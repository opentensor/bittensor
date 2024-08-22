import argparse
import bittensor
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import Dict
from .. import defaults


class VoteCommand:
    """
    Executes the ``senate_vote`` command to cast a vote on an active proposal in Bittensor's governance protocol.

    This command is used by Senate members to vote on various proposals that shape the network's future.

    Usage:
        The user needs to specify the hash of the proposal they want to vote on. The command then allows the Senate member to cast an 'Aye' or 'Nay' vote, contributing to the decision-making process.

    Optional arguments:
        - ``--proposal`` (str): The hash of the proposal to vote on.

    Example usage::

        btcli root senate_vote --proposal <proposal_hash>

    Note:
        This command is crucial for Senate members to exercise their voting rights on key proposals. It plays a vital role in the governance and evolution of the Bittensor network.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Vote in Bittensor's governance protocol proposals"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            VoteCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Vote in Bittensor's governance protocol proposals"""
        wallet = bittensor.wallet(config=cli.config)

        proposal_hash = cli.config.proposal_hash
        if len(proposal_hash) == 0:
            console.print(
                'Aborting: Proposal hash not specified. View all proposals with the "proposals" command.'
            )
            return

        if not subtensor.is_senate_member(hotkey_ss58=wallet.hotkey.ss58_address):
            console.print(
                "Aborting: Hotkey {} isn't a senate member.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        vote_data = subtensor.get_vote_data(proposal_hash)
        if vote_data == None:
            console.print(":cross_mark: [red]Failed[/red]: Proposal not found.")
            return

        vote = Confirm.ask("Desired vote for proposal")
        subtensor.vote_senate(
            wallet=wallet,
            proposal_hash=proposal_hash,
            proposal_idx=vote_data["index"],
            vote=vote,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter [bold dark_green]coldkey[/bold dark_green] name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter [light_salmon3]hotkey[/light_salmon3] name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        vote_parser = parser.add_parser(
            "vote", help="""Vote on an active proposal by hash."""
        )
        vote_parser.add_argument(
            "--proposal",
            dest="proposal_hash",
            type=str,
            nargs="?",
            help="""Set the proposal to show votes for.""",
            default="",
        )
        bittensor.wallet.add_args(vote_parser)
        bittensor.subtensor.add_args(vote_parser)
