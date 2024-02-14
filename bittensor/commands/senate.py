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
from rich.prompt import Prompt, Confirm
from rich.table import Table
from typing import Optional, Dict
from .utils import get_delegates_details, DelegatesDetails
from . import defaults

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


def format_call_data(call_data: "bittensor.ProposalCallData") -> str:
    human_call_data = list()

    for arg in call_data["call_args"]:
        arg_value = arg["value"]

        # If this argument is a nested call
        func_args = (
            format_call_data(
                {
                    "call_function": arg_value["call_function"],
                    "call_args": arg_value["call_args"],
                }
            )
            if isinstance(arg_value, dict) and "call_function" in arg_value
            else str(arg_value)
        )

        human_call_data.append("{}: {}".format(arg["name"], func_args))

    return "{}({})".format(call_data["call_function"], ", ".join(human_call_data))


def display_votes(
    vote_data: "bittensor.ProposalVoteData", delegate_info: "bittensor.DelegateInfo"
) -> str:
    vote_list = list()

    for address in vote_data["ayes"]:
        vote_list.append(
            "{}: {}".format(
                delegate_info[address].name if address in delegate_info else address,
                "[bold green]Aye[/bold green]",
            )
        )

    for address in vote_data["nays"]:
        vote_list.append(
            "{}: {}".format(
                delegate_info[address].name if address in delegate_info else address,
                "[bold red]Nay[/bold red]",
            )
        )

    return "\n".join(vote_list)


class ProposalsCommand:
    """
    Executes the ``proposals`` command to view active proposals within Bittensor's governance protocol.

    This command displays the details of ongoing proposals, including votes, thresholds, and proposal data.

    Usage:
        The command lists all active proposals, showing their hash, voting threshold, number of ayes and nays, detailed votes by address, end block number, and call data associated with each proposal.

    Example usage::

        btcli root proposals

    Note:
        This command is essential for users who are actively participating in or monitoring the governance of the Bittensor network.
        It provides a detailed view of the proposals being considered, along with the community's response to each.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""View Bittensor's governance protocol proposals"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            ProposalsCommand._run(cli, subtensor)
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
                subtensor.network
            )
        )

        senate_members = subtensor.get_senate_members()
        proposals = subtensor.get_proposals()

        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=bittensor.__delegates_details_url__)

        table = Table(show_footer=False)
        table.title = (
            "[white]Proposals\t\tActive Proposals: {}\t\tSenate Size: {}".format(
                len(proposals), len(senate_members)
            )
        )
        table.add_column(
            "[overline white]HASH",
            footer_style="overline white",
            style="yellow",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]THRESHOLD", footer_style="overline white", style="white"
        )
        table.add_column(
            "[overline white]AYES", footer_style="overline white", style="green"
        )
        table.add_column(
            "[overline white]NAYS", footer_style="overline white", style="red"
        )
        table.add_column(
            "[overline white]VOTES",
            footer_style="overline white",
            style="rgb(50,163,219)",
        )
        table.add_column(
            "[overline white]END", footer_style="overline white", style="blue"
        )
        table.add_column(
            "[overline white]CALLDATA", footer_style="overline white", style="white"
        )
        table.show_footer = True

        for hash in proposals:
            call_data, vote_data = proposals[hash]

            table.add_row(
                hash,
                str(vote_data["threshold"]),
                str(len(vote_data["ayes"])),
                str(len(vote_data["nays"])),
                display_votes(vote_data, registered_delegate_info),
                str(vote_data["end"]),
                format_call_data(call_data),
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
        proposals_parser = parser.add_parser(
            "proposals", help="""View active triumvirate proposals and their status"""
        )

        bittensor.wallet.add_args(proposals_parser)
        bittensor.subtensor.add_args(proposals_parser)


class ShowVotesCommand:
    """
    Executes the ``proposal_votes`` command to view the votes for a specific proposal in Bittensor's governance protocol.

    IMPORTANT
        **THIS COMMAND IS DEPRECATED**. Use ``btcli root proposals`` to see vote status.

    This command provides a detailed breakdown of the votes cast by the senators for a particular proposal.

    Usage:
        Users need to specify the hash of the proposal they are interested in. The command then displays the voting addresses and their respective votes (Aye or Nay) for the specified proposal.

    Optional arguments:
        - ``--proposal`` (str): The hash of the proposal for which votes need to be displayed.

    Example usage::

        btcli root proposal_votes --proposal <proposal_hash>

    Note:
        This command is crucial for users seeking detailed insights into the voting behavior of the Senate on specific governance proposals.
        It helps in understanding the level of consensus or disagreement within the Senate on key decisions.

    **THIS COMMAND IS DEPRECATED**. Use ``btcli root proposals`` to see vote status.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""View Bittensor's governance protocol proposals active votes"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            ShowVotesCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""View Bittensor's governance protocol proposals active votes"""
        console.print(
            ":satellite: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.subtensor.network
            )
        )

        proposal_hash = cli.config.proposal_hash
        if len(proposal_hash) == 0:
            console.print(
                'Aborting: Proposal hash not specified. View all proposals with the "proposals" command.'
            )
            return

        proposal_vote_data = subtensor.get_vote_data(proposal_hash)
        if proposal_vote_data == None:
            console.print(":cross_mark: [red]Failed[/red]: Proposal not found.")
            return

        registered_delegate_info: Optional[
            Dict[str, DelegatesDetails]
        ] = get_delegates_details(url=bittensor.__delegates_details_url__)

        table = Table(show_footer=False)
        table.title = "[white]Votes for Proposal {}".format(proposal_hash)
        table.add_column(
            "[overline white]ADDRESS",
            footer_style="overline white",
            style="yellow",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]VOTE", footer_style="overline white", style="white"
        )
        table.show_footer = True

        votes = display_votes(proposal_vote_data, registered_delegate_info).split("\n")
        for vote in votes:
            split_vote_data = vote.split(": ")  # Nasty, but will work.
            table.add_row(split_vote_data[0], split_vote_data[1])

        table.box = None
        table.pad_edge = False
        table.min_width = 64
        console.print(table)

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        show_votes_parser = parser.add_parser(
            "proposal_votes", help="""View an active proposal's votes by address."""
        )
        show_votes_parser.add_argument(
            "--proposal",
            dest="proposal_hash",
            type=str,
            nargs="?",
            help="""Set the proposal to show votes for.""",
            default="",
        )
        bittensor.wallet.add_args(show_votes_parser)
        bittensor.subtensor.add_args(show_votes_parser)


class SenateRegisterCommand:
    """
    Executes the ``senate_register`` command to register as a member of the Senate in Bittensor's governance protocol.

    This command is used by delegates who wish to participate in the governance and decision-making process of the network.

    Usage:
        The command checks if the user's hotkey is a delegate and not already a Senate member before registering them to the Senate.
        Successful execution allows the user to participate in proposal voting and other governance activities.

    Example usage::

        btcli root senate_register

    Note:
        This command is intended for delegates who are interested in actively participating in the governance of the Bittensor network.
        It is a significant step towards engaging in network decision-making processes.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register to participate in Bittensor's governance protocol proposals"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            SenateRegisterCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register to participate in Bittensor's governance protocol proposals"""
        wallet = bittensor.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        # Check if the hotkey is a delegate.
        if not subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
            console.print(
                "Aborting: Hotkey {} isn't a delegate.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        if subtensor.is_senate_member(hotkey_ss58=wallet.hotkey.ss58_address):
            console.print(
                "Aborting: Hotkey {} is already a senate member.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        subtensor.register_senate(wallet=wallet, prompt=not cli.config.no_prompt)

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_register_parser = parser.add_parser(
            "senate_register",
            help="""Register as a senate member to participate in proposals""",
        )

        bittensor.wallet.add_args(senate_register_parser)
        bittensor.subtensor.add_args(senate_register_parser)


class SenateLeaveCommand:
    """
    Executes the ``senate_leave`` command to discard membership in Bittensor's Senate.

    This command allows a Senate member to voluntarily leave the governance body.

    Usage:
        The command checks if the user's hotkey is currently a Senate member before processing the request to leave the Senate.
        It effectively removes the user from participating in future governance decisions.

    Example usage::

        btcli root senate_leave

    Note:
        This command is relevant for Senate members who wish to step down from their governance responsibilities within the Bittensor network.
        It should be used when a member no longer desires to participate in the Senate activities.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Discard membership in Bittensor's governance protocol proposals"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            SenateLeaveCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.cli"):
        r"""Discard membership in Bittensor's governance protocol proposals"""
        wallet = bittensor.wallet(config=cli.config)

        # Unlock the wallet.
        wallet.hotkey
        wallet.coldkey

        if not subtensor.is_senate_member(hotkey_ss58=wallet.hotkey.ss58_address):
            console.print(
                "Aborting: Hotkey {} isn't a senate member.".format(
                    wallet.hotkey.ss58_address
                )
            )
            return

        subtensor.leave_senate(wallet=wallet, prompt=not cli.config.no_prompt)

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_leave_parser = parser.add_parser(
            "senate_leave",
            help="""Discard senate membership in the governance protocol""",
        )

        bittensor.wallet.add_args(senate_leave_parser)
        bittensor.subtensor.add_args(senate_leave_parser)


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
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        vote_parser = parser.add_parser(
            "senate_vote", help="""Vote on an active proposal by hash."""
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
