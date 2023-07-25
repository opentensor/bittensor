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
from rich.table import Table
from typing import List, Union, Optional, Dict, Tuple
from .utils import get_delegates_details, DelegatesDetails

console = bittensor.__console__


class SenateCommand:
    @staticmethod
    def run(cli):
        r"""View Bittensor's governance protocol proposals"""
        config = cli.config.copy()
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

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
                delegate_info[ss58_address].name
                if ss58_address in delegate_info
                else "",
                ss58_address,
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_parser = parser.add_parser(
            "senate", help="""View senate and it's members"""
        )
        senate_parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        senate_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
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
    @staticmethod
    def run(cli):
        r"""View Bittensor's governance protocol proposals"""
        config = cli.config.copy()
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

        console.print(
            ":satellite: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.subtensor.network
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
    def check_config(cls, config: "bittensor.Config"):
        None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        proposals_parser = parser.add_parser(
            "proposals", help="""View active triumvirate proposals and their status"""
        )
        proposals_parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        proposals_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.wallet.add_args(proposals_parser)
        bittensor.subtensor.add_args(proposals_parser)


class ShowVotesCommand:
    @staticmethod
    def run(cli):
        r"""View Bittensor's governance protocol proposals active votes"""
        config = cli.config.copy()
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

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
    def check_config(cls, config: "bittensor.Config"):
        if config.proposal_hash == "" and not config.no_prompt:
            proposal_hash = Prompt.ask("Enter proposal hash")
            config.proposal_hash = str(proposal_hash)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        show_votes_parser = parser.add_parser(
            "proposal_votes", help="""View an active proposal's votes by address."""
        )
        show_votes_parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        show_votes_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
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
    @staticmethod
    def run(cli):
        r"""Register to participate in Bittensor's governance protocol proposals"""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

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
    def check_config(cls, config: "bittensor.Config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter wallet name", default=bittensor.defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter hotkey name", default=bittensor.defaults.wallet.hotkey
            )
            config.wallet.hotkey = str(hotkey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_register_parser = parser.add_parser(
            "senate_register",
            help="""Register as a senate member to participate in proposals""",
        )
        senate_register_parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        senate_register_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.wallet.add_args(senate_register_parser)
        bittensor.subtensor.add_args(senate_register_parser)


class SenateLeaveCommand:
    @staticmethod
    def run(cli):
        r"""Discard membership in Bittensor's governance protocol proposals"""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

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
    def check_config(cls, config: "bittensor.Config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter wallet name", default=bittensor.defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter hotkey name", default=bittensor.defaults.wallet.hotkey
            )
            config.wallet.hotkey = str(hotkey)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        senate_leave_parser = parser.add_parser(
            "senate_leave",
            help="""Discard senate membership in the governance protocol""",
        )
        senate_leave_parser.add_argument(
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        senate_leave_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.wallet.add_args(senate_leave_parser)
        bittensor.subtensor.add_args(senate_leave_parser)


class VoteCommand:
    @staticmethod
    def run(cli):
        r"""Vote in Bittensor's governance protocol proposals"""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.Subtensor = bittensor.subtensor(config=config)

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
    def check_config(cls, config: "bittensor.Config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter wallet name", default=bittensor.defaults.wallet.name
            )
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask(
                "Enter hotkey name", default=bittensor.defaults.wallet.hotkey
            )
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
            "--no_version_checking",
            action="store_true",
            help="""Set false to stop cli version checking""",
            default=False,
        )
        vote_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
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
