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
from rich.table import Table
from typing import List, Union, Optional, Dict, Tuple
console = bittensor.__console__

class ProposalsCommand:

    @staticmethod
    def run( cli ):
        r""" View Bittensor's governance protocol proposals
        """
        config = cli.config.copy()
        subtensor: bittensor.Subtensor = bittensor.subtensor( config = config )

        console.print(":satellite: Syncing with chain: [white]{}[/white] ...".format(cli.config.subtensor.network))

        proposals = dict()
        proposal_hashes = subtensor.query_module("Triumvirate", "Proposals")

        for hash in proposal_hashes:
            proposals[hash] = [
                subtensor.query_module("Triumvirate", "ProposalOf", None, [hash]), 
                subtensor.query_module("Triumvirate", "Voting", None, [hash])
            ]

        table = Table(show_footer=False)
        table.title = (
            "[white]Proposals:"
        )
        table.add_column("[overline white]HASH", footer_style = "overline white", style='yellow', no_wrap=True)
        table.add_column("[overline white]THRESHOLD", footer_style = "overline white", style='white')
        table.add_column("[overline white]AYES", footer_style = "overline white", style='green')
        table.add_column("[overline white]NAYS", footer_style = "overline white", style='red')
        table.add_column("[overline white]END", footer_style = "overline white", style='blue')
        table.add_column("[overline white]CALLDATA", footer_style = "overline white", style='white')
        table.show_footer = True

        for hash in proposals:
            call_data = proposals[hash][0].serialize()
            vote_data = proposals[hash][1].serialize()

            human_call_data = list()
            for arg in call_data["call_args"]:
                human_call_data.append("{}: {}".format(arg["name"], str(arg["value"])))

            table.add_row(
                hash,
                str(vote_data["threshold"]),
                str(len(vote_data["ayes"])),
                str(len(vote_data["nays"])),
                str(vote_data["end"]),
                "{}({})".format(call_data["call_function"], ", ".join(human_call_data))
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        None

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        proposals_parser = parser.add_parser(
            'proposals',
            help='''View active triumvirate proposals and their status'''
        )
        proposals_parser.add_argument(
            '--no_version_checking',
            action='store_true',
            help='''Set false to stop cli version checking''',
            default = False
        )
        proposals_parser.add_argument(
            '--no_prompt',
            dest='no_prompt',
            action='store_true',
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( proposals_parser )
        bittensor.subtensor.add_args( proposals_parser )