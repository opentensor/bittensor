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

import os
import sys
import argparse
import bittensor
from typing import List
from rich.prompt import Prompt
from rich.table import Table

console = bittensor.__console__


class UpdateCommand:
    @staticmethod
    def run(cli):
        if cli.config.no_prompt or cli.config.answer == "Y":
            os.system(
                " (cd ~/.bittensor/bittensor/ ; git checkout master ; git pull --ff-only )"
            )
            os.system("pip install -e ~/.bittensor/bittensor/")

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.no_prompt:
            answer = Prompt.ask(
                "This will update the local bittensor package",
                choices=["Y", "N"],
                default="Y",
            )
            config.answer = answer

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        update_parser = parser.add_parser(
            "update", add_help=False, help="""Update bittensor """
        )

        bittensor.subtensor.add_args(update_parser)


class ListSubnetsCommand:
    @staticmethod
    def run(cli):
        r"""List all subnet netuids in the network."""
        subtensor = bittensor.subtensor(config=cli.config)
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        rows = []
        total_neurons = 0

        for subnet in subnets:
            total_neurons += subnet.max_n
            # netuid, N, Max N, difficulty, network connect, tempo, emission, burn rate
            rows.append(
                (
                    str(subnet.netuid),
                    str(subnet.subnetwork_n),
                    str(bittensor.utils.formatting.millify(subnet.max_n)),
                    str(bittensor.utils.formatting.millify(subnet.difficulty)),
                    str(subnet.tempo),
                    str(
                        [
                            f"{cr[0]}: {cr[1] * 100:.1f}%"
                            for cr in subnet.connection_requirements.items()
                        ]
                        if len(subnet.connection_requirements) > 0
                        else None
                    ),
                    f"{subnet.emission_value / bittensor.utils.RAOPERTAO * 100:0.2f}%",
                    f"{subnet.burn!s:8.8}",
                    f"{subnet.owner_ss58}",
                )
            )

        table = Table(
            show_footer=True,
            width=cli.config.get("width", None),
            pad_edge=True,
            box=None,
            show_edge=True,
        )
        table.title = "[white]Subnets - {}".format(subtensor.network)
        # netuid, N, Max N, difficulty, network connect, tempo, emission, burn rate
        table.add_column(
            "[overline white]NETUID",
            str(len(subnets)),
            footer_style="overline white",
            style="bold green",
            justify="center",
        )
        table.add_column(
            "[overline white]NEURONS",
            str(total_neurons),
            footer_style="overline white",
            style="white",
            justify="center",
        )
        table.add_column("[overline white]MAX_N", style="white", justify="center")
        table.add_column("[overline white]DIFFICULTY", style="white", justify="center")
        # table.add_column("[overline white]IMMUNITY", style='white')
        # table.add_column("[overline white]BATCH SIZE", style='white')
        # table.add_column("[overline white]SEQ_LEN", style='white')
        table.add_column("[overline white]TEMPO", style="white", justify="center")
        # table.add_column("[overline white]MODALITY", style='white')
        table.add_column("[overline white]CON_REQ", style="white", justify="center")
        # table.add_column("[overline white]STAKE", style="green", justify="center")
        table.add_column(
            "[overline white]EMISSION", style="white", justify="center"
        )  # sums to 100%
        table.add_column("[overline white]BURN(\u03C4)", style="white")
        table.add_column("[overline white]OWNER(\u03C4)", style="white")

        for row in rows:
            table.add_row(*row)

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list_subnets", help="""List all subnets on the network"""
        )

        bittensor.subtensor.add_args(list_subnets_parser)
