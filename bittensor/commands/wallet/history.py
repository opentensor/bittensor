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
from typing import List

import requests
from rich.prompt import Prompt
from rich.table import Table

import bittensor
from bittensor.utils import RAOPERTAO
from .. import defaults

API_URL = "https://api.subquery.network/sq/TaoStats/bittensor-indexer"
MAX_TXN = 1000
GRAPHQL_QUERY = """
query ($first: Int!, $after: Cursor, $filter: TransferFilter, $order: [TransfersOrderBy!]!) {
    transfers(first: $first, after: $after, filter: $filter, orderBy: $order) {
        nodes {
            id
            from
            to
            amount
            extrinsicId
            blockNumber
        }
        pageInfo {
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
    }
}
"""


class GetWalletHistoryCommand:
    """
    Executes the ``history`` command to fetch the latest transfers of the provided wallet on the Bittensor network.

    This command provides a detailed view of the transfers carried out on the wallet.

    Usage:
        The command lists the latest transfers of the provided wallet, showing the From, To, Amount, Extrinsic Id and Block Number.

    Optional arguments:
        None. The command uses the wallet and subtensor configurations to fetch latest transfer data associated with a wallet.

    Example usage::

        btcli wallet history

    Note:
        This command is essential for users to monitor their financial status on the Bittensor network.
        It helps in fetching info on all the transfers so that user can easily tally and cross check the transactions.
    """

    @staticmethod
    def run(cli):
        r"""Check the transfer history of the provided wallet."""
        wallet = bittensor.wallet(config=cli.config)
        wallet_address = wallet.get_coldkeypub().ss58_address
        # Fetch all transfers
        transfers = get_wallet_transfers(wallet_address)

        # Create output table
        table = create_transfer_history_table(transfers)

        bittensor.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        history_parser = parser.add_parser(
            "history",
            help="""Fetch transfer history associated with the provided wallet""",
        )
        bittensor.wallet.add_args(history_parser)
        bittensor.subtensor.add_args(history_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
            config.wallet.name = str(wallet_name)


def get_wallet_transfers(wallet_address) -> List[dict]:
    """Get all transfers associated with the provided wallet address."""

    variables = {
        "first": MAX_TXN,
        "filter": {
            "or": [
                {"from": {"equalTo": wallet_address}},
                {"to": {"equalTo": wallet_address}},
            ]
        },
        "order": "BLOCK_NUMBER_DESC",
    }

    response = requests.post(
        API_URL, json={"query": GRAPHQL_QUERY, "variables": variables}
    )
    data = response.json()

    # Extract nodes and pageInfo from the response
    transfer_data = data.get("data", {}).get("transfers", {})
    transfers = transfer_data.get("nodes", [])

    return transfers


def create_transfer_history_table(transfers):
    """Get output transfer table"""

    table = Table(show_footer=False)
    # Define the column names
    column_names = [
        "Id",
        "From",
        "To",
        "Amount (Tao)",
        "Extrinsic Id",
        "Block Number",
        "URL (taostats)",
    ]
    taostats_url_base = "https://x.taostats.io/extrinsic"

    # Create a table
    table = Table(show_footer=False)
    table.title = "[white]Wallet Transfers"

    # Define the column styles
    header_style = "overline white"
    footer_style = "overline white"
    column_style = "rgb(50,163,219)"
    no_wrap = True

    # Add columns to the table
    for column_name in column_names:
        table.add_column(
            f"[white]{column_name}",
            header_style=header_style,
            footer_style=footer_style,
            style=column_style,
            no_wrap=no_wrap,
            justify="left" if column_name == "Id" else "right",
        )

    # Add rows to the table
    for item in transfers:
        try:
            tao_amount = int(item["amount"]) / RAOPERTAO
        except:
            tao_amount = item["amount"]
        table.add_row(
            item["id"],
            item["from"],
            item["to"],
            f"{tao_amount:.3f}",
            str(item["extrinsicId"]),
            item["blockNumber"],
            f"{taostats_url_base}/{item['blockNumber']}-{item['extrinsicId']}",
        )
    table.add_row()
    table.show_footer = True
    table.box = None
    table.pad_edge = False
    table.width = None
    return table
