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
import json
import bittensor
import requests
from rich.table import Table

from bittensor.commands.models.subnets import FetchMetagraphData
from bittensor.utils import U16_NORMALIZED_FLOAT
from .utils import check_netuid_set

console = bittensor.__console__  # type: ignore


API_URL = "https://api.taomarketcap.com/graphql"

FETCH_SUBNETS_QUERY = """
query FetchSubnets($netUid: [Int!]!) {
  subnets(netUid: $netUid) {
    uids {
      stake(limit: 1) {
        data {
          value
        }
        uid
      }
      rank(limit: 1) {
        uid
        data {
          value
        }
      }
      trust(limit: 1) {
        data {
          value
        }
        uid
      }
      consensus(limit: 1) {
        uid
        data {
          value
        }
      }
      incentive(limit: 1) {
        data {
          value
        }
        uid
      }
      dividends(limit: 1) {
        uid
        data {
          value
        }
      }
      emission(limit: 1) {
        data {
          value
        }
        uid
      }
      validatorTrust(limit: 1) {
        data {
          value
        }
        uid
      }
      axons(limit: 1) {
        data {
          value
        }
        uid
      }
      active(limit: 1) {
        data {
          value
        }
        uid
      }
      lastUpdate(limit: 1) {
        data {
          value
        }
        uid
      }
      coldkey(limit: 1) {
        data {
          value
        }
        uid
      }
      validatorPermit {
        data {
          value
        }
        uid
      }
      hotkey {
        key
        uid
      }
    }
    difficulty(limit: 1) {
      value
    }
  }
  totalIssuance(limit: 1) {
    value
    blockNumber
  }
}
"""


class MetagraphCommand:
    """
    Executes the ``metagraph`` command to retrieve and display the entire metagraph for a specified network.

    This metagraph contains detailed information about
    all the neurons (nodes) participating in the network, including their stakes,
    trust scores, and more.

    Optional arguments:
        - ``--netuid``: The netuid of the network to query. Defaults to the default network UID.
        - ``--subtensor.network``: The name of the network to query. Defaults to the default network name.

    The table displayed includes the following columns for each neuron:

    - UID: Unique identifier of the neuron.
    - STAKE(τ): Total stake of the neuron in Tau (τ).
    - RANK: Rank score of the neuron.
    - TRUST: Trust score assigned to the neuron by other neurons.
    - CONSENSUS: Consensus score of the neuron.
    - INCENTIVE: Incentive score representing the neuron's incentive alignment.
    - DIVIDENDS: Dividends earned by the neuron.
    - EMISSION(p): Emission in Rho (p) received by the neuron.
    - VTRUST: Validator trust score indicating the network's trust in the neuron as a validator.
    - VAL: Validator status of the neuron.
    - UPDATED: Number of blocks since the neuron's last update.
    - ACTIVE: Activity status of the neuron.
    - AXON: Network endpoint information of the neuron.
    - HOTKEY: Partial hotkey (public key) of the neuron.
    - COLDKEY: Partial coldkey (public key) of the neuron.

    The command also prints network-wide statistics such as total stake, issuance, and difficulty.

    Usage:
        The user must specify the network UID to query the metagraph. If not specified, the default network UID is used.

    Example usage::

        btcli subnet metagraph --netuid 0 # Root network
        btcli subnet metagraph --netuid 1 --subtensor.network test

    Note:
        This command provides a snapshot of the network's state at the time of calling.
        It is useful for network analysis and diagnostics. It is intended to be used as
        part of the Bittensor CLI and not as a standalone function within user code.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Prints an entire metagraph."""
        try:
            response = requests.post(
                url=API_URL,
                json={
                    "query": FETCH_SUBNETS_QUERY,
                    "variables": {"netUid": cli.config.netuid},
                },
            )
            metagraph_data = FetchMetagraphData.validate(response.json())
            MetagraphCommand._run(cli, metagraph_data)
        except Exception as e:
            bittensor.logging.exception(f"An error occurred: {e}")

    def _run(cli: "bittensor.cli", metagraph_data: FetchMetagraphData):
        r"""Prints an entire metagraph."""
        console = bittensor.__console__

        # TODO fix it
        network = "finney"

        difficulty = metagraph_data.data.subnets[0].difficulty[0].value
        total_issuance = bittensor.Balance.from_rao(
            int(metagraph_data.data.totalIssuance[0].value)
        )
        uids = metagraph_data.data.subnets[0].uids
        TABLE_DATA = []
        total_stake = 0
        total_rank = 0
        total_validator_trust = 0
        total_trust = 0
        total_consensus = 0
        total_incentive = 0
        total_dividends = 0
        total_emission = 0
        total_active = 0
        for item in uids.rank:
            uid = item.uid
            active = uids.active[uid].data[0].value
            axons = json.loads(uids.axons[uid].data[0].value)
            stake = int(uids.stake[uid].data[0].value) / 1000000000
            row = [
                str(uid),
                "{:.5f}".format(stake),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.rank[uid].data[0].value))
                ),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.trust[uid].data[0].value))
                ),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.consensus[uid].data[0].value))
                ),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.incentive[uid].data[0].value))
                ),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.dividends[uid].data[0].value))
                ),
                "{}".format(int(uids.emission[uid].data[0].value)),
                "{:.5f}".format(
                    U16_NORMALIZED_FLOAT(int(uids.validatorTrust[uid].data[0].value))
                ),
                "*" if uids.validatorPermit[uid].data[0].value else "",
                str(uids.lastUpdate[uid].data[0].value),
                str(1 if active else 0),
                ("{ip}:{port}".format(**axons) if axons else "[yellow]none[/yellow]"),
                uids.hotkey[uid].key[:10],
                uids.coldkey[uid].data[0].value[:10],
            ]
            total_stake += stake
            total_rank += int(uids.rank[uid].data[0].value)
            total_validator_trust += int(uids.validatorTrust[uid].data[0].value)
            total_trust += int(uids.trust[uid].data[0].value)
            total_consensus += int(uids.consensus[uid].data[0].value)
            total_incentive += int(uids.incentive[uid].data[0].value)
            total_dividends += int(uids.dividends[uid].data[0].value)
            total_emission += int(uids.emission[uid].data[0].value)
            total_active += 1 if active else 0
            TABLE_DATA.append(row)
        total_neurons = len(uids.rank)
        table = Table(show_footer=False)
        table.title = "[white]Metagraph: net: {}:{}, block: {}, N: {}/{}, stake: {}, issuance: {}, difficulty: {}".format(
            network,
            cli.config.netuid,
            metagraph_data.data.totalIssuance[0].blockNumber,
            total_active,
            len(uids.rank),
            bittensor.Balance.from_tao(total_stake),
            total_issuance,
            difficulty,
        )
        table.add_column(
            "[overline white]UID",
            str(total_neurons),
            footer_style="overline white",
            style="yellow",
        )
        table.add_column(
            "[overline white]STAKE(\u03C4)",
            "\u03C4{:.5f}".format(total_stake),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]RANK",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_rank)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]TRUST",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_trust)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]CONSENSUS",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_consensus)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]INCENTIVE",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_incentive)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]DIVIDENDS",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_dividends)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]EMISSION(\u03C1)",
            "\u03C1{}".format(U16_NORMALIZED_FLOAT(total_emission)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]VTRUST",
            "{:.5f}".format(U16_NORMALIZED_FLOAT(total_validator_trust)),
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]VAL", justify="right", style="green", no_wrap=True
        )
        table.add_column("[overline white]UPDATED", justify="right", no_wrap=True)
        table.add_column(
            "[overline white]ACTIVE", justify="right", style="green", no_wrap=True
        )
        table.add_column(
            "[overline white]AXON", justify="left", style="dim blue", no_wrap=True
        )
        table.add_column("[overline white]HOTKEY", style="dim blue", no_wrap=False)
        table.add_column("[overline white]COLDKEY", style="dim purple", no_wrap=False)
        table.show_footer = True

        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        # TODO ckeck netuid with GQL
        # check_netuid_set(
        #     config, subtensor=bittensor.subtensor(config=config, log_verbose=False)
        # )
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        metagraph_parser = parser.add_parser(
            "metagraph", help="""View a subnet metagraph information."""
        )
        metagraph_parser.add_argument(
            "--netuid",
            dest="netuid",
            type=int,
            help="""Set the netuid to get the metagraph of""",
            default=False,
        )
        metagraph_parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )

        bittensor.subtensor.add_args(metagraph_parser)
