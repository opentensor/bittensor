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
from rich.table import Table
from ..utils import check_netuid_set


class SubnetHyperparamsCommand:
    """
    Executes the '``hyperparameters``' command to view the current hyperparameters of a specific subnet on the Bittensor network.

    This command is useful for users who wish to understand the configuration and
    operational parameters of a particular subnet.

    Usage:
        Upon invocation, the command fetches and displays a list of all hyperparameters for the specified subnet.
        These include settings like tempo, emission rates, and other critical network parameters that define
        the subnet's behavior.

    Example usage::

        $ btcli subnets hyperparameters --netuid 1

        Subnet Hyperparameters - NETUID: 1 - finney
        HYPERPARAMETER            VALUE
        rho                       10
        kappa                     32767
        immunity_period           7200
        min_allowed_weights       8
        max_weight_limit          455
        tempo                     99
        min_difficulty            1000000000000000000
        max_difficulty            1000000000000000000
        weights_version           2013
        weights_rate_limit        100
        adjustment_interval       112
        activity_cutoff           5000
        registration_allowed      True
        target_regs_per_interval  2
        min_burn                  1000000000
        max_burn                  100000000000
        bonds_moving_avg          900000
        max_regs_per_block        1

    Note:
        The user must specify the subnet identifier (``netuid``) for which they want to view the hyperparameters.
        This command is read-only and does not modify the network state or configurations.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""View hyperparameters of a subnetwork."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SubnetHyperparamsCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""View hyperparameters of a subnetwork."""
        subnet: bittensor.SubnetHyperparameters = subtensor.get_subnet_hyperparameters(
            cli.config.netuid
        )

        table = Table(
            show_footer=True,
            width=cli.config.get("width", None),
            pad_edge=True,
            box=None,
            show_edge=True,
        )
        table.title = "[white]Subnet Hyperparameters - NETUID: {} - {}".format(
            cli.config.netuid, subtensor.network
        )
        table.add_column("[overline white]HYPERPARAMETER", style="bold white")
        table.add_column("[overline white]VALUE", style="green")

        for param in subnet.__dict__:
            table.add_row("  " + param, str(subnet.__dict__[param]))

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, bittensor.subtensor(config=config, log_verbose=False)
            )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "hyperparameters", help="""View subnet hyperparameters"""
        )
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        bittensor.subtensor.add_args(parser)
