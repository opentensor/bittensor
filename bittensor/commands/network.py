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

import time
import argparse
import bittensor
from . import defaults
from rich.prompt import Prompt
from rich.table import Table
from typing import List, Optional, Dict
from .utils import get_delegates_details, DelegatesDetails, check_netuid_set

console = bittensor.__console__


class RegisterSubnetworkCommand:
    @staticmethod
    def run(cli):
        r"""Register a subnetwork"""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        # Call register command.
        subtensor.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "create",
            help="""Create a new bittensor subnetwork on this chain.""",
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)


class SubnetLockCostCommand:
    @staticmethod
    def run(cli):
        r"""View locking cost of creating a new subnetwork"""
        config = cli.config.copy()
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        try:
            bittensor.__console__.print(
                f"Subnet lock cost: [green]{bittensor.utils.balance.Balance( subtensor.get_subnet_burn_cost() )}[/green]"
            )
        except Exception as e:
            bittensor.__console__.print(
                f"Subnet lock cost: [red]Failed to get subnet lock cost[/red]"
                f"Error: {e}"
            )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "lock_cost",
            help=""" Return the lock cost to register a subnet""",
        )

        bittensor.subtensor.add_args(parser)


class SubnetListCommand:
    @staticmethod
    def run(cli):
        r"""List all subnet netuids in the network."""
        subtensor = bittensor.subtensor(config=cli.config)
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        rows = []
        total_neurons = 0
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        for subnet in subnets:
            total_neurons += subnet.max_n
            rows.append(
                (
                    str(subnet.netuid),
                    str(subnet.subnetwork_n),
                    str(bittensor.utils.formatting.millify(subnet.max_n)),
                    f"{subnet.emission_value / bittensor.utils.RAOPERTAO * 100:0.2f}%",
                    str(subnet.tempo),
                    f"{subnet.burn!s:8.8}",
                    str(bittensor.utils.formatting.millify(subnet.difficulty)),
                    f"{delegate_info[subnet.owner_ss58].name if subnet.owner_ss58 in delegate_info else subnet.owner_ss58}",
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
        table.add_column(
            "[overline white]NETUID",
            str(len(subnets)),
            footer_style="overline white",
            style="bold green",
            justify="center",
        )
        table.add_column(
            "[overline white]N",
            str(total_neurons),
            footer_style="overline white",
            style="green",
            justify="center",
        )
        table.add_column("[overline white]MAX_N", style="white", justify="center")
        table.add_column("[overline white]EMISSION", style="white", justify="center")
        table.add_column("[overline white]TEMPO", style="white", justify="center")
        table.add_column("[overline white]BURN", style="white", justify="center")
        table.add_column("[overline white]POW", style="white", justify="center")
        table.add_column("[overline white]SUDO", style="white")
        for row in rows:
            table.add_row(*row)
        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_subnets_parser = parser.add_parser(
            "list", help="""List all subnets on the network"""
        )
        bittensor.subtensor.add_args(list_subnets_parser)


HYPERPARAMS = {
    "serving_rate_limit": "sudo_set_serving_rate_limit",
    "weights_version": "sudo_set_weights_version_key",
    "weights_rate_Limit": "sudo_set_weights_set_rate_limit",
    "max_weight_limit": "sudo_set_max_weight_limit",
    "immunity_period": "sudo_set_immunity_period",
    "min_allowed_weights": "sudo_set_min_allowed_weights",
    "activity_cutoff": "sudo_set_activity_cutoff",
    "max_validators": "sudo_set_max_allowed_validators",
}


class SubnetSudoCommand:
    @staticmethod
    def run(cli):
        r"""Set subnet hyperparameters."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=cli.config)
        subtensor: bittensor.subtensor = bittensor.subtensor(config=config)
        print("\n")
        SubnetHyperparamsCommand.run(cli)
        if not config.is_set("param") and not config.no_prompt:
            param = Prompt.ask("Enter hyperparameter", choices=HYPERPARAMS)
            config.param = str(param)
        if not config.is_set("value") and not config.no_prompt:
            value = Prompt.ask("Enter new value")
            config.value = value

        subtensor.set_hyperparameter(
            wallet,
            netuid=cli.config.netuid,
            parameter=config.param,
            value=config.value,
            prompt=not cli.config.no_prompt,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(config, bittensor.subtensor(config=config))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("set", help="""Set hyperparameters for a subnet""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        parser.add_argument("--param", dest="param", type=str, required=False)
        parser.add_argument("--value", dest="value", type=str, required=False)

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)


class SubnetHyperparamsCommand:
    @staticmethod
    def run(cli):
        r"""View hyperparameters of a subnetwork."""
        subtensor = bittensor.subtensor(config=cli.config)
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
            check_netuid_set(config, bittensor.subtensor(config=config))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "hyperparameters", help="""View subnet hyperparameters"""
        )
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        bittensor.subtensor.add_args(parser)


class SubnetGetHyperparamsCommand:
    @staticmethod
    def run(cli):
        r"""View hyperparameters of a subnetwork."""
        subtensor = bittensor.subtensor(config=cli.config)
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
        table.add_column("[overline white]HYPERPARAMETER", style="white")
        table.add_column("[overline white]VALUE", style="green")

        for param in subnet.__dict__:
            table.add_row(param, str(subnet.__dict__[param]))

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(config, bittensor.subtensor(config=config))

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("get", help="""View subnet hyperparameters""")
        parser.add_argument(
            "--netuid", dest="netuid", type=int, required=False, default=False
        )
        bittensor.subtensor.add_args(parser)
