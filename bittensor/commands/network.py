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
from . import defaults
from rich.prompt import Prompt
from rich.table import Table
from typing import List, Dict
from .utils import check_netuid_set
from .identity import SetIdentityCommand

console = bittensor.__console__


class RegisterSubnetworkCommand:
    """
    Executes the ``register_subnetwork`` command to register a new subnetwork on the Bittensor network.

    This command facilitates the creation and registration of a subnetwork, which involves interaction with the user's wallet and the Bittensor subtensor. It ensures that the user has the necessary credentials and configurations to successfully register a new subnetwork.

    Usage:
        Upon invocation, the command performs several key steps to register a subnetwork:

        1. It copies the user's current configuration settings.
        2. It accesses the user's wallet using the provided configuration.
        3. It initializes the Bittensor subtensor object with the user's configuration.
        4. It then calls the ``register_subnetwork`` function of the subtensor object, passing the user's wallet and a prompt setting based on the user's configuration.

    If the user's configuration does not specify a wallet name and ``no_prompt`` is not set, the command will prompt the user to enter a wallet name. This name is then used in the registration process.

    The command structure includes:

    - Copying the user's configuration.
    - Accessing and preparing the user's wallet.
    - Initializing the Bittensor subtensor.
    - Registering the subnetwork with the necessary credentials.

    Example usage::

        btcli subnets create

    Note:
        This command is intended for advanced users of the Bittensor network who wish to contribute by adding new subnetworks. It requires a clear understanding of the network's functioning and the roles of subnetworks. Users should ensure that they have secured their wallet and are aware of the implications of adding a new subnetwork to the Bittensor ecosystem.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register a subnetwork"""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RegisterSubnetworkCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register a subnetwork"""
        wallet = bittensor.wallet(config=cli.config)

        # Call register command.
        success = subtensor.register_subnetwork(
            wallet=wallet,
            prompt=not cli.config.no_prompt,
        )
        if success and not cli.config.no_prompt:
            # Prompt for user to set identity.
            do_set_identity = Prompt.ask(
                f"Subnetwork registered successfully. Would you like to set your identity? [y/n]",
                choices=["y", "n"],
            )

            if do_set_identity.lower() == "y":
                subtensor.close()
                config = cli.config.copy()
                SetIdentityCommand.check_config(config)
                cli.config = config
                SetIdentityCommand.run(cli)

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


HYPERPARAMS = {
    "serving_rate_limit": "sudo_set_serving_rate_limit",
    "min_difficulty": "sudo_set_min_difficulty",
    "max_difficulty": "sudo_set_max_difficulty",
    "weights_version": "sudo_set_weights_version_key",
    "weights_rate_limit": "sudo_set_weights_set_rate_limit",
    "max_weight_limit": "sudo_set_max_weight_limit",
    "immunity_period": "sudo_set_immunity_period",
    "min_allowed_weights": "sudo_set_min_allowed_weights",
    "activity_cutoff": "sudo_set_activity_cutoff",
    "network_registration_allowed": "sudo_set_network_registration_allowed",
    "network_pow_registration_allowed": "sudo_set_network_pow_registration_allowed",
    "min_burn": "sudo_set_min_burn",
    "max_burn": "sudo_set_max_burn",
    "adjustment_alpha": "sudo_set_adjustment_alpha",
    "rho": "sudo_set_rho",
    "kappa": "sudo_set_kappa",
    "difficulty": "sudo_set_difficulty",
    "bonds_moving_avg": "sudo_set_bonds_moving_average",
}


class SubnetSudoCommand:
    """
    Executes the ``set`` command to set hyperparameters for a specific subnet on the Bittensor network.

    This command allows subnet owners to modify various hyperparameters of theirs subnet, such as its tempo,
    emission rates, and other network-specific settings.

    Usage:
        The command first prompts the user to enter the hyperparameter they wish to change and its new value.
        It then uses the user's wallet and configuration settings to authenticate and send the hyperparameter update
        to the specified subnet.

    Example usage::

        btcli sudo set --netuid 1 --param 'tempo' --value '0.5'

    Note:
        This command requires the user to specify the subnet identifier (``netuid``) and both the hyperparameter
        and its new value. It is intended for advanced users who are familiar with the network's functioning
        and the impact of changing these parameters.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set subnet hyperparameters."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SubnetSudoCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(
        cli: "bittensor.cli",
        subtensor: "bittensor.subtensor",
    ):
        r"""Set subnet hyperparameters."""
        wallet = bittensor.wallet(config=cli.config)
        print("\n")
        SubnetHyperparamsCommand.run(cli)
        if not cli.config.is_set("param") and not cli.config.no_prompt:
            param = Prompt.ask("Enter hyperparameter", choices=HYPERPARAMS)
            cli.config.param = str(param)
        if not cli.config.is_set("value") and not cli.config.no_prompt:
            value = Prompt.ask("Enter new value")
            cli.config.value = value

        if (
            cli.config.param == "network_registration_allowed"
            or cli.config.param == "network_pow_registration_allowed"
        ):
            cli.config.value = True if cli.config.value.lower() == "true" else False

        subtensor.set_hyperparameter(
            wallet,
            netuid=cli.config.netuid,
            parameter=cli.config.param,
            value=cli.config.value,
            prompt=not cli.config.no_prompt,
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("netuid") and not config.no_prompt:
            check_netuid_set(
                config, bittensor.subtensor(config=config, log_verbose=False)
            )

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
