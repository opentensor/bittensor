# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

from rich.prompt import Prompt

import bittensor
from .hyperparameters import SubnetHyperparamsCommand
from .. import defaults
from ..utils import check_netuid_set
from ...params.subnets import HYPERPARAMS


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
            wallet_name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name",
                default=defaults.wallet.name,
            )
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
