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

import re
import typing
import argparse
import numpy as np
import bittensor
from typing import List, Optional, Dict
from rich.prompt import Prompt
from rich.table import Table
from .utils import get_delegates_details, DelegatesDetails

from . import defaults

console = bittensor.__console__


class RootRegisterCommand:
    """
    Executes the ``register`` command to register a wallet to the root network of the Bittensor network.

    This command is used to formally acknowledge a wallet's participation in the network's root layer.

    Usage:
        The command registers the user's wallet with the root network, which is a crucial step for participating in network governance and other advanced functions.

    Optional arguments:
        - None. The command primarily uses the wallet and subtensor configurations.

    Example usage::

        btcli root register

    Note:
        This command is important for users seeking to engage deeply with the Bittensor network, particularly in aspects related to network governance and decision-making.

        It is a straightforward process but requires the user to have an initialized and configured wallet.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Register to root network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootRegisterCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Register to root network."""
        wallet = bittensor.wallet(config=cli.config)

        subtensor.root_register(wallet=wallet, prompt=not cli.config.no_prompt)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "register", help="""Register a wallet to the root network."""
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class RootList:
    """
    Executes the ``list`` command to display the members of the root network on the Bittensor network.

    This command provides an overview of the neurons that constitute the network's foundational layer.

    Usage:
        Upon execution, the command fetches and lists the neurons in the root network, showing their unique identifiers (UIDs), names, addresses, stakes, and whether they are part of the senate (network governance body).

    Optional arguments:
        - None. The command uses the subtensor configuration to retrieve data.

    Example usage::

        $ btcli root list

        UID  NAME                             ADDRESS                                                STAKE(τ)  SENATOR
        0                                     5CaCUPsSSdKWcMJbmdmJdnWVa15fJQuz5HsSGgVdZffpHAUa    27086.37070  Yes
        1    RaoK9                            5GmaAk7frPXnAxjbQvXcoEzMGZfkrDee76eGmKoB3wxUburE      520.24199  No
        2    Openτensor Foundaτion            5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3  1275437.45895  Yes
        3    RoundTable21                     5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v    84718.42095  Yes
        4                                     5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN   168897.40859  Yes
        5    Rizzo                            5CXRfP2ekFhe62r7q3vppRajJmGhTi7vwvb2yr79jveZ282w    53383.34400  No
        6    τaosτaτs and BitAPAI             5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8   646944.73569  Yes
        ...

    Note:
        This command is useful for users interested in understanding the composition and governance structure of the Bittensor network's root layer. It provides insights into which neurons hold significant influence and responsibility within the network.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""List the root network"""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootList._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""List the root network"""
        console.print(
            ":satellite: Syncing with chain: [white]{}[/white] ...".format(
                subtensor.network
            )
        )

        senate_members = subtensor.get_senate_members()
        root_neurons: typing.List[bittensor.NeuronInfoLite] = subtensor.neurons_lite(
            netuid=0
        )
        delegate_info: Optional[Dict[str, DelegatesDetails]] = get_delegates_details(
            url=bittensor.__delegates_details_url__
        )

        table = Table(show_footer=False)
        table.title = "[white]Root Network"
        table.add_column(
            "[overline white]UID",
            footer_style="overline white",
            style="rgb(50,163,219)",
            no_wrap=True,
        )
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
        table.add_column(
            "[overline white]STAKE(\u03C4)",
            footer_style="overline white",
            justify="right",
            style="green",
            no_wrap=True,
        )
        table.add_column(
            "[overline white]SENATOR",
            footer_style="overline white",
            style="green",
            no_wrap=True,
        )
        table.show_footer = True

        for neuron_data in root_neurons:
            table.add_row(
                str(neuron_data.uid),
                (
                    delegate_info[neuron_data.hotkey].name
                    if neuron_data.hotkey in delegate_info
                    else ""
                ),
                neuron_data.hotkey,
                "{:.5f}".format(
                    float(subtensor.get_total_stake_for_hotkey(neuron_data.hotkey))
                ),
                "Yes" if neuron_data.hotkey in senate_members else "No",
            )

        table.box = None
        table.pad_edge = False
        table.width = None
        bittensor.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("list", help="""List the root network""")
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass


class RootSetBoostCommand:
    """
    Executes the ``boost`` command to boost the weights for a specific subnet within the root network on the Bittensor network.

    Usage:
        The command allows boosting the weights for different subnets within the root network.

    Optional arguments:
        - ``--netuid`` (int): A single netuid for which weights are to be boosted.
        - ``--increase`` (float): The cooresponding increase in the weight for this subnet.

    Example usage::

        $ btcli root boost --netuid 1 --increase 0.01

        Enter netuid (e.g. 1): 1
        Enter amount (e.g. 0.01): 0.1
        Boosting weight for subnet: 1 by amount: 0.1

        Normalized weights:
                tensor([
                0.0000, 0.5455, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.4545, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000]) -> tensor([0.0000, 0.5455, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.4545, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
            )

        Do you want to set the following root weights?:
        weights: tensor([
                0.0000, 0.5455, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.4545, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        uids: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40])? [y/n]: y
        True None
        ✅ Finalized
        ⠙ 📡 Setting root weights on test ...2023-11-28 22:09:14.001 |     SUCCESS      | Set weights                   Finalized: True

    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set weights for root network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootSetBoostCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set weights for root network."""
        wallet = bittensor.wallet(config=cli.config)

        root = subtensor.metagraph(0, lite=False)
        try:
            my_uid = root.hotkeys.index(wallet.hotkey.ss58_address)
        except ValueError:
            bittensor.__console__.print(
                "Wallet hotkey: {} not found in root metagraph".format(wallet.hotkey)
            )
            exit()
        my_weights = root.weights[my_uid]
        prev_weight = my_weights[cli.config.netuid]
        new_weight = prev_weight + cli.config.amount

        bittensor.__console__.print(
            f"Boosting weight for netuid {cli.config.netuid} from {prev_weight} -> {new_weight}"
        )
        my_weights[cli.config.netuid] = new_weight
        all_netuids = np.arange(len(my_weights))

        bittensor.__console__.print("Setting root weights...")
        subtensor.root_set_weights(
            wallet=wallet,
            netuids=all_netuids,
            weights=my_weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "boost", help="""Boost weight for a specific subnet by increase amount."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--increase", dest="amount", type=float, required=False)

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
        if not config.is_set("netuid") and not config.no_prompt:
            config.netuid = int(Prompt.ask(f"Enter netuid (e.g. 1)"))
        if not config.is_set("amount") and not config.no_prompt:
            config.amount = float(Prompt.ask(f"Enter amount (e.g. 0.01)"))


class RootSetSlashCommand:
    """
    Executes the ``slash`` command to decrease the weights for a specific subnet within the root network on the Bittensor network.

    Usage:
        The command allows slashing (decreasing) the weights for different subnets within the root network.

    Optional arguments:
        - ``--netuid`` (int): A single netuid for which weights are to be slashed.
        - ``--decrease`` (float): The corresponding decrease in the weight for this subnet.

    Example usage::

        $ btcli root slash --netuid 1 --decrease 0.01

        Enter netuid (e.g. 1): 1
        Enter decrease amount (e.g. 0.01): 0.2
        Slashing weight for subnet: 1 by amount: 0.2

        Normalized weights:
                tensor([
                0.0000, 0.4318, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.5682, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000]) -> tensor([
                0.0000, 0.4318, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.5682, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
            )

        Do you want to set the following root weights?:
        weights: tensor([
                0.0000, 0.4318, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.5682, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        uids: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40])? [y/n]: y
        ⠙ 📡 Setting root weights on test ...2023-11-28 22:09:14.001 |     SUCCESS      | Set weights                   Finalized: True
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Set weights for root network with decreased values."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootSetSlashCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)

        bittensor.__console__.print(
            "Slashing weight for subnet: {} by amount: {}".format(
                cli.config.netuid, cli.config.amount
            )
        )
        root = subtensor.metagraph(0, lite=False)
        try:
            my_uid = root.hotkeys.index(wallet.hotkey.ss58_address)
        except ValueError:
            bittensor.__console__.print(
                "Wallet hotkey: {} not found in root metagraph".format(wallet.hotkey)
            )
            exit()
        my_weights = root.weights[my_uid]
        my_weights[cli.config.netuid] -= cli.config.amount
        my_weights[my_weights < 0] = 0  # Ensure weights don't go negative
        all_netuids = np.arange(len(my_weights))

        subtensor.root_set_weights(
            wallet=wallet,
            netuids=all_netuids,
            weights=my_weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "slash", help="""Slash weight for a specific subnet by decrease amount."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--decrease", dest="amount", type=float, required=False)

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
        if not config.is_set("netuid") and not config.no_prompt:
            config.netuid = int(Prompt.ask(f"Enter netuid (e.g. 1)"))
        if not config.is_set("amount") and not config.no_prompt:
            config.amount = float(Prompt.ask(f"Enter decrease amount (e.g. 0.01)"))


class RootSetWeightsCommand:
    """
    Executes the ``weights`` command to set the weights for the root network on the Bittensor network.

    This command is used by network senators to influence the distribution of network rewards and responsibilities.

    Usage:
        The command allows setting weights for different subnets within the root network. Users need to specify the netuids (network unique identifiers) and corresponding weights they wish to assign.

    Optional arguments:
        - ``--netuids`` (str): A comma-separated list of netuids for which weights are to be set.
        - ``--weights`` (str): Corresponding weights for the specified netuids, in comma-separated format.

    Example usage::

        btcli root weights --netuids 1,2,3 --weights 0.3,0.3,0.4

    Note:
        This command is particularly important for network senators and requires a comprehensive understanding of the network's dynamics.
        It is a powerful tool that directly impacts the network's operational mechanics and reward distribution.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Set weights for root network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootSetWeightsCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Set weights for root network."""
        wallet = bittensor.wallet(config=cli.config)
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        # Get values if not set.
        if not cli.config.is_set("netuids"):
            example = (
                ", ".join(map(str, [subnet.netuid for subnet in subnets][:3])) + " ..."
            )
            cli.config.netuids = Prompt.ask(f"Enter netuids (e.g. {example})")

        if not cli.config.is_set("weights"):
            example = (
                ", ".join(
                    map(
                        str,
                        [
                            "{:.2f}".format(float(1 / len(subnets)))
                            for subnet in subnets
                        ][:3],
                    )
                )
                + " ..."
            )
            cli.config.weights = Prompt.ask(f"Enter weights (e.g. {example})")

        # Parse from string
        matched_netuids = list(map(int, re.split(r"[ ,]+", cli.config.netuids)))
        netuids = np.array(matched_netuids, dtype=np.int64)

        matched_weights = [
            float(weight) for weight in re.split(r"[ ,]+", cli.config.weights)
        ]
        weights = np.array(matched_weights, dtype=np.float32)

        # Run the set weights operation.
        subtensor.root_set_weights(
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=0,
            prompt=not cli.config.no_prompt,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser("weights", help="""Set weights for root network.""")
        parser.add_argument("--netuids", dest="netuids", type=str, required=False)
        parser.add_argument("--weights", dest="weights", type=str, required=False)

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)


class RootGetWeightsCommand:
    """
    Executes the ``get_weights`` command to retrieve the weights set for the root network on the Bittensor network.

    This command provides visibility into how network responsibilities and rewards are distributed among various subnets.

    Usage:
        The command outputs a table listing the weights assigned to each subnet within the root network. This information is crucial for understanding the current influence and reward distribution among the subnets.

    Optional arguments:
        - None. The command fetches weight information based on the subtensor configuration.

    Example usage::

        $ btcli root get_weights

                                                Root Network Weights
        UID        0        1        2       3        4        5       8        9       11     13      18       19
        1    100.00%        -        -       -        -        -       -        -        -      -       -        -
        2          -   40.00%    5.00%  10.00%   10.00%   10.00%  10.00%    5.00%        -      -  10.00%        -
        3          -        -   25.00%       -   25.00%        -  25.00%        -        -      -  25.00%        -
        4          -        -    7.00%   7.00%   20.00%   20.00%  20.00%        -    6.00%      -  20.00%        -
        5          -   20.00%        -  10.00%   15.00%   15.00%  15.00%    5.00%        -      -  10.00%   10.00%
        6          -        -        -       -   10.00%   10.00%  25.00%   25.00%        -      -  30.00%        -
        7          -   60.00%        -       -   20.00%        -       -        -   20.00%      -       -        -
        8          -   49.35%        -   7.18%   13.59%   21.14%   1.53%    0.12%    7.06%  0.03%       -        -
        9    100.00%        -        -       -        -        -       -        -        -      -       -        -
        ...

    Note:
        This command is essential for users interested in the governance and operational dynamics of the Bittensor network. It offers transparency into how network rewards and responsibilities are allocated across different subnets.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Get weights for root network."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RootGetWeightsCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Get weights for root network."""
        weights = subtensor.weights(0)

        table = Table(show_footer=False)
        table.title = "[white]Root Network Weights"
        table.add_column(
            "[white]UID",
            header_style="overline white",
            footer_style="overline white",
            style="rgb(50,163,219)",
            no_wrap=True,
        )

        uid_to_weights = {}
        netuids = set()
        for matrix in weights:
            [uid, weights_data] = matrix

            if not len(weights_data):
                uid_to_weights[uid] = {}
                normalized_weights = []
            else:
                normalized_weights = np.array(weights_data)[:, 1] / max(
                    np.sum(weights_data, axis=0)[1], 1
                )

            for weight_data, normalized_weight in zip(weights_data, normalized_weights):
                [netuid, _] = weight_data
                netuids.add(netuid)
                if uid not in uid_to_weights:
                    uid_to_weights[uid] = {}

                uid_to_weights[uid][netuid] = normalized_weight

        for netuid in netuids:
            table.add_column(
                f"[white]{netuid}",
                header_style="overline white",
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )

        for uid in uid_to_weights:
            row = [str(uid)]

            uid_weights = uid_to_weights[uid]
            for netuid in netuids:
                if netuid in uid_weights:
                    normalized_weight = uid_weights[netuid]
                    row.append("{:0.2f}%".format(normalized_weight * 100))
                else:
                    row.append("-")
            table.add_row(*row)

        table.show_footer = True

        table.box = None
        table.pad_edge = False
        table.width = None
        bittensor.__console__.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "get_weights", help="""Get weights for root network."""
        )

        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        pass
