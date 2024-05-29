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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from fuzzywuzzy import fuzz
from rich.align import Align
from rich.table import Table
from rich.prompt import Prompt
from typing import List, Optional, Dict, Tuple
from .utils import (
    get_hotkey_wallets_for_wallet,
    get_coldkey_wallets_for_path,
    get_all_wallets_for_path,
    filter_netuids_by_registered_hotkeys,
)
from . import defaults

console = bittensor.__console__


class OverviewCommand:
    """
    Executes the ``overview`` command to present a detailed overview of the user's registered accounts on the Bittensor network.

    This command compiles and displays comprehensive information about each neuron associated with the user's wallets,
    including both hotkeys and coldkeys. It is especially useful for users managing multiple accounts or seeking a summary
    of their network activities and stake distributions.

    Usage:
        The command offers various options to customize the output. Users can filter the displayed data by specific netuids,
        sort by different criteria, and choose to include all wallets in the user's configuration directory. The output is
        presented in a tabular format with the following columns:

        - COLDKEY: The SS58 address of the coldkey.
        - HOTKEY: The SS58 address of the hotkey.
        - UID: Unique identifier of the neuron.
        - ACTIVE: Indicates if the neuron is active.
        - STAKE(τ): Amount of stake in the neuron, in Tao.
        - RANK: The rank of the neuron within the network.
        - TRUST: Trust score of the neuron.
        - CONSENSUS: Consensus score of the neuron.
        - INCENTIVE: Incentive score of the neuron.
        - DIVIDENDS: Dividends earned by the neuron.
        - EMISSION(p): Emission received by the neuron, in Rho.
        - VTRUST: Validator trust score of the neuron.
        - VPERMIT: Indicates if the neuron has a validator permit.
        - UPDATED: Time since last update.
        - AXON: IP address and port of the neuron.
        - HOTKEY_SS58: Human-readable representation of the hotkey.

    Example usage::

        btcli wallet overview
        btcli wallet overview --all --sort_by stake --sort_order descending

    Note:
        This command is read-only and does not modify the network state or account configurations. It provides a quick and
        comprehensive view of the user's network presence, making it ideal for monitoring account status, stake distribution,
        and overall contribution to the Bittensor network.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Prints an overview for the wallet's colkey."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            OverviewCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _get_total_balance(
        total_balance: "bittensor.Balance",
        subtensor: "bittensor.subtensor",
        cli: "bittensor.cli",
    ) -> Tuple[List["bittensor.wallet"], "bittensor.Balance"]:
        if cli.config.get("all", d=None):
            cold_wallets = get_coldkey_wallets_for_path(cli.config.wallet.path)
            for cold_wallet in tqdm(cold_wallets, desc="Pulling balances"):
                if (
                    cold_wallet.coldkeypub_file.exists_on_device()
                    and not cold_wallet.coldkeypub_file.is_encrypted()
                ):
                    total_balance = total_balance + subtensor.get_balance(
                        cold_wallet.coldkeypub.ss58_address
                    )
            all_hotkeys = get_all_wallets_for_path(cli.config.wallet.path)
        else:
            # We are only printing keys for a single coldkey
            coldkey_wallet = bittensor.wallet(config=cli.config)
            if (
                coldkey_wallet.coldkeypub_file.exists_on_device()
                and not coldkey_wallet.coldkeypub_file.is_encrypted()
            ):
                total_balance = subtensor.get_balance(
                    coldkey_wallet.coldkeypub.ss58_address
                )
            if not coldkey_wallet.coldkeypub_file.exists_on_device():
                console.print("[bold red]No wallets found.")
                return [], None
            all_hotkeys = get_hotkey_wallets_for_wallet(coldkey_wallet)

        return all_hotkeys, total_balance

    @staticmethod
    def _get_hotkeys(
        cli: "bittensor.cli", all_hotkeys: List["bittensor.wallet"]
    ) -> List["bittensor.wallet"]:
        if not cli.config.get("all_hotkeys", False):
            # We are only showing hotkeys that are specified.
            all_hotkeys = [
                hotkey
                for hotkey in all_hotkeys
                if hotkey.hotkey_str in cli.config.hotkeys
            ]
        else:
            # We are excluding the specified hotkeys from all_hotkeys.
            all_hotkeys = [
                hotkey
                for hotkey in all_hotkeys
                if hotkey.hotkey_str not in cli.config.hotkeys
            ]
        return all_hotkeys

    @staticmethod
    def _get_key_address(all_hotkeys: List["bittensor.wallet"]):
        hotkey_coldkey_to_hotkey_wallet = {}
        for hotkey_wallet in all_hotkeys:
            if hotkey_wallet.hotkey.ss58_address not in hotkey_coldkey_to_hotkey_wallet:
                hotkey_coldkey_to_hotkey_wallet[hotkey_wallet.hotkey.ss58_address] = {}

            hotkey_coldkey_to_hotkey_wallet[hotkey_wallet.hotkey.ss58_address][
                hotkey_wallet.coldkeypub.ss58_address
            ] = hotkey_wallet

        all_hotkey_addresses = list(hotkey_coldkey_to_hotkey_wallet.keys())

        return all_hotkey_addresses, hotkey_coldkey_to_hotkey_wallet

    @staticmethod
    def _process_neuron_results(
        results: List[Tuple[int, List["bittensor.NeuronInfoLite"], Optional[str]]],
        neurons: Dict[str, List["bittensor.NeuronInfoLite"]],
        netuids: List[int],
    ) -> Dict[str, List["bittensor.NeuronInfoLite"]]:
        for result in results:
            netuid, neurons_result, err_msg = result
            if err_msg is not None:
                console.print(f"netuid '{netuid}': {err_msg}")

            if len(neurons_result) == 0:
                # Remove netuid from overview if no neurons are found.
                netuids.remove(netuid)
                del neurons[str(netuid)]
            else:
                # Add neurons to overview.
                neurons[str(netuid)] = neurons_result
        return neurons

    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Prints an overview for the wallet's colkey."""
        console = bittensor.__console__
        wallet = bittensor.wallet(config=cli.config)

        all_hotkeys = []
        total_balance = bittensor.Balance(0)

        # We are printing for every coldkey.
        all_hotkeys, total_balance = OverviewCommand._get_total_balance(
            total_balance, subtensor, cli
        )

        # We are printing for a select number of hotkeys from all_hotkeys.
        if cli.config.get("hotkeys"):
            all_hotkeys = OverviewCommand._get_hotkeys(cli, all_hotkeys)

        # Check we have keys to display.
        if len(all_hotkeys) == 0:
            console.print("[red]No wallets found.[/red]")
            return

        # Pull neuron info for all keys.
        neurons: Dict[str, List[bittensor.NeuronInfoLite]] = {}
        block = subtensor.block

        netuids = subtensor.get_all_subnet_netuids()
        netuids = filter_netuids_by_registered_hotkeys(
            cli, subtensor, netuids, all_hotkeys
        )
        bittensor.logging.debug(f"Netuids to check: {netuids}")

        for netuid in netuids:
            neurons[str(netuid)] = []

        all_wallet_names = {wallet.name for wallet in all_hotkeys}
        all_coldkey_wallets = [
            bittensor.wallet(name=wallet_name) for wallet_name in all_wallet_names
        ]

        (
            all_hotkey_addresses,
            hotkey_coldkey_to_hotkey_wallet,
        ) = OverviewCommand._get_key_address(all_hotkeys)

        with console.status(
            ":satellite: Syncing with chain: [white]{}[/white] ...".format(
                cli.config.subtensor.get(
                    "network", bittensor.defaults.subtensor.network
                )
            )
        ):
            # Create a copy of the config without the parser and formatter_class.
            ## This is needed to pass to the ProcessPoolExecutor, which cannot pickle the parser.
            copy_config = cli.config.copy()
            copy_config["__parser"] = None
            copy_config["formatter_class"] = None

            # Pull neuron info for all keys.
            ## Max len(netuids) or 5 threads.
            with ProcessPoolExecutor(max_workers=max(len(netuids), 5)) as executor:
                results = executor.map(
                    OverviewCommand._get_neurons_for_netuid,
                    [(copy_config, netuid, all_hotkey_addresses) for netuid in netuids],
                )
                executor.shutdown(wait=True)  # wait for all complete

                neurons = OverviewCommand._process_neuron_results(
                    results, neurons, netuids
                )

            total_coldkey_stake_from_metagraph = defaultdict(
                lambda: bittensor.Balance(0.0)
            )
            checked_hotkeys = set()
            for neuron_list in neurons.values():
                for neuron in neuron_list:
                    if neuron.hotkey in checked_hotkeys:
                        continue
                    total_coldkey_stake_from_metagraph[
                        neuron.coldkey
                    ] += neuron.stake_dict[neuron.coldkey]
                    checked_hotkeys.add(neuron.hotkey)

            alerts_table = Table(show_header=True, header_style="bold magenta")
            alerts_table.add_column("🥩 alert!")

            coldkeys_to_check = []
            for coldkey_wallet in all_coldkey_wallets:
                # Check if we have any stake with hotkeys that are not registered.
                total_coldkey_stake_from_chain = subtensor.get_total_stake_for_coldkey(
                    ss58_address=coldkey_wallet.coldkeypub.ss58_address
                )
                difference = (
                    total_coldkey_stake_from_chain
                    - total_coldkey_stake_from_metagraph[
                        coldkey_wallet.coldkeypub.ss58_address
                    ]
                )
                if difference == 0:
                    continue  # We have all our stake registered.

                coldkeys_to_check.append(coldkey_wallet)
                alerts_table.add_row(
                    "Found {} stake with coldkey {} that is not registered.".format(
                        difference, coldkey_wallet.coldkeypub.ss58_address
                    )
                )

            if coldkeys_to_check:
                # We have some stake that is not with a registered hotkey.
                if "-1" not in neurons:
                    neurons["-1"] = []

            # Use process pool to check each coldkey wallet for de-registered stake.
            with ProcessPoolExecutor(
                max_workers=max(len(coldkeys_to_check), 5)
            ) as executor:
                results = executor.map(
                    OverviewCommand._get_de_registered_stake_for_coldkey_wallet,
                    [
                        (cli.config, all_hotkey_addresses, coldkey_wallet)
                        for coldkey_wallet in coldkeys_to_check
                    ],
                )
                executor.shutdown(wait=True)  # wait for all complete

            for result in results:
                coldkey_wallet, de_registered_stake, err_msg = result
                if err_msg is not None:
                    console.print(err_msg)

                if len(de_registered_stake) == 0:
                    continue  # We have no de-registered stake with this coldkey.

                de_registered_neurons = []
                for hotkey_addr, our_stake in de_registered_stake:
                    # Make a neuron info lite for this hotkey and coldkey.
                    de_registered_neuron = bittensor.NeuronInfoLite._null_neuron()
                    de_registered_neuron.hotkey = hotkey_addr
                    de_registered_neuron.coldkey = (
                        coldkey_wallet.coldkeypub.ss58_address
                    )
                    de_registered_neuron.total_stake = bittensor.Balance(our_stake)

                    de_registered_neurons.append(de_registered_neuron)

                    # Add this hotkey to the wallets dict
                    wallet_ = bittensor.wallet(
                        name=wallet,
                    )
                    wallet_.hotkey_ss58 = hotkey_addr
                    wallet.hotkey_str = hotkey_addr[:5]  # Max length of 5 characters
                    # Indicates a hotkey not on local machine but exists in stake_info obj on-chain
                    if hotkey_coldkey_to_hotkey_wallet.get(hotkey_addr) is None:
                        hotkey_coldkey_to_hotkey_wallet[hotkey_addr] = {}
                    hotkey_coldkey_to_hotkey_wallet[hotkey_addr][
                        coldkey_wallet.coldkeypub.ss58_address
                    ] = wallet_

                # Add neurons to overview.
                neurons["-1"].extend(de_registered_neurons)

        # Setup outer table.
        grid = Table.grid(pad_edge=False)

        # If there are any alerts, add them to the grid
        if len(alerts_table.rows) > 0:
            grid.add_row(alerts_table)

        title: str = ""
        if not cli.config.get("all", d=None):
            title = "[bold white italic]Wallet - {}:{}".format(
                cli.config.wallet.name, wallet.coldkeypub.ss58_address
            )
        else:
            title = "[bold whit italic]All Wallets:"

        # Add title
        grid.add_row(Align(title, vertical="middle", align="center"))

        # Generate rows per netuid
        hotkeys_seen = set()
        total_neurons = 0
        total_stake = 0.0
        for netuid in netuids:
            subnet_tempo = subtensor.tempo(netuid=netuid)
            last_subnet = netuid == netuids[-1]
            TABLE_DATA = []
            total_rank = 0.0
            total_trust = 0.0
            total_consensus = 0.0
            total_validator_trust = 0.0
            total_incentive = 0.0
            total_dividends = 0.0
            total_emission = 0

            for nn in neurons[str(netuid)]:
                hotwallet = hotkey_coldkey_to_hotkey_wallet.get(nn.hotkey, {}).get(
                    nn.coldkey, None
                )
                if not hotwallet:
                    # Indicates a mismatch between what the chain says the coldkey
                    # is for this hotkey and the local wallet coldkey-hotkey pair
                    hotwallet = argparse.Namespace()
                    hotwallet.name = nn.coldkey[:7]
                    hotwallet.hotkey_str = nn.hotkey[:7]
                nn: bittensor.NeuronInfoLite
                uid = nn.uid
                active = nn.active
                stake = nn.total_stake.tao
                rank = nn.rank
                trust = nn.trust
                consensus = nn.consensus
                validator_trust = nn.validator_trust
                incentive = nn.incentive
                dividends = nn.dividends
                emission = int(nn.emission / (subnet_tempo + 1) * 1e9)
                last_update = int(block - nn.last_update)
                validator_permit = nn.validator_permit
                row = [
                    hotwallet.name,
                    hotwallet.hotkey_str,
                    str(uid),
                    str(active),
                    "{:.5f}".format(stake),
                    "{:.5f}".format(rank),
                    "{:.5f}".format(trust),
                    "{:.5f}".format(consensus),
                    "{:.5f}".format(incentive),
                    "{:.5f}".format(dividends),
                    "{:_}".format(emission),
                    "{:.5f}".format(validator_trust),
                    "*" if validator_permit else "",
                    str(last_update),
                    (
                        bittensor.utils.networking.int_to_ip(nn.axon_info.ip)
                        + ":"
                        + str(nn.axon_info.port)
                        if nn.axon_info.port != 0
                        else "[yellow]none[/yellow]"
                    ),
                    nn.hotkey,
                ]

                total_rank += rank
                total_trust += trust
                total_consensus += consensus
                total_incentive += incentive
                total_dividends += dividends
                total_emission += emission
                total_validator_trust += validator_trust

                if not (nn.hotkey, nn.coldkey) in hotkeys_seen:
                    # Don't double count stake on hotkey-coldkey pairs.
                    hotkeys_seen.add((nn.hotkey, nn.coldkey))
                    total_stake += stake

                # netuid -1 are neurons that are de-registered.
                if netuid != "-1":
                    total_neurons += 1

                TABLE_DATA.append(row)

            # Add subnet header
            if netuid == "-1":
                grid.add_row(f"Deregistered Neurons")
            else:
                grid.add_row(f"Subnet: [bold white]{netuid}[/bold white]")

            table = Table(
                show_footer=False,
                width=cli.config.get("width", None),
                pad_edge=False,
                box=None,
            )
            if last_subnet:
                table.add_column(
                    "[overline white]COLDKEY",
                    str(total_neurons),
                    footer_style="overline white",
                    style="bold white",
                )
                table.add_column(
                    "[overline white]HOTKEY",
                    str(total_neurons),
                    footer_style="overline white",
                    style="white",
                )
            else:
                # No footer for non-last subnet.
                table.add_column("[overline white]COLDKEY", style="bold white")
                table.add_column("[overline white]HOTKEY", style="white")
            table.add_column(
                "[overline white]UID",
                str(total_neurons),
                footer_style="overline white",
                style="yellow",
            )
            table.add_column(
                "[overline white]ACTIVE", justify="right", style="green", no_wrap=True
            )
            if last_subnet:
                table.add_column(
                    "[overline white]STAKE(\u03C4)",
                    "\u03C4{:.5f}".format(total_stake),
                    footer_style="overline white",
                    justify="right",
                    style="green",
                    no_wrap=True,
                )
            else:
                # No footer for non-last subnet.
                table.add_column(
                    "[overline white]STAKE(\u03C4)",
                    justify="right",
                    style="green",
                    no_wrap=True,
                )
            table.add_column(
                "[overline white]RANK",
                "{:.5f}".format(total_rank),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]TRUST",
                "{:.5f}".format(total_trust),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]CONSENSUS",
                "{:.5f}".format(total_consensus),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]INCENTIVE",
                "{:.5f}".format(total_incentive),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]DIVIDENDS",
                "{:.5f}".format(total_dividends),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]EMISSION(\u03C1)",
                "\u03C1{:_}".format(total_emission),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column(
                "[overline white]VTRUST",
                "{:.5f}".format(total_validator_trust),
                footer_style="overline white",
                justify="right",
                style="green",
                no_wrap=True,
            )
            table.add_column("[overline white]VPERMIT", justify="right", no_wrap=True)
            table.add_column("[overline white]UPDATED", justify="right", no_wrap=True)
            table.add_column(
                "[overline white]AXON", justify="left", style="dim blue", no_wrap=True
            )
            table.add_column(
                "[overline white]HOTKEY_SS58", style="dim blue", no_wrap=False
            )
            table.show_footer = True

            sort_by: Optional[str] = cli.config.get("sort_by", None)
            sort_order: Optional[str] = cli.config.get("sort_order", None)

            if sort_by is not None and sort_by != "":
                column_to_sort_by: int = 0
                highest_matching_ratio: int = 0
                sort_descending: bool = False  # Default sort_order to ascending

                for index, column in zip(range(len(table.columns)), table.columns):
                    # Fuzzy match the column name. Default to the first column.
                    column_name = column.header.lower().replace("[overline white]", "")
                    match_ratio = fuzz.ratio(sort_by.lower(), column_name)
                    # Finds the best matching column
                    if match_ratio > highest_matching_ratio:
                        highest_matching_ratio = match_ratio
                        column_to_sort_by = index

                if sort_order.lower() in {"desc", "descending", "reverse"}:
                    # Sort descending if the sort_order matches desc, descending, or reverse
                    sort_descending = True

                def overview_sort_function(row):
                    data = row[column_to_sort_by]
                    # Try to convert to number if possible
                    try:
                        data = float(data)
                    except ValueError:
                        pass
                    return data

                TABLE_DATA.sort(key=overview_sort_function, reverse=sort_descending)

            for row in TABLE_DATA:
                table.add_row(*row)

            grid.add_row(table)

        console.clear()

        caption = "[italic][dim][white]Wallet balance: [green]\u03C4" + str(
            total_balance.tao
        )
        grid.add_row(Align(caption, vertical="middle", align="center"))

        # Print the entire table/grid
        console.print(grid, width=cli.config.get("width", None))

    @staticmethod
    def _get_neurons_for_netuid(
        args_tuple: Tuple["bittensor.Config", int, List[str]]
    ) -> Tuple[int, List["bittensor.NeuronInfoLite"], Optional[str]]:
        subtensor_config, netuid, hot_wallets = args_tuple

        result: List["bittensor.NeuronInfoLite"] = []

        try:
            subtensor = bittensor.subtensor(config=subtensor_config, log_verbose=False)

            all_neurons: List["bittensor.NeuronInfoLite"] = subtensor.neurons_lite(
                netuid=netuid
            )
            # Map the hotkeys to uids
            hotkey_to_neurons = {n.hotkey: n.uid for n in all_neurons}
            for hot_wallet_addr in hot_wallets:
                uid = hotkey_to_neurons.get(hot_wallet_addr)
                if uid is not None:
                    nn = all_neurons[uid]
                    result.append(nn)
        except Exception as e:
            return netuid, [], "Error: {}".format(e)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

        return netuid, result, None

    @staticmethod
    def _get_de_registered_stake_for_coldkey_wallet(
        args_tuple,
    ) -> Tuple[
        "bittensor.Wallet", List[Tuple[str, "bittensor.Balance"]], Optional[str]
    ]:
        subtensor_config, all_hotkey_addresses, coldkey_wallet = args_tuple

        # List of (hotkey_addr, our_stake) tuples.
        result: List[Tuple[str, "bittensor.Balance"]] = []

        try:
            subtensor = bittensor.subtensor(config=subtensor_config, log_verbose=False)

            # Pull all stake for our coldkey
            all_stake_info_for_coldkey = subtensor.get_stake_info_for_coldkey(
                coldkey_ss58=coldkey_wallet.coldkeypub.ss58_address
            )

            ## Filter out hotkeys that are in our wallets
            ## Filter out hotkeys that are delegates.
            def _filter_stake_info(stake_info: "bittensor.StakeInfo") -> bool:
                if stake_info.stake == 0:
                    return False  # Skip hotkeys that we have no stake with.
                if stake_info.hotkey_ss58 in all_hotkey_addresses:
                    return False  # Skip hotkeys that are in our wallets.
                if subtensor.is_hotkey_delegate(hotkey_ss58=stake_info.hotkey_ss58):
                    return False  # Skip hotkeys that are delegates, they show up in btcli my_delegates table.

                return True

            all_staked_hotkeys = filter(_filter_stake_info, all_stake_info_for_coldkey)
            result = [
                (
                    stake_info.hotkey_ss58,
                    stake_info.stake.tao,
                )  # stake is a Balance object
                for stake_info in all_staked_hotkeys
            ]

        except Exception as e:
            return coldkey_wallet, [], "Error: {}".format(e)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

        return coldkey_wallet, result, None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        overview_parser = parser.add_parser(
            "overview", help="""Show registered account overview."""
        )
        overview_parser.add_argument(
            "--all",
            dest="all",
            action="store_true",
            help="""View overview for all wallets.""",
            default=False,
        )
        overview_parser.add_argument(
            "--width",
            dest="width",
            action="store",
            type=int,
            help="""Set the output width of the overview. Defaults to automatic width from terminal.""",
            default=None,
        )
        overview_parser.add_argument(
            "--sort_by",
            "--wallet.sort_by",
            dest="sort_by",
            required=False,
            action="store",
            default="",
            type=str,
            help="""Sort the hotkeys by the specified column title (e.g. name, uid, axon).""",
        )
        overview_parser.add_argument(
            "--sort_order",
            "--wallet.sort_order",
            dest="sort_order",
            required=False,
            action="store",
            default="ascending",
            type=str,
            help="""Sort the hotkeys in the specified ordering. (ascending/asc or descending/desc/reverse)""",
        )
        overview_parser.add_argument(
            "--hotkeys",
            "--exclude_hotkeys",
            "--wallet.hotkeys",
            "--wallet.exclude_hotkeys",
            required=False,
            action="store",
            default=[],
            type=str,
            nargs="*",
            help="""Specify the hotkeys by name or ss58 address. (e.g. hk1 hk2 hk3)""",
        )
        overview_parser.add_argument(
            "--all_hotkeys",
            "--wallet.all_hotkeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all hotkeys. Specifying hotkeys will exclude them from this all.""",
        )
        overview_parser.add_argument(
            "--netuids",
            dest="netuids",
            type=int,
            nargs="*",
            help="""Set the netuid(s) to filter by.""",
            default=None,
        )
        bittensor.wallet.add_args(overview_parser)
        bittensor.subtensor.add_args(overview_parser)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if (
            not config.is_set("wallet.name")
            and not config.no_prompt
            and not config.get("all", d=None)
        ):
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.netuids != [] and config.netuids != None:
            if not isinstance(config.netuids, list):
                config.netuids = [int(config.netuids)]
            else:
                config.netuids = [int(netuid) for netuid in config.netuids]
