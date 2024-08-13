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
from typing import List, Union, Optional, Tuple

from rich.prompt import Confirm, Prompt
from tqdm import tqdm

import bittensor
from bittensor.utils.balance import Balance
from . import defaults, GetChildrenCommand
from .utils import get_hotkey_wallets_for_wallet

console = bittensor.__console__


class UnStakeCommand:
    """
    Executes the ``remove`` command to unstake TAO tokens from one or more hotkeys and transfer them back to the user's coldkey on the Bittensor network.

    This command is used to withdraw tokens previously staked to different hotkeys.

    Usage:
        Users can specify the amount to unstake, the hotkeys to unstake from (either by name or ``SS58`` address), and whether to unstake from all hotkeys. The command checks for sufficient stake and prompts for confirmation before proceeding with the unstaking process.

    Optional arguments:
        - ``--all`` (bool): When set, unstakes all staked tokens from the specified hotkeys.
        - ``--amount`` (float): The amount of TAO tokens to unstake.
        - --hotkey_ss58address (str): The SS58 address of the hotkey to unstake from.
        - ``--max_stake`` (float): Sets the maximum amount of TAO to remain staked in each hotkey.
        - ``--hotkeys`` (list): Specifies hotkeys by name or SS58 address to unstake from.
        - ``--all_hotkeys`` (bool): When set, unstakes from all hotkeys associated with the wallet, excluding any specified in --hotkeys.

    The command prompts for confirmation before executing the unstaking operation.

    Example usage::

        btcli stake remove --amount 100 --hotkeys hk1,hk2

    Note:
        This command is important for users who wish to reallocate their stakes or withdraw them from the network.
        It allows for flexible management of token stakes across different neurons (hotkeys) on the network.
    """

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.get("hotkey_ss58address", d=None)
            and not config.is_set("wallet.hotkey")
            and not config.no_prompt
            and not config.get("all_hotkeys")
            and not config.get("hotkeys")
        ):
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        # Get amount.
        if (
            not config.get("hotkey_ss58address")
            and not config.get("amount")
            and not config.get("unstake_all")
            and not config.get("max_stake")
        ):
            hotkeys: str = ""
            if config.get("all_hotkeys"):
                hotkeys = "all hotkeys"
            elif config.get("hotkeys"):
                hotkeys = str(config.hotkeys).replace("[", "").replace("]", "")
            else:
                hotkeys = str(config.wallet.hotkey)
            if config.no_prompt:
                config.unstake_all = True
            else:
                # I really don't like this logic flow. It can be a bit confusing to read for something
                # as serious as unstaking all.
                if Confirm.ask(f"Unstake all Tao from: [bold]'{hotkeys}'[/bold]?"):
                    config.unstake_all = True
                else:
                    config.unstake_all = False
                    amount = Prompt.ask("Enter Tao amount to unstake")
                    try:
                        config.amount = float(amount)
                    except ValueError:
                        console.print(
                            f":cross_mark:[red] Invalid Tao amount[/red] [bold white]{amount}[/bold white]"
                        )
                        sys.exit()

    @staticmethod
    def add_args(command_parser):
        unstake_parser = command_parser.add_parser(
            "remove",
            help="""Remove stake from the specified hotkey into the coldkey balance.""",
        )
        unstake_parser.add_argument(
            "--all", dest="unstake_all", action="store_true", default=False
        )
        unstake_parser.add_argument(
            "--amount", dest="amount", type=float, required=False
        )
        unstake_parser.add_argument(
            "--hotkey_ss58address", dest="hotkey_ss58address", type=str, required=False
        )
        unstake_parser.add_argument(
            "--max_stake",
            dest="max_stake",
            type=float,
            required=False,
            action="store",
            default=None,
            help="""Specify the maximum amount of Tao to have staked in each hotkey.""",
        )
        unstake_parser.add_argument(
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
        unstake_parser.add_argument(
            "--all_hotkeys",
            "--wallet.all_hotkeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all hotkeys. Specifying hotkeys will exclude them from this all.""",
        )
        bittensor.wallet.add_args(unstake_parser)
        bittensor.subtensor.add_args(unstake_parser)

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Unstake token of amount from hotkey(s)."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            UnStakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Unstake token of amount from hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        # Get the hotkey_names (if any) and the hotkey_ss58s.
        hotkeys_to_unstake_from: List[Tuple[Optional[str], str]] = []
        if cli.config.get("hotkey_ss58address"):
            # Stake to specific hotkey.
            hotkeys_to_unstake_from = [(None, cli.config.get("hotkey_ss58address"))]
        elif cli.config.get("all_hotkeys"):
            # Stake to all hotkeys.
            all_hotkeys: List[bittensor.wallet] = get_hotkey_wallets_for_wallet(
                wallet=wallet
            )
            # Get the hotkeys to exclude. (d)efault to no exclusions.
            hotkeys_to_exclude: List[str] = cli.config.get("hotkeys", d=[])
            # Exclude hotkeys that are specified.
            hotkeys_to_unstake_from = [
                (wallet.hotkey_str, wallet.hotkey.ss58_address)
                for wallet in all_hotkeys
                if wallet.hotkey_str not in hotkeys_to_exclude
            ]  # definitely wallets

        elif cli.config.get("hotkeys"):
            # Stake to specific hotkeys.
            for hotkey_ss58_or_hotkey_name in cli.config.get("hotkeys"):
                if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_hotkey_name):
                    # If the hotkey is a valid ss58 address, we add it to the list.
                    hotkeys_to_unstake_from.append((None, hotkey_ss58_or_hotkey_name))
                else:
                    # If the hotkey is not a valid ss58 address, we assume it is a hotkey name.
                    #  We then get the hotkey from the wallet and add it to the list.
                    wallet_ = bittensor.wallet(
                        config=cli.config, hotkey=hotkey_ss58_or_hotkey_name
                    )
                    hotkeys_to_unstake_from.append(
                        (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                    )
        elif cli.config.wallet.get("hotkey"):
            # Only cli.config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            hotkey_ss58_or_name = cli.config.wallet.get("hotkey")
            if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_name):
                hotkeys_to_unstake_from = [(None, hotkey_ss58_or_name)]
            else:
                # Hotkey is not a valid ss58 address, so we assume it is a hotkey name.
                wallet_ = bittensor.wallet(
                    config=cli.config, hotkey=hotkey_ss58_or_name
                )
                hotkeys_to_unstake_from = [
                    (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                ]
        else:
            # Only cli.config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            assert cli.config.wallet.hotkey is not None
            hotkeys_to_unstake_from = [
                (None, bittensor.wallet(config=cli.config).hotkey.ss58_address)
            ]

        final_hotkeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for hotkey in tqdm(hotkeys_to_unstake_from):
            hotkey: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)
            unstake_amount_tao: float = cli.config.get(
                "amount"
            )  # The amount specified to unstake.
            hotkey_stake: Balance = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=hotkey[1], coldkey_ss58=wallet.coldkeypub.ss58_address
            )
            if unstake_amount_tao == None:
                unstake_amount_tao = hotkey_stake.tao
            if cli.config.get("max_stake"):
                # Get the current stake of the hotkey from this coldkey.
                unstake_amount_tao: float = hotkey_stake.tao - cli.config.get(
                    "max_stake"
                )
                cli.config.amount = unstake_amount_tao
                if unstake_amount_tao < 0:
                    # Skip if max_stake is greater than current stake.
                    continue
            else:
                if unstake_amount_tao is not None:
                    # There is a specified amount to unstake.
                    if unstake_amount_tao > hotkey_stake.tao:
                        # Skip if the specified amount is greater than the current stake.
                        continue

            final_amounts.append(unstake_amount_tao)
            final_hotkeys.append(hotkey)  # add both the name and the ss58 address.

        if len(final_hotkeys) == 0:
            # No hotkeys to unstake from.
            bittensor.__console__.print(
                "Not enough stake to unstake from any hotkeys or max_stake is more than current stake."
            )
            return None

        # Ask to unstake
        if not cli.config.no_prompt:
            if not Confirm.ask(
                f"Do you want to unstake from the following keys to {wallet.name}:\n"
                + "".join(
                    [
                        f"    [bold white]- {hotkey[0] + ':' if hotkey[0] else ''}{hotkey[1]}: {f'{amount} {bittensor.__tao_symbol__}' if amount else 'All'}[/bold white]\n"
                        for hotkey, amount in zip(final_hotkeys, final_amounts)
                    ]
                )
            ):
                return None

        if len(final_hotkeys) == 1:
            # do regular unstake
            return subtensor.unstake(
                wallet=wallet,
                hotkey_ss58=final_hotkeys[0][1],
                amount=None if cli.config.get("unstake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not cli.config.no_prompt,
            )

        subtensor.unstake_multiple(
            wallet=wallet,
            hotkey_ss58s=[hotkey_ss58 for _, hotkey_ss58 in final_hotkeys],
            amounts=None if cli.config.get("unstake_all") else final_amounts,
            wait_for_inclusion=True,
            prompt=False,
        )


class RevokeChildrenCommand:
    """
    Executes the ``revoke_children`` command to remove all children hotkeys on a specified subnet on the Bittensor network.

    This command is used to remove delegated authority from all child hotkeys, removing their position and influence on the subnet.

    Usage:
        Users need to specify the parent hotkey and the subnet ID (netuid).
        The user needs to have sufficient authority to make this call.

    The command prompts for confirmation before executing the revoke_children operation.

    Example usage::

        btcli stake revoke_children --hotkey <parent_hotkey> --netuid 1

    Note:
        This command is critical for users who wish to remove children hotkeys on the network.
        It allows for a complete removal of delegated authority to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        """Revokes all children hotkeys."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            RevokeChildrenCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        wallet = bittensor.wallet(config=cli.config)

        # Get values if not set.
        if not cli.config.is_set("netuid"):
            cli.config.netuid = int(Prompt.ask("Enter netuid"))

        if not cli.config.is_set("hotkey"):
            cli.config.hotkey = Prompt.ask("Enter parent hotkey (ss58)")

        # Get and display current children information
        current_children = GetChildrenCommand.retrieve_children(
            subtensor=subtensor,
            hotkey=cli.config.hotkey,
            netuid=cli.config.netuid,
            render_table=False,
        )

        # Parse from strings
        netuid = cli.config.netuid

        # Prepare children with zero proportions
        children_with_zero_proportions = [(0.0, child[1]) for child in current_children]

        success, message = subtensor.set_children(
            wallet=wallet,
            netuid=netuid,
            children_with_proportions=children_with_zero_proportions,
            hotkey=cli.config.hotkey,
            wait_for_inclusion=cli.config.wait_for_inclusion,
            wait_for_finalization=cli.config.wait_for_finalization,
            prompt=cli.config.prompt,
        )

        # Result
        if success:
            if cli.config.wait_for_finalization and cli.config.wait_for_inclusion:
                GetChildrenCommand.retrieve_children(
                    subtensor=subtensor,
                    hotkey=cli.config.hotkey,
                    netuid=cli.config.netuid,
                    render_table=True,
                )
            console.print(
                ":white_heavy_check_mark: [green]Revoked all children hotkeys.[/green]"
            )
        else:
            console.print(
                f":cross_mark:[red] Unable to revoke children hotkeys.[/red] {message}"
            )

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser = parser.add_parser(
            "revoke_children", help="""Revoke all children hotkeys."""
        )
        parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        parser.add_argument("--hotkey", dest="hotkey", type=str, required=False)
        parser.add_argument(
            "--wait_for_inclusion",
            dest="wait_for_inclusion",
            action="store_true",
            default=False,
            help="""Wait for the transaction to be included in a block.""",
        )
        parser.add_argument(
            "--wait_for_finalization",
            dest="wait_for_finalization",
            action="store_true",
            default=False,
            help="""Wait for the transaction to be finalized.""",
        )
        parser.add_argument(
            "--prompt",
            dest="prompt",
            action="store_true",
            default=False,
            help="""Prompt for confirmation before proceeding.""",
        )
        bittensor.wallet.add_args(parser)
        bittensor.subtensor.add_args(parser)
