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
import os
import sys
from typing import List, Union, Optional, Tuple

from rich.prompt import Confirm, Prompt
from tqdm import tqdm

import bittensor
from bittensor.utils.balance import Balance
from .. import defaults  # type: ignore
from ..utils import (
    get_hotkey_wallets_for_wallet,
)

console = bittensor.__console__


class StakeCommand:
    """
    Executes the ``add`` command to stake tokens to one or more hotkeys from a user's coldkey on the Bittensor network.

    This command is used to allocate tokens to different hotkeys, securing their position and influence on the network.

    Usage:
        Users can specify the amount to stake, the hotkeys to stake to (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the staking process.

    Optional arguments:
        - ``--all`` (bool): When set, stakes all available tokens from the coldkey.
        - ``--uid`` (int): The unique identifier of the neuron to which the stake is to be added.
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--max_stake`` (float): Sets the maximum amount of TAO to have staked in each hotkey.
        - ``--hotkeys`` (list): Specifies hotkeys by name or SS58 address to stake to.
        - ``--all_hotkeys`` (bool): When set, stakes to all hotkeys associated with the wallet, excluding any specified in --hotkeys.

    The command prompts for confirmation before executing the staking operation.

    Example usage::

        btcli stake add --amount 100 --wallet.name <my_wallet> --wallet.hotkey <my_hotkey>

    Note:
        This command is critical for users who wish to distribute their stakes among different neurons (hotkeys) on the network.
        It allows for a strategic allocation of tokens to enhance network participation and influence.
    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Stake token of amount to hotkey(s)."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            StakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Stake token of amount to hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        # Get the hotkey_names (if any) and the hotkey_ss58s.
        hotkeys_to_stake_to: List[Tuple[Optional[str], str]] = []
        if config.get("all_hotkeys"):
            # Stake to all hotkeys.
            all_hotkeys: List[bittensor.wallet] = get_hotkey_wallets_for_wallet(
                wallet=wallet
            )
            # Get the hotkeys to exclude. (d)efault to no exclusions.
            hotkeys_to_exclude: List[str] = cli.config.get("hotkeys", d=[])
            # Exclude hotkeys that are specified.
            hotkeys_to_stake_to = [
                (wallet.hotkey_str, wallet.hotkey.ss58_address)
                for wallet in all_hotkeys
                if wallet.hotkey_str not in hotkeys_to_exclude
            ]  # definitely wallets

        elif config.get("hotkeys"):
            # Stake to specific hotkeys.
            for hotkey_ss58_or_hotkey_name in config.get("hotkeys"):
                if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_hotkey_name):
                    # If the hotkey is a valid ss58 address, we add it to the list.
                    hotkeys_to_stake_to.append((None, hotkey_ss58_or_hotkey_name))
                else:
                    # If the hotkey is not a valid ss58 address, we assume it is a hotkey name.
                    #  We then get the hotkey from the wallet and add it to the list.
                    wallet_ = bittensor.wallet(
                        config=config, hotkey=hotkey_ss58_or_hotkey_name
                    )
                    hotkeys_to_stake_to.append(
                        (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                    )
        elif config.wallet.get("hotkey"):
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            hotkey_ss58_or_name = config.wallet.get("hotkey")
            if bittensor.utils.is_valid_ss58_address(hotkey_ss58_or_name):
                hotkeys_to_stake_to = [(None, hotkey_ss58_or_name)]
            else:
                # Hotkey is not a valid ss58 address, so we assume it is a hotkey name.
                wallet_ = bittensor.wallet(config=config, hotkey=hotkey_ss58_or_name)
                hotkeys_to_stake_to = [
                    (wallet_.hotkey_str, wallet_.hotkey.ss58_address)
                ]
        else:
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            assert config.wallet.hotkey is not None
            hotkeys_to_stake_to = [
                (None, bittensor.wallet(config=config).hotkey.ss58_address)
            ]

        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        final_hotkeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for hotkey in tqdm(hotkeys_to_stake_to):
            hotkey: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)
            if not subtensor.is_hotkey_registered_any(hotkey_ss58=hotkey[1]):
                # Hotkey is not registered.
                if len(hotkeys_to_stake_to) == 1:
                    # Only one hotkey, error
                    bittensor.__console__.print(
                        f"[red]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Aborting.[/red]"
                    )
                    return None
                else:
                    # Otherwise, print warning and skip
                    bittensor.__console__.print(
                        f"[yellow]Hotkey [bold]{hotkey[1]}[/bold] is not registered. Skipping.[/yellow]"
                    )
                    continue

            stake_amount_tao: float = config.get("amount")
            if config.get("max_stake"):
                # Get the current stake of the hotkey from this coldkey.
                hotkey_stake: Balance = subtensor.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=hotkey[1], coldkey_ss58=wallet.coldkeypub.ss58_address
                )
                stake_amount_tao: float = config.get("max_stake") - hotkey_stake.tao

                # If the max_stake is greater than the current wallet balance, stake the entire balance.
                stake_amount_tao: float = min(stake_amount_tao, wallet_balance.tao)
                if (
                    stake_amount_tao <= 0.00001
                ):  # Threshold because of fees, might create a loop otherwise
                    # Skip hotkey if max_stake is less than current stake.
                    continue
                wallet_balance = Balance.from_tao(wallet_balance.tao - stake_amount_tao)

                if wallet_balance.tao < 0:
                    # No more balance to stake.
                    break

            final_amounts.append(stake_amount_tao)
            final_hotkeys.append(hotkey)  # add both the name and the ss58 address.

        if len(final_hotkeys) == 0:
            # No hotkeys to stake to.
            bittensor.__console__.print(
                "Not enough balance to stake to any hotkeys or max_stake is less than current stake."
            )
            return None

        # Ask to stake
        if not config.no_prompt:
            if not Confirm.ask(
                f"Do you want to stake to the following keys from {wallet.name}:\n"
                + "".join(
                    [
                        f"    [bold white]- {hotkey[0] + ':' if hotkey[0] else ''}{hotkey[1]}: {f'{amount} {bittensor.__tao_symbol__}' if amount else 'All'}[/bold white]\n"
                        for hotkey, amount in zip(final_hotkeys, final_amounts)
                    ]
                )
            ):
                return None

        if len(final_hotkeys) == 1:
            # do regular stake
            return subtensor.add_stake(
                wallet=wallet,
                hotkey_ss58=final_hotkeys[0][1],
                amount=None if config.get("stake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )

        subtensor.add_stake_multiple(
            wallet=wallet,
            hotkey_ss58s=[hotkey_ss58 for _, hotkey_ss58 in final_hotkeys],
            amounts=None if config.get("stake_all") else final_amounts,
            wait_for_inclusion=True,
            prompt=False,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.is_set("wallet.hotkey")
            and not config.no_prompt
            and not config.wallet.get("all_hotkeys")
            and not config.wallet.get("hotkeys")
        ):
            hotkey = Prompt.ask("Enter hotkey name", default=defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        # Get amount.
        if (
            not config.get("amount")
            and not config.get("stake_all")
            and not config.get("max_stake")
        ):
            if not Confirm.ask(
                "Stake all Tao from account: [bold]'{}'[/bold]?".format(
                    config.wallet.get("name", defaults.wallet.name)
                )
            ):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(
                        ":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(
                            amount
                        )
                    )
                    sys.exit()
            else:
                config.stake_all = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "add", help="""Add stake to your hotkey accounts from your coldkey."""
        )
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
        stake_parser.add_argument("--uid", dest="uid", type=int, required=False)
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--max_stake",
            dest="max_stake",
            type=float,
            required=False,
            action="store",
            default=None,
            help="""Specify the maximum amount of Tao to have staked in each hotkey.""",
        )
        stake_parser.add_argument(
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
        stake_parser.add_argument(
            "--all_hotkeys",
            "--wallet.all_hotkeys",
            required=False,
            action="store_true",
            default=False,
            help="""To specify all hotkeys. Specifying hotkeys will exclude them from this all.""",
        )
        bittensor.wallet.add_args(stake_parser)
        bittensor.subtensor.add_args(stake_parser)


def _get_coldkey_wallets_for_path(path: str) -> List["bittensor.wallet"]:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [bittensor.wallet(path=path, name=name) for name in wallet_names]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets


def _get_hotkey_wallets_for_wallet(wallet) -> List["bittensor.wallet"]:
    hotkey_wallets = []
    hotkeys_path = wallet.path + "/" + wallet.name + "/hotkeys"
    try:
        hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
    except StopIteration:
        hotkey_files = []
    for hotkey_file_name in hotkey_files:
        try:
            hotkey_for_name = bittensor.wallet(
                path=wallet.path, name=wallet.name, hotkey=hotkey_file_name
            )
            if (
                hotkey_for_name.hotkey_file.exists_on_device()
                and not hotkey_for_name.hotkey_file.is_encrypted()
            ):
                hotkey_wallets.append(hotkey_for_name)
        except Exception:
            pass
    return hotkey_wallets
