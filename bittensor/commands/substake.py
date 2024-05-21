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
import bittensor
from rich.prompt import Confirm, Prompt
from bittensor.commands.utils import get_hotkey_wallets_for_wallet
from bittensor.utils.balance import Balance
from typing import Optional, Tuple, List, Union
from tqdm import tqdm
from . import defaults

console = bittensor.__console__


class SubStakeCommand:
    """
        Adds stake to a specific hotkey account on a specific subnet, specified by `netuid`.

         Usage:
        Users can specify the amount to stake, the hotkeys to stake to (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the staking process.

    Optional arguments:
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--netuid`` (int): The subnet to stake to.
        - ``--max_stake`` (float): Sets the maximum amount of TAO to have staked in each hotkey.
        - ``--hotkey`` (str): Specifies hotkey by name or SS58 address to stake to.

    The command prompts for confirmation before executing the staking operation.

    Example usage (args):

        btcli substake add --amount <tao amt to stake> --wallet.name <wallet to pull tao from> --hotkey <ss58_address stake destination>

        btcli substake add --amount 100 --netuid 1 --wallet.name default --hotkey 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

    Example usage (prompt):

        btcli substake add

        Enter netuid (0): 1

        Enter wallet name (default):

        Enter hotkey name or ss58_address to stake to (default):

        Stake all Tao from account: 'default'? [y/n]: n

        Enter Tao amount to stake: 100

        Do you want to stake to the following hotkey on netuid 1:
          - from   default:5GeYLB44QY9wcqJmFZvJW8D3EYPDaJGSgGfkbJVxUbkVcU7C
          - to     default:5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH
          - amount 100.0 τ
          - for:
        [y/n]: y
        Enter password to unlock key:
        ✅ Finalized

        Balance:
          τ100 ➡ τ0
        Stake:
          τ1,234 ➡ τ1,334

    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Stake token of amount to hotkey on subnet of given netuid."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            SubStakeCommand._run(cli, subtensor)
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
            print(f"All hotkeys 1: {all_hotkeys}")
            print(f"Mock wallets in _run: {wallet}")
            # Get the hotkeys to exclude. (d)efault to no exclusions.
            hotkeys_to_exclude: List[str] = cli.config.get("hotkeys", d=[])
            # Exclude hotkeys that are specified.
            hotkeys_to_stake_to = [
                (wallet.hotkey_str, wallet.hotkey.ss58_address)
                for wallet in all_hotkeys
                if wallet.hotkey_str not in hotkeys_to_exclude
            ]  # definitely wallets
            print(f"Hotkeys to stake to 1: {hotkeys_to_stake_to}")
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
            print(f"Hotkeys to stake to 2: {hotkeys_to_stake_to}")
        else:
            # Only config.wallet.hotkey is specified.
            #  so we stake to that single hotkey.
            assert config.wallet.hotkey is not None
            hotkeys_to_stake_to = [
                (None, bittensor.wallet(config=config).hotkey.ss58_address)
            ]
            print(f"Hotkeys to stake to 3: {hotkeys_to_stake_to}")

        # Get coldkey balance
        wallet_balance: Balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        print(f"Hotkeys to stake to: {hotkeys_to_stake_to}")
        print(f"Wallet balance: {wallet_balance}")
        final_hotkeys: List[Tuple[str, str]] = []
        final_amounts: List[Union[float, Balance]] = []
        for hotkey in tqdm(hotkeys_to_stake_to):
            hotkey_balance = subtensor.get_balance(hotkey[1])
            print(f"Hotkey balance: {hotkey_balance}")
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
                hotkey_stake: Balance = (
                    subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                        hotkey_ss58=hotkey[1],
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        netuid=config.netuid,
                    )
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
                print(f"Wallet balance: {wallet_balance}")

                if wallet_balance.tao < 0:
                    # No more balance to stake.
                    break

            final_amounts.append(stake_amount_tao)
            final_hotkeys.append(hotkey)

        if len(final_hotkeys) == 0:
            # No hotkeys to stake to.
            bittensor.__console__.print(
                "Not enough balance to stake to any hotkeys or max_stake is less than current stake."
            )
            return None

        # Ask to stake
        if not config.no_prompt:
            if not Confirm.ask(
                f"Do you want to stake to the following keys from {wallet.name} on netuid {config.netuid}:\n"
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
            return subtensor.add_substake(
                wallet=wallet,
                hotkey_ss58=final_hotkeys[0][1],
                netuid=config.netuid,
                amount=None if config.get("stake_all") else final_amounts[0],
                wait_for_inclusion=True,
                prompt=not config.no_prompt,
            )
        else:
            return subtensor.add_substake_multiple(
                wallet=wallet,
                hotkey_ss58s=[hotkey_ss58 for _, hotkey_ss58 in final_hotkeys],
                netuid=config.netuid,
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
            not config.is_set("hotkey")
            and not config.is_set("wallet.hotkey")
            and not config.wallet.get("hotkey")
            and not config.no_prompt
        ):
            hotkey = Prompt.ask(
                "Enter hotkey name or ss58_address to stake to",
                default=defaults.wallet.hotkey,
            )
            if bittensor.is_valid_ss58_address(hotkey):
                config.hotkey = str(hotkey)
            else:
                config.wallet.hotkey = str(hotkey)

        if not config.is_set("netuid") and not config.no_prompt:
            netuid = Prompt.ask("Enter netuid", default="0")
            config.netuid = int(netuid)

        # Get amount.
        if not config.get("amount") and not config.get("max_stake"):
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
            "add",
            help="""Add stake to specific hotkeys on subnet `netuid` from your coldkey.""",
        )
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--all", dest="stake_all", action="store_true")
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


class RemoveSubStakeCommand:
    """
    Removes stake to a specific hotkey account on a specific subnet, specified by `netuid`.

    Usage:
        Users can specify the amount to unstake, the hotkey to unstake from (either by name or ``SS58`` address), and whether to stake to all hotkeys. The command checks for sufficient balance and hotkey registration
        before proceeding with the unstaking process.

    Optional arguments:
        - ``--wallet.name (str)``: The wallet return TAO to.
        - ``--amount`` (float): The amount of TAO tokens to stake.
        - ``--netuid`` (int): The subnet to stake to.
        - ``--hotkey`` (str): Specifies hotkey by name or SS58 address to stake to.
        - ``--all`` (bool): Unstake all TAO from the specified hotkey.

    The command prompts for confirmation before executing the staking operation.

    Example usage (args):

        btcli substake remove --amount <tao amt to unstake> --wallet.name <wallet coldkey to return tao to> --hotkey <ss58_address to unstake from>

        btcli substake remove --amount 111 --wallet.name default --netuid 1 --hotkey 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

    Example usage (prompt):

        btcli substake remove

        Enter netuid (0): 1

        Enter wallet name (default):

        Enter hotkey name or ss58_address to unstake from (default): 5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH

        Unstake all τao
        from account: 'default'
        and hotkey  : '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'
        from subnet : '1'
        [y/n]: n

        Enter Tao amount to unstake: 1337

        Enter password to unlock key:

        Do you want to unstake:
            amount: τ500
            from  : default
            netuid: 1
        [y/n]: y

        ✅ Finalized
        Balance:
            τ5000 ➡ τ6337
        Unstaked:
            τ16,961 ➡ τ15,623

    """

    @staticmethod
    def run(cli: "bittensor.cli"):
        r"""Unstake token of amount to hotkey on subnet of given netuid."""
        try:
            config = cli.config.copy()
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=config, log_verbose=False
            )
            RemoveSubStakeCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Unstake token amount from hotkey(s)."""
        config = cli.config.copy()
        wallet = bittensor.wallet(config=config)

        hotkey_tup: Tuple[Optional[str], str]  # (hotkey_name (or None), hotkey_ss58)

        if config.is_set("hotkey"):
            assert bittensor.is_valid_ss58_address(config.get("hotkey"))
            hotkey_tup = (None, config.get("hotkey"))
        else:
            wallet_ = bittensor.wallet(
                config=config, hotkey=config.wallet.get("hotkey")
            )
            hotkey_tup = (wallet_.hotkey_str, wallet_.hotkey.ss58_address)

        # Get coldkey balance
        if not subtensor.is_hotkey_registered_any(hotkey_ss58=hotkey_tup[1]):
            # Hotkey is not registered.
            bittensor.__console__.print(
                f"[red]Hotkey [bold]{hotkey_tup[1]}[/bold] is not registered. Aborting.[/red]"
            )
            return None

        # Calculate if able to unstake amount desired
        unstake_amount_tao: float = config.get("amount")

        # Get the current stake of the hotkey from this coldkey.
        hotkey_subnet_balance: Balance = (
            subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                hotkey_ss58=hotkey_tup[1],
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                netuid=config.netuid,
            )
        )

        # If we are unstaking all set that value to the amount currently on that account.
        if config.get("unstake_all"):
            unstake_amount_tao = hotkey_subnet_balance.tao

        balance_after_unstake_tao: float = (
            hotkey_subnet_balance.tao - unstake_amount_tao
        )
        if balance_after_unstake_tao < 0:
            bittensor.__console__.print(
                f"Unstake amount {unstake_amount_tao} is greater than current stake for hotkey [bold]{hotkey_tup[1]}[/bold]. Unstaking all."
            )
            unstake_amount_tao = hotkey_subnet_balance.tao

        return subtensor.remove_substake(
            wallet=wallet,
            hotkey_ss58=hotkey_tup[1],
            netuid=config.netuid,
            amount=unstake_amount_tao,
            wait_for_inclusion=True,
            prompt=not config.no_prompt,
        )

    @classmethod
    def check_config(cls, config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default=defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if (
            not config.is_set("hotkey")
            and not config.is_set("wallet.hotkey")
            and not config.wallet.get("hotkey")
            and not config.no_prompt
        ):
            hotkey = Prompt.ask(
                "Enter hotkey name or ss58_address to unstake from",
                default=defaults.wallet.hotkey,
            )
            if bittensor.is_valid_ss58_address(hotkey):
                config.hotkey = str(hotkey)
            else:
                config.hotkey = str(hotkey)
                config.wallet.hotkey = str(hotkey)

        if not config.is_set("netuid") and not config.no_prompt:
            netuid = Prompt.ask("Enter netuid", default="0")
            config.netuid = int(netuid)

        # Get amount.
        if not config.get("amount") and not config.get("unstake_all"):
            if not Confirm.ask(
                "Unstake all {}ao \n  [bold white]from account: '{}'[/bold white] \n  [bold white]and hotkey  : '{}'[/bold white] \n  [bold white]from subnet : '{}'[/bold white]\n".format(
                    bittensor.__tao_symbol__,
                    config.wallet.get("name", defaults.wallet.name),
                    config.get("hotkey"),
                    config.netuid,
                )
            ):
                amount = Prompt.ask("Enter Tao amount to unstake")
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
                config.unstake_all = True

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser(
            "remove",
            help="""Remove stake from a specific hotkey on subnet `netuid` from your coldkey.""",
        )
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--all", dest="unstake_all", action="store_true")
        stake_parser.add_argument("--amount", dest="amount", type=float, required=False)
        stake_parser.add_argument(
            "--hotkey",
            "--wallet.hotkey",
            required=False,
            type=str,
            help="""Specify the hotkey by name or ss58 address.""",
        )
        bittensor.wallet.add_args(stake_parser)
        bittensor.subtensor.add_args(stake_parser)
