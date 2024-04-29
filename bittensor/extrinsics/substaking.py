# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import bittensor
from rich.prompt import Confirm
from typing import Union, Optional
from bittensor.utils.balance import Balance
from loguru import logger


def add_substake_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    hotkey_ss58: Optional[str] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Adds the specified amount of stake to passed hotkey ``uid``.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        netuid (int):
            The subnetwork uid of to stake with.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        amount (Union[Balance, float]):
            Amount to stake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        bittensor.errors.NotRegisteredError:
            If the wallet is not registered on the chain.
        bittensor.errors.NotDelegateError:
            If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.coldkey
    dynamic_info = subtensor.get_dynamic_info_for_netuid(netuid)

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address

    # Flag to indicate if we are using the wallet's own hotkey.
    own_hotkey: bool

    with bittensor.__console__.status(
        ":satellite: Syncing with chain: [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        # Get hotkey owner
        hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
        own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
        if not own_hotkey:
            # This is not the wallet's own hotkey so we are delegating.
            if not subtensor.is_hotkey_delegate(hotkey_ss58):
                raise bittensor.errors.NotDelegateError(
                    "Hotkey: {} is not a delegate.".format(hotkey_ss58)
                )

            # Get hotkey take
            hotkey_take = subtensor.get_delegate_take(hotkey_ss58)

        # Get current stake
        old_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
        )

    # Convert to bittensor.Balance
    if amount == None:
        # Stake it all.
        staking_balance = bittensor.Balance.from_tao(old_balance.tao)
    elif not isinstance(amount, bittensor.Balance):
        staking_balance = bittensor.Balance.from_tao(amount)
    else:
        staking_balance = amount

    # Remove existential balance to keep key alive.
    if staking_balance > bittensor.Balance.from_rao(1000):
        staking_balance = staking_balance - bittensor.Balance.from_rao(1000)
    else:
        staking_balance = staking_balance

    # Check enough to stake.
    if staking_balance > old_balance:
        bittensor.__console__.print(
            ":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  coldkey: {}[/bold white]".format(
                old_balance, staking_balance, wallet.name
            )
        )
        return False

    # Set maximum slippage percentage
    max_slippage_pct = 5.0

    # Calculate slippage
    subnet_stake_amount_tao = bittensor.Balance.from_tao(staking_balance.tao)
    alpha_returned, slippage = dynamic_info.tao_to_alpha_with_slippage(
        subnet_stake_amount_tao
    )
    slippage_pct = 100 * (1 - float(alpha_returned) / float(subnet_stake_amount_tao))
    logger.debug(
        f"Slippage for subnet {netuid}: {slippage} TAO, Tao staked {subnet_stake_amount_tao}, Alpha returned {alpha_returned}, Slippage percent {slippage_pct:.2f}%"
    )

    # Check if slippage exceeds the maximum threshold
    if slippage_pct > max_slippage_pct:
        bittensor.__console__.print(
            f":warning: [yellow]Warning:[/yellow] Slippage exceeds {max_slippage_pct}% for subnet {netuid}: {bittensor.Balance.from_tao(slippage)} TAO ({slippage_pct:.2f}%)"
        )
        if not Confirm.ask(
            "Do you want to proceed with staking despite the high slippage?"
        ):
            return False

    # Check if any slippage exceeds the maximum threshold
    if slippage_pct > max_slippage_pct:
        bittensor.__console__.print(
            f":warning: [yellow]Warning:[/yellow] Slippage exceeds {max_slippage_pct}% for subnet {netuid}: {bittensor.Balance.from_tao(slippage)} TAO ({slippage_pct:.2f}%)"
        )
        if not Confirm.ask("Do you want to proceed with staking?"):
            return False

    try:
        with bittensor.__console__.status(
            ":satellite: Staking netuid:{} on network: [bold white]{}[/bold white] ...".format(
                netuid, subtensor.network
            )
        ):
            # Decrypt keys,
            wallet.coldkey

            hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
            own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
            if not own_hotkey:
                # We are delegating.
                # Verify that the hotkey is a delegate.
                if not subtensor.is_hotkey_delegate(hotkey_ss58=hotkey_ss58):
                    raise bittensor.errors.NotDelegateError(
                        "Hotkey: {} is not a delegate.".format(hotkey_ss58)
                    )

            if isinstance(amount, float):
                amount = bittensor.Balance.from_tao(amount)

            staking_response: bool = subtensor._do_subnet_stake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response == True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            with bittensor.__console__.status(
                ":satellite: Checking Balance on: [white]{}[/white] ...".format(
                    subtensor.network
                )
            ):
                new_balance = subtensor.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    block=block,
                    netuid=netuid,
                )  # Get current stake

                bittensor.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_balance, new_balance
                    )
                )
                bittensor.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_stake, new_stake
                    )
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except bittensor.errors.NotRegisteredError as e:
        bittensor.__console__.print(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except bittensor.errors.StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def remove_substake_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    hotkey_ss58: Optional[str] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Removes the specified amount of stake to passed hotkey ``hotkey_ss58``.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        netuid (int):
            The subnetwork uid of to stake with.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to unstake from. Defaults to the wallet's hotkey.
        amount (Union[Balance, float]):
            Amount to unstake as Bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        bittensor.errors.NotRegisteredError:
            If the wallet is not registered on the chain.
        bittensor.errors.NotDelegateError:
            If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.coldkey

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address

    # Flag to indicate if we are using the wallet's own hotkey.
    own_hotkey: bool

    with bittensor.__console__.status(
        ":satellite: Syncing with chain: [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        # Get currently staked on hotkey provided
        currently_staked = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            coldkey_ss58=wallet.coldkeypub.ss58_address,
        )

        old_balance = subtensor.get_balance(address=wallet.coldkeypub.ss58_address)

        # Get hotkey owner
        hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
        own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
        if not own_hotkey:
            # This is not the wallet's own hotkey so we are undelegating.
            if not subtensor.is_hotkey_delegate(hotkey_ss58):
                raise bittensor.errors.NotDelegateError(
                    "Hotkey: {} is not a delegate.".format(hotkey_ss58)
                )

            # Get hotkey take
            hotkey_take = subtensor.get_delegate_take(hotkey_ss58)

        # Get current stake
        old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=hotkey_ss58
        )

    # Convert to bittensor.Balance
    if amount == None:
        # Unstake it all.
        unstaking_balance = bittensor.Balance.from_tao(currently_staked.tao)
    elif not isinstance(amount, bittensor.Balance):
        unstaking_balance = bittensor.Balance.from_tao(amount)
    else:
        unstaking_balance = amount

    # Remove existential balance to keep key alive.
    if unstaking_balance > bittensor.Balance.from_rao(1000):
        unstaking_balance = unstaking_balance - bittensor.Balance.from_rao(1000)

    # Check enough to unstake.
    if unstaking_balance > currently_staked:
        bittensor.__console__.print(
            ":cross_mark: [red]Not enough stake[/red]:[bold white]\n  balance:{}\n  amount: {}\n  coldkey: {}[/bold white]".format(
                currently_staked, unstaking_balance, wallet.name
            )
        )
        return False

    # Set maximum slippage percentage
    max_slippage_pct = 5.0
    dynamic_info = subtensor.get_dynamic_info_for_netuid(netuid)

    # Calculate slippage
    subnet_stake_amount_alpha = bittensor.Balance.from_tao(unstaking_balance.tao)
    alpha_returned, slippage = dynamic_info.alpha_to_tao_with_slippage(
        subnet_stake_amount_alpha
    )
    slippage_pct = 100 * (1 - float(alpha_returned) / float(subnet_stake_amount_alpha))
    logger.debug(
        f"Slippage for subnet {netuid}: {slippage} TAO, Tao staked {subnet_stake_amount_alpha}, Alpha returned {alpha_returned}, Slippage percent {slippage_pct:.2f}%"
    )

    # Check if slippage exceeds the maximum threshold
    if slippage_pct > max_slippage_pct:
        bittensor.__console__.print(
            f":warning: [yellow]Warning:[/yellow] Slippage exceeds {max_slippage_pct}% for subnet {netuid}: {bittensor.Balance.from_tao(slippage)} TAO ({slippage_pct:.2f}%)"
        )
        if not Confirm.ask(
            "Do you want to proceed with staking despite the high slippage?"
        ):
            return False

    # Ask before moving on.
    if prompt:
        if not own_hotkey:
            # We are delegating.
            if not Confirm.ask(
                "Do you want to undelegate:[bold white]\n  amount:{}\n  from: {}\n  take: {}\n  owner: {}\n  on subnet: {}[/bold white]".format(
                    unstaking_balance,
                    wallet.hotkey_str,
                    hotkey_take,
                    hotkey_owner,
                    netuid,
                )
            ):
                return False
        else:
            if not Confirm.ask(
                "Do you want to unstake:[bold white]\n  amount: {}\n  from  : {}\n  netuid: {}[/bold white]\n".format(
                    unstaking_balance, wallet.hotkey_str, netuid
                )
            ):
                return False

    try:
        with bittensor.__console__.status(
            f":satellite: Unstaking [bold white]{unstaking_balance}[/bold white] from: [bold white]{hotkey_ss58}[/bold white] on [bold white]{subtensor.network}[/bold white]..."
        ):
            # Decrypt keys,
            wallet.coldkey

            hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
            own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
            if not own_hotkey:
                # We are delegating.
                # Verify that the hotkey is a delegate.
                if not subtensor.is_hotkey_delegate(hotkey_ss58=hotkey_ss58):
                    raise bittensor.errors.NotDelegateError(
                        "Hotkey: {} is not a delegate.".format(hotkey_ss58)
                    )

            staking_response: bool = subtensor._do_subnet_unstake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                amount=unstaking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response == True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            with bittensor.__console__.status(
                ":satellite: Checking Balance on: [white]{}[/white] ...".format(
                    subtensor.network
                )
            ):
                new_balance = subtensor.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    block=block,
                )  # Get current stake

                bittensor.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_balance, new_balance
                    )
                )
                bittensor.__console__.print(
                    "Unstaked:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        old_stake, new_stake
                    )
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except bittensor.errors.NotRegisteredError as e:
        bittensor.__console__.print(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except bittensor.errors.StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False
