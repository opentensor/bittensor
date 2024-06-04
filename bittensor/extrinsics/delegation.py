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

import logging
from typing import Optional, Union

from rich.prompt import Confirm

import bittensor
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME
from bittensor.utils.balance import Balance

from ..errors import (
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    StakeError,
    TakeError,
)

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


def nominate_extrinsic(
    subtensor: "bittensor.Subtensor",
    wallet: "bittensor.wallet",
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Becomes a delegate for the hotkey.

    Args:
        wallet (bittensor.wallet): The wallet to become a delegate for.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    # Check if the hotkey is already a delegate.
    if subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
        logger.error(f"Hotkey {wallet.hotkey.ss58_address} is already a delegate.")
        return False

    with bittensor.__console__.status(
        f":satellite: Sending nominate call on [white]{subtensor.network}[/white] ..."
    ):
        try:
            success = subtensor._do_nominate(
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Become Delegate",
                    suffix="<green>Finalized: </green>" + str(success),
                )

            # Raises NominationError if False
            return success

        except Exception as e:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: error:{e}")
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
        except NominationError as e:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: error:{e}")
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def delegate_extrinsic(
    subtensor: "bittensor.Subtensor",
    wallet: "bittensor.wallet",
    delegate_ss58: Optional[str] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Delegates the specified amount of stake to the passed delegate.

    Args:
        wallet (bittensor.wallet): Bittensor wallet object.
        delegate_ss58 (Optional[str]): The ``ss58`` address of the delegate.
        amount (Union[Balance, float]): Amount to stake as bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        NotRegisteredError: If the wallet is not registered on the chain.
        NotDelegateError: If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.coldkey
    if not subtensor.is_hotkey_delegate(delegate_ss58):
        raise NotDelegateError(f"Hotkey: {delegate_ss58} is not a delegate.")

    # Get state.
    my_prev_coldkey_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
    subtensor.get_delegate_take(delegate_ss58)
    delegate_owner = subtensor.get_hotkey_owner(delegate_ss58)
    my_prev_delegated_stake = subtensor.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=delegate_ss58
    )

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        staking_balance = bittensor.Balance.from_tao(my_prev_coldkey_balance.tao)
    elif not isinstance(amount, bittensor.Balance):
        staking_balance = bittensor.Balance.from_tao(amount)
    else:
        staking_balance = amount

    # Remove existential balance to keep key alive.
    if staking_balance > bittensor.Balance.from_rao(1000):
        staking_balance = staking_balance - bittensor.Balance.from_rao(1000)
    else:
        staking_balance = staking_balance

    # Check enough balance to stake.
    if staking_balance > my_prev_coldkey_balance:
        bittensor.__console__.print(
            f":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance:{my_prev_coldkey_balance}\n  amount: {staking_balance}\n  coldkey: {wallet.name}[/bold white]"
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            f"Do you want to delegate:[bold white]\n  amount: {staking_balance}\n  to: {delegate_ss58}\n owner: {delegate_owner}[/bold white]"
        ):
            return False

    try:
        with bittensor.__console__.status(
            f":satellite: Staking to: [bold white]{subtensor.network}[/bold white] ..."
        ):
            staking_response: bool = subtensor._do_delegation(
                wallet=wallet,
                delegate_ss58=delegate_ss58,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            with bittensor.__console__.status(
                f":satellite: Checking Balance on: [white]{subtensor.network}[/white] ..."
            ):
                new_balance = subtensor.get_balance(address=wallet.coldkey.ss58_address)
                block = subtensor.get_current_block()
                new_delegate_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                bittensor.__console__.print(
                    f"Balance:\n  [blue]{my_prev_coldkey_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                bittensor.__console__.print(
                    f"Stake:\n  [blue]{my_prev_delegated_stake}[/blue] :arrow_right: [green]{new_delegate_stake}[/green]"
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError:
        bittensor.__console__.print(
            f":cross_mark: [red]Hotkey: {wallet.hotkey_str} is not registered.[/red]"
        )
        return False
    except StakeError as e:
        bittensor.__console__.print(f":cross_mark: [red]Stake Error: {e}[/red]")
        return False


def undelegate_extrinsic(
    subtensor: "bittensor.Subtensor",
    wallet: "bittensor.wallet",
    delegate_ss58: Optional[str] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Un-delegates stake from the passed delegate.

    Args:
        wallet (bittensor.wallet): Bittensor wallet object.
        delegate_ss58 (Optional[str]): The ``ss58`` address of the delegate.
        amount (Union[Balance, float]): Amount to unstake as bittensor balance, or ``float`` interpreted as Tao.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.

    Raises:
        NotRegisteredError: If the wallet is not registered on the chain.
        NotDelegateError: If the hotkey is not a delegate on the chain.
    """
    # Decrypt keys,
    wallet.coldkey
    if not subtensor.is_hotkey_delegate(delegate_ss58):
        raise NotDelegateError(f"Hotkey: {delegate_ss58} is not a delegate.")

    # Get state.
    my_prev_coldkey_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
    subtensor.get_delegate_take(delegate_ss58)
    delegate_owner = subtensor.get_hotkey_owner(delegate_ss58)
    my_prev_delegated_stake = subtensor.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=delegate_ss58
    )

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        unstaking_balance = bittensor.Balance.from_tao(my_prev_delegated_stake.tao)

    elif not isinstance(amount, bittensor.Balance):
        unstaking_balance = bittensor.Balance.from_tao(amount)

    else:
        unstaking_balance = amount

    # Check enough stake to unstake.
    if unstaking_balance > my_prev_delegated_stake:
        bittensor.__console__.print(
            f":cross_mark: [red]Not enough delegated stake[/red]:[bold white]\n  stake:{my_prev_delegated_stake}\n  amount: {unstaking_balance}\n coldkey: {wallet.name}[/bold white]"
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            f"Do you want to un-delegate:[bold white]\n  amount: {unstaking_balance}\n  from: {delegate_ss58}\n  owner: {delegate_owner}[/bold white]"
        ):
            return False

    try:
        with bittensor.__console__.status(
            f":satellite: Unstaking from: [bold white]{subtensor.network}[/bold white] ..."
        ):
            staking_response: bool = subtensor._do_undelegation(
                wallet=wallet,
                delegate_ss58=delegate_ss58,
                amount=unstaking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        if staking_response is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            with bittensor.__console__.status(
                f":satellite: Checking Balance on: [white]{subtensor.network}[/white] ..."
            ):
                new_balance = subtensor.get_balance(address=wallet.coldkey.ss58_address)
                block = subtensor.get_current_block()
                new_delegate_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                bittensor.__console__.print(
                    f"Balance:\n  [blue]{my_prev_coldkey_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                bittensor.__console__.print(
                    f"Stake:\n  [blue]{my_prev_delegated_stake}[/blue] :arrow_right: [green]{new_delegate_stake}[/green]"
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError:
        bittensor.__console__.print(
            f":cross_mark: [red]Hotkey: {wallet.hotkey_str} is not registered.[/red]"
        )
        return False
    except StakeError as e:
        bittensor.__console__.print(f":cross_mark: [red]Stake Error: {e}[/red]")
        return False


def decrease_take_extrinsic(
    subtensor: "bittensor.Subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: Optional[str] = None,
    take: int = 0,
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Decrease delegate take for the hotkey.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        take (float):
            The ``take`` of the hotkey.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    with bittensor.__console__.status(
        f":satellite: Sending decrease_take_extrinsic call on [white]{subtensor.network}[/white] ..."
    ):
        try:
            success = subtensor._do_decrease_take(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                take=take,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Decrease Delegate Take",
                    suffix="<green>Finalized: </green>" + str(success),
                )

            return success

        except (TakeError, Exception) as e:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: error:{e}")
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def increase_take_extrinsic(
    subtensor: "bittensor.Subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: Optional[str] = None,
    take: int = 0,
    wait_for_finalization: bool = False,
    wait_for_inclusion: bool = True,
) -> bool:
    r"""Increase delegate take for the hotkey.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (Optional[str]):
            The ``ss58`` address of the hotkey account to stake to defaults to the wallet's hotkey.
        take (float):
            The ``take`` of the hotkey.
    Returns:
        success (bool): ``True`` if the transaction was successful.
    """
    # Unlock the coldkey.
    wallet.coldkey
    wallet.hotkey

    with bittensor.__console__.status(
        f":satellite: Sending increase_take_extrinsic call on [white]{subtensor.network}[/white] ..."
    ):
        try:
            success = subtensor._do_increase_take(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                take=take,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Increase Delegate Take",
                    suffix="<green>Finalized: </green>" + str(success),
                )

            return success

        except Exception as e:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: error:{e}")
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
        except TakeError as e:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: error:{e}")
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False
