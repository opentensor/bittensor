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
import bittensor
from ..errors import (
    NominationError,
    NotDelegateError,
    NotRegisteredError,
    StakeError,
    TakeError,
)
from rich.prompt import Confirm
from typing import Union, Optional
from bittensor.utils.balance import Balance
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


def nominate_extrinsic(
    subtensor: "bittensor.subtensor",
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

    try:
        wallet.coldkey

    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False

    wallet.hotkey
    # Check if the hotkey is already a delegate.
    if subtensor.is_hotkey_delegate(wallet.hotkey.ss58_address):
        logger.error(
            "Hotkey {} is already a delegate.".format(wallet.hotkey.ss58_address)
        )
        return False

    if not subtensor.is_hotkey_registered_any(wallet.hotkey.ss58_address):
        logger.error(
            "Hotkey {} is not registered to any network".format(
                wallet.hotkey.ss58_address
            )
        )
        return False

    with bittensor.__console__.status(
        ":satellite: Sending nominate call on [white]{}[/white] ...".format(
            subtensor.network
        )
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
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
        except NominationError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def delegate_extrinsic(
    subtensor: "bittensor.subtensor",
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
    try:
        wallet.coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False
    if not subtensor.is_hotkey_delegate(delegate_ss58):
        raise NotDelegateError("Hotkey: {} is not a delegate.".format(delegate_ss58))

    # Get state.
    my_prev_coldkey_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
    delegate_take = subtensor.get_delegate_take(delegate_ss58)
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
            ":cross_mark: [red]Not enough balance[/red]:[bold white]\n  balance:{}\n  amount: {}\n  coldkey: {}[/bold white]".format(
                my_prev_coldkey_balance, staking_balance, wallet.name
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n owner: {}[/bold white]".format(
                staking_balance, delegate_ss58, delegate_owner
            )
        ):
            return False

    try:
        with bittensor.__console__.status(
            ":satellite: Staking to: [bold white]{}[/bold white] ...".format(
                subtensor.network
            )
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
                ":satellite: Checking Balance on: [white]{}[/white] ...".format(
                    subtensor.network
                )
            ):
                new_balance = subtensor.get_balance(address=wallet.coldkey.ss58_address)
                block = subtensor.get_current_block()
                new_delegate_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                bittensor.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_coldkey_balance, new_balance
                    )
                )
                bittensor.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_delegated_stake, new_delegate_stake
                    )
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError as e:
        bittensor.__console__.print(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def undelegate_extrinsic(
    subtensor: "bittensor.subtensor",
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
        raise NotDelegateError("Hotkey: {} is not a delegate.".format(delegate_ss58))

    # Get state.
    my_prev_coldkey_balance = subtensor.get_balance(wallet.coldkey.ss58_address)
    delegate_take = subtensor.get_delegate_take(delegate_ss58)
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
            ":cross_mark: [red]Not enough delegated stake[/red]:[bold white]\n  stake:{}\n  amount: {}\n coldkey: {}[/bold white]".format(
                my_prev_delegated_stake, unstaking_balance, wallet.name
            )
        )
        return False

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to un-delegate:[bold white]\n  amount: {}\n  from: {}\n  owner: {}[/bold white]".format(
                unstaking_balance, delegate_ss58, delegate_owner
            )
        ):
            return False

    try:
        with bittensor.__console__.status(
            ":satellite: Unstaking from: [bold white]{}[/bold white] ...".format(
                subtensor.network
            )
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
                ":satellite: Checking Balance on: [white]{}[/white] ...".format(
                    subtensor.network
                )
            ):
                new_balance = subtensor.get_balance(address=wallet.coldkey.ss58_address)
                block = subtensor.get_current_block()
                new_delegate_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=delegate_ss58,
                    block=block,
                )  # Get current stake

                bittensor.__console__.print(
                    "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_coldkey_balance, new_balance
                    )
                )
                bittensor.__console__.print(
                    "Stake:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        my_prev_delegated_stake, new_delegate_stake
                    )
                )
                return True
        else:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: Error unknown."
            )
            return False

    except NotRegisteredError as e:
        bittensor.__console__.print(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def decrease_take_extrinsic(
    subtensor: "bittensor.subtensor",
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
    try:
        wallet.coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False

    wallet.hotkey

    with bittensor.__console__.status(
        ":satellite: Sending decrease_take_extrinsic call on [white]{}[/white] ...".format(
            subtensor.network
        )
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
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False


def increase_take_extrinsic(
    subtensor: "bittensor.subtensor",
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
    try:
        wallet.coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False

    wallet.hotkey

    with bittensor.__console__.status(
        ":satellite: Sending increase_take_extrinsic call on [white]{}[/white] ...".format(
            subtensor.network
        )
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
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
        except TakeError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )

    return False
