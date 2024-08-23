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

from rich.prompt import Confirm
from time import sleep
from typing import List, Union, Optional, Tuple

import bittensor
from ..utils.formatting import float_to_u64, float_to_u16

from bittensor.utils.balance import Balance

console = bittensor.__console__


def _check_threshold_amount(
    subtensor: "bittensor.subtensor", stake_balance: Balance
) -> Tuple[bool, Balance]:
    """
    Checks if the new stake balance will be above the minimum required stake threshold.

    Args:
        stake_balance (Balance):
            the balance to check for threshold limits.

    Returns:
        success, threshold (bool, Balance):
            ``true`` if the staking balance is above the threshold, or ``false`` if the
                staking balance is below the threshold.
            The threshold balance required to stake.
    """
    min_req_stake: Balance = subtensor.get_minimum_required_stake()

    if min_req_stake > stake_balance:
        return False, min_req_stake
    else:
        return True, min_req_stake


def add_stake_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
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
    try:
        wallet.coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False

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
        old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=hotkey_ss58
        )

        # Grab the existential deposit.
        existential_deposit = subtensor.get_existential_deposit()

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        staking_balance = bittensor.Balance.from_tao(old_balance.tao)
    elif not isinstance(amount, bittensor.Balance):
        staking_balance = bittensor.Balance.from_tao(amount)
    else:
        staking_balance = amount

    # Leave existential balance to keep key alive.
    if staking_balance > old_balance - existential_deposit:
        # If we are staking all, we need to leave at least the existential deposit.
        staking_balance = old_balance - existential_deposit
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

    # If nominating, we need to check if the new stake balance will be above the minimum required stake threshold.
    if not own_hotkey:
        new_stake_balance = old_stake + staking_balance
        is_above_threshold, threshold = _check_threshold_amount(
            subtensor, new_stake_balance
        )
        if not is_above_threshold:
            bittensor.__console__.print(
                f":cross_mark: [red]New stake balance of {new_stake_balance} is below the minimum required nomination stake threshold {threshold}.[/red]"
            )
            return False

    # Ask before moving on.
    if prompt:
        if not own_hotkey:
            # We are delegating.
            if not Confirm.ask(
                "Do you want to delegate:[bold white]\n  amount: {}\n  to: {}\n  take: {}\n  owner: {}[/bold white]".format(
                    staking_balance, wallet.hotkey_str, hotkey_take, hotkey_owner
                )
            ):
                return False
        else:
            if not Confirm.ask(
                "Do you want to stake:[bold white]\n  amount: {}\n  to: {}[/bold white]".format(
                    staking_balance, wallet.hotkey_str
                )
            ):
                return False

    try:
        with bittensor.__console__.status(
            ":satellite: Staking to: [bold white]{}[/bold white] ...".format(
                subtensor.network
            )
        ):
            staking_response: bool = __do_add_stake_single(
                subtensor=subtensor,
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
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
                new_balance = subtensor.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    block=block,
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

    except bittensor.errors.NotRegisteredError:
        bittensor.__console__.print(
            ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                wallet.hotkey_str
            )
        )
        return False
    except bittensor.errors.StakeError as e:
        bittensor.__console__.print(":cross_mark: [red]Stake Error: {}[/red]".format(e))
        return False


def add_stake_multiple_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58s: List[str],
    amounts: Optional[List[Union[Balance, float]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object for the coldkey.
        hotkey_ss58s (List[str]):
            List of hotkeys to stake to.
        amounts (List[Union[Balance, float]]):
            List of amounts to stake. If ``None``, stake all to the first hotkey.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was staked. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not isinstance(hotkey_ss58s, list) or not all(
        isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s
    ):
        raise TypeError("hotkey_ss58s must be a list of str")

    if len(hotkey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

    if amounts is not None and not all(
        isinstance(amount, (Balance, float)) for amount in amounts
    ):
        raise TypeError(
            "amounts must be a [list of bittensor.Balance or float] or None"
        )

    if amounts is None:
        amounts = [None] * len(hotkey_ss58s)
    else:
        # Convert to Balance
        amounts = [
            bittensor.Balance.from_tao(amount) if isinstance(amount, float) else amount
            for amount in amounts
        ]

        if sum(amount.tao for amount in amounts) == 0:
            # Staking 0 tao
            return True

    # Decrypt coldkey.
    wallet.coldkey

    old_stakes = []
    with bittensor.__console__.status(
        ":satellite: Syncing with chain: [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        # Get the old stakes.
        for hotkey_ss58 in hotkey_ss58s:
            old_stakes.append(
                subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address, hotkey_ss58=hotkey_ss58
                )
            )

    # Remove existential balance to keep key alive.
    ## Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in amounts]
    )
    if total_staking_rao == 0:
        # Staking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= bittensor.Balance.from_rao(1000)

    elif total_staking_rao < 1000:
        # Staking less than 1000 rao to the wallets.
        pass
    else:
        # Staking more than 1000 rao to the wallets.
        ## Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_staking_rao)
        amounts = [
            Balance.from_tao(amount.tao * percent_reduction) for amount in amounts
        ]

    successful_stakes = 0
    for idx, (hotkey_ss58, amount, old_stake) in enumerate(
        zip(hotkey_ss58s, amounts, old_stakes)
    ):
        staking_all = False
        # Convert to bittensor.Balance
        if amount == None:
            # Stake it all.
            staking_balance = bittensor.Balance.from_tao(old_balance.tao)
            staking_all = True
        else:
            # Amounts are cast to balance earlier in the function
            assert isinstance(amount, bittensor.Balance)
            staking_balance = amount

        # Check enough to stake
        if staking_balance > old_balance:
            bittensor.__console__.print(
                ":cross_mark: [red]Not enough balance[/red]: [green]{}[/green] to stake: [blue]{}[/blue] from coldkey: [white]{}[/white]".format(
                    old_balance, staking_balance, wallet.name
                )
            )
            continue

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to stake:\n[bold white]  amount: {}\n  hotkey: {}[/bold white ]?".format(
                    staking_balance, wallet.hotkey_str
                )
            ):
                continue

        try:
            staking_response: bool = __do_add_stake_single(
                subtensor=subtensor,
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                amount=staking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if staking_response == True:  # If we successfully staked.
                # We only wait here if we expect finalization.

                if idx < len(hotkey_ss58s) - 1:
                    # Wait for tx rate limit.
                    tx_rate_limit_blocks = subtensor.tx_rate_limit()
                    if tx_rate_limit_blocks > 0:
                        bittensor.__console__.print(
                            ":hourglass: [yellow]Waiting for tx rate limit: [white]{}[/white] blocks[/yellow]".format(
                                tx_rate_limit_blocks
                            )
                        )
                        sleep(tx_rate_limit_blocks * 12)  # 12 seconds per block

                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )

                block = subtensor.get_current_block()
                new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    block=block,
                )
                new_balance = subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block=block
                )
                bittensor.__console__.print(
                    "Stake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                        hotkey_ss58, old_stake, new_stake
                    )
                )
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break

            else:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: Error unknown."
                )
                continue

        except bittensor.errors.NotRegisteredError:
            bittensor.__console__.print(
                ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                    hotkey_ss58
                )
            )
            continue
        except bittensor.errors.StakeError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Stake Error: {}[/red]".format(e)
            )
            continue

    if successful_stakes != 0:
        with bittensor.__console__.status(
            ":satellite: Checking Balance on: ([white]{}[/white] ...".format(
                subtensor.network
            )
        ):
            new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        bittensor.__console__.print(
            "Balance: [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                old_balance, new_balance
            )
        )
        return True

    return False


def __do_add_stake_single(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: str,
    amount: "bittensor.Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    r"""
    Executes a stake call to the chain using the wallet and the amount specified.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (str):
            Hotkey to stake to.
        amount (bittensor.Balance):
            Amount to stake as Bittensor balance object.
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
        bittensor.errors.StakeError:
            If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotDelegateError:
            If the hotkey is not a delegate.
        bittensor.errors.NotRegisteredError:
            If the hotkey is not registered in any subnets.

    """
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

    success = subtensor._do_stake(
        wallet=wallet,
        hotkey_ss58=hotkey_ss58,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    return success


def set_childkey_take_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey: str,
    netuid: int,
    take: float,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    """
    Sets childkey take.

    Args:
        subtensor (bittensor.subtensor): Subtensor endpoint to use.
        wallet (bittensor.wallet): Bittensor wallet object.
        hotkey (str): Childkey hotkey.
        take (float): Childkey take value.
        netuid (int): Unique identifier of for the subnet.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    Raises:
        bittensor.errors.ChildHotkeyError: If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotRegisteredError: If the hotkey is not registered in any subnets.

    """

    # Decrypt coldkey.
    wallet.coldkey

    user_hotkey_ss58 = wallet.hotkey.ss58_address  # Default to wallet's own hotkey.
    if hotkey != user_hotkey_ss58:
        raise ValueError("You can only set childkey take for ss58 hotkey that you own.")

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            f"Do you want to set childkey take to: [bold white]{take*100}%[/bold white]?"
        ):
            return False, "Operation Cancelled"

    with bittensor.__console__.status(
        f":satellite: Setting childkey take on [white]{subtensor.network}[/white] ..."
    ):
        try:
            
            if 0 < take < 0.18:
                take_u16 = float_to_u16(take)
            else:
                return False, "Invalid take value"

            success, error_message = subtensor._do_set_childkey_take(
                wallet=wallet,
                hotkey=hotkey,
                netuid=netuid,
                take=take_u16,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization and not wait_for_inclusion:
                return (
                    True,
                    "Not waiting for finalization or inclusion. Set childkey take initiated.",
                )

            if success:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Setting childkey take",
                    suffix="<green>Finalized: </green>" + str(success),
                )
                return True, "Successfully set childkey take and Finalized."
            else:
                bittensor.__console__.print(
                    f":cross_mark: [red]Failed[/red]: {error_message}"
                )
                bittensor.logging.warning(
                    prefix="Setting childkey take",
                    suffix="<red>Failed: </red>" + str(error_message),
                )
                return False, error_message

        except Exception as e:
            return False, f"Exception occurred while setting childkey take: {str(e)}"


def set_children_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey: str,
    netuid: int,
    children_with_proportions: List[Tuple[float, str]],
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    """
    Sets children hotkeys with proportions assigned from the parent.

    Args:
        subtensor (bittensor.subtensor): Subtensor endpoint to use.
        wallet (bittensor.wallet): Bittensor wallet object.
        hotkey (str): Parent hotkey.
        children_with_proportions (List[str]): Children hotkeys.
        netuid (int): Unique identifier of for the subnet.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    Raises:
        bittensor.errors.ChildHotkeyError: If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotRegisteredError: If the hotkey is not registered in any subnets.

    """

    # Decrypt coldkey.
    wallet.coldkey

    user_hotkey_ss58 = wallet.hotkey.ss58_address  # Default to wallet's own hotkey.
    if hotkey != user_hotkey_ss58:
        raise ValueError("Cannot set/revoke child hotkeys for others.")

    # Check if all children are being revoked
    all_revoked = all(prop == 0.0 for prop, _ in children_with_proportions)

    operation = "Revoke children hotkeys" if all_revoked else "Set children hotkeys"

    # Ask before moving on.
    if prompt:
        if all_revoked:
            if not Confirm.ask(
                f"Do you want to revoke all children hotkeys for hotkey {hotkey}?"
            ):
                return False, "Operation Cancelled"
        else:
            if not Confirm.ask(
                "Do you want to set children hotkeys with proportions:\n[bold white]{}[/bold white]?".format(
                    "\n".join(
                        f"  {child[1]}: {child[0]}"
                        for child in children_with_proportions
                    )
                )
            ):
                return False, "Operation Cancelled"

    with bittensor.__console__.status(
        f":satellite: {operation} on [white]{subtensor.network}[/white] ..."
    ):
        try:
            normalized_children = (
                prepare_child_proportions(children_with_proportions)
                if not all_revoked
                else children_with_proportions
            )

            success, error_message = subtensor._do_set_children(
                wallet=wallet,
                hotkey=hotkey,
                netuid=netuid,
                children=normalized_children,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization and not wait_for_inclusion:
                return (
                    True,
                    f"Not waiting for finalization or inclusion. {operation} initiated.",
                )

            if success:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix=operation,
                    suffix="<green>Finalized: </green>" + str(success),
                )
                return True, f"Successfully {operation.lower()} and Finalized."
            else:
                bittensor.__console__.print(
                    f":cross_mark: [red]Failed[/red]: {error_message}"
                )
                bittensor.logging.warning(
                    prefix=operation,
                    suffix="<red>Failed: </red>" + str(error_message),
                )
                return False, error_message

        except Exception as e:
            return False, f"Exception occurred while {operation.lower()}: {str(e)}"


def prepare_child_proportions(children_with_proportions):
    """
    Convert proportions to u64 and normalize, ensuring total does not exceed u64 max.
    """
    children_u64 = [(float_to_u64(proportion), child) for proportion, child in children_with_proportions]
    total = sum(proportion for proportion, _ in children_u64)

    if total > (2 ** 64 - 1):
        excess = total - (2 ** 64 - 1)
        if excess > (2 ** 64 * 0.01):  # Example threshold of 1% of u64 max
            raise ValueError("Excess is too great to normalize proportions")
        largest_child_index = max(range(len(children_u64)), key=lambda i: children_u64[i][0])
        children_u64[largest_child_index] = (
            children_u64[largest_child_index][0] - excess,
            children_u64[largest_child_index][1]
        )

    return children_u64
