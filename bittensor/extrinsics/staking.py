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
from time import sleep
import numpy as np
from typing import List, Union, Optional, Tuple
from numpy.typing import NDArray

from bittensor.utils import weight_utils
from bittensor.utils.balance import Balance


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


def do_set_child_singular_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey: str,
    child: str,
    netuid: int,
    proportion: float,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""
    Sets child hotkey with a proportion assigned from the parent.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey (str):
            Parent hotkey.
        child (str):
            Child hotkey.
        netuid (int):
            Unique identifier of for the subnet.
        proportion (float):
            Proportion assigned to child hotkey.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    Raises:
        bittensor.errors.ChildHotkeyError:
            If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotRegisteredError:
            If the hotkey is not registered in any subnets.

    """
    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to add child hotkey:\n[bold white]  child: {}\n  proportion: {}[/bold white ]?".format(
                child, proportion
            )
        ):
            return False

    with bittensor.__console__.status(
        ":satellite: Setting child hotkey on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            # prepare values for emmit
            proportion = np.array([proportion], dtype=np.float32)
            netuids = np.full(proportion.shape, netuid, dtype=np.int64)

            uid_val, proportion_val = weight_utils.convert_values_and_ids_for_emit(
                netuids, proportion
            )

            success, error_message = subtensor._do_set_child_singular(
                wallet=wallet,
                hotkey=hotkey,
                child=child,
                netuid=netuid,
                proportion=proportion_val[0],
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            bittensor.__console__.print(success, error_message)

            if not wait_for_finalization and not wait_for_inclusion:
                return True

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Set child hotkey",
                    suffix="<green>Finalized: </green>" + str(success),
                )
                return True
            else:
                bittensor.__console__.print(
                    f":cross_mark: [red]Failed[/red]: {error_message}"
                )
                bittensor.logging.warning(
                    prefix="Set child hotkey",
                    suffix="<red>Failed: </red>" + str(error_message),
                )
                return False

        except Exception as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set child hotkey", suffix="<red>Failed: </red>" + str(e)
            )
            return False


def do_set_children_multiple_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey: str,
    children: Union[NDArray[str], list],
    netuid: int,
    proportions: Union[NDArray[np.float32], list],
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""
    Sets children hotkeys with a proportion assigned from the parent.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey (str):
            Parent hotkey.
        children (np.ndarray):
            Children hotkeys.
        netuid (int):
            Unique identifier of for the subnet.
        proportions (np.ndarray):
            Proportions assigned to children hotkeys.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    Raises:
        bittensor.errors.ChildHotkeyError:
            If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotRegisteredError:
            If the hotkey is not registered in any subnets.

    """
    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to add children hotkeys:\n[bold white]  children: {}\n  proportions: {}[/bold white ]?".format(
                children, proportions
            )
        ):
            return False

    with bittensor.__console__.status(
        ":satellite: Setting children hotkeys on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            # prepare values for emmit
            if isinstance(proportions, np.ndarray):
                uids = np.full(proportions.shape, netuid, dtype=np.int64)
            else:
                uids = [netuid] * len(proportions)

            uid_val, proportions_val = weight_utils.convert_values_and_ids_for_emit(
                uids, proportions
            )

            children_with_proportions = list(zip(children, proportions_val))

            success, error_message = subtensor._do_set_children_multiple(
                wallet=wallet,
                hotkey=hotkey,
                children_with_proportions=children_with_proportions,
                netuid=netuid,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            bittensor.__console__.print(success, error_message)

            if not wait_for_finalization and not wait_for_inclusion:
                return True

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Set children hotkeys",
                    suffix="<green>Finalized: </green>" + str(success),
                )
                return True
            else:
                bittensor.__console__.print(
                    f":cross_mark: [red]Failed[/red]: {error_message}"
                )
                bittensor.logging.warning(
                    prefix="Set children hotkeys",
                    suffix="<red>Failed: </red>" + str(error_message),
                )
                return False

        except Exception as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set children hotkeys", suffix="<red>Failed: </red>" + str(e)
            )
            return False
