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
from time import sleep
from rich.prompt import Confirm
from typing import Union, Optional, List
from bittensor.utils.balance import Balance
from bittensor.utils.user_io import (
    user_input_confirmation,
    print_summary_header,
    print_summary_footer,
    print_summary_message,
)
from loguru import logger

# Maximum slippage percentage
# MAX_SLIPPAGE_PCT = 5.0
MAX_SLIPPAGE_PCT = 0.01


def get_total_coldkey_stake_for_netuid(
    subtensor: "bittensor.subtensor",
    coldkey_ss58: str,
    netuid: int,
) -> bittensor.Balance:
    stake_info = subtensor.get_subnet_stake_info_for_coldkey(
        coldkey_ss58=coldkey_ss58, netuid=netuid
    )
    stake = 0
    for info in stake_info:
        stake += info.stake
    return stake


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
    # Get dynamic pool info for slippage calculation
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
        old_stake = get_total_coldkey_stake_for_netuid(
            subtensor, coldkey_ss58=wallet.coldkeypub.ss58_address, netuid=netuid
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

    # Calculate slippage
    subnet_stake_amount_tao = bittensor.Balance.from_tao(staking_balance.tao)
    alpha_returned, slippage = dynamic_info.tao_to_alpha_with_slippage(
        subnet_stake_amount_tao
    )
    slippage_pct = 0
    if slippage + alpha_returned != 0:
        slippage_pct = 100 * float(slippage) / float(slippage + alpha_returned)

    logger.debug(
        f"Slippage for subnet {netuid}: {slippage} TAO, Tao staked {subnet_stake_amount_tao}, Alpha returned {alpha_returned}, Slippage percent {slippage_pct:.2f}%"
    )

    # Check if slippage exceeds the maximum threshold
    if slippage_pct > MAX_SLIPPAGE_PCT:
        print_summary_header(f":warning: [yellow]Slippage Warning:[/yellow]")
        print_summary_message(
            f"Slippage exceeds {MAX_SLIPPAGE_PCT}% for subnet {netuid}: {bittensor.Balance.from_tao(slippage.tao).set_unit(netuid)} ({slippage_pct:.2f}%)"
        )
        estimated = (
            bittensor.Balance.from_tao(alpha_returned.tao).set_unit(netuid).__str__()
        )
        expected = (
            bittensor.Balance.from_tao(slippage.tao + alpha_returned.tao)
            .set_unit(netuid)
            .__str__()
        )
        print_summary_message(
            f"You will only receive [green][bold]{estimated}[/bold][/green] vs. expected [green][bold]{expected}[/bold][/green]"
        )
        print_summary_footer()
        if prompt:
            if not user_input_confirmation("proceed despite the high slippage"):
                return False

    # Decrypt keys
    wallet.coldkey

    try:
        with bittensor.__console__.status(
            ":satellite: Staking netuid:{} on network: [bold white]{}[/bold white] ...".format(
                netuid, subtensor.network
            )
        ):
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
                new_stake = get_total_coldkey_stake_for_netuid(
                    subtensor,
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    netuid=netuid,
                )
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


def add_substake_multiple_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58s: List[str],
    netuid: int,
    amounts: Optional[List[Union[Balance, float]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Adds substake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey on a specific subnet.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object for the coldkey.
        hotkey_ss58s (List[str]):
            List of hotkeys to substake to.
        netuid (int):
            The unique identifier of the subnet to substake on.
        amounts (List[Union[Balance, float]]):
            List of amounts to substake. If ``None``, substake all to the first hotkey.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. Flag is ``true`` if any wallet was substaked. If we did not wait for finalization / inclusion, the response is ``true``.
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
            # Substaking 0 tao
            return True

    # Decrypt coldkey.
    wallet.coldkey

    old_substakes = []
    with bittensor.__console__.status(
        ":satellite: Syncing with chain: [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        # Get the old substakes.
        for hotkey_ss58 in hotkey_ss58s:
            old_substakes.append(
                subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                )
            )

    # Remove existential balance to keep key alive.
    ## Keys must maintain a balance of at least 1000 rao to stay alive.
    total_substaking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in amounts]
    )
    if total_substaking_rao == 0:
        # Substaking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= bittensor.Balance.from_rao(1000)

    elif total_substaking_rao < 1000:
        # Substaking less than 1000 rao to the wallets.
        pass
    else:
        # Substaking more than 1000 rao to the wallets.
        ## Reduce the amount to substake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_substaking_rao)
        amounts = [
            Balance.from_tao(amount.tao * percent_reduction) for amount in amounts
        ]

    successful_substakes = 0
    for idx, (hotkey_ss58, amount, old_substake) in enumerate(
        zip(hotkey_ss58s, amounts, old_substakes)
    ):
        substaking_all = False
        # Convert to bittensor.Balance
        if amount == None:
            # Substake it all.
            substaking_balance = bittensor.Balance.from_tao(old_balance.tao)
            substaking_all = True
        else:
            # Amounts are cast to balance earlier in the function
            assert isinstance(amount, bittensor.Balance)
            substaking_balance = amount

        # Check enough to substake
        if substaking_balance > old_balance:
            bittensor.__console__.print(
                ":cross_mark: [red]Not enough balance[/red]: [green]{}[/green] to substake: [blue]{}[/blue] from coldkey: [white]{}[/white]".format(
                    old_balance, substaking_balance, wallet.name
                )
            )
            continue

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                "Do you want to substake:\n[bold white]  amount: {}\n  hotkey: {}[/bold white ]?".format(
                    substaking_balance, wallet.hotkey_str
                )
            ):
                continue

        try:
            substaking_response: bool = __do_add_substake_single(
                subtensor=subtensor,
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                amount=substaking_balance,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

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
                old_balance -= substaking_balance
                successful_substakes += 1
                if substaking_all:
                    # If substaked all, no need to continue
                    break

                continue

            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Finalized[/green]"
            )
            block = subtensor.get_current_block()
            new_substake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
            )
            new_balance = subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block=block
            )
            bittensor.__console__.print(
                "Substake ({}): [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                    hotkey_ss58, old_substake, new_substake
                )
            )
            old_balance = new_balance
            successful_substakes += 1
            if substaking_all:
                # If substaked all, no need to continue
                break

            else:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: Error unknown."
                )
                continue

        except bittensor.errors.NotRegisteredError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Hotkey: {} is not registered.[/red]".format(
                    hotkey_ss58
                )
            )
            continue
        except bittensor.errors.SubstakeError as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Substake Error: {}[/red]".format(e)
            )
            continue

    if successful_substakes != 0:
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


def __do_add_substake_single(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    hotkey_ss58: str,
    netuid: int,
    amount: "bittensor.Balance",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    r"""
    Executes a substake call to the chain using the wallet and the amount specified for a specific subnet.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        hotkey_ss58 (str):
            Hotkey to substake to.
        netuid (int):
            The unique identifier of the subnet to substake on.
        amount (bittensor.Balance):
            Amount to substake as Bittensor balance object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    Raises:
        bittensor.errors.SubstakeError:
            If the extrinsic fails to be finalized or included in the block.
        bittensor.errors.NotRegisteredError:
            If the hotkey is not registered in the specified subnet.

    """
    # Decrypt keys,
    wallet.coldkey

    hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
    own_hotkey = wallet.coldkeypub.ss58_address == hotkey_owner
    if not own_hotkey:
        # We are delegating.
        # Verify that the hotkey is registered in the specified subnet.
        if not subtensor.is_hotkey_registered(hotkey_ss58=hotkey_ss58, netuid=netuid):
            raise bittensor.errors.NotRegisteredError(
                "Hotkey: {} is not registered in subnet: {}.".format(
                    hotkey_ss58, netuid
                )
            )

    success = subtensor._do_subnet_stake(
        wallet=wallet,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        amount=amount,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    return success


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
    # Get dynamic pool info for slippage calculation
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
        old_stake = subtensor.get_stake_for_coldkey_and_hotkey_on_netuid(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
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

    # Calculate slippage
    subnet_stake_amount_alpha = bittensor.Balance.from_tao(unstaking_balance.tao)
    tao_returned, slippage = dynamic_info.alpha_to_tao_with_slippage(
        subnet_stake_amount_alpha
    )
    slippage_pct = 0
    if slippage + tao_returned != 0:
        slippage_pct = 100 * float(slippage) / float(slippage + tao_returned)
    logger.debug(
        f"Slippage for subnet {netuid}: {slippage} TAO, Tao staked {subnet_stake_amount_alpha}, TAO returned {tao_returned}, Slippage percent {slippage_pct:.2f}%"
    )

    # Check if slippage exceeds the maximum threshold
    if prompt and slippage_pct > MAX_SLIPPAGE_PCT:
        print_summary_header(f":warning: [yellow]Slippage Warning:[/yellow]")
        print_summary_message(
            f"Slippage exceeds {MAX_SLIPPAGE_PCT}% for subnet {netuid}: {bittensor.Balance.from_tao(slippage.tao)} TAO ({slippage_pct:.2f}%)"
        )
        estimated = bittensor.Balance.from_tao(tao_returned.tao).__str__()
        expected = bittensor.Balance.from_tao(slippage.tao + tao_returned.tao).__str__()
        print_summary_message(
            f"You will only receive [green][bold]{estimated}[/bold][/green] vs. expected [green][bold]{expected}[/bold][/green]"
        )
        print_summary_footer()
        if prompt:
            if not user_input_confirmation("proceed despite the high slippage"):
                return False

    # Decrypt keys
    wallet.coldkey

    try:
        with bittensor.__console__.status(
            f":satellite: Unstaking [bold white]{unstaking_balance.set_unit(netuid)}[/bold white] from: [bold white]{hotkey_ss58}[/bold white] on [bold white]{subtensor.network}[/bold white]..."
        ):
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

        if staking_response == True:  # If we successfully unstaked.
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
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    netuid=netuid,
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
