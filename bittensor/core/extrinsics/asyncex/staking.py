import asyncio
from typing import Optional, Sequence, TYPE_CHECKING

from async_substrate_interface.errors import SubstrateRequestException

from bittensor.core.extrinsics.utils import get_old_stakes
from bittensor.core.types import UIDs
from bittensor.utils import unlock_key, format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def add_stake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    amount: Balance,
    safe_staking: bool = False,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Adds a stake from the specified wallet to the neuron identified by the SS58 address of its hotkey in specified subnet.
    Staking is a fundamental process in the Bittensor network that enables neurons to participate actively and earn
    incentives.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object.
        netuid: The unique identifier of the subnet to which the neuron belongs.
        hotkey_ss58: The `ss58` address of the hotkey account to stake to default to the wallet's hotkey.
        amount: Amount to stake as Bittensor balance in TAO always.
        safe_staking: If True, enables price safety checks. Default is ``False``.
        allow_partial_stake: If True, allows partial unstaking if price tolerance exceeded. Default is ``False``.
        rate_tolerance: Maximum allowed price increase percentage (0.005 = 0.5%). Default is ``0.005``.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.

    Raises:
        SubstrateRequestException: Raised if the extrinsic fails to be included in the block within the timeout.
    """

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
    block_hash = await subtensor.substrate.get_chain_head()

    # Get current stake and existential deposit
    old_stake, existential_deposit = await asyncio.gather(
        subtensor.get_stake(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block_hash=block_hash,
        ),
        subtensor.get_existential_deposit(block_hash=block_hash),
    )

    # Leave existential balance to keep key alive.
    if amount > old_balance - existential_deposit:
        # If we are staking all, we need to leave at least the existential deposit.
        amount = old_balance - existential_deposit
    else:
        amount = amount

    # Check enough to stake.
    if amount > old_balance:
        logging.error(":cross_mark: [red]Not enough stake:[/red]")
        logging.error(f"\t\tbalance:{old_balance}")
        logging.error(f"\t\tamount: {amount}")
        logging.error(f"\t\twallet: {wallet.name}")
        return False

    call_params = {
        "hotkey": hotkey_ss58,
        "netuid": netuid,
        "amount_staked": amount.rao,
    }

    if safe_staking:
        pool = await subtensor.subnet(netuid=netuid)
        base_price = pool.price.tao

        price_with_tolerance = (
            base_price if pool.netuid == 0 else base_price * (1 + rate_tolerance)
        )

        logging.info(
            f":satellite: [magenta]Safe Staking to:[/magenta] "
            f"[blue]netuid: [green]{netuid}[/green], amount: [green]{amount}[/green], "
            f"tolerance percentage: [green]{rate_tolerance * 100}%[/green], "
            f"price limit: [green]{price_with_tolerance}[/green], "
            f"original price: [green]{base_price}[/green], "
            f"with partial stake: [green]{allow_partial_stake}[/green] "
            f"on [blue]{subtensor.network}[/blue][/magenta]...[/magenta]"
        )

        limit_price = Balance.from_tao(price_with_tolerance).rao
        call_params.update(
            {
                "limit_price": limit_price,
                "allow_partial": allow_partial_stake,
            }
        )
        call_function = "add_stake_limit"
    else:
        logging.info(
            f":satellite: [magenta]Staking to:[/magenta] "
            f"[blue]netuid: [green]{netuid}[/green], amount: [green]{amount}[/green] "
            f"on [blue]{subtensor.network}[/blue][magenta]...[/magenta]"
        )
        call_function = "add_stake"

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=call_params,
    )
    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        nonce_key="coldkeypub",
        sign_with="coldkey",
        use_nonce=True,
        period=period,
        raise_error=raise_error,
    )
    if success:  # If we successfully staked.
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True

        logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] "
            f"[blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
        )
        new_block_hash = await subtensor.substrate.get_chain_head()
        new_balance, new_stake = await asyncio.gather(
            subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block_hash=new_block_hash
            ),
            subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                block_hash=new_block_hash,
            ),
        )

        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        logging.info(
            f"Stake: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
        )
        return True

    if safe_staking and "Custom error: 8" in message:
        logging.error(
            ":cross_mark: [red]Failed[/red]: Price exceeded tolerance limit. Either increase price tolerance or enable partial staking."
        )
    else:
        logging.error(f":cross_mark: [red]Failed: {message}.[/red]")
    return False


async def add_stake_multiple_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: UIDs,
    amounts: list[Balance],
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey on subnet with
    corresponding netuid.

    Parameters:
        subtensor: Subtensor instance with the connection to the chain.
        wallet: Bittensor wallet object for the coldkey.
        netuids: List of netuids to stake to.
        hotkey_ss58s: List of hotkeys to stake to.
        amounts: List of corresponding TAO amounts to bet for each netuid and hotkey.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.
    """
    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    assert all(
        [
            isinstance(netuids, list),
            isinstance(hotkey_ss58s, list),
            isinstance(amounts, list),
        ]
    ), "The `netuids`, `hotkey_ss58s` and `amounts` must be lists."

    if len(hotkey_ss58s) == 0:
        return True

    assert len(netuids) == len(hotkey_ss58s) == len(amounts), (
        "The number of items in `netuids`, `hotkey_ss58s` and `amounts` must be the same."
    )

    if not all(isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s):
        raise TypeError("hotkey_ss58s must be a list of str")

    new_amounts: Sequence[Optional[Balance]] = [
        amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)
    ]

    if sum(amount.tao for amount in new_amounts) == 0:
        # Staking 0 tao
        return True

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    block_hash = await subtensor.substrate.get_chain_head()

    all_stakes = await subtensor.get_stake_for_coldkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address, block_hash=block_hash
    )
    old_stakes: list[Balance] = get_old_stakes(
        wallet=wallet, hotkey_ss58s=hotkey_ss58s, netuids=netuids, all_stakes=all_stakes
    )

    # Remove existential balance to keep key alive.
    # Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in new_amounts]
    )
    old_balance = initial_balance = await subtensor.get_balance(
        wallet.coldkeypub.ss58_address, block_hash=block_hash
    )

    if total_staking_rao == 0:
        # Staking all to the first wallet.
        if old_balance.rao > 1000:
            old_balance -= Balance.from_rao(1000)

    elif total_staking_rao < 1000:
        # Staking less than 1000 rao to the wallets.
        pass
    else:
        # Staking more than 1000 rao to the wallets.
        # Reduce the amount to stake to each wallet to keep the balance above 1000 rao.
        percent_reduction = 1 - (1000 / total_staking_rao)
        new_amounts = [
            Balance.from_tao(amount.tao * percent_reduction) for amount in new_amounts
        ]

    successful_stakes = 0
    for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
        zip(hotkey_ss58s, new_amounts, old_stakes, netuids)
    ):
        # Check enough to stake
        if amount > old_balance:
            logging.error(
                f":cross_mark: [red]Not enough balance[/red]: [green]{old_balance}[/green] to stake: "
                f"[blue]{amount}[/blue] from wallet: [white]{wallet.name}[/white]"
            )
            continue

        try:
            logging.info(
                f"Staking [blue]{amount}[/blue] to hotkey: [magenta]{hotkey_ss58}[/magenta] on netuid: "
                f"[blue]{netuid}[/blue]"
            )
            call = await subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="add_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_staked": amount.rao,
                    "netuid": netuid,
                },
            )
            success, message = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                nonce_key="coldkeypub",
                sign_with="coldkey",
                use_nonce=True,
                period=period,
                raise_error=raise_error,
            )

            # If we successfully staked.
            if success:
                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= amount
                    successful_stakes += 1
                    continue

                logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

                new_block_hash = await subtensor.substrate.get_chain_head()
                new_stake, new_balance = await asyncio.gather(
                    subtensor.get_stake(
                        coldkey_ss58=wallet.coldkeypub.ss58_address,
                        hotkey_ss58=hotkey_ss58,
                        netuid=netuid,
                        block_hash=new_block_hash,
                    ),
                    subtensor.get_balance(
                        wallet.coldkeypub.ss58_address, block_hash=new_block_hash
                    ),
                )
                logging.info(
                    f"Stake ({hotkey_ss58}) on netuid {netuid}: [blue]{old_stake}[/blue] :arrow_right: "
                    f"[green]{new_stake}[/green]"
                )
                logging.info(
                    f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                old_balance = new_balance
                successful_stakes += 1
            else:
                logging.error(f":cross_mark: [red]Failed: {message}.[/red]")
                continue

        except SubstrateRequestException as error:
            logging.error(
                f":cross_mark: [red]Add Stake Multiple error: {format_error_message(error)}[/red]"
            )

    if successful_stakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
            f"[magenta]...[/magenta]"
        )
        new_balance = await subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{initial_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
