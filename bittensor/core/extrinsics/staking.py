from typing import Optional, TYPE_CHECKING, Sequence

from async_substrate_interface.errors import SubstrateRequestException

from bittensor.core.extrinsics.utils import get_old_stakes
from bittensor.utils import unlock_key, format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def add_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[Balance] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    safe_staking: bool = False,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    period: Optional[int] = None,
) -> bool:
    """
    Adds the specified amount of stake to passed hotkey `uid`.

    Arguments:
        subtensor: the Subtensor object to use
        wallet: Bittensor wallet object.
        hotkey_ss58: The `ss58` address of the hotkey account to stake to default to the wallet's hotkey.
        netuid (Optional[int]): Subnet unique ID.
        amount: Amount to stake as Bittensor balance, `None` if staking all.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`,
            or returns `False` if the extrinsic fails to be finalized within the timeout.
        safe_staking (bool): If true, enables price safety checks
        allow_partial_stake (bool): If true, allows partial unstaking if price tolerance exceeded
        rate_tolerance (float): Maximum allowed price increase percentage (0.005 = 0.5%)
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success: Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
                      finalization/inclusion, the response is `True`.

    Raises:
        SubstrateRequestException: Raised if the extrinsic fails to be included in the block within the timeout.
    """

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # Default to wallet's own hotkey if the value is not passed.
    if hotkey_ss58 is None:
        hotkey_ss58 = wallet.hotkey.ss58_address

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    block = subtensor.get_current_block()

    # Get current stake and existential deposit
    old_stake = subtensor.get_stake(
        hotkey_ss58=hotkey_ss58,
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        netuid=netuid,
        block=block,
    )
    existential_deposit = subtensor.get_existential_deposit(block=block)

    # Convert to bittensor.Balance
    if amount is None:
        # Stake it all.
        staking_balance = Balance.from_tao(old_balance.tao)
        logging.warning(
            f"Didn't receive any staking amount. Staking all available balance: [blue]{staking_balance}[/blue] "
            f"from wallet: [blue]{wallet.name}[/blue]"
        )
    else:
        staking_balance = amount
    staking_balance.set_unit(netuid)

    # Leave existential balance to keep key alive.
    if staking_balance > old_balance - existential_deposit:
        # If we are staking all, we need to leave at least the existential deposit.
        staking_balance = old_balance - existential_deposit
    else:
        staking_balance = staking_balance

    # Check enough to stake.
    if staking_balance > old_balance:
        logging.error(":cross_mark: [red]Not enough stake:[/red]")
        logging.error(f"\t\tbalance:{old_balance}")
        logging.error(f"\t\tamount: {staking_balance}")
        logging.error(f"\t\twallet: {wallet.name}")
        return False

    try:
        call_params = {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "amount_staked": staking_balance.rao,
        }

        if safe_staking:
            pool = subtensor.subnet(netuid=netuid)
            base_price = pool.price.rao
            price_with_tolerance = base_price * (1 + rate_tolerance)

            # For logging
            base_rate = pool.price.tao
            rate_with_tolerance = base_rate * (1 + rate_tolerance)

            logging.info(
                f":satellite: [magenta]Safe Staking to:[/magenta] "
                f"[blue]netuid: [green]{netuid}[/green], amount: [green]{staking_balance}[/green], "
                f"tolerance percentage: [green]{rate_tolerance * 100}%[/green], "
                f"price limit: [green]{rate_with_tolerance}[/green], "
                f"original price: [green]{base_rate}[/green], "
                f"with partial stake: [green]{allow_partial_stake}[/green] "
                f"on [blue]{subtensor.network}[/blue][/magenta]...[/magenta]"
            )

            call_params.update(
                {
                    "limit_price": price_with_tolerance,
                    "allow_partial": allow_partial_stake,
                }
            )
            call_function = "add_stake_limit"
        else:
            logging.info(
                f":satellite: [magenta]Staking to:[/magenta] "
                f"[blue]netuid: [green]{netuid}[/green], amount: [green]{staking_balance}[/green] "
                f"on [blue]{subtensor.network}[/blue][magenta]...[/magenta]"
            )
            call_function = "add_stake"

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )

        success, message = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            use_nonce=True,
            sign_with="coldkey",
            nonce_key="coldkeypub",
            period=period,
        )
        if success is True:  # If we successfully staked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] "
                f"[blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
            )
            new_block = subtensor.get_current_block()
            new_balance = subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block=new_block
            )
            new_stake = subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=netuid,
                block=new_block,
            )
            logging.info(
                f"Balance: [blue]{old_balance}[/blue] :arrow_right: {new_balance}[/green]"
            )
            logging.info(
                f"Stake: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
            )
            return True
        else:
            if safe_staking and "Custom error: 8" in message:
                logging.error(
                    ":cross_mark: [red]Failed[/red]: Price exceeded tolerance limit. Either increase price tolerance or enable partial staking."
                )
            else:
                logging.error(f":cross_mark: [red]Failed: {message}.[/red]")
            return False

    except SubstrateRequestException as error:
        logging.error(
            f":cross_mark: [red]Add Stake Error: {format_error_message((error))}[/red]"
        )
        return False


def add_stake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    amounts: Optional[list[Balance]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> bool:
    """Adds stake to each ``hotkey_ss58`` in the list, using each amount, from a common coldkey.

    Arguments:
        subtensor: The initialized SubtensorInterface object.
        wallet: Bittensor wallet object for the coldkey.
        hotkey_ss58s: List of hotkeys to stake to.
        netuids: List of netuids to stake to.
        amounts: List of amounts to stake. If `None`, stake all to the first hotkey.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns `False`
            if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`, or
            returns `False` if the extrinsic fails to be finalized within the timeout.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success: `True` if extrinsic was finalized or included in the block. `True` if any wallet was staked. If we did
            not wait for finalization/inclusion, the response is `True`.
    """

    if not isinstance(hotkey_ss58s, list) or not all(
        isinstance(hotkey_ss58, str) for hotkey_ss58 in hotkey_ss58s
    ):
        raise TypeError("hotkey_ss58s must be a list of str")

    if len(hotkey_ss58s) == 0:
        return True

    if amounts is not None and len(amounts) != len(hotkey_ss58s):
        raise ValueError("amounts must be a list of the same length as hotkey_ss58s")

    if netuids is not None and len(netuids) != len(hotkey_ss58s):
        raise ValueError("netuids must be a list of the same length as hotkey_ss58s")

    new_amounts: Sequence[Optional[Balance]]

    if amounts is None:
        new_amounts = [None] * len(hotkey_ss58s)
    else:
        new_amounts = [
            amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)
        ]
        if sum(amount.tao for amount in new_amounts) == 0:
            # Staking 0 tao
            return True

    # Decrypt keys,
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    block = subtensor.get_current_block()
    all_stakes = subtensor.get_stake_for_coldkey(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
    )
    old_stakes: list[Balance] = get_old_stakes(
        wallet=wallet, hotkey_ss58s=hotkey_ss58s, netuids=netuids, all_stakes=all_stakes
    )

    # Remove existential balance to keep key alive.
    # Keys must maintain a balance of at least 1000 rao to stay alive.
    total_staking_rao = sum(
        [amount.rao if amount is not None else 0 for amount in new_amounts]
    )
    old_balance = initial_balance = subtensor.get_balance(
        wallet.coldkeypub.ss58_address, block=block
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
        staking_all = False
        if amount is None:
            # Stake it all.
            staking_balance = Balance.from_tao(old_balance.tao)
            staking_all = True
        else:
            staking_balance = amount

        # Check enough to stake
        if staking_balance > old_balance:
            logging.error(
                f":cross_mark: [red]Not enough balance[/red]: [green]{old_balance}[/green] to stake: "
                f"[blue]{staking_balance}[/blue] from wallet: [white]{wallet.name}[/white]"
            )
            continue

        try:
            logging.info(
                f"Staking [blue]{staking_balance}[/blue] to [magenta]{hotkey_ss58}[/magenta] on netuid [blue]{netuid}[/blue]"
            )
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="add_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_staked": staking_balance.rao,
                    "netuid": netuid,
                },
            )
            success, message = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                use_nonce=True,
                nonce_key="coldkeypub",
                sign_with="coldkey",
                period=period,
            )

            if success is True:  # If we successfully staked.
                # We only wait here if we expect finalization.

                if not wait_for_finalization and not wait_for_inclusion:
                    old_balance -= staking_balance
                    successful_stakes += 1
                    if staking_all:
                        # If staked all, no need to continue
                        break

                    continue

                logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

                new_block = subtensor.get_current_block()
                new_stake = subtensor.get_stake(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                    block=new_block,
                )
                new_balance = subtensor.get_balance(
                    wallet.coldkeypub.ss58_address, block=new_block
                )
                logging.info(
                    f"Stake ({hotkey_ss58}) on netuid {netuid}: [blue]{old_stake}[/blue] :arrow_right: [green]{new_stake}[/green]"
                )
                logging.info(
                    f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                old_balance = new_balance
                successful_stakes += 1
                if staking_all:
                    # If staked all, no need to continue
                    break

            else:
                logging.error(f":cross_mark: [red]Failed[/red]: {message}")
                continue

        except SubstrateRequestException as error:
            logging.error(
                f":cross_mark: [red]Add Stake Multiple error: {format_error_message(error)}[/red]"
            )
            continue

    if successful_stakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
            f"[magenta]...[/magenta]"
        )
        new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
        logging.info(
            f"Balance: [blue]{initial_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
