import asyncio
from typing import Optional, TYPE_CHECKING

from async_substrate_interface.errors import SubstrateRequestException
from bittensor.core.extrinsics.asyncex.utils import get_extrinsic_fee
from bittensor.core.extrinsics.utils import get_old_stakes
from bittensor.utils import unlock_key, format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def unstake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: str,
    amount: Balance,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    safe_staking: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Removes stake into the wallet coldkey from the specified hotkey ``uid``.

    Parameters:
        subtensor: Subtensor instance.
        wallet: Bittensor wallet object.
        netuid: Subnet unique id.
        hotkey_ss58: The ``ss58`` address of the hotkey to unstake from.
        amount: Amount to stake as Bittensor balance.
        allow_partial_stake: If true, allows partial unstaking if price tolerance exceeded.
        safe_staking: If true, enables price safety checks.
        rate_tolerance: Maximum allowed price decrease percentage (0.005 = 0.5%).
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

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    block_hash = await subtensor.substrate.get_chain_head()
    old_balance, old_stake = await asyncio.gather(
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        subtensor.get_stake(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block_hash=block_hash,
        ),
    )

    amount.set_unit(netuid)

    # Check enough to unstake.
    if amount > old_stake:
        logging.error(
            f":cross_mark: [red]Not enough stake[/red]: [green]{old_stake}[/green] to unstake: "
            f"[blue]{amount}[/blue] from hotkey: [yellow]{wallet.hotkey_str}[/yellow]"
        )
        return False

    try:
        call_params = {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "amount_unstaked": amount.rao,
        }
        if safe_staking:
            pool = await subtensor.subnet(netuid=netuid)
            base_price = pool.price.tao

            if pool.netuid == 0:
                price_with_tolerance = base_price
            else:
                price_with_tolerance = base_price * (1 - rate_tolerance)

            logging_info = (
                f":satellite: [magenta]Safe Unstaking from:[/magenta] "
                f"netuid: [green]{netuid}[/green], amount: [green]{amount}[/green], "
                f"tolerance percentage: [green]{rate_tolerance * 100}%[/green], "
                f"price limit: [green]{price_with_tolerance}[/green], "
                f"original price: [green]{base_price}[/green], "
                f"with partial unstake: [green]{allow_partial_stake}[/green] "
                f"on [blue]{subtensor.network}[/blue]"
            )

            limit_price = Balance.from_tao(price_with_tolerance).rao
            call_params.update(
                {
                    "limit_price": limit_price,
                    "allow_partial": allow_partial_stake,
                }
            )
            call_function = "remove_stake_limit"
        else:
            logging_info = (
                f":satellite: [magenta]Unstaking from:[/magenta] "
                f"netuid: [green]{netuid}[/green], amount: [green]{amount}[/green] "
                f"on [blue]{subtensor.network}[/blue]"
            )
            call_function = "remove_stake"

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )
        fee = await get_extrinsic_fee(
            subtensor=subtensor, call=call, keypair=wallet.coldkeypub, netuid=netuid
        )
        logging.info(f"{logging_info} for fee [blue]{fee}[/blue][magenta]...[/magenta]")
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

        if success:  # If we successfully unstaked.
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            logging.info(
                f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
                f"[magenta]...[/magenta]"
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

    except SubstrateRequestException as error:
        logging.error(
            f":cross_mark: [red]Unstake filed with error: {format_error_message(error)}[/red]"
        )
        return False


async def unstake_all_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey: str,
    netuid: int,
    rate_tolerance: Optional[float] = 0.005,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> tuple[bool, str]:
    """Unstakes all TAO/Alpha associated with a hotkey from the specified subnets on the Bittensor network.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet of the stake owner.
        hotkey: The SS58 address of the hotkey to unstake from.
        netuid: The unique identifier of the subnet.
        rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
            price decrease. If not passed (None), then unstaking goes without price limit. Default is `0.005`.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        tuple[bool, str]:
            A tuple containing:
            - `True` and a success message if the unstake operation succeeded;
            - `False` and an error message otherwise.
    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False, unlock.message

    call_params = {
        "hotkey": hotkey,
        "netuid": netuid,
        "limit_price": None,
    }

    if rate_tolerance:
        current_price = (await subtensor.subnet(netuid=netuid)).price
        limit_price = current_price * (1 - rate_tolerance)
        call_params.update({"limit_price": limit_price})

    async with subtensor.substrate as substrate:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="remove_stake_full_limit",
            call_params=call_params,
        )

        return await subtensor.sign_and_send_extrinsic(
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


async def unstake_multiple_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    amounts: Optional[list[Balance]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
    unstake_all: bool = False,
) -> bool:
    """Removes stake from each ``hotkey_ss58`` in the list, using each amount, to a common coldkey.

    Args:
        subtensor: Subtensor instance.
        wallet: The wallet with the coldkey to unstake to.
        hotkey_ss58s: List of hotkeys to unstake from.
        netuids: List of netuids to unstake from.
        amounts: List of amounts to unstake. If ``None``, unstake all.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning ``True``, or
            returns ``False`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning
            ``True``, or returns ``False`` if the extrinsic fails to be finalized within the timeout.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        unstake_all: If true, unstakes all tokens. Default is ``False``.

    Returns:
        tuple[bool, str]:
            A tuple containing:
            - `True` and a success message if the unstake operation succeeded;
            - `False` and an error message otherwise.
    """
    if amounts and unstake_all:
        raise ValueError("Cannot specify both `amounts` and `unstake_all`.")

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
        amounts = [amount.set_unit(netuid) for amount, netuid in zip(amounts, netuids)]
        if sum(amount.tao for amount in amounts) == 0:
            # Staking 0 tao
            return True

    # Unlock coldkey.
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.info(
        f":satellite: [magenta]Syncing with chain:[/magenta] [blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )

    block_hash = await subtensor.substrate.get_chain_head()

    all_stakes, old_balance = await asyncio.gather(
        subtensor.get_stake_for_coldkey(
            coldkey_ss58=wallet.coldkeypub.ss58_address, block_hash=block_hash
        ),
        subtensor.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
    )

    old_stakes: list[Balance] = get_old_stakes(
        wallet=wallet, hotkey_ss58s=hotkey_ss58s, netuids=netuids, all_stakes=all_stakes
    )

    successful_unstakes = 0
    for idx, (hotkey_ss58, amount, old_stake, netuid) in enumerate(
        zip(hotkey_ss58s, amounts, old_stakes, netuids)
    ):
        # Convert to bittensor.Balance
        if amount is None:
            # Unstake it all.
            unstaking_balance = old_stake
            logging.warning(
                f"Didn't receive any unstaking amount. Unstaking all existing stake: [blue]{old_stake}[/blue] "
                f"from hotkey: [blue]{hotkey_ss58}[/blue]"
            )
        else:
            unstaking_balance = amount

        # Check enough to unstake.
        stake_on_uid = old_stake
        if unstaking_balance > stake_on_uid:
            logging.error(
                f":cross_mark: [red]Not enough stake[/red]: [green]{stake_on_uid}[/green] to unstake: "
                f"[blue]{unstaking_balance}[/blue] from hotkey: [blue]{wallet.hotkey_str}[/blue]."
            )
            continue

        try:
            call = await subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="remove_stake",
                call_params={
                    "hotkey": hotkey_ss58,
                    "amount_unstaked": unstaking_balance.rao,
                    "netuid": netuid,
                },
            )
            fee = await get_extrinsic_fee(
                subtensor=subtensor, call=call, keypair=wallet.coldkeypub, netuid=netuid
            )
            logging.info(
                f"Unstaking [blue]{unstaking_balance}[/blue] from hotkey: [magenta]{hotkey_ss58}[/magenta] on netuid: "
                f"[blue]{netuid}[/blue] for fee [blue]{fee}[/blue]"
            )

            staking_response, err_msg = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                nonce_key="coldkeypub",
                sign_with="coldkey",
                use_nonce=True,
                period=period,
            )

            if staking_response is True:  # If we successfully unstaked.
                # We only wait here if we expect finalization.

                if not wait_for_finalization and not wait_for_inclusion:
                    successful_unstakes += 1
                    continue

                logging.info(":white_heavy_check_mark: [green]Finalized[/green]")

                logging.info(
                    f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
                    f"[magenta]...[/magenta]..."
                )
                block_hash = await subtensor.substrate.get_chain_head()
                new_stake = await subtensor.get_stake(
                    coldkey_ss58=wallet.coldkeypub.ss58_address,
                    hotkey_ss58=hotkey_ss58,
                    netuid=netuid,
                    block_hash=block_hash,
                )
                logging.info(
                    f"Stake ({hotkey_ss58}): [blue]{stake_on_uid}[/blue] :arrow_right: [green]{new_stake}[/green]"
                )
                successful_unstakes += 1
            else:
                logging.error(f":cross_mark: [red]Failed: {err_msg}.[/red]")
                continue

        except SubstrateRequestException as error:
            logging.error(
                f":cross_mark: [red]Multiple unstake filed with error: {format_error_message(error)}[/red]"
            )
            return False

    if successful_unstakes != 0:
        logging.info(
            f":satellite: [magenta]Checking Balance on:[/magenta] [blue]{subtensor.network}[/blue] "
            f"[magenta]...[/magenta]"
        )
        block_hash = await subtensor.substrate.get_chain_head()
        new_balance = await subtensor.get_balance(
            wallet.coldkeypub.ss58_address, block_hash=block_hash
        )
        logging.info(
            f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
        )
        return True

    return False
