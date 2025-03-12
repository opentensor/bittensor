import asyncio
from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def _get_stake_in_origin_and_dest(
    subtensor: "AsyncSubtensor",
    origin_hotkey_ss58: str,
    destination_hotkey_ss58: str,
    origin_coldkey_ss58: str,
    destination_coldkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
) -> tuple[Balance, Balance]:
    block_hash = await subtensor.substrate.get_chain_head()
    stake_in_origin, stake_in_destination = await asyncio.gather(
        subtensor.get_stake(
            coldkey_ss58=origin_coldkey_ss58,
            hotkey_ss58=origin_hotkey_ss58,
            netuid=origin_netuid,
            block_hash=block_hash,
        ),
        subtensor.get_stake(
            coldkey_ss58=destination_coldkey_ss58,
            hotkey_ss58=destination_hotkey_ss58,
            netuid=destination_netuid,
            block_hash=block_hash,
        ),
    )
    return stake_in_origin, stake_in_destination


async def transfer_stake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    destination_coldkey_ss58: str,
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Transfers stake from one coldkey to another in the Bittensor network.

    Args:
        subtensor (AsyncSubtensor): The subtensor instance to interact with the blockchain.
        wallet (Wallet): The wallet containing the coldkey to authorize the transfer.
        destination_coldkey_ss58 (str): SS58 address of the destination coldkey.
        hotkey_ss58 (str): SS58 address of the hotkey associated with the stake.
        origin_netuid (int): Network UID of the origin subnet.
        destination_netuid (int): Network UID of the destination subnet.
        amount (Balance): The amount of stake to transfer as a `Balance` object.
        wait_for_inclusion (bool): If True, waits for transaction inclusion in a block. Defaults to `True`.
        wait_for_finalization (bool): If True, waits for transaction finalization. Defaults to `False`.

    Returns:
        bool: True if the transfer was successful, False otherwise.
    """

    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = await subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: "
            f"{wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin, stake_in_destination = await _get_stake_in_origin_and_dest(
        subtensor,
        origin_hotkey_ss58=hotkey_ss58,
        destination_hotkey_ss58=hotkey_ss58,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=destination_coldkey_ss58,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {hotkey_ss58}. "
            f"Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Transferring stake from coldkey [blue]{wallet.coldkeypub.ss58_address}[/blue] to coldkey "
            f"[blue]{destination_coldkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid "
            f"[yellow]{destination_netuid}[/yellow]"
        )
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="transfer_stake",
            call_params={
                "destination_coldkey": destination_coldkey_ss58,
                "hotkey": hotkey_ss58,
                "origin_netuid": origin_netuid,
                "destination_netuid": destination_netuid,
                "alpha_amount": amount.rao,
            },
        )

        success, err_msg = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = await _get_stake_in_origin_and_dest(
                subtensor,
                origin_hotkey_ss58=hotkey_ss58,
                destination_hotkey_ss58=hotkey_ss58,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=destination_coldkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
            )
            logging.info(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.info(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            return True
        else:
            logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
            return False

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed[/red]: {str(e)}")
        return False


async def swap_stake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    safe_staking: bool = False,
    allow_partial_stake: bool = False,
    rate_threshold: float = 0.005,
) -> bool:
    """
    Swaps stake from one subnet to another for a given hotkey in the Bittensor network.

    Args:
        subtensor (AsyncSubtensor): The subtensor instance to interact with the blockchain.
        wallet (Wallet): The wallet containing the coldkey to authorize the swap.
        hotkey_ss58 (str): SS58 address of the hotkey associated with the stake.
        origin_netuid (int): Network UID of the origin subnet.
        destination_netuid (int): Network UID of the destination subnet.
        amount (Balance): The amount of stake to swap as a `Balance` object.
        wait_for_inclusion (bool): If True, waits for transaction inclusion in a block. Defaults to True.
        wait_for_finalization (bool): If True, waits for transaction finalization. Defaults to False.
        safe_staking (bool): If true, enables price safety checks to protect against price impact.
        allow_partial_stake (bool): If true, allows partial stake swaps when the full amount would exceed the price threshold.
        rate_threshold (float): Maximum allowed increase in price ratio (0.005 = 0.5%).

    Returns:
        bool: True if the swap was successful, False otherwise.
    """
    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = await subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: "
            f"{wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin, stake_in_destination = await _get_stake_in_origin_and_dest(
        subtensor,
        origin_hotkey_ss58=hotkey_ss58,
        destination_hotkey_ss58=hotkey_ss58,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {hotkey_ss58}. "
            f"Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        call_params = {
            "hotkey": hotkey_ss58,
            "origin_netuid": origin_netuid,
            "destination_netuid": destination_netuid,
            "alpha_amount": amount.rao,
        }

        if safe_staking:
            origin_pool, destination_pool = await asyncio.gather(
                subtensor.subnet(netuid=origin_netuid),
                subtensor.subnet(netuid=destination_netuid),
            )
            swap_rate_ratio = origin_pool.price.rao / destination_pool.price.rao
            swap_rate_ratio_with_tolerance = swap_rate_ratio * (1 + rate_threshold)

            logging.info(
                f"Swapping stake with safety for hotkey [blue]{hotkey_ss58}[/blue]\n"
                f"Amount: [green]{amount}[/green] from netuid [green]{origin_netuid}[/green] to netuid "
                f"[green]{destination_netuid}[/green]\n"
                f"Current price ratio: [green]{swap_rate_ratio:.4f}[/green], "
                f"Ratio with tolerance: [green]{swap_rate_ratio_with_tolerance:.4f}[/green]"
            )
            call_params.update(
                {
                    "limit_price": swap_rate_ratio_with_tolerance,
                    "allow_partial": allow_partial_stake,
                }
            )
            call_function = "swap_stake_limit"
        else:
            logging.info(
                f"Swapping stake for hotkey [blue]{hotkey_ss58}[/blue]\n"
                f"Amount: [green]{amount}[/green] from netuid [green]{origin_netuid}[/green] to netuid "
                f"[green]{destination_netuid}[/green]"
            )
            call_function = "swap_stake"

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )

        success, err_msg = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = await _get_stake_in_origin_and_dest(
                subtensor,
                origin_hotkey_ss58=hotkey_ss58,
                destination_hotkey_ss58=hotkey_ss58,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
            )
            logging.info(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.info(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            return True
        else:
            if safe_staking and "Custom error: 8" in err_msg:
                logging.error(
                    ":cross_mark: [red]Failed[/red]: Price ratio exceeded tolerance limit. Either increase price tolerance or enable partial staking."
                )
            else:
                logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
            return False

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed[/red]: {str(e)}")
        return False


async def move_stake_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    origin_hotkey: str,
    origin_netuid: int,
    destination_hotkey: str,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Moves stake from one hotkey to another within subnets in the Bittensor network.

    Args:
        subtensor (Subtensor): The subtensor instance to interact with the blockchain.
        wallet (Wallet): The wallet containing the coldkey to authorize the move.
        origin_hotkey (str): SS58 address of the origin hotkey associated with the stake.
        origin_netuid (int): Network UID of the origin subnet.
        destination_hotkey (str): SS58 address of the destination hotkey.
        destination_netuid (int): Network UID of the destination subnet.
        amount (Balance): The amount of stake to move as a `Balance` object.
        wait_for_inclusion (bool): If True, waits for transaction inclusion in a block. Defaults to True.
        wait_for_finalization (bool): If True, waits for transaction finalization. Defaults to False.

    Returns:
        bool: True if the move was successful, False otherwise.
    """
    amount.set_unit(netuid=origin_netuid)

    # Check sufficient stake
    stake_in_origin, stake_in_destination = await _get_stake_in_origin_and_dest(
        subtensor,
        origin_hotkey_ss58=origin_hotkey,
        destination_hotkey_ss58=destination_hotkey,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {origin_hotkey}. "
            f"Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Moving stake from hotkey [blue]{origin_hotkey}[/blue] to hotkey [blue]{destination_hotkey}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid "
            f"[yellow]{destination_netuid}[/yellow]"
        )
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="move_stake",
            call_params={
                "origin_hotkey": origin_hotkey,
                "origin_netuid": origin_netuid,
                "destination_hotkey": destination_hotkey,
                "destination_netuid": destination_netuid,
                "alpha_amount": amount.rao,
            },
        )

        success, err_msg = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = await _get_stake_in_origin_and_dest(
                subtensor,
                origin_hotkey_ss58=origin_hotkey,
                destination_hotkey_ss58=destination_hotkey,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
            )
            logging.info(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.info(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            return True
        else:
            logging.error(f":cross_mark: [red]Failed[/red]: {err_msg}")
            return False

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed[/red]: {str(e)}")
        return False
