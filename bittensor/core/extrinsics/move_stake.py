from typing import Optional, TYPE_CHECKING

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _get_stake_in_origin_and_dest(
    subtensor: "Subtensor",
    origin_hotkey_ss58: str,
    destination_hotkey_ss58: str,
    origin_coldkey_ss58: str,
    destination_coldkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
) -> tuple[Balance, Balance]:
    """Gets the current stake balances for both origin and destination addresses in their respective subnets."""
    block = subtensor.get_current_block()
    stake_in_origin = subtensor.get_stake(
        coldkey_ss58=origin_coldkey_ss58,
        hotkey_ss58=origin_hotkey_ss58,
        netuid=origin_netuid,
        block=block,
    )
    stake_in_destination = subtensor.get_stake(
        coldkey_ss58=destination_coldkey_ss58,
        hotkey_ss58=destination_hotkey_ss58,
        netuid=destination_netuid,
        block=block,
    )
    return stake_in_origin, stake_in_destination


def transfer_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    destination_coldkey_ss58: str,
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Optional[Balance] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> bool:
    """
    Transfers stake from one subnet to another while changing the coldkey owner.

    Args:
        subtensor (Subtensor): Subtensor instance.
        wallet (bittensor.wallet): The wallet to transfer stake from.
        destination_coldkey_ss58 (str): The destination coldkey SS58 address.
        hotkey_ss58 (str): The hotkey SS58 address associated with the stake.
        origin_netuid (int): The source subnet UID.
        destination_netuid (int): The destination subnet UID.
        amount (Union[Balance, float, int]): Amount to transfer.
        wait_for_inclusion (bool): If true, waits for inclusion before returning.
        wait_for_finalization (bool): If true, waits for finalization before returning.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): True if the transfer was successful.
    """

    amount.set_unit(netuid=origin_netuid)

    # Check sufficient stake
    stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
        subtensor,
        origin_hotkey_ss58=hotkey_ss58,
        destination_hotkey_ss58=hotkey_ss58,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=destination_coldkey_ss58,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {hotkey_ss58}. "
            f"Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Transferring stake from coldkey [blue]{wallet.coldkeypub.ss58_address}[/blue] to coldkey ["
            f"blue]{destination_coldkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid "
            f"[yellow]{destination_netuid}[/yellow]"
        )
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="transfer_stake_aggregate",
            call_params={
                "destination_coldkey": destination_coldkey_ss58,
                "hotkey": hotkey_ss58,
                "origin_netuid": origin_netuid,
                "destination_netuid": destination_netuid,
                "alpha_amount": amount.rao,
            },
        )

        success, err_msg = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor,
                origin_hotkey_ss58=hotkey_ss58,
                destination_hotkey_ss58=hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=destination_coldkey_ss58,
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


def swap_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Optional[Balance] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    safe_staking: bool = False,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    period: Optional[int] = None,
) -> bool:
    """
    Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.

    Args:
        subtensor (Subtensor): Subtensor instance.
        wallet (bittensor.wallet): The wallet to swap stake from.
        hotkey_ss58 (str): The hotkey SS58 address associated with the stake.
        origin_netuid (int): The source subnet UID.
        destination_netuid (int): The destination subnet UID.
        amount (Union[Balance, float]): Amount to swap.
        wait_for_inclusion (bool): If true, waits for inclusion before returning.
        wait_for_finalization (bool): If true, waits for finalization before returning.
        safe_staking (bool): If true, enables price safety checks to protect against price impact.
        allow_partial_stake (bool): If true, allows partial stake swaps when the full amount would exceed the price tolerance.
        rate_tolerance (float): Maximum allowed increase in a price ratio (0.005 = 0.5%).
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): True if the swap was successful.
    """

    amount.set_unit(netuid=origin_netuid)

    # Check sufficient stake
    stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
        subtensor=subtensor,
        origin_hotkey_ss58=hotkey_ss58,
        destination_hotkey_ss58=hotkey_ss58,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
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
            origin_pool = subtensor.subnet(netuid=origin_netuid)
            destination_pool = subtensor.subnet(netuid=destination_netuid)
            swap_rate_ratio = origin_pool.price.rao / destination_pool.price.rao
            swap_rate_ratio_with_tolerance = swap_rate_ratio * (1 + rate_tolerance)

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
            call_function = "swap_stake_limit_aggregate"
        else:
            logging.info(
                f"Swapping stake for hotkey [blue]{hotkey_ss58}[/blue]\n"
                f"Amount: [green]{amount}[/green] from netuid [green]{origin_netuid}[/green] to netuid "
                f"[green]{destination_netuid}[/green]"
            )
            call_function = "swap_stake_aggregate"

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function=call_function,
            call_params=call_params,
        )

        success, err_msg = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor=subtensor,
                origin_hotkey_ss58=hotkey_ss58,
                destination_hotkey_ss58=hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
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


def move_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    origin_hotkey: str,
    origin_netuid: int,
    destination_hotkey: str,
    destination_netuid: int,
    amount: Optional[Balance] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> bool:
    """
    Moves stake to a different hotkey and/or subnet while keeping the same coldkey owner.

    Args:
        subtensor (Subtensor): Subtensor instance.
        wallet (bittensor.wallet): The wallet to move stake from.
        origin_hotkey (str): The SS58 address of the source hotkey.
        origin_netuid (int): The netuid of the source subnet.
        destination_hotkey (str): The SS58 address of the destination hotkey.
        destination_netuid (int): The netuid of the destination subnet.
        amount (Union[Balance, float]): Amount to move.
        wait_for_inclusion (bool): If true, waits for inclusion before returning.
        wait_for_finalization (bool): If true, waits for finalization before returning.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): True if the move was successful.
    """

    amount.set_unit(netuid=origin_netuid)

    # Check sufficient stake
    stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
        subtensor=subtensor,
        origin_hotkey_ss58=origin_hotkey,
        destination_hotkey_ss58=destination_hotkey,
        origin_netuid=origin_netuid,
        destination_netuid=destination_netuid,
        origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
        destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {origin_hotkey}. Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Moving stake from hotkey [blue]{origin_hotkey}[/blue] to hotkey [blue]{destination_hotkey}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid [yellow]{destination_netuid}[/yellow]"
        )
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="move_stake_aggregate",
            call_params={
                "origin_hotkey": origin_hotkey,
                "origin_netuid": origin_netuid,
                "destination_hotkey": destination_hotkey,
                "destination_netuid": destination_netuid,
                "alpha_amount": amount.rao,
            },
        )

        success, err_msg = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor=subtensor,
                origin_hotkey_ss58=origin_hotkey,
                destination_hotkey_ss58=destination_hotkey,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
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
