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

    Returns:
        success (bool): True if the transfer was successful.
    """

    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: "
            f"{wallet.coldkeypub.ss58_address}"
        )
        return False

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
            call_function="transfer_stake",
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

    Returns:
        success (bool): True if the swap was successful.
    """

    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: "
            f"{wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
        subtensor,
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
        logging.info(
            f"Swapping stake for hotkey [blue]{hotkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid "
            f"[yellow]{destination_netuid}[/yellow]"
        )
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="swap_stake",
            call_params={
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

    Returns:
        success (bool): True if the move was successful.
    """

    amount.set_unit(netuid=origin_netuid)
    # Verify ownership of origin hotkey
    origin_owner = subtensor.get_hotkey_owner(origin_hotkey)
    if origin_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Origin hotkey: {origin_hotkey} does not belong to the coldkey owner: {wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
        subtensor,
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
            call_function="move_stake",
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
        )

        if success:
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            logging.success(":white_heavy_check_mark: [green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor,
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
