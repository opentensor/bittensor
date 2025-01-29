from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


async def transfer_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    destination_coldkey_ss58: str,
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = await subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: {wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin = await subtensor.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        netuid=origin_netuid,
    )
    stake_in_destination = await subtensor.get_stake(
        coldkey_ss58=destination_coldkey_ss58,
        hotkey_ss58=hotkey_ss58,
        netuid=destination_netuid,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {hotkey_ss58}. Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Transferring stake from coldkey [blue]{wallet.coldkeypub.ss58_address}[/blue] to coldkey [blue]{destination_coldkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid [yellow]{destination_netuid}[/yellow]"
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
            block = await subtensor.get_current_block()
            origin_stake = await subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=origin_netuid,
                block=block,
            )
            dest_stake = await subtensor.get_stake(
                coldkey_ss58=destination_coldkey_ss58,
                hotkey_ss58=hotkey_ss58,
                netuid=destination_netuid,
                block=block,
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
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    amount.set_unit(netuid=origin_netuid)
    # Verify ownership
    hotkey_owner = await subtensor.get_hotkey_owner(hotkey_ss58)
    if hotkey_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Hotkey: {hotkey_ss58} does not belong to the origin coldkey owner: {wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin = await subtensor.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        netuid=origin_netuid,
    )
    stake_in_destination = await subtensor.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=hotkey_ss58,
        netuid=destination_netuid,
    )
    if stake_in_origin < amount:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Insufficient stake in origin hotkey: {hotkey_ss58}. Stake: {stake_in_origin}, amount: {amount}"
        )
        return False

    try:
        logging.info(
            f"Swapping stake for hotkey [blue]{hotkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid [yellow]{destination_netuid}[/yellow]"
        )
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="swap_stake",
            call_params={
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
            block = await subtensor.get_current_block()
            origin_stake = await subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=origin_netuid,
                block=block,
            )
            dest_stake = await subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=hotkey_ss58,
                netuid=destination_netuid,
                block=block,
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


async def move_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    origin_hotkey: str,
    origin_netuid: int,
    destination_hotkey: str,
    destination_netuid: int,
    amount: Balance,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    amount.set_unit(netuid=origin_netuid)
    # Verify ownership of origin hotkey
    origin_owner = await subtensor.get_hotkey_owner(origin_hotkey)
    if origin_owner != wallet.coldkeypub.ss58_address:
        logging.error(
            f":cross_mark: [red]Failed[/red]: Origin hotkey: {origin_hotkey} does not belong to the coldkey owner: {wallet.coldkeypub.ss58_address}"
        )
        return False

    # Check sufficient stake
    stake_in_origin = await subtensor.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=origin_hotkey,
        netuid=origin_netuid,
    )
    stake_in_destination = await subtensor.get_stake(
        coldkey_ss58=wallet.coldkeypub.ss58_address,
        hotkey_ss58=destination_hotkey,
        netuid=destination_netuid,
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
            block = await subtensor.get_current_block()
            origin_stake = await subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=origin_hotkey,
                netuid=origin_netuid,
                block=block,
            )
            dest_stake = await subtensor.get_stake(
                coldkey_ss58=wallet.coldkeypub.ss58_address,
                hotkey_ss58=destination_hotkey,
                netuid=destination_netuid,
                block=block,
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
