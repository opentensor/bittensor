from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
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


def move_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    origin_netuid: int,
    origin_hotkey_ss58: str,
    destination_netuid: int,
    destination_hotkey_ss58: str,
    amount: Optional[Balance] = None,
    move_all_stake: bool = False,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Moves stake to a different hotkey and/or subnet while keeping the same coldkey owner.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet to move stake from.
        origin_netuid: The netuid of the source subnet.
        origin_hotkey_ss58: The SS58 address of the source hotkey.
        destination_netuid: The netuid of the destination subnet.
        destination_hotkey_ss58: The SS58 address of the destination hotkey.
        amount: Amount to move.
        move_all_stake: If true, moves all stake from the source hotkey to the destination hotkey.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        if not amount and not move_all_stake:
            return ExtrinsicResponse(
                False,
                "Please specify an `amount` or `move_all_stake` argument to move stake.",
            ).with_log()

        # Check sufficient stake
        stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
            subtensor=subtensor,
            origin_netuid=origin_netuid,
            origin_hotkey_ss58=origin_hotkey_ss58,
            destination_netuid=destination_netuid,
            destination_hotkey_ss58=destination_hotkey_ss58,
            origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
            destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
        )
        if move_all_stake:
            amount = stake_in_origin

        elif stake_in_origin < amount:
            return ExtrinsicResponse(
                False,
                f"Insufficient stake in origin hotkey: {origin_hotkey_ss58}. Stake: {stake_in_origin}, amount: {amount}.",
            ).with_log()

        amount.set_unit(netuid=origin_netuid)

        logging.debug(
            f"Moving stake from hotkey [blue]{origin_hotkey_ss58}[/blue] to hotkey [blue]{destination_hotkey_ss58}[/blue]\n"
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid [yellow]{destination_netuid}[/yellow]"
        )
        call = SubtensorModule(subtensor).move_stake(
            origin_netuid=origin_netuid,
            origin_hotkey_ss58=origin_hotkey_ss58,
            destination_netuid=destination_netuid,
            destination_hotkey_ss58=destination_hotkey_ss58,
            alpha_amount=amount.rao,
        )
        block_before = subtensor.block
        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            sim_swap = subtensor.sim_swap(
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                amount=amount,
                block=block_before,
            )
            response.transaction_tao_fee = sim_swap.tao_fee
            response.transaction_alpha_fee = sim_swap.alpha_fee.set_unit(origin_netuid)

            if not wait_for_finalization and not wait_for_inclusion:
                return response

            logging.debug("[green]Finalized[/green]")

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor=subtensor,
                origin_hotkey_ss58=origin_hotkey_ss58,
                destination_hotkey_ss58=destination_hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=wallet.coldkeypub.ss58_address,
            )
            logging.debug(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.debug(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            response.data = {
                "origin_stake_before": stake_in_origin,
                "origin_stake_after": origin_stake,
                "destination_stake_before": stake_in_destination,
                "destination_stake_after": dest_stake,
            }
            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def transfer_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    destination_coldkey_ss58: str,
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Transfers stake from one subnet to another while changing the coldkey owner.

    Parameters:
        subtensor: The subtensor instance to interact with the blockchain.
        wallet: The wallet containing the coldkey to authorize the transfer.
        destination_coldkey_ss58: SS58 address of the destination coldkey.
        hotkey_ss58: SS58 address of the hotkey associated with the stake.
        origin_netuid: Network UID of the origin subnet.
        destination_netuid: Network UID of the destination subnet.
        amount: The amount of stake to transfer as a `Balance` object.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        amount.set_unit(netuid=origin_netuid)

        # Check sufficient stake
        stake_in_origin, stake_in_destination = _get_stake_in_origin_and_dest(
            subtensor=subtensor,
            origin_hotkey_ss58=hotkey_ss58,
            destination_hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
            destination_coldkey_ss58=destination_coldkey_ss58,
        )
        if stake_in_origin < amount:
            return ExtrinsicResponse(
                False,
                f"Insufficient stake in origin hotkey: {hotkey_ss58}. Stake: {stake_in_origin}, amount: {amount}.",
            ).with_log()

        logging.debug(
            f"Transferring stake from coldkey [blue]{wallet.coldkeypub.ss58_address}[/blue] to coldkey "
            f"[blue]{destination_coldkey_ss58}[/blue]"
        )
        logging.debug(
            f"Amount: [green]{amount}[/green] from netuid [yellow]{origin_netuid}[/yellow] to netuid "
            f"[yellow]{destination_netuid}[/yellow]"
        )
        call = SubtensorModule(subtensor).transfer_stake(
            destination_coldkey=destination_coldkey_ss58,
            hotkey=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            alpha_amount=amount.rao,
        )
        block_before = subtensor.block
        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            sim_swap = subtensor.sim_swap(
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                amount=amount,
                block=block_before,
            )
            response.transaction_tao_fee = sim_swap.tao_fee
            response.transaction_alpha_fee = sim_swap.alpha_fee.set_unit(origin_netuid)

            if not wait_for_finalization and not wait_for_inclusion:
                return response

            # Get updated stakes
            origin_stake, dest_stake = _get_stake_in_origin_and_dest(
                subtensor=subtensor,
                origin_hotkey_ss58=hotkey_ss58,
                destination_hotkey_ss58=hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                origin_coldkey_ss58=wallet.coldkeypub.ss58_address,
                destination_coldkey_ss58=destination_coldkey_ss58,
            )
            logging.debug(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.debug(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def swap_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    origin_netuid: int,
    destination_netuid: int,
    amount: Balance,
    safe_swapping: bool = False,
    allow_partial_stake: bool = False,
    rate_tolerance: float = 0.005,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet to swap stake from.
        hotkey_ss58: The hotkey SS58 address associated with the stake.
        origin_netuid: The source subnet UID.
        destination_netuid: The destination subnet UID.
        amount: Amount to swap.
        safe_swapping: If true, enables price safety checks to protect against price impact.
        allow_partial_stake: If true, allows partial stake swaps when the full amount would exceed the price tolerance.
        rate_tolerance: Maximum allowed increase in a price ratio (0.005 = 0.5%).
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

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
            return ExtrinsicResponse(
                False,
                f"Insufficient stake in origin hotkey: {hotkey_ss58}. Stake: {stake_in_origin}, amount: {amount}.",
            ).with_log()

        if safe_swapping:
            origin_pool = subtensor.subnet(netuid=origin_netuid)
            destination_pool = subtensor.subnet(netuid=destination_netuid)
            swap_rate_ratio = origin_pool.price.rao / destination_pool.price.rao
            swap_rate_ratio_with_tolerance = swap_rate_ratio * (1 + rate_tolerance)

            logging.debug(
                f"Swapping stake with safety for hotkey [blue]{hotkey_ss58}[/blue]\n"
                f"Amount: [green]{amount}[/green] from netuid [green]{origin_netuid}[/green] to netuid "
                f"[green]{destination_netuid}[/green]\n"
                f"Current price ratio: [green]{swap_rate_ratio:.4f}[/green], "
                f"Ratio with tolerance: [green]{swap_rate_ratio_with_tolerance:.4f}[/green]"
            )

            call = SubtensorModule(subtensor).swap_stake_limit(
                hotkey=hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                alpha_amount=amount.rao,
                limit_price=swap_rate_ratio_with_tolerance,
                allow_partial=allow_partial_stake,
            )

        else:
            logging.debug(
                f"Swapping stake for hotkey [blue]{hotkey_ss58}[/blue]\n"
                f"Amount: [green]{amount}[/green] from netuid [green]{origin_netuid}[/green] to netuid "
                f"[green]{destination_netuid}[/green]"
            )

            call = SubtensorModule(subtensor).swap_stake(
                hotkey=hotkey_ss58,
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                alpha_amount=amount.rao,
            )

        block_before = subtensor.block
        if mev_protection:
            response = submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            sim_swap = subtensor.sim_swap(
                origin_netuid=origin_netuid,
                destination_netuid=destination_netuid,
                amount=amount,
                block=block_before,
            )
            response.transaction_tao_fee = sim_swap.tao_fee
            response.transaction_alpha_fee = sim_swap.alpha_fee.set_unit(origin_netuid)

            if not wait_for_finalization and not wait_for_inclusion:
                return response

            logging.debug("[green]Finalized[/green]")

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

            logging.debug(
                f"Origin Stake: [blue]{stake_in_origin}[/blue] :arrow_right: [green]{origin_stake}[/green]"
            )
            logging.debug(
                f"Destination Stake: [blue]{stake_in_destination}[/blue] :arrow_right: [green]{dest_stake}[/green]"
            )

            response.data = {
                "origin_stake_before": stake_in_origin,
                "origin_stake_after": origin_stake,
                "destination_stake_before": stake_in_destination,
                "destination_stake_after": dest_stake,
            }
            return response

        if safe_swapping and "Custom error: 8" in response.message:
            response.message = "Price ratio exceeded tolerance limit. Either increase price tolerance or enable partial staking."

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
