from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import SubtensorModule
from bittensor.core.extrinsics.utils import (
    compute_coldkey_hash,
    verify_coldkey_hash,
)
from bittensor_wallet import Keypair
from bittensor.core.settings import DEFAULT_MEV_PROTECTION
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def announce_coldkey_swap_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    new_coldkey_ss58: str,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Announces a coldkey swap by submitting the BlakeTwo256 hash of the new coldkey.

    This extrinsic allows a coldkey to declare its intention to swap to a new coldkey address. The announcement
    must be made before the actual swap can be executed, and a delay period must pass before execution is allowed.
    After making an announcement, all transactions from the coldkey are blocked except for `swap_coldkey_announced`.

    Parameters:
        subtensor: AsyncSubtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the current coldkey wallet).
        new_coldkey_ss58: SS58 address of the new coldkey that will replace the current one.
        mev_protection: If ``True``, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If ``False``, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning ``False`` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - A swap cost is charged when making the first announcement (not when reannouncing).
        - After making an announcement, all transactions from the coldkey are blocked except for `swap_coldkey_announced`.
        - The swap can only be executed after the delay period has passed (check via `get_coldkey_swap_announcement`).
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # Compute hash of new coldkey
        new_coldkey = Keypair(ss58_address=new_coldkey_ss58)
        new_coldkey_hash = compute_coldkey_hash(new_coldkey)

        logging.debug(
            f"Announcing coldkey swap: current=[blue]{wallet.coldkeypub.ss58_address}[/blue], "
            f"new=[blue]{new_coldkey_ss58}[/blue], "
            f"hash=[blue]{new_coldkey_hash}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = await SubtensorModule(subtensor).announce_coldkey_swap(
            new_coldkey_hash=new_coldkey_hash
        )

        if mev_protection:
            response = await submit_encrypted_extrinsic(
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
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            logging.debug("[green]Coldkey swap announced successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def swap_coldkey_announced_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    new_coldkey_ss58: str,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Executes a previously announced coldkey swap.

    This extrinsic executes a coldkey swap that was previously announced via `announce_coldkey_swap_extrinsic`.
    The new coldkey address must match the hash that was announced, and the delay period must have passed.

    Parameters:
        subtensor: AsyncSubtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (should be the current coldkey wallet that made the announcement).
        new_coldkey_ss58: SS58 address of the new coldkey to swap to. This must match the hash that was announced.
        mev_protection: If ``True``, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If ``False``, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning ``False`` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - The new coldkey hash must match the hash that was announced.
        - The delay period must have passed (check via `get_coldkey_swap_announcement`).
        - All assets, stakes, subnet ownerships, and hotkey associations are transferred from the old coldkey to the new
            one.
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # Verify announcement exists and hash matches
        announcement = await subtensor.get_coldkey_swap_announcement(
            coldkey_ss58=wallet.coldkeypub.ss58_address
        )

        if announcement is None:
            error_msg = "No coldkey swap announcement found. Make an announcement first using announce_coldkey_swap_extrinsic."
            if raise_error:
                raise ValueError(error_msg)
            return ExtrinsicResponse(
                success=False,
                message=error_msg,
                extrinsic_receipt=None,
            )

        new_coldkey = Keypair(ss58_address=new_coldkey_ss58)
        if not verify_coldkey_hash(new_coldkey, announcement.new_coldkey_hash):
            error_msg = (
                f"New coldkey hash does not match announcement. "
                f"Expected: {announcement.new_coldkey_hash}, "
                f"Got: {compute_coldkey_hash(new_coldkey)}"
            )
            if raise_error:
                raise ValueError(error_msg)
            return ExtrinsicResponse(
                success=False,
                message=error_msg,
                extrinsic_receipt=None,
            )

        # Check if delay has passed
        current_block = await subtensor.get_current_block()
        if current_block < announcement.execution_block:
            error_msg = (
                f"Swap too early. Current block: {current_block}, "
                f"Execution block: {announcement.execution_block}. "
                f"Wait for {announcement.execution_block - current_block} more blocks."
            )
            if raise_error:
                raise ValueError(error_msg)
            return ExtrinsicResponse(
                success=False,
                message=error_msg,
                extrinsic_receipt=None,
            )

        logging.debug(
            f"Executing coldkey swap: current=[blue]{wallet.coldkeypub.ss58_address}[/blue], "
            f"new=[blue]{new_coldkey_ss58}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = await SubtensorModule(subtensor).swap_coldkey_announced(
            new_coldkey=new_coldkey_ss58
        )

        if mev_protection:
            response = await submit_encrypted_extrinsic(
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
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            logging.debug("[green]Coldkey swap executed successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def remove_coldkey_swap_announcement_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    coldkey_ss58: str,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Removes a coldkey swap announcement.

    This extrinsic can only called by root. It removes a pending coldkey swap announcement for the specified coldkey.

    Parameters:
        subtensor: AsyncSubtensor instance with the connection to the chain.
        wallet: Bittensor wallet object (must be root/admin wallet).
        coldkey_ss58: SS58 address of the coldkey to remove the swap announcement for.
        mev_protection: If ``True``, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If ``False``, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You
            can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning ``False`` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Notes:
        - This function can only called by root.
        - See: <https://docs.learnbittensor.org/keys/coldkey-swap>
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Removing coldkey swap announcement: coldkey=[blue]{coldkey_ss58}[/blue] "
            f"on [blue]{subtensor.network}[/blue]."
        )

        call = await SubtensorModule(subtensor).remove_coldkey_swap_announcement(
            coldkey=coldkey_ss58
        )

        if mev_protection:
            response = await submit_encrypted_extrinsic(
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
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            logging.debug(
                "[green]Coldkey swap announcement removed successfully.[/green]"
            )
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
