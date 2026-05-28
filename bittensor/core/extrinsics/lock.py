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


def lock_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: str,
    netuid: int,
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
    Locks alpha stake on a hotkey within a subnet, building conviction over time.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet whose coldkey owns the stake to lock.
        hotkey_ss58: The SS58 address of the hotkey to lock stake on.
        netuid: The subnet UID on which to lock.
        amount: Amount of alpha to lock.
        mev_protection: If True, encrypts and submits the transaction through MEV Shield.
        period: Number of blocks during which the transaction remains valid.
        raise_error: Raises exception rather than returning failure response.
        wait_for_inclusion: Whether to wait for inclusion in a block.
        wait_for_finalization: Whether to wait for finalization.
        wait_for_revealed_execution: Whether to wait for revealed execution if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Locking stake on hotkey [blue]{hotkey_ss58}[/blue] "
            f"on subnet [yellow]{netuid}[/yellow], amount: [green]{amount}[/green]"
        )

        call = SubtensorModule(subtensor).lock_stake(
            hotkey=hotkey_ss58,
            netuid=netuid,
            amount=amount.rao,
        )

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
            logging.debug("[green]Lock stake finalized[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def move_lock_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    destination_hotkey_ss58: str,
    netuid: int,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Moves an existing lock from its current hotkey to a different hotkey on the same subnet.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet whose coldkey owns the lock.
        destination_hotkey_ss58: The SS58 address of the hotkey to move the lock to.
        netuid: The subnet UID on which the lock exists.
        mev_protection: If True, encrypts and submits the transaction through MEV Shield.
        period: Number of blocks during which the transaction remains valid.
        raise_error: Raises exception rather than returning failure response.
        wait_for_inclusion: Whether to wait for inclusion in a block.
        wait_for_finalization: Whether to wait for finalization.
        wait_for_revealed_execution: Whether to wait for revealed execution if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Moving lock to hotkey [blue]{destination_hotkey_ss58}[/blue] "
            f"on subnet [yellow]{netuid}[/yellow]"
        )

        call = SubtensorModule(subtensor).move_lock(
            destination_hotkey=destination_hotkey_ss58,
            netuid=netuid,
        )

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
            logging.debug("[green]Move lock finalized[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def set_perpetual_lock_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    enabled: bool,
    *,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Sets or clears the perpetual lock flag for the caller's lock on a subnet.

    When enabled, the lock does not decay over time. When disabled, normal decay resumes.

    Parameters:
        subtensor: Subtensor instance.
        wallet: The wallet whose coldkey owns the lock.
        netuid: The subnet UID for which to set the perpetual lock flag.
        enabled: If True, the lock will not decay. If False, normal decay resumes.
        period: Number of blocks during which the transaction remains valid.
        raise_error: Raises exception rather than returning failure response.
        wait_for_inclusion: Whether to wait for inclusion in a block.
        wait_for_finalization: Whether to wait for finalization.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        logging.debug(
            f"Setting perpetual lock to [green]{enabled}[/green] "
            f"on subnet [yellow]{netuid}[/yellow]"
        )

        call = SubtensorModule(subtensor).set_perpetual_lock(
            netuid=netuid,
            enabled=enabled,
        )

        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

        if response.success:
            logging.debug("[green]Set perpetual lock finalized[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
