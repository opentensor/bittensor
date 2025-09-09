from typing import TYPE_CHECKING, Optional

from bittensor_wallet.bittensor_wallet import Wallet

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import unlock_key, get_function_name
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def increase_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Sets the delegate 'take' percentage for a neuron identified by its hotkey.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to sign the extrinsic.
        hotkey_ss58: SS58 address of the hotkey to set take for.
        take: The percentage of rewards that the delegate claims from nominators.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        Tuple[bool, str]:
            - True and a success message if the extrinsic is successfully submitted or processed.
            - False and an error message if the submission fails or the wallet cannot be unlocked.
    """

    unlock = unlock_key(wallet, raise_error=raise_error)
    if not unlock.success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="increase_take",
        call_params={
            "hotkey": hotkey_ss58,
            "take": take,
        },
    )

    return await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )


async def decrease_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Sets the delegate 'take' percentage for a neuron identified by its hotkey.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to sign the extrinsic.
        hotkey_ss58: SS58 address of the hotkey to set take for.
        take: The percentage of rewards that the delegate claims from nominators.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        Tuple[bool, str]:
            - True and a success message if the extrinsic is successfully submitted or processed.
            - False and an error message if the submission fails or the wallet cannot be unlocked.
    """
    unlock = unlock_key(wallet, raise_error=raise_error)
    if not unlock.success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="decrease_take",
        call_params={
            "hotkey": hotkey_ss58,
            "take": take,
        },
    )

    return await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        period=period,
        raise_error=raise_error,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        calling_function=get_function_name(),
    )
