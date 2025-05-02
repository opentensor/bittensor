from typing import TYPE_CHECKING, Optional

from bittensor_wallet.bittensor_wallet import Wallet

from bittensor.utils import unlock_key

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def increase_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    raise_error: bool = False,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """Sets the delegate 'take' percentage for a neuron identified by its hotkey.

    Args:
        subtensor (Subtensor): Blockchain connection.
        wallet (Wallet): The wallet to sign the extrinsic.
        hotkey_ss58 (str): SS58 address of the hotkey to set take for.
        take (int): The percentage of rewards that the delegate claims from nominators.
        wait_for_inclusion (bool, optional): Wait for inclusion before returning. Defaults to True.
        wait_for_finalization (bool, optional): Wait for finalization before returning. Defaults to True.
        raise_error (bool, optional): Raise error on failure. Defaults to False.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]: Success flag and status message.
    """

    unlock = unlock_key(wallet, raise_error=raise_error)

    if not unlock.success:
        return False, unlock.message

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
    )


async def decrease_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    raise_error: bool = False,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """Sets the delegate 'take' percentage for a neuron identified by its hotkey.

    Args:
        subtensor (Subtensor): Blockchain connection.
        wallet (Wallet): The wallet to sign the extrinsic.
        hotkey_ss58 (str): SS58 address of the hotkey to set take for.
        take (int): The percentage of rewards that the delegate claims from nominators.
        wait_for_inclusion (bool, optional): Wait for inclusion before returning. Defaults to True.
        wait_for_finalization (bool, optional): Wait for finalization before returning. Defaults to True.
        raise_error (bool, optional): Raise error on failure. Defaults to False.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]: Success flag and status message.
    """
    unlock = unlock_key(wallet, raise_error=raise_error)

    if not unlock.success:
        return False, unlock.message

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
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )
