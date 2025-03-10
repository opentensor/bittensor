from typing import TYPE_CHECKING

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
) -> tuple[bool, str]:
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
        call,
        wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
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
) -> tuple[bool, str]:
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
        call,
        wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        raise_error=raise_error,
    )
