from typing import TYPE_CHECKING

from bittensor_wallet.bittensor_wallet import Wallet

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def increase_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    wait_for_inclusion=True,
    wait_for_finalization=True,
) -> None:
    wallet.unlock_coldkey()

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="increase_take",
        call_params={
            "hotkey": hotkey_ss58,
            "take": take,
        },
    )

    await subtensor.sign_and_send_extrinsic(
        call,
        wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        raise_error=True,
    )


async def decrease_take_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: Wallet,
    hotkey_ss58: str,
    take: int,
    wait_for_inclusion=True,
    wait_for_finalization=True,
) -> None:
    wallet.unlock_coldkey()

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="decrease_take",
        call_params={
            "hotkey": hotkey_ss58,
            "take": take,
        },
    )

    await subtensor.sign_and_send_extrinsic(
        call,
        wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        raise_error=True,
    )
