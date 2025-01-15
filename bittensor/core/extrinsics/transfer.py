from typing import Union, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.transfer import (
    transfer_extrinsic as async_transfer_extrinsic,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.balance import Balance


def transfer_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    dest: str,
    amount: Union["Balance", float],
    transfer_all: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    keep_alive: bool = True,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_transfer_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            destination=dest,
            amount=amount,
            transfer_all=transfer_all,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            keep_alive=keep_alive,
        )
    )
