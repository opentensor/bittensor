from typing import Union, Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.unstaking import (
    unstake_extrinsic as async_unstake_extrinsic,
    unstake_multiple_extrinsic as async_unstake_multiple_extrinsic,
)
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def unstake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    amount: Optional[Union[Balance, float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_unstake_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )


def unstake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    amounts: Optional[list[Union[Balance, float]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_unstake_multiple_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )
