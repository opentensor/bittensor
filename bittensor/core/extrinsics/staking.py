from typing import Union, Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.staking import (
    add_stake_extrinsic as async_add_stake_extrinsic,
    add_stake_multiple_extrinsic as async_add_stake_multiple_extrinsic,
)
from bittensor.utils import execute_coroutine

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.balance import Balance


def add_stake_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58: Optional[str] = None,
    amount: Optional[Union["Balance", float]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    return execute_coroutine(
        coroutine=async_add_stake_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=subtensor.event_loop,
    )


def add_stake_multiple_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    amounts: Optional[list[Union["Balance", float]]] = None,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    return execute_coroutine(
        coroutine=async_add_stake_multiple_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=subtensor.event_loop,
    )
