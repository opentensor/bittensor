"""
This module provides functionalities for registering a wallet with the subtensor network using Proof-of-Work (PoW).

Extrinsics:
- register_extrinsic: Registers the wallet to the subnet.
- burned_register_extrinsic: Registers the wallet to chain by recycling TAO.
"""

from typing import Union, Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.registration import (
    burned_register_extrinsic as async_burned_register_extrinsic,
    register_extrinsic as async_register_extrinsic,
)

# For annotation and lazy import purposes
if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def burned_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_burned_register_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )


def register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[list[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_register_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
    )
