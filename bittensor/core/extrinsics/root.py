from typing import Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.extrinsics.asyncex.root import (
    root_register_extrinsic as async_root_register_extrinsic,
    set_root_weights_extrinsic as async_set_root_weights_extrinsic,
)
from bittensor.utils import execute_coroutine
from bittensor.utils.registration import torch

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def root_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    return execute_coroutine(
        coroutine=async_root_register_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=0,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=subtensor.event_loop,
    )


def set_root_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuids: Union[NDArray[np.int64], "torch.LongTensor", list[int]],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list[float]],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> bool:
    return execute_coroutine(
        coroutine=async_set_root_weights_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=subtensor.event_loop,
    )
