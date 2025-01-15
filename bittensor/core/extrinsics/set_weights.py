"""Module sync setting weights extrinsic."""

from typing import Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.extrinsics.asyncex.weights import (
    set_weights_extrinsic as async_set_weights_extrinsic,
)
from bittensor.utils.registration import torch

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet


def set_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    return subtensor.execute_coroutine(
        coroutine=async_set_weights_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )
