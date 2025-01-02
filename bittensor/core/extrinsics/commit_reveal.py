"""This module provides sync functionality for commit reveal in the Bittensor network."""

from typing import Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.extrinsics.asyncex.commit_reveal import (
    commit_reveal_v3_extrinsic as async_commit_reveal_v3_extrinsic,
)
from bittensor.core.settings import version_as_int
from bittensor.utils import execute_coroutine

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.registration import torch


def commit_reveal_v3_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    return execute_coroutine(
        coroutine=async_commit_reveal_v3_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=subtensor.event_loop,
    )
