"""Module sync commit weights and reveal weights extrinsic."""

from typing import TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.weights import (
    reveal_weights_extrinsic as async_reveal_weights_extrinsic,
    commit_weights_extrinsic as async_commit_weights_extrinsic,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def commit_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    return subtensor.execute_coroutine(
        coroutine=async_commit_weights_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            commit_hash=commit_hash,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )


def reveal_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    weights: list[int],
    salt: list[int],
    version_key: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    return subtensor.execute_coroutine(
        coroutine=async_reveal_weights_extrinsic(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )
