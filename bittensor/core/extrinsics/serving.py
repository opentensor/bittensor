from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.serving import (
    do_serve_axon as async_do_serve_axon,
    serve_axon_extrinsic as async_serve_axon_extrinsic,
    publish_metadata as async_publish_metadata,
    get_metadata as async_get_metadata,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.axon import Axon
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.types import AxonServeCallParams
    from bittensor.utils import Certificate


def do_serve_axon(
    subtensor: "Subtensor",
    wallet: "Wallet",
    call_params: "AxonServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[dict]]:
    return subtensor.execute_coroutine(
        coroutine=async_do_serve_axon(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            call_params=call_params,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )


def serve_axon_extrinsic(
    subtensor: "Subtensor",
    netuid: int,
    axon: "Axon",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    certificate: Optional["Certificate"] = None,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_serve_axon_extrinsic(
            subtensor=subtensor.async_subtensor,
            netuid=netuid,
            axon=axon,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            certificate=certificate,
        )
    )


def publish_metadata(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: bytes,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    return subtensor.execute_coroutine(
        coroutine=async_publish_metadata(
            subtensor=subtensor.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            data_type=data_type,
            data=data,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    )


def get_metadata(
    subtensor: "Subtensor", netuid: int, hotkey: str, block: Optional[int] = None
) -> str:
    return subtensor.execute_coroutine(
        coroutine=async_get_metadata(
            subtensor=subtensor.async_subtensor,
            netuid=netuid,
            hotkey=hotkey,
            block=block,
        )
    )
