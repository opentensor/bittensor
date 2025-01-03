from typing import Optional, TYPE_CHECKING

from bittensor.core.extrinsics.asyncex.serving import (
    do_serve_axon as async_do_serve_axon,
    serve_axon_extrinsic as async_serve_axon_extrinsic,
    publish_metadata as async_publish_metadata,
    get_metadata as async_get_metadata,
)
from bittensor.utils import execute_coroutine

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.axon import Axon
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.types import AxonServeCallParams
    from bittensor.utils import Certificate


def do_serve_axon(
    self: "Subtensor",
    wallet: "Wallet",
    call_params: "AxonServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[dict]]:
    return execute_coroutine(
        coroutine=async_do_serve_axon(
            subtensor=self.async_subtensor,
            wallet=wallet,
            call_params=call_params,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=self.event_loop,
    )


def serve_axon_extrinsic(
    subtensor: "Subtensor",
    netuid: int,
    axon: "Axon",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    certificate: Optional["Certificate"] = None,
) -> bool:
    return execute_coroutine(
        coroutine=async_serve_axon_extrinsic(
            subtensor=subtensor.async_subtensor,
            netuid=netuid,
            axon=axon,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            certificate=certificate,
        ),
        event_loop=subtensor.event_loop,
    )


def publish_metadata(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: bytes,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    return execute_coroutine(
        coroutine=async_publish_metadata(
            subtensor=self.async_subtensor,
            wallet=wallet,
            netuid=netuid,
            data_type=data_type,
            data=data,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        ),
        event_loop=self.event_loop,
    )


def get_metadata(
    self: "Subtensor", netuid: int, hotkey: str, block: Optional[int] = None
) -> str:
    return execute_coroutine(
        coroutine=async_get_metadata(
            subtensor=self.async_subtensor,
            netuid=netuid,
            hotkey=hotkey,
            block=block,
        ),
        event_loop=self.event_loop,
    )
