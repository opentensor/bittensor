from typing import TYPE_CHECKING

from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from scalecodec import GenericCall
    from bittensor_wallet import Keypair
    from bittensor.core.async_subtensor import AsyncSubtensor


async def get_unstaking_fee(
    subtensor: "AsyncSubtensor", netuid: int, call: "GenericCall", keypair: "Keypair"
):
    """
    Get unstaking fee for a given extrinsic call and keypair for a given SN's netuid.

    Arguments:
        subtensor: The Subtensor instance.
        netuid: The SN's netuid.
        call: The extrinsic call.
        keypair: The keypair associated with the extrinsic.

    Returns:
        Balance object representing the unstaking fee in RAO.
    """
    payment_info = await subtensor.substrate.get_payment_info(
        call=call, keypair=keypair
    )
    return Balance.from_rao(amount=payment_info["partial_fee"]).set_unit(netuid=netuid)
