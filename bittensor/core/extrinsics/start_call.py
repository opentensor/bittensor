from typing import TYPE_CHECKING, Optional

from bittensor.core.extrinsics.params import StartCallParams
from bittensor.core.types import ExtrinsicResponse

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def start_call_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> ExtrinsicResponse:
    """
    Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start a
    new subnet's emission mechanism).

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked).
        netuid: The UID of the target subnet for which the call is being initiated.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        start_call = subtensor.compose_call(
            call_module="SubtensorModule",
            call_function="start_call",
            call_params=StartCallParams.start_call(netuid=netuid),
        )

        return subtensor.sign_and_send_extrinsic(
            call=start_call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
            raise_error=raise_error,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
