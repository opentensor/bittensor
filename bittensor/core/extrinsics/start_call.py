from typing import TYPE_CHECKING, Optional

from bittensor.utils import unlock_key
from bittensor.utils.btlogging import logging

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
) -> tuple[bool, str]:
    """
    Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start a
    new subnet's emission mechanism).

    Parameters:
        subtensor (Subtensor): The Subtensor client instance used for blockchain interaction.
        wallet (Wallet): The wallet used to sign the extrinsic (must be unlocked).
        netuid (int): The UID of the target subnet for which the call is being initiated.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for finalization of the extrinsic.

    Returns:
        Tuple[bool, str]:
            - True and a success message if the extrinsic is successfully submitted or processed.
            - False and an error message if the submission fails or the wallet cannot be unlocked.
    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False, unlock.message

    start_call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="start_call",
        call_params={"netuid": netuid},
    )

    success, message = subtensor.sign_and_send_extrinsic(
        call=start_call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, message

    if success:
        return True, "Success with `start_call` response."

    return True, message
