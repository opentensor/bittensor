from typing import TYPE_CHECKING

from bittensor.utils import unlock_key, format_error_message
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def start_call_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start a
    new subnet's emission mechanism).

    Args:
        subtensor (Subtensor): The Subtensor client instance used for blockchain interaction.
        wallet (Wallet): The wallet used to sign the extrinsic (must be unlocked).
        netuid (int): The UID of the target subnet for which the call is being initiated.
        wait_for_inclusion (bool, optional): Whether to wait for the extrinsic to be included in a block. Defaults to True.
        wait_for_finalization (bool, optional): Whether to wait for finalization of the extrinsic. Defaults to False.

    Returns:
        Tuple[bool, str]:
            - True and a success message if the extrinsic is successfully submitted or processed.
            - False and an error message if the submission fails or the wallet cannot be unlocked.
    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False, unlock.message

    with subtensor.substrate as substrate:
        start_call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function="start_call",
            call_params={"netuid": netuid},
        )

        signed_ext = substrate.create_signed_extrinsic(
            call=start_call,
            keypair=wallet.coldkey,
        )

        response = substrate.submit_extrinsic(
            extrinsic=signed_ext,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True, "Not waiting for finalization or inclusion."

        if response.is_success:
            return True, "Success with `start_call` response."

        return False, format_error_message(response.error_message)
