import bittensor
from typing import Tuple
from rich.prompt import Confirm


def schedule_coldkey_swap_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    new_coldkey: str,
    work: bytes,
    block_number: int,
    nonce: int,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    """
    Schedules a coldkey swap on the Bittensor network.

    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the current coldkey.
        new_coldkey (str): The SS58 address of the new coldkey.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.

    Returns:
        Tuple[bool, str]: ``True`` if the operation was successful, False otherwise. And `msg`, a string
        value describing the success or potential error.
    """
    # Decrypt keys
    wallet.coldkey

    if prompt and not Confirm.ask(
        f"Would you like to schedule a coldkey swap to {new_coldkey}?"
    ):
        return False, "User cancelled the operation."

    with bittensor.__console__.status(":satellite: Scheduling coldkey swap..."):
        try:
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="schedule_coldkey_swap",
                call_params={
                    "new_coldkey": new_coldkey,
                    "work": [int(byte_) for byte_ in work],
                    "block_number": block_number,
                    "nonce": nonce,
                },
            )
            extrinsic = subtensor.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            if wait_for_inclusion or wait_for_finalization:
                response.process_events()
                if response.is_success:
                    return True, "Successfully scheduled coldkey swap."
                else:
                    return False, response.error_message
            else:
                return (
                    True,
                    "Scheduled coldkey swap without waiting for inclusion or finalization.",
                )
        except Exception as e:
            return False, str(e)
