# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from typing import Union, Optional, TYPE_CHECKING
from retry import retry

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.utils.networking import ensure_connected

# For annotation purposes
if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet


@ensure_connected
def register_neuron_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    hotkey_ss58: Optional[str] = None,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[int], str]:
    """
    Registers a neuron on a specified subnet by submitting a `burned_register` extrinsic to the Bittensor blockchain.

    Args:
        subtensor (Subtensor): The Subtensor interface for blockchain interaction.
        wallet (Wallet): The wallet containing the hotkey to register.
        netuid (int): The unique identifier of the subnet to register on.
        hotkey_ss58 (Optional[str]): The SS58 address of the hotkey to register. Defaults to the wallet's hotkey.
        wait_for_inclusion (bool): Wait for the extrinsic to be included in a block.
        wait_for_finalization (bool): Wait for the extrinsic to be finalized.

    Returns:
        tuple[bool, Optional[int], str]: A tuple containing a success flag, the neuron's UID if successful, and a response message.

    Example:
        ```python
        success, uid, message = register_extrinsic(
            subtensor=subtensor,
            wallet=wallet,
            netuid=1,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if success:
            print(f"Successfully registered neuron with UID: {uid}")
        else:
            print(f"Registration failed: {message}")
        ```

    Notes:
        - This function is crucial for adding new neurons to the Bittensor network, allowing participation in subnet consensus and learning processes.
        - Ensure the wallet has sufficient balance for the registration burn.

    """
    # Use the wallet's hotkey if none is provided
    if hotkey_ss58 is None:
        hotkey_ss58: str = wallet.hotkey.ss58_address

    # Function to submit the extrinsic with retries
    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def submit_register_extrinsic() -> tuple[bool, Optional[int], str]:
        """
        Submits the `burned_register` extrinsic to the blockchain.

        Returns:
            tuple[bool, Optional[int], str]: A tuple containing a success flag, the neuron's UID if successful, and a response message.
        """
        # Compose the call to the 'burned_register' extrinsic
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="burned_register",
            call_params={
                "netuid": netuid,
                "hotkey": hotkey_ss58,
            },
        )

        # Create a signed extrinsic using the wallet's coldkey
        extrinsic = subtensor.substrate.create_signed_extrinsic(
            call=call,
            keypair=wallet.coldkey,
            era={"period": 64},  # The era defines how long the extrinsic is valid
        )

        # Submit the extrinsic to the blockchain
        response = submit_extrinsic(
            substrate=subtensor.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        # Handle cases where we're not waiting for inclusion or finalization
        if not wait_for_finalization and not wait_for_inclusion:
            return True, None, "Not waiting for finalization or inclusion."

        # Process the response events
        response.process_events()
        if response.is_success:
            # Retrieve the UID of the newly registered neuron
            uid: Optional[int] = subtensor.get_uid_for_hotkey_on_subnet(
                hotkey_ss58, netuid
            )
            return True, uid, "Successfully registered neuron."
        else:
            return False, None, response.error_message

    # Execute the extrinsic submission with error handling
    try:
        success, uid, message = submit_register_extrinsic()
        return success, uid, message
    except Exception as e:
        # Handle exceptions and return the error message
        return False, None, str(e)
