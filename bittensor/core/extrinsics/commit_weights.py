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

"""Module commit weights and reveal weights extrinsic."""

from typing import Optional, TYPE_CHECKING

from retry import retry
from rich.prompt import Confirm

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected

# For annotation purposes
if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


# # Chain call for `commit_weights_extrinsic`
@ensure_connected
def do_commit_weights(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[dict]]:
    """
    Internal method to send a transaction to the Bittensor blockchain, committing the hash of a neuron's weights.
    This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        self (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This method ensures that the weight commitment is securely recorded on the Bittensor blockchain, providing a verifiable record of the neuron's weight distribution at a specific point in time.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        call = self.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_weights",
            call_params={
                "netuid": netuid,
                "commit_hash": commit_hash,
            },
        )
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call,
            keypair=wallet.hotkey,
        )
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True, None

        response.process_events()
        if response.is_success:
            return True, None
        else:
            return False, response.error_message

    return make_substrate_call_with_retry()


def commit_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> tuple[bool, str]:
    """
    Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `do_commit_weights` method, handling user prompts and error messages.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.
        prompt (bool): If ``True``, prompts for user confirmation before proceeding.

    Returns:
        tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.

    This function provides a user-friendly interface for committing weights to the Bittensor blockchain, ensuring proper error handling and user interaction when required.
    """
    if prompt and not Confirm.ask(f"Would you like to commit weights?"):
        return False, "User cancelled the operation."

    success, error_message = do_commit_weights(
        self=subtensor,
        wallet=wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if success:
        success_message = "Successfully committed weights."
        logging.info(success_message)
        return True, success_message
    else:
        error_message = format_error_message(error_message, substrate=subtensor.substrate)
        logging.error(f"Failed to commit weights: {error_message}")
        return False, error_message


# Chain call for `reveal_weights_extrinsic`
@ensure_connected
def do_reveal_weights(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    values: list[int],
    salt: list[int],
    version_key: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[dict]]:
    """
    Internal method to send a transaction to the Bittensor blockchain, revealing the weights for a specific subnet.
    This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        self (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[int]): List of neuron UIDs for which weights are being revealed.
        values (list[int]): List of weight values corresponding to each UID.
        salt (list[int]): List of salt values corresponding to the hash function.
        version_key (int): Version key for compatibility with the network.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This method ensures that the weight revelation is securely recorded on the Bittensor blockchain, providing transparency and accountability for the neuron's weight distribution.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        call = self.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="reveal_weights",
            call_params={
                "netuid": netuid,
                "uids": uids,
                "values": values,
                "salt": salt,
                "version_key": version_key,
            },
        )
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call,
            keypair=wallet.hotkey,
        )
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True, None

        response.process_events()
        if response.is_success:
            return True, None
        else:
            return False, response.error_message

    return make_substrate_call_with_retry()


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
    prompt: bool = False,
) -> tuple[bool, str]:
    """
    Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `_do_reveal_weights` method, handling user prompts and error messages.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[int]): List of neuron UIDs for which weights are being revealed.
        weights (list[int]): List of weight values corresponding to each UID.
        salt (list[int]): List of salt values corresponding to the hash function.
        version_key (int): Version key for compatibility with the network.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.
        prompt (bool): If ``True``, prompts for user confirmation before proceeding.

    Returns:
        tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.

    This function provides a user-friendly interface for revealing weights on the Bittensor blockchain, ensuring proper error handling and user interaction when required.
    """

    if prompt and not Confirm.ask(f"Would you like to reveal weights?"):
        return False, "User cancelled the operation."

    success, error_message = do_reveal_weights(
        self=subtensor,
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        values=weights,
        salt=salt,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if success:
        success_message = "Successfully revealed weights."
        logging.info(success_message)
        return True, success_message
    else:
        error_message = format_error_message(error_message, substrate=subtensor.substrate)
        logging.error(f"Failed to reveal weights: {error_message}")
        return False, error_message
