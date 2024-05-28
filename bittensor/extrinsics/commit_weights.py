# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

""" Module commit weights and reveal weights extrinsic. """

from typing import Tuple, List

from rich.prompt import Confirm

import bittensor


def commit_weights_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    """
    Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `_do_commit_weights` method, handling user prompts and error messages.
    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
    Returns:
        Tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.
    This function provides a user-friendly interface for committing weights to the Bittensor blockchain, ensuring proper
    error handling and user interaction when required.
    """
    if prompt and not Confirm.ask(f"Would you like to commit weights?"):
        return False, "User cancelled the operation."

    success, error_message = subtensor._do_commit_weights(
        wallet=wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if success:
        bittensor.logging.info("Successfully committed weights.")
        return True, "Successfully committed weights."
    else:
        bittensor.logging.error(f"Failed to commit weights: {error_message}")
        return False, error_message


def reveal_weights_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    uids: List[int],
    weights: List[int],
    salt: List[int],
    version_key: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    """
    Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `_do_reveal_weights` method, handling user prompts and error messages.
    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (List[int]): List of neuron UIDs for which weights are being revealed.
        weights (List[int]): List of weight values corresponding to each UID.
        salt (List[int]): List of salt values corresponding to the hash function.
        version_key (int): Version key for compatibility with the network.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
    Returns:
        Tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.
    This function provides a user-friendly interface for revealing weights on the Bittensor blockchain, ensuring proper
    error handling and user interaction when required.
    """

    if prompt and not Confirm.ask(f"Would you like to reveal weights?"):
        return False, "User cancelled the operation."

    success, error_message = subtensor._do_reveal_weights(
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
        bittensor.logging.info("Successfully revealed weights.")
        return True, "Successfully revealed weights."
    else:
        bittensor.logging.error(f"Failed to reveal weights: {error_message}")
        return False, error_message
