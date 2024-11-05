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

import json
from typing import Optional, TYPE_CHECKING
import socket
from retry import retry
from rich.prompt import Confirm

from bittensor.core import settings
from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected
from bittensor.utils.weight_utils import generate_weight_hash

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
    def make_substrate_call_with_retry(extrinsic_):
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic_,
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
    return make_substrate_call_with_retry(extrinsic)


def commit_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
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

    Returns:
        tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.

    This function provides a user-friendly interface for committing weights to the Bittensor blockchain, ensuring proper error handling and user interaction when required.
    """

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
        logging.success(success_message)
        return True, success_message
    else:
        error_message = format_error_message(
            error_message, substrate=subtensor.substrate
        )
        logging.error(f"Failed to commit weights: {error_message}")
        return False, error_message


def commit_weights_process(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    uids: list[int],
    weights: list[int],
    salt: list[int],
    version_key: int = settings.version_as_int,
    block: Optional[int] = None,
):
    """
    Lets the background_subprocess know what a commit was submitted to the chain.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        uids (list[int]): List of neuron UIDs for which weights are being committed.
        weights (list[int]): List of weight values corresponding to each UID.
        salt (list[int]): List of salt values for the hash function.
        version_key (int): Version key for network compatibility (default is settings.version_as_int).
        block (Optional[int]): Specific block number to use (default is None).

    The function calculates the necessary blocks until the next epoch and the reveal block, then the background_subprocess will
    wait until the appropriate time to reveal the weights.
    """

    def send_command(command_):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 9949))
        client.send(command_.encode())
        client.close()

    curr_block = block if block is not None else subtensor.get_current_block()
    blocks_until_next_epoch = subtensor.blocks_until_next_epoch(netuid=netuid)
    subnet_hyperparams = subtensor.get_subnet_hyperparameters(netuid=netuid)
    if subnet_hyperparams is None:
        raise ValueError(f"Subnet hyperparameters for netuid {netuid} are None.")
    subnet_tempo_blocks = subnet_hyperparams.tempo
    epoch_start_block = curr_block + blocks_until_next_epoch
    cr_periods = subnet_hyperparams.commit_reveal_weights_interval
    reveal_block = epoch_start_block + ((cr_periods - 1) * subnet_tempo_blocks) + 1
    expire_block = reveal_block + subnet_tempo_blocks

    command = f'committed "{wallet.name}" "{wallet.path}" "{wallet.hotkey_str}" "{wallet.hotkey.ss58_address}" "{curr_block}" "{reveal_block}" "{expire_block}" "{commit_hash}" "{netuid}" "{uids}" "{weights}" "{salt}" "{version_key}"'
    send_command(command)


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
    def make_substrate_call_with_retry(extrinsic_):
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic_,
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
    return make_substrate_call_with_retry(extrinsic)


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

    Returns:
        tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string
        value describing the success or potential error.

    This function provides a user-friendly interface for revealing weights on the Bittensor blockchain, ensuring proper error handling and user interaction when required.
    """
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
        logging.success(success_message)
        return True, success_message
    else:
        error_message = format_error_message(
            error_message, substrate=subtensor.substrate
        )
        logging.error(f"Failed to reveal weights: {error_message}")
        return False, error_message


def reveal_weights_process(
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    weights: list[int],
    salt: list[int],
    version_key: int = settings.version_as_int,
):
    """
    Coordinates the process of revealing weights with the background background_subprocess.

    This method generates a hash of the weights using the provided wallet and network
    parameters, and sends a command to a local background_subprocess that this commit was revealed.
    In case of any exception during hash generation, it sends a command with detailed information
    including wallet details and weight parameters.

    Args:
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[int]): List of neuron UIDs for which weights are being revealed.
        weights (list[int]): List of weight values corresponding to each UID.
        salt (list[int]): List of salt values corresponding to the hash function.
        version_key (int): Version key for compatibility with the network. Defaults to `settings.version_as_int`.
    """

    def send_command(command_):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 9949))
        client.send(command_.encode())
        client.close()

    try:
        # Generate the hash of the weights - so we can remove from local reveal background_subprocess
        commit_hash = generate_weight_hash(
            address=wallet.hotkey.ss58_address,
            netuid=netuid,
            uids=list(uids),
            values=list(weights),
            salt=salt,
            version_key=version_key,
        )
        command = f'revealed_hash "{commit_hash}"'
        send_command(command)
    except Exception as e:
        logging.error(
            f"Not able to generate hash to reveal weights on background_subprocess: {e}"
        )


# Chain call for `batch_reveal_weights_extrinsic`
@ensure_connected
def do_batch_reveal_weights(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[list[int]],
    values: list[list[int]],
    salt: list[list[int]],
    version_keys: list[int],
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[dict]]:
    """
    Internal method to send a batch transaction to the Bittensor blockchain, revealing the weights for a specific subnet.
    This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        self (bittensor.core.subtensor.Subtensor): The Subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[list[int]]): List of neuron UIDs for which weights are being revealed.
        values (list[list[int]]): List of weight values corresponding to each UID.
        salt (list[list[int]]): List of salt values corresponding to the hash function.
        version_keys (list[int]): Version key for compatibility with the network.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to False.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Defaults to False.

    Returns:
        tuple[bool, Optional[dict]]: A tuple containing a success flag and an optional error message.

    This method ensures that the weight revelation is securely recorded on the Bittensor blockchain, providing transparency and accountability for the neuron's weight distribution.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry(extrinsic_):
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic_,
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

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="batch_reveal_weights",
        call_params={
            "netuid": netuid,
            "uids_list": uids,
            "values_list": values,
            "salts_list": salt,
            "version_keys": version_keys,
        },
    )
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.hotkey,
    )
    return make_substrate_call_with_retry(extrinsic)


def batch_reveal_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[list[int]],
    weights: list[list[int]],
    salt: list[list[int]],
    version_keys: list[int],
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `do_batch_reveal_weights` method, handling user prompts and error messages.

    Args:
        version_keys:
        subtensor (bittensor.core.subtensor.Subtensor): The Subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[list[int]]): List of neuron UID lists for which weights are being revealed in batch.
        weights (list[list[int]]): List of weight value lists corresponding to each UID list.
        salt (list[list[int]]): List of salt value lists corresponding to the hash function for each batch.
        version_keys (list[int]): List of version keys for compatibility with the network for each batch.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults to False.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Defaults to False.

    Returns:
        tuple[bool, str]: ``True`` if the weight revelation is successful, ``False`` otherwise. And `msg`, a string
        describing the success or potential error.

    This function provides a user-friendly interface for revealing weights in batch on the Bittensor blockchain,
    ensuring proper error handling and user interaction when required.
    """

    success, error_message = do_batch_reveal_weights(
        self=subtensor,
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        values=weights,
        salt=salt,
        version_keys=version_keys,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if success:
        success_message = "Successfully batch revealed weights."
        logging.success(success_message)
        return True, success_message
    else:
        error_message = format_error_message(
            error_message, substrate=subtensor.substrate
        )
        logging.error(f"Failed batch reveal weights extrinsic: {error_message}")
        return False, error_message


def batch_reveal_weights_process(
    wallet: "Wallet",
    netuid: int,
    uids: list[list[int]],
    weights: list[list[int]],
    salt: list[list[int]],
    version_keys: list[int],
):
    """
    Processes a batch reveal of weights for a specific subnet on the Bittensor blockchain using the provided wallet.
    This function generates the hash of weights for each batch and sends the corresponding command.

    Args:
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[list[int]]): List of neuron UID lists for which weights are being revealed in batch.
        weights (list[list[int]]): List of weight value lists corresponding to each UID list.
        salt (list[list[int]]): List of salt value lists corresponding to the hash function for each batch.
        version_keys (list[int]): List of version keys for compatibility with the network for each batch.

    This function facilitates the batch reveal process, ensuring that the hashed weights are properly recorded and sent.
    """

    def send_command(command_):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 9949))
        client.send(command_.encode())
        client.close()

    try:
        commit_hashes = []
        for batch_uids, batch_weights, batch_salt, batch_version_key in zip(
            uids, weights, salt, version_keys
        ):
            # Generate the hash of the weights for each individual batch
            commit_hash = generate_weight_hash(
                address=wallet.hotkey.ss58_address,
                netuid=netuid,
                uids=batch_uids,
                values=batch_weights,
                salt=batch_salt,
                version_key=batch_version_key,
            )
            commit_hashes.append(commit_hash)

        command = f"revealed_hash_batch {json.dumps(commit_hashes)}"
        send_command(command)
    except Exception as e:
        logging.error(f"Failed batch reveal weights background_subprocess: {e}")
