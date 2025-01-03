"""This module provides sync functionality for working with weights in the Bittensor network."""

from typing import Union, TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

import bittensor.utils.weight_utils as weight_utils
from bittensor.core.settings import version_as_int
from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.utils.registration import torch


async def _do_commit_weights(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Internal method to send a transaction to the Bittensor blockchain, committing the hash of a neuron's weights.
    This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This method ensures that the weight commitment is securely recorded on the Bittensor blockchain, providing a verifiable record of the neuron's weight distribution at a specific point in time.
    """
    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={
            "netuid": netuid,
            "commit_hash": commit_hash,
        },
    )

    next_nonce = await subtensor.substrate.get_account_next_index(
        wallet.hotkey.ss58_address
    )

    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.hotkey,
        nonce=next_nonce,
    )
    response = await subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    await response.process_events()
    if await response.is_success:
        return True, None
    else:
        return False, format_error_message(response.error_message)


async def commit_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `do_commit_weights` method.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The subtensor instance used for blockchain interaction.
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

    success, error_message = await _do_commit_weights(
        subtensor=subtensor,
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
        logging.error(f"Failed to commit weights: {error_message}")
        return False, error_message


async def _do_reveal_weights(
    subtensor: "AsyncSubtensor",
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
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The subtensor instance used for blockchain interaction.
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

    This method ensures that the weight revelation is securely recorded on the Bittensor blockchain, providing
        transparency and accountability for the neuron's weight distribution.
    """

    call = await subtensor.substrate.compose_call(
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
    next_nonce = await subtensor.substrate.get_account_next_index(
        wallet.hotkey.ss58_address
    )
    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.hotkey,
        nonce=next_nonce,
    )
    response = await subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    if await response.is_success:
        return True, None
    else:
        return False, await response.error_message


async def reveal_weights_extrinsic(
    subtensor: "AsyncSubtensor",
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
    This function is a wrapper around the `_do_reveal_weights` method.

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
        tuple[bool, str]: ``True`` if the weight revelation is successful, False otherwise. And `msg`, a string value describing the success or potential error.

    This function provides a user-friendly interface for revealing weights on the Bittensor blockchain, ensuring proper error handling and user interaction when required.
    """

    success, error_message = await _do_reveal_weights(
        subtensor=subtensor,
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
        error_message = format_error_message(error_message)
        logging.error(f"Failed to reveal weights: {error_message}")
        return False, error_message


async def _do_set_weights(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    vals: list[int],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: int = 5,
) -> tuple[bool, Optional[str]]:  # (success, error_message)
    """
    Internal method to send a transaction to the Bittensor blockchain, setting weights
    for specified neurons. This method constructs and submits the transaction, handling
    retries and blockchain communication.

    Args:
        subtensor (subtensor.core.async_subtensor.AsyncSubtensor): Async Subtensor instance.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
        uids (List[int]): List of neuron UIDs for which weights are being set.
        vals (List[int]): List of weight values corresponding to each UID.
        netuid (int): Unique identifier for the network.
        version_key (int, optional): Version key for compatibility with the network.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        period (int, optional): The period in seconds to wait for extrinsic inclusion or finalization. Defaults to 5.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their
        trust in other neurons based on observed performance and contributions.
    """

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": uids,
            "weights": vals,
            "netuid": netuid,
            "version_key": version_key,
        },
    )

    next_nonce = await subtensor.substrate.get_account_next_index(
        wallet.hotkey.ss58_address
    )

    # Period dictates how long the extrinsic will stay as part of waiting pool
    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.hotkey,
        era={"period": period},
        nonce=next_nonce,
    )
    response = await subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, "Not waiting for finalization or inclusion."

    await response.process_events()
    if await response.is_success:
        return True, "Successfully set weights."
    else:
        return False, format_error_message(response.error_message)


async def set_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """Sets the given weights and values on chain for wallet hotkey account.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): Bittensor subtensor object.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to set weights for.
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights to set. These must be ``float`` s and correspond to the passed ``uid`` s.
        version_key (int): The version key of the validator.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or included in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # First convert types.
    if isinstance(uids, list):
        uids = np.array(uids, dtype=np.int64)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float32)

    # Reformat and normalize.
    weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
        uids, weights
    )

    logging.info(
        ":satellite: [magenta]Setting weights on [/magenta][blue]{subtensor.network}[/blue] [magenta]...[/magenta]"
    )
    try:
        success, error_message = await _do_set_weights(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            uids=weight_uids,
            vals=weight_vals,
            version_key=version_key,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True, "Not waiting for finalization or inclusion."

        if success is True:
            message = "Successfully set weights and Finalized."
            logging.success(f":white_heavy_check_mark: [green]{message}[/green]")
            return True, message
        else:
            logging.error(f"[red]Failed[/red] set weights. Error: {error_message}")
            return False, error_message

    except Exception as error:
        logging.error(f":cross_mark: [red]Failed[/red] set weights. Error: {error}")
        return False, str(error)
