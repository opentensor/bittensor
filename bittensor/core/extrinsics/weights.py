"""Module sync commit weights and reveal weights extrinsic."""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from bittensor.core.settings import version_as_int
from bittensor.utils import get_function_name
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import (
    convert_and_normalize_weights_and_uids,
    convert_uids_and_weights,
)

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet
    from bittensor.utils.registration import torch


def commit_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `do_commit_weights` method.

    Parameters:
        subtensor: The subtensor instance used for blockchain interaction.
        wallet: The wallet associated with the neuron committing the weights.
        netuid: The unique identifier of the subnet.
        commit_hash: The hash of the neuron's weights to be committed.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Whether to raise an error if the transaction fails.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, str]:
            `True` if the weight commitment is successful, `False` otherwise.
            `msg` is a string value describing the success or potential error.

    This function provides a user-friendly interface for committing weights to the Bittensor blockchain, ensuring proper
    error handling and user interaction when required.
    """

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={
            "netuid": netuid,
            "commit_hash": commit_hash,
        },
    )
    success, message = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        use_nonce=True,
        period=period,
        sign_with="hotkey",
        nonce_key="hotkey",
        raise_error=raise_error,
    )

    if success:
        logging.info(message)
    else:
        logging.error(f"{get_function_name()}: {message}")
    return success, message


# TODO: deprecate in SDKv10
def reveal_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    weights: list[int],
    salt: list[int],
    version_key: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `_do_reveal_weights` method.

    Parameters:
        subtensor: The subtensor instance used for blockchain interaction.
        wallet: The wallet associated with the neuron revealing the weights.
        netuid: The unique identifier of the subnet.
        uids: List of neuron UIDs for which weights are being revealed.
        weights: List of weight values corresponding to each UID.
        salt: List of salt values corresponding to the hash function.
        version_key: Version key for compatibility with the network.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Whether to raise an error if the transaction fails.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, str]:
            `True` if the weight commitment is successful, `False` otherwise.
            `msg` is a string value describing the success or potential error.

    This function provides a user-friendly interface for revealing weights on the Bittensor blockchain, ensuring proper
        error handling and user interaction when required.
    """

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="reveal_weights",
        call_params={
            "netuid": netuid,
            "uids": uids,
            "values": weights,
            "salt": salt,
            "version_key": version_key,
        },
    )

    success, message = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        use_nonce=True,
        period=period,
        sign_with="hotkey",
        nonce_key="hotkey",
        raise_error=raise_error,
    )

    if success:
        logging.info(message)
    else:
        logging.error(f"{get_function_name()}: {message}")
    return success, message


def set_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
    period: Optional[int] = 8,
    raise_error: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Sets the given weights and values on a chain for a wallet hotkey account.

    Parameters:
        subtensor: Bittensor subtensor object.
        wallet: Bittensor wallet object.
        netuid: The ``netuid`` of the subnet to set weights for.
        uids: The ``uint64`` uids of destination neurons.
        weights: The weights to set. These must be ``float``s and correspond to the passed ``uid``s.
        version_key: The version key of the validator.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Whether to raise an error if the transaction fails.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, str]:
            `True` if the weight commitment is successful, `False` otherwise.
            `msg` is a string value describing the success or potential error.
    """
    # Convert types.
    uids, weights = convert_uids_and_weights(uids, weights)

    # Reformat and normalize.
    weight_uids, weight_vals = convert_and_normalize_weights_and_uids(uids, weights)

    logging.info(
        f":satellite: [magenta]Setting weights on [/magenta]"
        f"[blue]{subtensor.network}[/blue] "
        f"[magenta]...[/magenta]"
    )

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": weight_uids,
            "weights": weight_vals,
            "netuid": netuid,
            "version_key": version_key,
        },
    )
    success, message = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        use_nonce=True,
        nonce_key="hotkey",
        sign_with="hotkey",
        raise_error=raise_error,
    )

    if success:
        logging.info("Successfully set weights and Finalized.")
    else:
        logging.error(f"{get_function_name}: {message}")
    return success, message
