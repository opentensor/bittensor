"""Module sync commit weights and reveal weights extrinsic."""

from typing import TYPE_CHECKING, Optional

from bittensor.utils import get_function_name
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def commit_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
) -> tuple[bool, str]:
    """
    Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
    This function is a wrapper around the `do_commit_weights` method.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
        netuid (int): The unique identifier of the subnet.
        commit_hash (str): The hash of the neuron's weights to be committed.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error (bool): Whether to raise an error if the transaction fails.

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
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
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
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        raise_error: raises the relevant exception rather than returning `False` if unsuccessful.

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
