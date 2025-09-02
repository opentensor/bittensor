"""Module sync commit weights and reveal weights extrinsic."""

from typing import TYPE_CHECKING, Optional

from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _do_commit_weights(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit_hash: str,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
    raise_error: bool = True,
) -> tuple[bool, str]:
    """
    Internal method to send a transaction to the Bittensor blockchain, committing the hash of a neuron's weights.
    This method constructs and submits the transaction, handling retries and blockchain communication.

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
        raise_error: raises the relevant exception rather than returning `True` if unsuccessful.

    Returns:
        tuple[bool, str]:
            `True` if the weight commitment is successful, `False` otherwise.
            `msg` is a string value describing the success or potential error.

    This method ensures that the weight commitment is securely recorded on the Bittensor blockchain, providing a
        verifiable record of the neuron's weight distribution at a specific point in time.
    """
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="commit_weights",
        call_params={
            "netuid": netuid,
            "commit_hash": commit_hash,
        },
    )
    return subtensor.sign_and_send_extrinsic(
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

    success, error_message = _do_commit_weights(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        commit_hash=commit_hash,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )

    if success:
        success_message = "Successfully committed weights."
        logging.info(success_message)
        return True, success_message

    logging.error(f"Failed to commit weights: {error_message}")
    return False, error_message


def _do_reveal_weights(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    values: list[int],
    salt: list[int],
    version_key: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
) -> tuple[bool, str]:
    """
    Internal method to send a transaction to the Bittensor blockchain, revealing the weights for a specific subnet.
    This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron revealing the weights.
        netuid (int): The unique identifier of the subnet.
        uids (list[int]): List of neuron UIDs for which weights are being revealed.
        values (list[int]): List of weight values corresponding to each UID.
        salt (list[int]): List of salt values corresponding to the hash function.
        version_key (int): Version key for compatibility with the network.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]:
            `True` if the weight commitment is successful, `False` otherwise.
            `msg` is a string value describing the success or potential error.

    This method ensures that the weight revelation is securely recorded on the Bittensor blockchain, providing
        transparency and accountability for the neuron's weight distribution.
    """

    call = subtensor.substrate.compose_call(
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
    return subtensor.sign_and_send_extrinsic(
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

    success, error_message = _do_reveal_weights(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        values=weights,
        salt=salt,
        version_key=version_key,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )

    if success:
        success_message = "Successfully revealed weights."
        logging.info(success_message)
        return True, success_message

    error_message = error_message
    logging.error(f"Failed to reveal weights: {error_message}")
    return False, error_message
