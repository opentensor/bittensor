from typing import TYPE_CHECKING, Optional, Union

from bittensor_drand import get_encrypted_commit

from bittensor.core.types import ExtrinsicResponse
from bittensor.core.settings import version_as_int
from bittensor.core.types import Salt, UIDs, Weights
from bittensor.utils import (
    unlock_key,
    get_mechid_storage_index,
    get_function_name,
    format_error_message,
)
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import (
    convert_and_normalize_weights_and_uids,
    generate_weight_hash,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor


async def commit_mechanism_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    mechid: int,
    uids: UIDs,
    weights: Weights,
    salt: Salt,
    version_key: int = version_as_int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Commits the weights for a specific sub subnet on the Bittensor blockchain using the provided wallet.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The subnet unique identifier.
        mechid: The subnet mechanism unique identifier.
        uids: NumPy array of neuron UIDs for which weights are being committed.
        weights: NumPy array of weight values corresponding to each UID.
        salt: list of randomly generated integers as salt to generated weighted hash.
        version_key: Version key for compatibility with the network.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        unlock = unlock_key(wallet, raise_error=raise_error)
        if not unlock.success:
            logging.error(unlock.message)
            return ExtrinsicResponse(
                False, unlock.message, extrinsic_function=get_function_name()
            )

        storage_index = get_mechid_storage_index(netuid=netuid, mechid=mechid)
        # Generate the hash of the weights
        commit_hash = generate_weight_hash(
            address=wallet.hotkey.ss58_address,
            netuid=storage_index,
            uids=list(uids),
            values=list(weights),
            salt=salt,
            version_key=version_key,
        )

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_mechanism_weights",
            call_params={
                "netuid": netuid,
                "mecid": mechid,
                "commit_hash": commit_hash,
            },
        )
        response = await subtensor.sign_and_send_extrinsic(
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

        if response.success:
            logging.debug(response.message)
            return response

        logging.error(response.message)
        return response

    except Exception as error:
        if raise_error:
            raise error

        logging.error(str(error))
        return ExtrinsicResponse(
            success=False,
            message=format_error_message(error),
            error=error,
            extrinsic_function=get_function_name(),
        )


async def commit_timelocked_mechanism_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    mechid: int,
    uids: UIDs,
    weights: Weights,
    block_time: Union[int, float],
    commit_reveal_version: int = 4,
    version_key: int = version_as_int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """Commits the weights for a specific sub subnet on the Bittensor blockchain using the provided wallet.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The unique identifier of the subnet.
        mechid: The subnet mechanism unique identifier.
        uids: The list of neuron UIDs that the weights are being set for.
        weights: The corresponding weights to be set for each UID.
        block_time: The number of seconds for block duration.
        commit_reveal_version: The version of the commit-reveal in the chain.
        version_key: Version key for compatibility with the network.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        unlock = unlock_key(wallet, raise_error=raise_error)
        if not unlock.success:
            logging.error(unlock.message)
            return ExtrinsicResponse(
                False, unlock.message, extrinsic_function=get_function_name()
            )

        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        current_block = await subtensor.block
        subnet_hyperparameters = await subtensor.get_subnet_hyperparameters(
            netuid, block=current_block
        )
        tempo = subnet_hyperparameters.tempo
        subnet_reveal_period_epochs = subnet_hyperparameters.commit_reveal_period

        storage_index = get_mechid_storage_index(netuid=netuid, mechid=mechid)

        # Encrypt `commit_hash` with t-lock and `get reveal_round`
        commit_for_reveal, reveal_round = get_encrypted_commit(
            uids=uids,
            weights=weights,
            version_key=version_key,
            tempo=tempo,
            current_block=current_block,
            netuid=storage_index,
            subnet_reveal_period_epochs=subnet_reveal_period_epochs,
            block_time=block_time,
            hotkey=wallet.hotkey.public_key,
        )

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_timelocked_mechanism_weights",
            call_params={
                "netuid": netuid,
                "mecid": mechid,
                "commit": commit_for_reveal,
                "reveal_round": reveal_round,
                "commit_reveal_version": commit_reveal_version,
            },
        )
        response = await subtensor.sign_and_send_extrinsic(
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

        if response.success:
            logging.debug(response.message)
            response.data = {
                "commit_for_reveal": commit_for_reveal,
                "reveal_round": reveal_round,
            }
            return response

        logging.error(response.message)
        return response

    except Exception as error:
        if raise_error:
            raise error

        logging.error(str(error))
        return ExtrinsicResponse(
            success=False,
            message=format_error_message(error),
            error=error,
            extrinsic_function=get_function_name(),
        )


async def reveal_mechanism_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    mechid: int,
    uids: UIDs,
    weights: Weights,
    salt: Salt,
    version_key: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Reveals the weights for a specific sub subnet on the Bittensor blockchain using the provided wallet.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The unique identifier of the subnet.
        mechid: The subnet mechanism unique identifier.
        uids: List of neuron UIDs for which weights are being revealed.
        weights: List of weight values corresponding to each UID.
        salt: List of salt values corresponding to the hash function.
        version_key: Version key for compatibility with the network.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        unlock = unlock_key(wallet, raise_error=raise_error)
        if not unlock.success:
            logging.error(unlock.message)
            return ExtrinsicResponse(
                False, unlock.message, extrinsic_function=get_function_name()
            )

        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="reveal_mechanism_weights",
            call_params={
                "netuid": netuid,
                "mecid": mechid,
                "uids": uids,
                "values": weights,
                "salt": salt,
                "version_key": version_key,
            },
        )
        response = await subtensor.sign_and_send_extrinsic(
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

        if response.success:
            logging.debug(response.message)
            return response

        logging.error(response.message)
        return response

    except Exception as error:
        if raise_error:
            raise error

        logging.error(str(error))
        return ExtrinsicResponse(
            success=False,
            message=format_error_message(error),
            error=error,
            extrinsic_function=get_function_name(),
        )


async def set_mechanism_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    mechid: int,
    uids: UIDs,
    weights: Weights,
    version_key: int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Sets the passed weights in the chain for hotkeys in the sub-subnet of the passed subnet.

    Parameters:
        subtensor: AsyncSubtensor instance.
        wallet: Bittensor Wallet instance.
        netuid: The unique identifier of the subnet.
        mechid: The subnet mechanism unique identifier.
        uids: List of neuron UIDs for which weights are being revealed.
        weights: List of weight values corresponding to each UID.
        version_key: Version key for compatibility with the network.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        unlock = unlock_key(wallet, raise_error=raise_error)
        if not unlock.success:
            logging.error(unlock.message)
            return ExtrinsicResponse(
                False, unlock.message, extrinsic_function=get_function_name()
            )

        # Convert, reformat and normalize.
        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_mechanism_weights",
            call_params={
                "netuid": netuid,
                "mecid": mechid,
                "dests": uids,
                "weights": weights,
                "version_key": version_key,
            },
        )
        response = await subtensor.sign_and_send_extrinsic(
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

        if response.success:
            logging.debug("Successfully set weights and Finalized.")
            return response

        logging.error(response.message)
        return response

    except Exception as error:
        if raise_error:
            raise error

        logging.error(str(error))
        return ExtrinsicResponse(
            success=False,
            message=format_error_message(error),
            error=error,
            extrinsic_function=get_function_name(),
        )
