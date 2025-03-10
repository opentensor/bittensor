import asyncio
from typing import Union, TYPE_CHECKING

from bittensor_wallet import Wallet
import numpy as np
from numpy.typing import NDArray

from bittensor.core.errors import SubstrateRequestException
from bittensor.utils import u16_normalized_float, format_error_message, unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import (
    normalize_max_weight,
    convert_weights_and_uids_for_emit,
)

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor


async def _get_limits(subtensor: "AsyncSubtensor") -> tuple[int, float]:
    """
    Retrieves the minimum allowed weights and maximum weight limit for the given subnet.

    These values are fetched asynchronously using `asyncio.gather` to run both requests concurrently.

    Args:
        subtensor (AsyncSubtensor): The AsyncSubtensor object used to interface with the network's substrate node.

    Returns:
        tuple[int, float]: A tuple containing:
            - `min_allowed_weights` (int): The minimum allowed weights.
            - `max_weight_limit` (float): The maximum weight limit, normalized to a float value.
    """
    # Get weight restrictions.
    maw, mwl = await asyncio.gather(
        subtensor.get_hyperparameter("MinAllowedWeights", netuid=0),
        subtensor.get_hyperparameter("MaxWeightsLimit", netuid=0),
    )
    min_allowed_weights = int(maw)
    max_weight_limit = u16_normalized_float(int(mwl))
    return min_allowed_weights, max_weight_limit


async def root_register_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """Registers the wallet to root network.

    Arguments:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The AsyncSubtensor object
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion,
            the response is `True`.
    """
    netuid = 0
    logging.info(
        f"Registering on netuid [blue]{netuid}[/blue] on network: [blue]{subtensor.network}[/blue]"
    )

    logging.info("Fetching recycle amount & balance.")
    block_hash = await subtensor.get_block_hash()

    try:
        recycle_call, balance = await asyncio.gather(
            subtensor.get_hyperparameter(
                param_name="Burn",
                netuid=netuid,
                block_hash=block_hash,
            ),
            subtensor.get_balance(
                wallet.coldkeypub.ss58_address,
                block_hash=block_hash,
            ),
        )
    except TypeError as e:
        logging.error(f"Unable to retrieve current recycle. {e}")
        return False

    current_recycle = Balance.from_rao(int(recycle_call))

    if balance < current_recycle:
        logging.error(
            f"[red]Insufficient balance {balance} to register neuron. "
            f"Current recycle is {current_recycle} TAO[/red]."
        )
        return False

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    logging.debug(
        f"Checking if hotkey ([blue]{wallet.hotkey_str}[/blue]) is registered on root."
    )
    is_registered = await subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
    )
    if is_registered:
        logging.error(
            ":white_heavy_check_mark: [green]Already registered on root network.[/green]"
        )
        return True

    logging.info(":satellite: [magenta]Registering to root network...[/magenta]")
    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={"hotkey": wallet.hotkey.ss58_address},
    )
    success, err_msg = await subtensor.sign_and_send_extrinsic(
        call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {err_msg}")
        await asyncio.sleep(0.5)
        return False

    # Successful registration, final check for neuron and pubkey
    else:
        uid = await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, wallet.hotkey.ss58_address],
        )
        if uid is not None:
            logging.info(
                f":white_heavy_check_mark: [green]Registered with UID[/green] [blue]{uid}[/blue]."
            )
            return True
        else:
            # neuron not found, try again
            logging.error(":cross_mark: [red]Unknown error. Neuron not found.[/red]")
            return False


async def _do_set_root_weights(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuids: Union[NDArray[np.int64], list[int]],
    weights: Union[NDArray[np.float32], list[float]],
    netuid: int = 0,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: int = 5,
) -> tuple[bool, str]:
    """
    Sets the root weights on the Subnet for the given wallet hotkey account.

    This function constructs and submits an extrinsic to set the root weights for the given wallet hotkey account.
    It waits for inclusion or finalization of the extrinsic based on the provided parameters.

    Arguments:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The AsyncSubtensor object used to interact with the
            blockchain.
        wallet (bittensor_wallet.Wallet): The wallet containing the hotkey and coldkey for the transaction.
        netuids (Union[NDArray[np.int64], list[int]]): List of UIDs to set weights for.
        weights (Union[NDArray[np.float32], list[float]]): Corresponding weights to set for each UID.
        netuid (int): The netuid of the subnet to set weights for. Defaults to 0.
        version_key (int, optional): The version key of the validator. Defaults to 0.
        wait_for_inclusion (bool, optional): If True, waits for the extrinsic to be included in a block. Defaults to
            False.
        wait_for_finalization (bool, optional): If True, waits for the extrinsic to be finalized on the chain. Defaults
            to False.
        period (int, optional): The period in seconds to wait for extrinsic inclusion or finalization. Defaults to 5.

    Returns:
        tuple: Returns a tuple containing a boolean indicating success and a message describing the result of the
            operation.
    """
    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_root_weights",
        call_params={
            "dests": netuids,
            "weights": weights,
            "netuid": netuid,
            "version_key": version_key,
            "hotkey": wallet.hotkey.ss58_address,
        },
    )

    next_nonce = await subtensor.substrate.get_account_next_index(
        wallet.hotkey.ss58_address
    )

    # Period dictates how long the extrinsic will stay as part of waiting pool
    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.coldkey,
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

    if await response.is_success:
        return True, "Successfully set weights."

    return False, format_error_message(await response.error_message)


async def set_root_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuids: Union[NDArray[np.int64], list[int]],
    weights: Union[NDArray[np.float32], list[float]],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> bool:
    """Sets the given weights and values on chain for wallet hotkey account.

    Arguments:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The AsyncSubtensor object
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuids (Union[NDArray[np.int64], list[int]]): The `netuid` of the subnet to set weights for.
        weights (Union[NDArray[np.float32], list[float]]): Weights to set. These must be `float` s and must correspond
            to the passed `netuid` s.
        version_key (int): The version key of the validator.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning `
            True`, or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the
            response is `True`.
    """
    my_uid = await subtensor.substrate.query(
        "SubtensorModule", "Uids", [0, wallet.hotkey.ss58_address]
    )

    if my_uid is None:
        logging.error("Your hotkey is not registered to the root network.")
        return False

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # First convert types.
    if isinstance(netuids, list):
        netuids = np.array(netuids, dtype=np.int64)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float32)

    logging.debug("Fetching weight limits")
    min_allowed_weights, max_weight_limit = await _get_limits(subtensor)

    # Get non zero values.
    non_zero_weight_idx = np.argwhere(weights > 0).squeeze(axis=1)
    non_zero_weights = weights[non_zero_weight_idx]
    if non_zero_weights.size < min_allowed_weights:
        raise ValueError(
            "The minimum number of weights required to set weights is {}, got {}".format(
                min_allowed_weights, non_zero_weights.size
            )
        )

    # Normalize the weights to max value.
    logging.info("Normalizing weights")
    formatted_weights = normalize_max_weight(x=weights, limit=max_weight_limit)
    logging.info(
        f"Raw weights -> Normalized weights: [blue]{weights}[/blue] -> [green]{formatted_weights}[/green]"
    )

    try:
        logging.info(":satellite: [magenta]Setting root weights...[magenta]")
        weight_uids, weight_vals = convert_weights_and_uids_for_emit(netuids, weights)

        success, error_message = await _do_set_root_weights(
            subtensor=subtensor,
            wallet=wallet,
            netuids=weight_uids,
            weights=weight_vals,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True

        if success is True:
            logging.info(":white_heavy_check_mark: [green]Finalized[/green]")
            return True
        else:
            fmt_err = error_message
            logging.error(f":cross_mark: [red]Failed error:[/red] {fmt_err}")
            return False

    except SubstrateRequestException as e:
        fmt_err = format_error_message(e)
        logging.error(f":cross_mark: [red]Failed error:[/red] {fmt_err}")
        return False
