import asyncio
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.errors import SubstrateRequestException
from bittensor.utils import u16_normalized_float, format_error_message, unlock_key
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import (
    normalize_max_weight,
    convert_weights_and_uids_for_emit,
    convert_uids_and_weights,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
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
    period: Optional[int] = None,
) -> bool:
    """Registers the wallet to the root network.

    Arguments:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The AsyncSubtensor object
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

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
    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {message}")
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


async def set_root_weights_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuids: Union[NDArray[np.int64], list[int]],
    weights: Union[NDArray[np.float32], list[float]],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> bool:
    """Sets the given weights and values on a chain for a wallet hotkey account.

    Arguments:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): The AsyncSubtensor object
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuids (Union[NDArray[np.int64], list[int]]): The `netuid` of the subnet to set weights for.
        weights (Union[NDArray[np.float32], list[Float]]): Weights to set. These must be `Float`s and must correspond
            to the passed `netuid` s.
        version_key (int): The version key of the validator.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning `
            True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

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

    # Convert types.
    netuids, weights = convert_uids_and_weights(netuids, weights)

    logging.debug("[magenta]Fetching weight limits ...[/magenta]")
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
    logging.info("[magenta]Normalizing weights ...[/magenta]")
    formatted_weights = normalize_max_weight(x=weights, limit=max_weight_limit)
    logging.info(
        f"Raw weights -> Normalized weights: [blue]{weights}[/blue] -> [green]{formatted_weights}[/green]"
    )

    try:
        logging.info(":satellite: [magenta]Setting root weights...[magenta]")
        weight_uids, weight_vals = convert_weights_and_uids_for_emit(netuids, weights)

        # Since this extrinsic is only for the root network, we can set netuid to 0.
        netuid = 0
        call = await subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_root_weights",
            call_params={
                "dests": weight_uids,
                "weights": weight_vals,
                "netuid": netuid,
                "version_key": version_key,
                "hotkey": wallet.hotkey.ss58_address,
            },
        )

        success, message = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            use_nonce=True,
            period=period,
        )

        if success:
            logging.info(":white_heavy_check_mark: [green]Finalized[/green]")
            return True

        logging.error(f":cross_mark: [red]Failed error:[/red] {message}")
        return False

    except SubstrateRequestException as e:
        fmt_err = format_error_message(e)
        logging.error(f":cross_mark: [red]Failed error:[/red] {fmt_err}")
        return False
