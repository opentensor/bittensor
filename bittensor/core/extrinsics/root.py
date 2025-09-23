import time
from typing import Optional, TYPE_CHECKING

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import u16_normalized_float, unlock_key, get_function_name
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _get_limits(subtensor: "Subtensor") -> tuple[int, float]:
    """
    Retrieves the minimum allowed weights and maximum weight limit for the given subnet.

    These values are fetched asynchronously using `asyncio.gather` to run both requests concurrently.

    Parameters:
        subtensor: The AsyncSubtensor instance.

    Returns:
        tuple[int, float]:
            - `min_allowed_weights`: The minimum allowed weights.
            - `max_weight_limit`: The maximum weight limit, normalized to a float value.
    """
    # Get weight restrictions.
    maw = subtensor.get_hyperparameter("MinAllowedWeights", netuid=0)
    mwl = subtensor.get_hyperparameter("MaxWeightsLimit", netuid=0)
    min_allowed_weights = int(maw)
    max_weight_limit = u16_normalized_float(int(mwl))
    return min_allowed_weights, max_weight_limit


def root_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Registers the neuron to the root network.

    Parameters:
        subtensor: Subtensor instance to interact with the blockchain.
        wallet: Bittensor Wallet instance.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    netuid = 0
    logging.info(
        f"Registering on netuid [blue]{netuid}[/blue] on network: [blue]{subtensor.network}[/blue]"
    )

    logging.info("Fetching recycle amount & balance.")
    block = subtensor.get_current_block()
    recycle_call = subtensor.get_hyperparameter(
        param_name="Burn",
        netuid=netuid,
        block=block,
    )
    balance = subtensor.get_balance(
        address=wallet.coldkeypub.ss58_address,
        block=block,
    )

    current_recycle = Balance.from_rao(int(recycle_call))

    if balance < current_recycle:
        message = f"Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO"
        logging.error(f"[red]{message}[/red].")
        return ExtrinsicResponse(False, message, extrinsic_function=get_function_name())

    logging.debug(
        f"Checking if hotkey ([blue]{wallet.hotkey_str}[/blue]) is registered on root."
    )
    is_registered = subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
    )
    if is_registered:
        message = "Already registered on root network."
        logging.error(f":white_heavy_check_mark: [green]{message}[/green]")
        return ExtrinsicResponse(
            message=message, extrinsic_function=get_function_name()
        )

    logging.info(":satellite: [magenta]Registering to root network...[/magenta]")
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={"hotkey": wallet.hotkey.ss58_address},
    )
    response = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
    )

    if not response.success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {response.message}")
        time.sleep(0.5)
        return response

    # Successful registration, final check for neuron and pubkey
    uid = subtensor.substrate.query(
        module="SubtensorModule",
        storage_function="Uids",
        params=[netuid, wallet.hotkey.ss58_address],
    )
    if uid is not None:
        response.data = {"uid": uid}
        logging.info(
            f":white_heavy_check_mark: [green]Registered with UID: {uid}[/green]."
        )
        return response

    # neuron not found
    # neuron not found, try again
    message = "Unknown error. Neuron not found."
    logging.error(f":cross_mark: [red]{message}[/red]")
    return ExtrinsicResponse(False, message, extrinsic_function=get_function_name())
