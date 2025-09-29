import time
from typing import Optional, TYPE_CHECKING

from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import u16_normalized_float
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
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        netuid = 0
        logging.debug(
            f"Registering on netuid [blue]{netuid}[/blue] on network: [blue]{subtensor.network}[/blue]."
        )

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
            return ExtrinsicResponse(
                False,
                f"Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO.",
            ).with_log()

        logging.debug(
            f"Checking if hotkey ([blue]{wallet.hotkey.ss58_address}[/blue]) is registered on root."
        )
        is_registered = subtensor.is_hotkey_registered(
            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
        )
        if is_registered:
            return ExtrinsicResponse(message="Already registered on root network.")

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="root_register",
            call_params={"hotkey": wallet.hotkey.ss58_address},
        )
        response = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not response.success:
            logging.error(f"[red]{response.message}[/red]")
            time.sleep(0.5)
            return response

        # Successful registration, final check for neuron and pubkey
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=wallet.hotkey.ss58_address, netuid=netuid
        )
        if uid is not None:
            response.data = {
                "hotkey_ss58": wallet.hotkey.ss58_address,
                "netuid": netuid,
                "uid": uid,
            }
            logging.debug(
                f"Hotkey {wallet.hotkey.ss58_address} registered in subnet {netuid} with UID: {uid}."
            )
            return response

        # neuron not found, try again
        return ExtrinsicResponse(False, "Unknown error. Neuron not found.").with_log()

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
