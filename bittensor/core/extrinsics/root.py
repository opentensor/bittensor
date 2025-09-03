import time
from typing import Optional, TYPE_CHECKING

from bittensor.utils import (
    u16_normalized_float,
    unlock_key,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _get_limits(subtensor: "Subtensor") -> tuple[int, float]:
    """
    Retrieves the minimum allowed weights and maximum weight limit for the given subnet.

    These values are fetched asynchronously using `asyncio.gather` to run both requests concurrently.

    Args:
        subtensor (Subtensor): The AsyncSubtensor object used to interface with the network's substrate node.

    Returns:
        tuple[int, float]: A tuple containing:
            - `min_allowed_weights` (int): The minimum allowed weights.
            - `max_weight_limit` (float): The maximum weight limit, normalized to a float value.
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
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> bool:
    """Registers the wallet to the root network.

    Arguments:
        subtensor (bittensor.core.subtensor.Subtensor): The Subtensor object
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            `True`, or returns `False` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        `True` if extrinsic was finalized or included in the block. If we did not wait for finalization/inclusion, the
            response is `True`.
    """
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
    is_registered = subtensor.is_hotkey_registered(
        netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
    )
    if is_registered:
        logging.error(
            ":white_heavy_check_mark: [green]Already registered on root network.[/green]"
        )
        return True

    logging.info(":satellite: [magenta]Registering to root network...[/magenta]")
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={"hotkey": wallet.hotkey.ss58_address},
    )
    success, err_msg = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
    )

    if not success:
        logging.error(f":cross_mark: [red]Failed error:[/red] {err_msg}")
        time.sleep(0.5)
        return False

    # Successful registration, final check for neuron and pubkey
    else:
        uid = subtensor.substrate.query(
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
