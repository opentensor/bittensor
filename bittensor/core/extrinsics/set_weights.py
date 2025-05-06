"""Module sync setting weights extrinsic."""

from typing import Union, TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from bittensor.core.settings import version_as_int
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import (
    convert_and_normalize_weights_and_uids,
    convert_uids_and_weights,
)

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet
    from bittensor.utils.registration import torch


def _do_set_weights(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: list[int],
    vals: list[int],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Internal method to send a transaction to the Bittensor blockchain, setting weights
    for specified neurons. This method constructs and submits the transaction, handling
    retries and blockchain communication.

    Args:
        subtensor (subtensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
        uids (List[int]): List of neuron UIDs for which weights are being set.
        vals (List[int]): List of weight values corresponding to each UID.
        netuid (int): Unique identifier for the network.
        version_key (int, optional): Version key for compatibility with the network.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        Tuple[bool, str]: A tuple containing a success flag and an optional error message.

    This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their
        trust in other neurons based on observed performance and contributions.
    """

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_weights",
        call_params={
            "dests": uids,
            "weights": vals,
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
    )
    return success, message


def set_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    period: Optional[int] = 8,
) -> tuple[bool, str]:
    """Sets the given weights and values on a chain for a wallet hotkey account.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): Bittensor subtensor object.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to set weights for.
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights to set. These must be ``float`` s
            and correspond to the passed ``uid`` s.
        version_key (int): The version key of the validator.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``True``, or
            returns ``False`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning
            ``True``, or returns ``False`` if the extrinsic fails to be finalized within the timeout.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted.
            If the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        success (bool): Flag is ``True`` if extrinsic was finalized or included in the block. If we did not wait for
            finalization / inclusion, the response is ``True``.
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
    try:
        success, message = _do_set_weights(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            uids=weight_uids,
            vals=weight_vals,
            version_key=version_key,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
            period=period,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True, message

        if success is True:
            message = "Successfully set weights and Finalized."
            logging.success(f":white_heavy_check_mark: [green]{message}[/green]")
            return True, message

        logging.error(f"[red]Failed[/red] set weights. Error: {message}")
        return False, message

    except Exception as error:
        logging.error(f":cross_mark: [red]Failed[/red] set weights. Error: {error}")
        return False, str(error)
