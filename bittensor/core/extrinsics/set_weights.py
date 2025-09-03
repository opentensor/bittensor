"""Module sync setting weights extrinsic."""

from typing import Union, TYPE_CHECKING, Optional

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


def set_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
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
    )

    if success:
        logging.info(message)
    else:
        logging.error(f"{get_function_name}: {message}")
    return success, message
