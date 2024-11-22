import time
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.settings import version_as_int
from bittensor.utils import format_error_message, weight_utils, unlock_key
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected
from bittensor.utils.registration import torch, legacy_torch_api_compat

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


@ensure_connected
def _do_root_register(
    self: "Subtensor",
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[str]]:
    # create extrinsic call
    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={"hotkey": wallet.hotkey.ss58_address},
    )
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.coldkey
    )
    response = self.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    # process if registration successful, try again if pow is still valid
    response.process_events()
    if not response.is_success:
        return False, format_error_message(
            response.error_message, substrate=self.substrate
        )
    # Successful registration
    else:
        return True, None


def root_register_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    """Registers the wallet to root network.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout. Default is ``False``.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout. Default is ``True``.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """

    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    is_registered = subtensor.is_hotkey_registered(
        netuid=0, hotkey_ss58=wallet.hotkey.ss58_address
    )
    if is_registered:
        logging.info(
            ":white_heavy_check_mark: <green>Already registered on root network.</green>"
        )
        return True

    logging.info(":satellite: <magenta>Registering to root network...</magenta>")
    success, err_msg = _do_root_register(
        self=subtensor,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not success:
        logging.error(f":cross_mark: <red>Failed</red>: {err_msg}")
        time.sleep(0.5)

    # Successful registration, final check for neuron and pubkey
    else:
        is_registered = subtensor.is_hotkey_registered(
            netuid=0, hotkey_ss58=wallet.hotkey.ss58_address
        )
        if is_registered:
            logging.success(":white_heavy_check_mark: <green>Registered</green>")
            return True
        else:
            # neuron not found, try again
            logging.error(":cross_mark: <red>Unknown error. Neuron not found.</red>")


@ensure_connected
def _do_set_root_weights(
    self: "Subtensor",
    wallet: "Wallet",
    uids: list[int],
    vals: list[int],
    netuid: int = 0,
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Internal method to send a transaction to the Bittensor blockchain, setting weights for specified neurons on root. This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        self (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
        uids (List[int]): List of neuron UIDs for which weights are being set.
        vals (List[int]): List of weight values corresponding to each UID.
        netuid (int): Unique identifier for the network.
        version_key (int, optional): Version key for compatibility with the network. Defaults is a current ``version_as_int``.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block. Defaults is ``False``.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain. Defaults is ``False``.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their trust in other neurons based on observed performance and contributions on the root network.
    """

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="set_root_weights",
        call_params={
            "dests": uids,
            "weights": vals,
            "netuid": netuid,
            "version_key": version_key,
            "hotkey": wallet.hotkey.ss58_address,
        },
    )
    # Period dictates how long the extrinsic will stay as part of waiting pool
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.coldkey,
        era={"period": 5},
    )
    response = self.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    # We only wait here if we expect finalization.
    if not wait_for_finalization and not wait_for_inclusion:
        return True, "Not waiting for finalziation or inclusion."

    response.process_events()
    if response.is_success:
        return True, "Successfully set weights."
    else:
        return False, response.error_message


@legacy_torch_api_compat
def set_root_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuids: Union[NDArray[np.int64], "torch.LongTensor", list[int]],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list[float]],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> bool:
    """Sets the given weights and values on chain for wallet hotkey account.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object. Bittensor wallet object.
        netuids (Union[NDArray[np.int64], torch.LongTensor, list[int]]): The ``netuid`` of the subnet to set weights for.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list[float]]): Weights to set. These must be ``float`` s and must correspond to the passed ``netuid`` s.
        version_key (int): The version key of the validator.  Default is ``0``.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.  Default is ``False``.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout. Default is ``False``.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not (unlock := unlock_key(wallet)).success:
        logging.error(unlock.message)
        return False

    # First convert types.
    if isinstance(netuids, list):
        netuids = np.array(netuids, dtype=np.int64)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float32)

    # Get weight restrictions.
    min_allowed_weights = subtensor.min_allowed_weights(netuid=0)
    max_weight_limit = subtensor.max_weight_limit(netuid=0)

    # Get non zero values.
    non_zero_weight_idx = np.argwhere(weights > 0).squeeze(axis=1)
    non_zero_weight_uids = netuids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]
    if non_zero_weights.size < min_allowed_weights:
        raise ValueError(
            "The minimum number of weights required to set weights is {}, got {}".format(
                min_allowed_weights, non_zero_weights.size
            )
        )

    # Normalize the weights to max value.
    formatted_weights = weight_utils.normalize_max_weight(
        x=weights, limit=max_weight_limit
    )
    logging.info(
        f"Raw Weights -> Normalized weights: <blue>{weights}</blue> -> <green>{formatted_weights}</green>"
    )

    logging.info(
        f":satellite: <magenta>Setting root weights on</magenta> <blue>{subtensor.network}</blue> <magenta>...</magenta>"
    )
    try:
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            netuids, weights
        )
        success, error_message = _do_set_root_weights(
            self=subtensor,
            wallet=wallet,
            netuid=0,
            uids=weight_uids,
            vals=weight_vals,
            version_key=version_key,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if not wait_for_finalization and not wait_for_inclusion:
            return True

        if success is True:
            logging.info(":white_heavy_check_mark: <green>Finalized</green>")
            logging.success(f"Set weights {str(success)}")
            return True
        else:
            logging.error(
                f":cross_mark: <red>Failed </red> set weights. {str(error_message)}"
            )
            return False

    except Exception as e:
        logging.error(f":cross_mark: <red>Failed </red> set weights. {str(e)}")
        return False
