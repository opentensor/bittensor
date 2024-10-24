# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from typing import Union, Optional, TYPE_CHECKING
import random
import numpy as np
from numpy.typing import NDArray
from retry import retry
from rich.prompt import Confirm

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.core.settings import bt_console, version_as_int
from bittensor.utils import format_error_message, weight_utils
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected
from bittensor.utils.registration import torch, use_torch
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit

# For annotation purposes
if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet


# Chain call for `do_set_weights`
@ensure_connected
def do_set_weights(
    self: "Subtensor",
    wallet: "Wallet",
    uids: list[int],
    vals: list[int],
    netuid: int,
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[dict]]:  # (success, error_message)
    """
    Internal method to send a transaction to the Bittensor blockchain, setting weights for specified neurons. This method constructs and submits the transaction, handling retries and blockchain communication.

    Args:
        self (bittensor.core.subtensor.Subtensor): Subtensor interface
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
        uids (list[int]): List of neuron UIDs for which weights are being set.
        vals (list[int]): List of weight values corresponding to each UID.
        netuid (int): Unique identifier for the network.
        version_key (int): Version key for compatibility with the network.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing a success flag and an optional response message.

    This method is vital for the dynamic weighting mechanism in Bittensor, where neurons adjust their trust in other neurons based on observed performance and contributions.
    """

    @retry(delay=1, tries=3, backoff=2, max_delay=4)
    def make_substrate_call_with_retry():
        call = self.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_weights",
            call_params={
                "dests": uids,
                "weights": vals,
                "netuid": netuid,
                "version_key": version_key,
            },
        )
        # Period dictates how long the extrinsic will stay as part of waiting pool
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call,
            keypair=wallet.hotkey,
            era={"period": 5},
        )
        response = submit_extrinsic(
            substrate=self.substrate,
            extrinsic=extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True, "Not waiting for finalization or inclusion."

        response.process_events()
        if response.is_success:
            return True, "Successfully set weights."
        else:
            return False, response.error_message

    return make_substrate_call_with_retry()


# Community uses this extrinsic directly and via `subtensor.set_weights`
def set_weights_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> tuple[bool, str]:
    """Sets the given weights and values on chain for wallet hotkey account.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor endpoint to use.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        netuid (int): The ``netuid`` of the subnet to set weights for.
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights to set. These must be ``float`` s and correspond to the passed ``uid`` s.
        version_key (int): The version key of the validator.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool): If ``true``, the call waits for confirmation from the user before proceeding.

    Returns:
        tuple[bool, str]: A tuple containing a success flag and an optional response message.
    """

    if subtensor.get_subnet_hyperparameters(
        netuid=netuid
    ).commit_reveal_weights_enabled:
        # if cr is enabled, commit instead of setting the weights.
        salt = [random.randint(0, 350) for _ in range(8)]

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                f"Do you want to commit weights:\n[bold white]  weights: {weights}\n"
                f"uids: {uids}[/bold white ]?"
            ):
                return False, "Prompt refused."

        with bt_console.status(
            f":satellite: Committing weights on [white]{subtensor.network}[/white] ..."
        ):
            try:

                # First convert types.
                if use_torch():
                    if isinstance(uids, list):
                        uids = torch.tensor(uids, dtype=torch.int64)
                    if isinstance(weights, list):
                        weights = torch.tensor(weights, dtype=torch.float32)
                else:
                    if isinstance(uids, list):
                        uids = np.array(uids, dtype=np.int64)
                    if isinstance(weights, list):
                        weights = np.array(weights, dtype=np.float32)

                # Reformat and normalize.
                weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
                    uids, weights
                )

                success, message = subtensor.commit_weights(
                    wallet=wallet,
                    netuid=netuid,
                    salt=salt,
                    uids=weight_uids,
                    weights=weight_vals,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
                if not wait_for_finalization and not wait_for_inclusion:
                    return True, "Not waiting for finalization or inclusion."

                if success is True:
                    bt_console.print(
                        ":white_heavy_check_mark: [green]Finalized[/green]"
                    )
                    logging.success(
                        msg=str(success),
                        prefix="Committed weights",
                        suffix="<green>Finalized: </green>",
                    )
                    return True, "Successfully committed weights and Finalized."
                else:
                    logging.error(message)
                    return False, message

            except Exception as e:
                bt_console.print(f":cross_mark: [red]Failed[/red]: error:{e}")
                logging.debug(str(e))
            return False, str(e)
    else:
        # First convert types.
        if use_torch():
            if isinstance(uids, list):
                uids = torch.tensor(uids, dtype=torch.int64)
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
        else:
            if isinstance(uids, list):
                uids = np.array(uids, dtype=np.int64)
            if isinstance(weights, list):
                weights = np.array(weights, dtype=np.float32)

        # Reformat and normalize.
        weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
            uids, weights
        )

        # Ask before moving on.
        if prompt:
            if not Confirm.ask(
                f"Do you want to set weights:\n[bold white]  weights: {[float(v / 65535) for v in weight_vals]}\n"
                f"uids: {weight_uids}[/bold white ]?"
            ):
                return False, "Prompt refused."

        with bt_console.status(
            f":satellite: Setting weights on [white]{subtensor.network}[/white] ..."
        ):
            try:
                success, error_message = do_set_weights(
                    self=subtensor,
                    wallet=wallet,
                    netuid=netuid,
                    uids=weight_uids,
                    vals=weight_vals,
                    version_key=version_key,
                    wait_for_finalization=wait_for_finalization,
                    wait_for_inclusion=wait_for_inclusion,
                )

                if not wait_for_finalization and not wait_for_inclusion:
                    return True, "Not waiting for finalization or inclusion."

                if success is True:
                    bt_console.print(
                        ":white_heavy_check_mark: [green]Finalized[/green]"
                    )
                    logging.success(
                        msg=str(success),
                        prefix="Set weights",
                        suffix="<green>Finalized: </green>",
                    )
                    return True, "Successfully set weights and Finalized."
                else:
                    error_message = format_error_message(error_message)
                    logging.error(error_message)
                    return False, error_message

            except Exception as e:
                bt_console.print(f":cross_mark: [red]Failed[/red]: error:{e}")
                logging.debug(str(e))
                return False, str(e)
