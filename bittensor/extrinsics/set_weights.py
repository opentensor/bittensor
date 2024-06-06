# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor

import logging
import numpy as np
from numpy.typing import NDArray
from rich.prompt import Confirm
from typing import Union, Tuple
import bittensor.utils.weight_utils as weight_utils
from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME
from bittensor.utils.registration import torch, use_torch

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


def set_weights_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> Tuple[bool, str]:
    r"""Sets the given weights and values on chain for wallet hotkey account.

    Args:
        subtensor (bittensor.subtensor):
            Subtensor endpoint to use.
        wallet (bittensor.wallet):
            Bittensor wallet object.
        netuid (int):
            The ``netuid`` of the subnet to set weights for.
        uids (Union[NDArray[np.int64], torch.LongTensor, list]):
            The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]):
            The weights to set. These must be ``float`` s and correspond to the passed ``uid`` s.
        version_key (int):
            The version key of the validator.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
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
            "Do you want to set weights:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format(
                [float(v / 65535) for v in weight_vals], weight_uids
            )
        ):
            return False, "Prompt refused."

    with bittensor.__console__.status(
        ":satellite: Setting weights on [white]{}[/white] ...".format(subtensor.network)
    ):
        try:
            success, error_message = subtensor._do_set_weights(
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
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Set weights",
                    suffix="<green>Finalized: </green>" + str(success),
                )
                return True, "Successfully set weights and Finalized."
            else:
                bittensor.logging.error(
                    msg=error_message,
                    prefix="Set weights",
                    suffix="<red>Failed: </red>",
                )
                return False, error_message

        except Exception as e:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", suffix="<red>Failed: </red>" + str(e)
            )
            return False, str(e)
