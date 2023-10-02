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

import time
import torch
from rich.prompt import Confirm
from typing import Union, List
import bittensor.utils.weight_utils as weight_utils

from loguru import logger

logger = logger.opt(colors=True)


def root_register_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Registers the wallet to root network.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true,
            or returns false if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block.
            If we did not wait for finalization / inclusion, the response is true.
    """

    wallet.coldkey  # unlock coldkey

    is_registered = subtensor.is_hotkey_registered(
        netuid=0, hotkey_ss58=wallet.hotkey.ss58_address
    )
    if is_registered:
        bittensor.__console__.print(
            f":white_heavy_check_mark: [green]Already registered on root network.[/green]"
        )
        return True

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Register to root network?"):
            return False

    with bittensor.__console__.status(":satellite: Registering to root network..."):
        success, err_msg = subtensor._do_root_register(
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success != True or success == False:
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(err_msg)
            )
            time.sleep(0.5)

        # Successful registration, final check for neuron and pubkey
        else:
            is_registered = subtensor.is_hotkey_registered(
                netuid=0, hotkey_ss58=wallet.hotkey.ss58_address
            )
            if is_registered:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Registered[/green]"
                )
                return True
            else:
                # neuron not found, try again
                bittensor.__console__.print(
                    ":cross_mark: [red]Unknown error. Neuron not found.[/red]"
                )


def set_root_weights_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuids: Union[torch.LongTensor, list],
    weights: Union[torch.FloatTensor, list],
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    prompt: bool = False,
) -> bool:
    r"""Sets the given weights and values on chain for wallet hotkey account.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        netuids (List[int]):
            netuid of the subent to set weights for.
        weights ( Union[torch.FloatTensor, list]):
            weights to set which must floats and correspond to the passed uids.
        version_key (int):
            version key of the validator.
        wait_for_inclusion (bool):
            if set, waits for the extrinsic to enter a block before returning true,
            or returns false if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            if set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or uncluded in the block.
            If we did not wait for finalization / inclusion, the response is true.
    """
    # First convert types.
    if isinstance(netuids, list):
        netuids = torch.tensor(netuids, dtype=torch.int64)
    if isinstance(weights, list):
        weights = torch.tensor(weights, dtype=torch.float32)

    # Get weight restrictions.
    min_allowed_weights = subtensor.min_allowed_weights(netuid=0)
    max_weight_limit = subtensor.max_weight_limit(netuid=0)

    # Get non zero values.
    non_zero_weight_idx = torch.argwhere(weights > 0).squeeze(dim=1)
    non_zero_weight_uids = netuids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]
    if non_zero_weights.numel() < min_allowed_weights:
        raise ValueError(
            "The minimum number of weights required to set weights is {}, got {}".format(
                min_allowed_weights, non_zero_weights.numel()
            )
        )

    # Normalize the weights to max value.
    formatted_weights = bittensor.utils.weight_utils.normalize_max_weight(
        x=weights, limit=max_weight_limit
    )
    bittensor.__console__.print(
        f"\nNormalized weights: \n\t{weights} -> {formatted_weights}\n"
    )

    # Ask before moving on.
    if prompt:
        if not Confirm.ask(
            "Do you want to set the following root weights?:\n[bold white]  weights: {}\n  uids: {}[/bold white ]?".format(
                formatted_weights, netuids
            )
        ):
            return False

    with bittensor.__console__.status(
        ":satellite: Setting root weights on [white]{}[/white] ...".format(
            subtensor.network
        )
    ):
        try:
            weight_uids, weight_vals = weight_utils.convert_weights_and_uids_for_emit(
                netuids, weights
            )
            success, error_message = subtensor._do_set_weights(
                wallet=wallet,
                netuid=0,
                uids=weight_uids,
                vals=weight_vals,
                version_key=version_key,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
            )

            bittensor.__console__.print(success, error_message)

            if not wait_for_finalization and not wait_for_inclusion:
                return True

            if success == True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Finalized[/green]"
                )
                bittensor.logging.success(
                    prefix="Set weights",
                    sufix="<green>Finalized: </green>" + str(success),
                )
                return True
            else:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(error_message)
                )
                bittensor.logging.warning(
                    prefix="Set weights",
                    sufix="<red>Failed: </red>" + str(error_message),
                )
                return False

        except Exception as e:
            # TODO( devs ): lets remove all of the bittensor.__console__ calls and replace with loguru.
            bittensor.__console__.print(
                ":cross_mark: [red]Failed[/red]: error:{}".format(e)
            )
            bittensor.logging.warning(
                prefix="Set weights", sufix="<red>Failed: </red>" + str(e)
            )
            return False
