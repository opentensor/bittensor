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
from rich.prompt import Confirm

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