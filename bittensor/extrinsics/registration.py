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

import time
from typing import List, Union, Optional, Tuple

from rich.prompt import Confirm

import bittensor
from bittensor.utils import format_error_message

from bittensor.utils.registration import (
    POWSolution,
    create_pow,
    torch,
    log_no_torch_error,
)


def register_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> bool:
    r"""Registers the wallet to the chain.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        netuid (int):
            The ``netuid`` of the subnet to register on.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
        max_allowed_attempts (int):
            Maximum number of attempts to register the wallet.
        cuda (bool):
            If ``true``, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]):
            The CUDA device id to use, or a list of device ids.
        tpb (int):
            The number of threads per block (CUDA).
        num_processes (int):
            The number of processes to use to register.
        update_interval (int):
            The number of nonces to solve between updates.
        log_verbose (bool):
            If ``true``, the registration process will log more information.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not subtensor.subnet_exists(netuid):
        bittensor.__console__.print(
            ":cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.".format(
                netuid
            )
        )
        return False

    with bittensor.__console__.status(
        f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."
    ):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid
        )
        if not neuron.is_null:
            bittensor.logging.debug(
                f"Wallet {wallet} is already registered on {neuron.netuid} with {neuron.uid}"
            )
            return True

    if prompt:
        if not Confirm.ask(
            "Continue Registration?\n  hotkey:     [bold white]{}[/bold white]\n  coldkey:    [bold white]{}[/bold white]\n  network:    [bold white]{}[/bold white]".format(
                wallet.hotkey.ss58_address,
                wallet.coldkeypub.ss58_address,
                subtensor.network,
            )
        ):
            return False

    if not torch:
        log_no_torch_error()
        return False

    # Attempt rolling registration.
    attempts = 1
    while True:
        bittensor.__console__.print(
            ":satellite: Registering...({}/{})".format(attempts, max_allowed_attempts)
        )
        # Solve latest POW.
        if cuda:
            if not torch.cuda.is_available():
                if prompt:
                    bittensor.__console__.print("CUDA is not available.")
                return False
            pow_result: Optional[POWSolution] = create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        else:
            pow_result: Optional[POWSolution] = create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )

        # pow failed
        if not pow_result:
            # might be registered already on this subnet
            is_registered = subtensor.is_hotkey_registered(
                netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
            )
            if is_registered:
                bittensor.__console__.print(
                    f":white_heavy_check_mark: [green]Already registered on netuid:{netuid}[/green]"
                )
                return True

        # pow successful, proceed to submit pow to chain for registration
        else:
            with bittensor.__console__.status(":satellite: Submitting POW..."):
                # check if pow result is still valid
                while not pow_result.is_stale(subtensor=subtensor):
                    result: Tuple[bool, Optional[str]] = subtensor._do_pow_register(
                        netuid=netuid,
                        wallet=wallet,
                        pow_result=pow_result,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                    )
                    success, err_msg = result

                    if not success:
                        # Look error here
                        # https://github.com/opentensor/subtensor/blob/development/pallets/subtensor/src/errors.rs
                        if "HotKeyAlreadyRegisteredInSubNet" in err_msg:
                            bittensor.__console__.print(
                                f":white_heavy_check_mark: [green]Already Registered on [bold]subnet:{netuid}[/bold][/green]"
                            )
                            return True

                        bittensor.__console__.print(
                            f":cross_mark: [red]Failed[/red]: {err_msg}"
                        )
                        time.sleep(0.5)

                    # Successful registration, final check for neuron and pubkey
                    else:
                        bittensor.__console__.print(":satellite: Checking Balance...")
                        is_registered = subtensor.is_hotkey_registered(
                            netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
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
                            continue
                else:
                    # Exited loop because pow is no longer valid.
                    bittensor.__console__.print("[red]POW is stale.[/red]")
                    # Try again.
                    continue

        if attempts < max_allowed_attempts:
            # Failed registration, retry pow
            attempts += 1
            bittensor.__console__.print(
                ":satellite: Failed registration, retrying pow ...({}/{})".format(
                    attempts, max_allowed_attempts
                )
            )
        else:
            # Failed to register after max attempts.
            bittensor.__console__.print("[red]No more attempts.[/red]")
            return False


def burned_register_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    netuid: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Registers the wallet to chain by recycling TAO.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        netuid (int):
            The ``netuid`` of the subnet to register on.
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
    if not subtensor.subnet_exists(netuid):
        bittensor.__console__.print(
            ":cross_mark: [red]Failed[/red]: error: [bold white]subnet:{}[/bold white] does not exist.".format(
                netuid
            )
        )
        return False

    try:
        wallet.coldkey  # unlock coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False
    with bittensor.__console__.status(
        f":satellite: Checking Account on [bold]subnet:{netuid}[/bold]..."
    ):
        neuron = subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid
        )

        old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

        recycle_amount = subtensor.recycle(netuid=netuid)
        if not neuron.is_null:
            bittensor.__console__.print(
                ":white_heavy_check_mark: [green]Already Registered[/green]:\n"
                "uid: [bold white]{}[/bold white]\n"
                "netuid: [bold white]{}[/bold white]\n"
                "hotkey: [bold white]{}[/bold white]\n"
                "coldkey: [bold white]{}[/bold white]".format(
                    neuron.uid, neuron.netuid, neuron.hotkey, neuron.coldkey
                )
            )
            return True

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Recycle {recycle_amount} to register on subnet:{netuid}?"):
            return False

    with bittensor.__console__.status(":satellite: Recycling TAO for Registration..."):
        success, err_msg = subtensor._do_burned_register(
            netuid=netuid,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not success:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: {err_msg}")
            time.sleep(0.5)
            return False
        # Successful registration, final check for neuron and pubkey
        else:
            bittensor.__console__.print(":satellite: Checking Balance...")
            block = subtensor.get_current_block()
            new_balance = subtensor.get_balance(
                wallet.coldkeypub.ss58_address, block=block
            )

            bittensor.__console__.print(
                "Balance:\n  [blue]{}[/blue] :arrow_right: [green]{}[/green]".format(
                    old_balance, new_balance
                )
            )
            is_registered = subtensor.is_hotkey_registered(
                netuid=netuid, hotkey_ss58=wallet.hotkey.ss58_address
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
                return False


class MaxSuccessException(Exception):
    pass


class MaxAttemptsException(Exception):
    pass


def run_faucet_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Tuple[bool, str]:
    r"""Runs a continual POW to get a faucet of TAO on the test net.

    Args:
        wallet (bittensor.wallet):
            Bittensor wallet object.
        prompt (bool):
            If ``true``, the call waits for confirmation from the user before proceeding.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.
        max_allowed_attempts (int):
            Maximum number of attempts to register the wallet.
        cuda (bool):
            If ``true``, the wallet should be registered using CUDA device(s).
        dev_id (Union[List[int], int]):
            The CUDA device id to use, or a list of device ids.
        tpb (int):
            The number of threads per block (CUDA).
        num_processes (int):
            The number of processes to use to register.
        update_interval (int):
            The number of nonces to solve between updates.
        log_verbose (bool):
            If ``true``, the registration process will log more information.
    Returns:
        success (bool):
            Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if prompt:
        if not Confirm.ask(
            "Run Faucet ?\n coldkey:    [bold white]{}[/bold white]\n network:    [bold white]{}[/bold white]".format(
                wallet.coldkeypub.ss58_address,
                subtensor.network,
            )
        ):
            return False, ""

    if not torch:
        log_no_torch_error()
        return False, "Requires torch"

    # Unlock coldkey
    try:
        wallet.coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False, ""

    # Get previous balance.
    old_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Attempt rolling registration.
    attempts = 1
    successes = 1
    while True:
        try:
            pow_result = None
            while pow_result is None or pow_result.is_stale(subtensor=subtensor):
                # Solve latest POW.
                if cuda:
                    if not torch.cuda.is_available():
                        if prompt:
                            bittensor.__console__.print("CUDA is not available.")
                        return False, "CUDA is not available."
                    pow_result: Optional[POWSolution] = create_pow(
                        subtensor,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        dev_id=dev_id,
                        tpb=tpb,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
                else:
                    pow_result: Optional[POWSolution] = create_pow(
                        subtensor,
                        wallet,
                        -1,
                        output_in_place,
                        cuda=cuda,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        log_verbose=log_verbose,
                    )
            call = subtensor.substrate.compose_call(
                call_module="SubtensorModule",
                call_function="faucet",
                call_params={
                    "block_number": pow_result.block_number,
                    "nonce": pow_result.nonce,
                    "work": [int(byte_) for byte_ in pow_result.seal],
                },
            )
            extrinsic = subtensor.substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = subtensor.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # process if registration successful, try again if pow is still valid
            response.process_events()
            if not response.is_success:
                bittensor.__console__.print(
                    f":cross_mark: [red]Failed[/red]: {format_error_message(response.error_message)}"
                )
                if attempts == max_allowed_attempts:
                    raise MaxAttemptsException
                attempts += 1
                # Wait a bit before trying again
                time.sleep(1)

            # Successful registration
            else:
                new_balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
                bittensor.__console__.print(
                    f"Balance: [blue]{old_balance}[/blue] :arrow_right: [green]{new_balance}[/green]"
                )
                old_balance = new_balance

                if successes == 3:
                    raise MaxSuccessException

                attempts = 1  # Reset attempts on success
                successes += 1

        except KeyboardInterrupt:
            return True, "Done"

        except MaxSuccessException:
            return True, f"Max successes reached: {3}"

        except MaxAttemptsException:
            return False, f"Max attempts reached: {max_allowed_attempts}"


def swap_hotkey_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    new_wallet: "bittensor.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    try:
        wallet.coldkey  # unlock coldkey
    except bittensor.KeyFileError:
        bittensor.__console__.print(
            ":cross_mark: [red]Keyfile is corrupt, non-writable, non-readable or the password used to decrypt is invalid[/red]:[bold white]\n  [/bold white]"
        )
        return False
    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(
            f"Swap {wallet.hotkey} for new hotkey: {new_wallet.hotkey}?"
        ):
            return False

    with bittensor.__console__.status(":satellite: Swapping hotkeys..."):
        success, err_msg = subtensor._do_swap_hotkey(
            wallet=wallet,
            new_wallet=new_wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if not success:
            bittensor.__console__.print(f":cross_mark: [red]Failed[/red]: {err_msg}")
            time.sleep(0.5)
            return False

        else:
            bittensor.__console__.print(
                f"Hotkey {wallet.hotkey} swapped for new hotkey: {new_wallet.hotkey}"
            )
            return True
