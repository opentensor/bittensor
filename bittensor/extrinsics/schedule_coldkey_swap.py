# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
from typing import Tuple, Optional, Union, List
from rich.prompt import Confirm

from bittensor.utils.coldkey_swap_pow import (
    create_pow_for_coldkey_swap,
    SwapPOWSolution,
)


def schedule_coldkey_swap_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    new_coldkey: str,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    prompt: bool = True,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Schedules a coldkey swap on the Bittensor network.

    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the current coldkey.
        new_coldkey (str): The SS58 address of the new coldkey.
        wait_for_inclusion (bool, optional): Waits for the transaction to be included in a block.
        wait_for_finalization (bool, optional): Waits for the transaction to be finalized on the blockchain.
        prompt (bool, optional): If ``True``, prompts for user confirmation before proceeding.
        max_allowed_attempts (int, optional): Maximum attempts to generate POW
        output_in_place (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If true, prints the progress of the proof of work to the console
                in-place. Meaning the progress is printed on the same lines.
        cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If true, uses CUDA to solve the proof of work.
        dev_id (:obj:`Union[List[int], int]`, `optional`, defaults to :obj:`0`):
            The CUDA device id(s) to use. If cuda is true and dev_id is a list,
                then multiple CUDA devices will be used to solve the proof of work.
        tpb (:obj:`int`, `optional`, defaults to :obj:`256`):
            The number of threads per block to use when solving the proof of work.
            Should be a multiple of 32.
        num_processes (:obj:`int`, `optional`, defaults to :obj:`None`):
            The number of processes to use when solving the proof of work.
            If None, then the number of processes is equal to the number of
                CPU cores.
        update_interval (:obj:`int`, `optional`, defaults to :obj:`None`):
            The number of nonces to run before checking for a new block.
        log_verbose (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If true, prints the progress of the proof of work more verbosely.

    Returns:
        Tuple[bool, str]: ``True`` if the operation was successful, ``False`` otherwise. And ``msg``, a string
        value describing the success or potential error.
    """
    # Decrypt keys
    wallet.coldkey

    if prompt and not Confirm.ask(
        f"Would you like to schedule a coldkey swap to {new_coldkey}?"
    ):
        return False, "User cancelled the operation."

    try:
        # Generate the POW
        pow_result = _generate_pow_for_coldkey_swap(
            subtensor=subtensor,
            wallet=wallet,
            old_coldkey=wallet.coldkeypub.ss58_address,  # swapping from the signer coldkey
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )
        with bittensor.__console__.status(":satellite: Scheduling coldkey swap..."):
            # Call Subtensor with POW
            success, error_message = subtensor._do_schedule_coldkey_swap(
                wallet=wallet,
                new_coldkey=new_coldkey,
                pow_result=pow_result,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            if not wait_for_finalization and not wait_for_inclusion:
                return (
                    True,
                    "Scheduled coldkey swap without waiting for inclusion or finalization.",
                )

            if success is True:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Scheduled[/green]"
                )
                bittensor.logging.success(
                    prefix="Schedule Coldkey Swap",
                    suffix="<green>Scheduled: </green>" + str(success),
                )
                return True, "Successfully scheduled cold key swap."
            else:
                bittensor.logging.error(
                    msg=error_message,
                    prefix="Schedule Coldkey Swap",
                    suffix="<red>Failed: </red>",
                )
                return False, error_message

    except Exception as e:
        import traceback

        # Print the full stack trace
        traceback.print_exc()
        bittensor.__console__.print(
            ":cross_mark: [red]Failed[/red]: error:{}".format(e)
        )
        bittensor.logging.warning(
            prefix="Schedule Coldkey Swap", suffix="<red>Failed: </red>" + str(e)
        )
        return False, str(e)


def _generate_pow_for_coldkey_swap(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    old_coldkey: str,
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Optional[SwapPOWSolution]:
    """
    Generate the proof of work for a scheduled cold-key swap.

    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the current coldkey.
        old_coldkey (str): The SS58 address of the old coldkey.
        max_allowed_attempts (int, optional): Maximum attempts to generate POW
        output_in_place (bool, optional): If true, prints the progress of the proof of work to the console in-place.
        cuda (bool, optional): If true, uses CUDA to solve the proof of work.
        dev_id (Union[List[int], int], optional): The CUDA device id(s) to use.
        tpb (int, optional): The number of threads per block to use when solving the proof of work.
        num_processes (int, optional): The number of processes to use when solving the proof of work.
        update_interval (int, optional): The number of nonces to run before checking for a new block.
        log_verbose (bool, optional): If true, prints the progress of the proof of work more verbosely.

    Returns:
        Optional[SwapPOWSolution]: The proof of work solution if found, None otherwise.

    Raises:
        ValueError: If unable to solve POW after maximum attempts.
    """
    for attempts in range(1, max_allowed_attempts + 1):
        bittensor.__console__.print(
            ":satellite: Generating POW for coldkey swap...({}/{})".format(
                attempts, max_allowed_attempts
            )
        )

        try:
            pow_result = create_pow_for_coldkey_swap(
                subtensor=subtensor,
                wallet=wallet,
                old_coldkey=old_coldkey,
                output_in_place=output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
            if pow_result:
                return pow_result

        except RuntimeError as e:
            bittensor.__console__.print(f"Error during PoW generation: {str(e)}")
            if "CUDA is not available" in str(e):
                bittensor.__console__.print("Falling back to CPU...")
                cuda = False
            else:
                raise

    bittensor.__console__.print("Unable to solve POW.")
    raise ValueError("Unable to solve POW required to schedule a coldkey swap.")
