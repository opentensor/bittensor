import bittensor
from typing import Tuple, Optional, Union, List
from rich.prompt import Confirm

from bittensor.utils import torch
from bittensor.utils.registration import POWSolution, create_pow


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

    with bittensor.__console__.status(":satellite: Scheduling coldkey swap..."):
        try:
            # Generate the POW
            pow_result = _generate_pow_for_coldkey_swap(
                subtensor=subtensor,
                wallet=wallet,
                max_allowed_attempts=max_allowed_attempts,
                output_in_place=output_in_place,
                cuda=cuda,
                dev_id=dev_id,
                tpb=tpb,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )

            # Call Subtensor with POW
            success, error_message = subtensor._do_schedule_coldkey_swap(
                wallet=wallet,
                new_coldkey=new_coldkey,
                pow_result=pow_result,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization
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
    max_allowed_attempts: int = 3,
    output_in_place: bool = True,
    cuda: bool = False,
    dev_id: Union[List[int], int] = 0,
    tpb: int = 256,
    num_processes: Optional[int] = None,
    update_interval: Optional[int] = None,
    log_verbose: bool = False,
) -> Optional[POWSolution]:
    """
    Generate the proof of work for a scheduled cold-key swap.

    Args:
        subtensor (bittensor.subtensor): The subtensor instance used for blockchain interaction.
        wallet (bittensor.wallet): The wallet associated with the current coldkey.
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
    """

    attempts = 1
    # it is not related to any netuid
    netuid = -1
    while True:
        bittensor.__console__.print(
            ":satellite: Generating POW for coldkey swap...({}/{})".format(
                attempts, max_allowed_attempts
            )
        )
        # Solve latest POW.
        pow_result: Optional[POWSolution]
        if cuda:
            if not torch.cuda.is_available():
                bittensor.__console__.print("CUDA use requested, but not available.")
                raise EnvironmentError("CUDA use requested, but not available.")

            pow_result = create_pow(
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
            pow_result = create_pow(
                subtensor,
                wallet,
                netuid,
                output_in_place,
                cuda=cuda,
                num_processes=num_processes,
                update_interval=update_interval,
                log_verbose=log_verbose,
            )
        if pow_result or attempts >= max_allowed_attempts:
            break
        attempts += 1

    if pow_result:
        return pow_result
    else:
        bittensor.__console__.print("Unable to solve POW.")
        raise ValueError("Unable to solve POW required to schedule a coldkey swap.")
