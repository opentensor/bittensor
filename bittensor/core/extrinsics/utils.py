"""Module with helper functions for extrinsics."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

from substrateinterface.exceptions import SubstrateRequestException, ExtrinsicNotFound

from bittensor.utils.btlogging import logging
from bittensor.utils import format_error_message

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from substrateinterface import ExtrinsicReceipt
    from scalecodec.types import GenericExtrinsic

try:
    EXTRINSIC_SUBMISSION_TIMEOUT = float(os.getenv("EXTRINSIC_SUBMISSION_TIMEOUT", 200))
except ValueError:
    raise ValueError(
        "EXTRINSIC_SUBMISSION_TIMEOUT environment variable must be a float."
    )

if EXTRINSIC_SUBMISSION_TIMEOUT < 0:
    raise ValueError("EXTRINSIC_SUBMISSION_TIMEOUT cannot be negative.")


def extrinsic_recovery(
    extrinsic_hash_hex: str, subtensor: "Subtensor", starting_block: dict[str, Any]
) -> Optional["ExtrinsicReceipt"]:
    """
    Attempts to recover an extrinsic from the chain that was previously submitted

    Args:
        extrinsic_hash_hex: the hex representation (including '0x' prefix) of the extrinsic hash
        subtensor: the Subtensor object to interact with the chain
        starting_block: the initial block dict at the time the extrinsic was submitted

    Returns:
        ExtrinsicReceipt of the extrinsic if recovered, None otherwise.
    """

    after_timeout_block = subtensor.substrate.get_block()
    response = None
    for block_num in range(
        starting_block["header"]["number"],
        after_timeout_block["header"]["number"] + 1,
    ):
        block_hash = subtensor.substrate.get_block_hash(block_num)
        try:
            response = subtensor.substrate.retrieve_extrinsic_by_hash(
                block_hash, extrinsic_hash_hex
            )
            response.process_events()
        except (ExtrinsicNotFound, SubstrateRequestException):
            continue
        if response:
            break
    return response


def submit_extrinsic(
    subtensor: "Subtensor",
    extrinsic: "GenericExtrinsic",
    wait_for_inclusion: bool,
    wait_for_finalization: bool,
) -> "ExtrinsicReceipt":
    event = threading.Event()
    extrinsic_hash = extrinsic.extrinsic_hash
    starting_block = subtensor.substrate.get_block()
    timeout = EXTRINSIC_SUBMISSION_TIMEOUT
    """
    Submits an extrinsic to the substrate blockchain and handles potential exceptions.

    This function attempts to submit an extrinsic to the substrate blockchain with specified options
    for waiting for inclusion in a block and/or finalization. If an exception occurs during submission,
    it logs the error and re-raises the exception.

    Args:
        subtensor: The Subtensor instance used to interact with the blockchain.
        extrinsic (scalecodec.types.GenericExtrinsic): The extrinsic to be submitted to the blockchain.
        wait_for_inclusion (bool): Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization (bool): Whether to wait for the extrinsic to be finalized on the blockchain.

    Returns:
        response: The response from the substrate after submitting the extrinsic.

    Raises:
        SubstrateRequestException: If the submission of the extrinsic fails, the error is logged and re-raised.
    """

    def try_submission():
        def submit():
            try:
                response__ = subtensor.substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
            except SubstrateRequestException as e:
                logging.error(
                    format_error_message(e.args[0], substrate=subtensor.substrate)
                )
                raise
            finally:
                event.set()
            return response__

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(submit)
            if not event.wait(timeout):
                logging.error(
                    "Timed out waiting for extrinsic submission. Reconnecting."
                )
                response_ = None
            else:
                response_ = future.result()
        return response_

    response = try_submission()
    if response is None:
        subtensor._get_substrate(force=True)
        response = extrinsic_recovery(
            f"0x{extrinsic_hash.hex()}", subtensor, starting_block
        )
        if response is None:
            raise SubstrateRequestException

    return response
