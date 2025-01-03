"""Module with helper functions for extrinsics."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import TYPE_CHECKING

from substrateinterface.exceptions import SubstrateRequestException

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


def submit_extrinsic(
    subtensor: "Subtensor",
    extrinsic: "GenericExtrinsic",
    wait_for_inclusion: bool,
    wait_for_finalization: bool,
) -> "ExtrinsicReceipt":
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
    extrinsic_hash = extrinsic.extrinsic_hash
    starting_block = subtensor.substrate.get_block()

    timeout = EXTRINSIC_SUBMISSION_TIMEOUT
    event = threading.Event()

    def submit():
        try:
            response_ = subtensor.substrate.submit_extrinsic(
                extrinsic=extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        except SubstrateRequestException as e:
            logging.error(format_error_message(e.args[0]))
            # Re-raise the exception for retrying of the extrinsic call. If we remove the retry logic,
            # the raise will need to be removed.
            raise
        finally:
            event.set()
        return response_

    with ThreadPoolExecutor(max_workers=1) as executor:
        response = None
        future = executor.submit(submit)
        if not event.wait(timeout):
            logging.error("Timed out waiting for extrinsic submission. Reconnecting.")
            # force reconnection of the websocket
            subtensor._get_substrate(force=True)
            raise SubstrateRequestException

        else:
            response = future.result()

    return response
