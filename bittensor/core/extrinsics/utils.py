"""Module with helper functions for extrinsics."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import TYPE_CHECKING

from substrateinterface.exceptions import SubstrateRequestException, ExtrinsicNotFound

from bittensor.utils.btlogging import logging
from bittensor.utils import format_error_message

if TYPE_CHECKING:
    from substrateinterface import SubstrateInterface, ExtrinsicReceipt
    from scalecodec.types import GenericExtrinsic

EXTRINSIC_SUBMISSION_TIMEOUT = int(os.getenv("EXTRINSIC_SUBMISSION_TIMEOUT", 200))


def submit_extrinsic(
    substrate: "SubstrateInterface",
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
        substrate (substrateinterface.SubstrateInterface): The substrate interface instance used to interact with the blockchain.
        extrinsic (scalecodec.types.GenericExtrinsic): The extrinsic to be submitted to the blockchain.
        wait_for_inclusion (bool): Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization (bool): Whether to wait for the extrinsic to be finalized on the blockchain.

    Returns:
        response: The response from the substrate after submitting the extrinsic.

    Raises:
        SubstrateRequestException: If the submission of the extrinsic fails, the error is logged and re-raised.
    """
    extrinsic_hash = extrinsic.extrinsic_hash
    starting_block = substrate.get_block()

    timeout = EXTRINSIC_SUBMISSION_TIMEOUT
    event = threading.Event()

    def submit():
        try:
            response_ = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        except SubstrateRequestException as e:
            logging.error(format_error_message(e.args[0], substrate=substrate))
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
            logging.error("Timed out waiting for extrinsic submission.")
            after_timeout_block = substrate.get_block()

            for block_num in range(
                starting_block["header"]["number"],
                after_timeout_block["header"]["number"] + 1,
            ):
                block_hash = substrate.get_block_hash(block_num)
                try:
                    response = substrate.retrieve_extrinsic_by_hash(
                        block_hash, f"0x{extrinsic_hash.hex()}"
                    )
                except ExtrinsicNotFound:
                    continue
                if response:
                    break
            if response is None:
                logging.error(
                    f"Extrinsic '0x{extrinsic_hash.hex()}' not submitted. "
                    f"Initially attempted to submit at block {starting_block['header']['number']}."
                )
                raise SubstrateRequestException

        else:
            response = future.result()

    return response
