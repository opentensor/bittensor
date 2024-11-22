"""Module with helper functions for extrinsics."""

import threading
from typing import TYPE_CHECKING

from substrateinterface.exceptions import SubstrateRequestException, ExtrinsicNotFound

from bittensor.utils.btlogging import logging
from bittensor.utils import format_error_message

if TYPE_CHECKING:
    from substrateinterface import SubstrateInterface, ExtrinsicReceipt
    from scalecodec.types import GenericExtrinsic


class _ThreadingTimeoutException(Exception):
    """
    Exception raised for timeout. Different from TimeoutException because this also triggers
    a websocket failure. This exception should only be used with `threading` timer..
    """

    pass


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

    def _handler():
        """
        Timeout handler for threading. Will raise a TimeoutError if timeout is exceeded.
        """
        logging.error("Timed out waiting for extrinsic submission.")
        raise _ThreadingTimeoutException

    # sets a timeout timer for the next call to 200 seconds
    # will raise a _ThreadingTimeoutException if it reaches this point
    timer = threading.Timer(200, _handler)

    try:
        timer.start()
        response = substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    except SubstrateRequestException as e:
        logging.error(format_error_message(e.args[0], substrate=substrate))
        # Re-rise the exception for retrying of the extrinsic call. If we remove the retry logic, the raise will need
        # to be removed.
        raise

    except _ThreadingTimeoutException:
        after_timeout_block = substrate.get_block()

        response = None
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
    finally:
        timer.cancel()

    if response is None:
        logging.error(
            f"Extrinsic '0x{extrinsic_hash.hex()}' not submitted. "
            f"Initially attempted to submit at block {starting_block['header']['number']}."
        )
        raise SubstrateRequestException

    return response
