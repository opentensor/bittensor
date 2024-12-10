"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING

from substrateinterface.exceptions import SubstrateRequestException

from bittensor.utils.btlogging import logging
from bittensor.utils import format_error_message

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from substrateinterface import ExtrinsicReceipt
    from scalecodec.types import GenericExtrinsic


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
    try:
        return subtensor.substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    except SubstrateRequestException as e:
        logging.error(format_error_message(e.args[0]))
        # Re-raise the exception for retrying of the extrinsic call. If we remove the retry logic,
        # the raise will need to be removed.
        raise
