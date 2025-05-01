"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING, Union

import numpy as np
from async_substrate_interface.errors import SubstrateRequestException
from numpy.typing import NDArray

from bittensor.utils import format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor
    from async_substrate_interface import (
        AsyncExtrinsicReceipt,
        ExtrinsicReceipt,
    )
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.chain_data import StakeInfo
    from scalecodec.types import GenericExtrinsic
    from bittensor.utils.registration import torch


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


async def async_submit_extrinsic(
    subtensor: "AsyncSubtensor",
    extrinsic: "GenericExtrinsic",
    wait_for_inclusion: bool,
    wait_for_finalization: bool,
) -> "AsyncExtrinsicReceipt":
    """
    Submits an extrinsic to the substrate blockchain and handles potential exceptions.

    This function attempts to submit an extrinsic to the substrate blockchain with specified options
    for waiting for inclusion in a block and/or finalization. If an exception occurs during submission,
    it logs the error and re-raises the exception.

    Args:
        subtensor: The AsyncSubtensor instance used to interact with the blockchain.
        extrinsic: The extrinsic to be submitted to the blockchain.
        wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
        wait_for_finalization: Whether to wait for the extrinsic to be finalized on the blockchain.

    Returns:
        response: The response from the substrate after submitting the extrinsic.

    Raises:
        SubstrateRequestException: If the submission of the extrinsic fails, the error is logged and re-raised.
    """
    try:
        return await subtensor.substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
    except SubstrateRequestException as e:
        logging.error(format_error_message(e.args[0]))
        # Re-raise the exception for retrying of the extrinsic call. If we remove the retry logic,
        # the raise will need to be removed.
        raise


def get_old_stakes(
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    all_stakes: list["StakeInfo"],
) -> list[Balance]:
    """
    Retrieve the previous staking balances for a wallet's hotkeys across given netuids.

    This function searches through the provided staking data to find the stake amounts for the specified hotkeys and
    netuids associated with the wallet's coldkey. If no match is found for a particular hotkey and netuid combination,
    a default balance of zero is returned.

    Args:
        wallet (Wallet): The wallet containing the coldkey to compare with stake data.
        hotkey_ss58s (list[str]): List of hotkey SS58 addresses for which stakes are retrieved.
        netuids (list[int]): List of network unique identifiers (netuids) corresponding to the hotkeys.
        all_stakes (list[StakeInfo]): A collection of all staking information to search through.

    Returns:
        list[Balance]: A list of Balances, each representing the stake for a given hotkey and netuid.
    """
    stake_lookup = {
        (stake.hotkey_ss58, stake.coldkey_ss58, stake.netuid): stake.stake
        for stake in all_stakes
    }
    return [
        stake_lookup.get(
            (hotkey_ss58, wallet.coldkeypub.ss58_address, netuid),
            Balance.from_tao(0),  # Default to 0 balance if no match found
        )
        for hotkey_ss58, netuid in zip(hotkey_ss58s, netuids)
    ]


def convert_and_normalize_weights_and_uids(
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
) -> tuple[list[int], list[int]]:
    """Converts weights and uids to numpy arrays if they are not already.

    Arguments:
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The ``uint64`` uids of destination neurons.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights to set. These must be ``float`` s
            and correspond to the passed ``uid`` s.

    Returns:
        weight_uids, weight_vals: Bytes converted weights and uids
    """
    if isinstance(uids, list):
        uids = np.array(uids, dtype=np.int64)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float32)

    # Reformat and normalize.
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids, weights
    )
    return weight_uids, weight_vals
