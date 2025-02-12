"""Module with helper functions for extrinsics."""

from typing import TYPE_CHECKING

from async_substrate_interface.errors import SubstrateRequestException

from bittensor.utils import format_error_message
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import AsyncSubtensor
    from async_substrate_interface import (
        AsyncExtrinsicReceipt,
        ExtrinsicReceipt,
    )
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.chain_data import StakeInfo
    from scalecodec.types import GenericExtrinsic, GenericCall


def sign_and_send_with_nonce(
    subtensor: "Subtensor",
    call: "GenericCall",
    wallet: "Wallet",
    wait_for_inclusion: bool,
    wait_for_finalization,
    nonce_key: str = "hotkey",
    signing_key: str = "hotkey",
):
    keypair = getattr(wallet, nonce_key)
    next_nonce = subtensor.substrate.get_account_next_index(keypair.ss58_address)
    signing_keypair = getattr(wallet, signing_key)
    extrinsic = subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=signing_keypair,
        nonce=next_nonce,
    )
    response = subtensor.substrate.submit_extrinsic(
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, None

    if response.is_success:
        return True, None

    return False, format_error_message(response.error_message)


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

    This function searches through the provided staking data to find the stake amounts
    for the specified hotkeys and netuids associated with the wallet's coldkey. If no match
    is found for a particular hotkey and netuid combination, a default balance of zero is returned.

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
