"""Module provides async MEV Shield extrinsics."""

from typing import TYPE_CHECKING, Optional

from async_substrate_interface import AsyncExtrinsicReceipt

from bittensor.core.extrinsics.pallets import MevShield
from bittensor.core.extrinsics.utils import (
    get_event_attributes_by_event_name,
    get_mev_commitment_and_ciphertext,
    post_process_mev_response,
    POST_SUBMIT_MEV_EVENTS,
    MEV_SUBMITTED_EVENT,
)
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet, Keypair
    from scalecodec.types import GenericCall, GenericExtrinsic


async def find_revealed_extrinsic(
    subtensor: "AsyncSubtensor",
    event_names: list[str],
    event_hash_id: str,
    start_block_hash: str,
    blocks_ahead: int = 5,
) -> tuple[str, "AsyncExtrinsicReceipt"] | tuple[None, None]:
    """
    Searches for an extrinsic containing a specific MEV Shield event in subsequent blocks.

    This function iterates through blocks starting from the specified block hash and searches for extrinsics that
    contain a MEV Shield event (DecryptedExecuted or DecryptedRejected) matching the provided wrapper_id and signer. It
    checks each extrinsic's triggered events to find the matching event.

    Parameters:
        subtensor: The Subtensor instance used for blockchain queries.
        event_names: The event identifiers to search for. Typically "DecryptedExecuted" or "DecryptedRejected" for MEV
            Shield transactions.
        event_hash_id: The wrapper_id (hash of (author, commitment, ciphertext)) to match. This uniquely identifies a
            specific MEV Shield submission.
        start_block_hash: The hash of the block where the search should begin. Usually the block where submit_encrypted
            was included.
        blocks_ahead: Maximum number of blocks to search ahead from the start block. Defaults to 5 blocks. The function
            will check blocks from start_block + 1 to start_block + blocks_ahead (the start block itself is not checked,
            as execute_revealed will be in subsequent blocks).

    Returns:
        Tuple with event name and ExtrinsicReceipt object.
    """
    start_block_number = await subtensor.substrate.get_block_number(start_block_hash)

    for offset in range(1, blocks_ahead + 1):
        current_block_number = start_block_number + offset

        try:
            current_block_hash = await subtensor.substrate.get_block_hash(
                current_block_number
            )
            events = await subtensor.substrate.get_events(current_block_hash)
        except Exception as e:
            logging.debug(
                f"Error getting extrinsics for block `{current_block_number}`: {e}"
            )
            continue

        for event_name in event_names:
            if event := get_event_attributes_by_event_name(events, event_name):
                if event["attributes"]["id"] == event_hash_id:
                    return event_name, AsyncExtrinsicReceipt(
                        substrate=subtensor.substrate,
                        block_hash=current_block_hash,
                        extrinsic_idx=event["extrinsic_idx"],
                    )

        logging.debug(f"No {event_names} event found in block {current_block_number}.")
        await subtensor.wait_for_block()

    return None, None


async def submit_encrypted_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    signed_ext: "GenericExtrinsic",
    signer_keypair: Optional["Keypair"] = None,
    period: Optional[int] = None,
    nonce: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    wait_for_revealed_execution: bool = True,
    blocks_for_revealed_execution: int = 5,
) -> ExtrinsicResponse:
    """
    Submits an encrypted extrinsic to the MEV Shield pallet.

    This function encrypts a call using ML-KEM-768 + XChaCha20Poly1305 and submits it to the MevShield pallet. The
    extrinsic remains encrypted in the transaction pool until it is included in a block and decrypted by validators.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked, coldkey will be used for signing).
        signed_ext: The signed GenericExtrinsic object to encrypt and submit.
        signer_keypair: The Keypair object used for signing the inner call.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the DecryptedExecuted event, indicating that validators have
            successfully decrypted and executed the inner call. If True, the function will poll subsequent blocks for
            the event matching this submission's commitment.
        blocks_for_revealed_execution: Maximum number of blocks to poll for the DecryptedExecuted event after inclusion.
            The function checks blocks from start_block + 1 to start_block + blocks_for_revealed_execution. Returns
            immediately if the event is found before the block limit is reached.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        ValueError: If NextKey is not available in storage or encryption fails.
        SubstrateRequestException: If the extrinsic fails to be submitted or included.

    Note:
        The encryption uses the public key from NextKey storage, which rotates every block. The payload structure is:
        payload_core = signer_bytes (32B) + key_hash (32B Blake2-256 hash of NextKey) + SCALE(call)
        plaintext = payload_core + b"\\x01" + signature (64B for sr25519)
        commitment = blake2_256(payload_core)

        The key_hash binds the transaction to the key epoch at submission time and replaces nonce-based replay
        protection.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        if wait_for_revealed_execution and not wait_for_inclusion:
            return ExtrinsicResponse.from_exception(
                raise_error=raise_error,
                error=ValueError(
                    "`wait_for_inclusion` must be `True` if `wait_for_revealed_execution` is `True`."
                ),
            )

        # Use wallet.coldkey as default signer if signer_keypair is not provided
        if signer_keypair is None:
            signer_keypair = wallet.coldkey

        ml_kem_768_public_key = await subtensor.get_mev_shield_next_key()
        if ml_kem_768_public_key is None:
            return ExtrinsicResponse.from_exception(
                raise_error=raise_error,
                error=ValueError("MEV Shield NextKey not available in storage."),
            )

        mev_commitment, mev_ciphertext, payload_core = (
            get_mev_commitment_and_ciphertext(
                signed_ext=signed_ext,
                signer_keypair=signer_keypair,
                ml_kem_768_public_key=ml_kem_768_public_key,
            )
        )

        extrinsic_call = await MevShield(subtensor).submit_encrypted(
            commitment=mev_commitment,
            ciphertext=mev_ciphertext,
        )

        response = await subtensor.sign_and_send_extrinsic(
            wallet=wallet,
            call=extrinsic_call,
            period=period,
            nonce=nonce,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if response.success:
            response.data = {
                "commitment": mev_commitment,
                "ciphertext": mev_ciphertext,
                "ml_kem_768_public_key": ml_kem_768_public_key,
                "payload_core": payload_core,
                "submitting_id": extrinsic_call.call_hash,
            }
            if wait_for_revealed_execution:
                triggered_events = await response.extrinsic_receipt.triggered_events

                event_hash_id = get_event_attributes_by_event_name(
                    events=triggered_events, event_name=MEV_SUBMITTED_EVENT
                )["attributes"]["id"]

                reveled_event, reveled_extrinsic = await find_revealed_extrinsic(
                    subtensor=subtensor,
                    event_names=POST_SUBMIT_MEV_EVENTS,
                    event_hash_id=event_hash_id,
                    start_block_hash=response.extrinsic_receipt.block_hash,
                    blocks_ahead=blocks_for_revealed_execution,
                )

                post_process_mev_response(
                    response=response,
                    revealed_name=reveled_event,
                    revealed_extrinsic=reveled_extrinsic,
                    raise_error=raise_error,
                )

            logging.debug("[green]Encrypted extrinsic submitted successfully.[/green]")
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
