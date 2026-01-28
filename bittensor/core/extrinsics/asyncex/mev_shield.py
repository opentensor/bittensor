"""Module provides async MEV Shield extrinsics."""

from typing import TYPE_CHECKING, Optional

from async_substrate_interface import AsyncExtrinsicReceipt

from bittensor.core.extrinsics.pallets import MevShield
from bittensor.core.extrinsics.utils import (
    get_mev_commitment_and_ciphertext,
    get_event_data_by_event_name,
)
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet
    from scalecodec.types import GenericCall


async def wait_for_extrinsic_by_hash(
    subtensor: "AsyncSubtensor",
    extrinsic_hash: str,
    shield_id: str,
    submit_block_hash: str,
    timeout_blocks: int = 3,
) -> Optional["AsyncExtrinsicReceipt"]:
    """
    Wait for the result of a MeV Shield encrypted extrinsic.

    After submit_encrypted succeeds, the block author will decrypt and submit the inner extrinsic directly. This
    function polls subsequent blocks looking for either:
    - an extrinsic matching the provided hash (success)
    OR
    - a markDecryptionFailed extrinsic with matching shield ID (failure)

    Args:
        subtensor: SubtensorInterface instance.
        extrinsic_hash: The hash of the inner extrinsic to find.
        shield_id: The wrapper ID from EncryptedSubmitted event (for detecting decryption failures).
        submit_block_hash: Block hash where submit_encrypted was included.
        timeout_blocks: Max blocks to wait.

    Returns:
        Optional ExtrinsicReceipt.
    """

    starting_block = await subtensor.substrate.get_block_number(submit_block_hash)
    current_block = starting_block

    while current_block - starting_block <= timeout_blocks:
        logging.debug(
            f"Waiting for MEV Protection (checking block {current_block - starting_block} of {timeout_blocks})..."
        )

        await subtensor.wait_for_block()

        block_hash = await subtensor.substrate.get_block_hash(current_block)
        extrinsics = await subtensor.substrate.get_extrinsics(block_hash)

        result_idx = None
        for idx, extrinsic in enumerate(extrinsics):
            # Success: Inner extrinsic executed
            if f"0x{extrinsic.extrinsic_hash.hex()}" == extrinsic_hash:
                result_idx = idx
                break

            # Failure: Decryption failed
            call = extrinsic.value.get("call", {})
            if (
                call.get("call_module") == "MevShield"
                and call.get("call_function") == "mark_decryption_failed"
            ):
                call_args = call.get("call_args", [])
                for arg in call_args:
                    if arg.get("name") == "id" and arg.get("value") == shield_id:
                        result_idx = idx
                        break
                if result_idx is not None:
                    break

        if result_idx is not None:
            return AsyncExtrinsicReceipt(
                substrate=subtensor.substrate,
                block_hash=block_hash,
                block_number=current_block,
                extrinsic_idx=result_idx,
            )

        current_block += 1

    return None


async def submit_encrypted_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    call: "GenericCall",
    sign_with: str = "coldkey",
    *,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
    wait_for_revealed_execution: bool = True,
    blocks_for_revealed_execution: int = 3,
) -> ExtrinsicResponse:
    """
    Submits an encrypted extrinsic to the MEV Shield pallet.

    This function encrypts a call using ML-KEM-768 + XChaCha20Poly1305 and submits it to the MevShield pallet. The
    extrinsic remains encrypted in the transaction pool until it is included in a block and decrypted by validators.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked, coldkey will be used for signing).
        call: The GenericCall object to encrypt and submit.
        sign_with: The keypair to use for signing the inner call/extrinsic. Can be either "coldkey" or "hotkey".
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the executed event, indicating that validators have
            successfully decrypted and executed the inner call. If True, the function will poll subsequent blocks for
            the event matching this submission's commitment.
        blocks_for_revealed_execution: Maximum number of blocks to poll for the executed event after inclusion.
            The function checks blocks from start_block to start_block + blocks_for_revealed_execution. Returns
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
        if sign_with not in ("coldkey", "hotkey"):
            raise AttributeError(
                f"'sign_with' must be either 'coldkey' or 'hotkey', not '{sign_with}'"
            )

        if wait_for_revealed_execution and not wait_for_inclusion:
            return ExtrinsicResponse.from_exception(
                raise_error=raise_error,
                error=ValueError(
                    "`wait_for_inclusion` must be `True` if `wait_for_revealed_execution` is `True`."
                ),
            )

        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error, sign_with)
        ).success:
            return unlocked

        ml_kem_768_public_key = await subtensor.get_mev_shield_next_key()
        if ml_kem_768_public_key is None:
            return ExtrinsicResponse.from_exception(
                raise_error=raise_error,
                error=ValueError("MEV Shield NextKey not available in storage."),
            )

        inner_signing_keypair = getattr(wallet, sign_with)

        era = "00" if period is None else {"period": period}

        current_nonce = await subtensor.substrate.get_account_next_index(
            account_address=inner_signing_keypair.ss58_address
        )
        next_nonce = await subtensor.substrate.get_account_next_index(
            account_address=inner_signing_keypair.ss58_address
        )
        signed_extrinsic = await subtensor.substrate.create_signed_extrinsic(
            call=call, keypair=inner_signing_keypair, nonce=next_nonce, era=era
        )

        mev_commitment, mev_ciphertext, payload_core = (
            get_mev_commitment_and_ciphertext(
                signed_ext=signed_extrinsic,
                ml_kem_768_public_key=ml_kem_768_public_key,
            )
        )

        extrinsic_call = await MevShield(subtensor).submit_encrypted(
            commitment=mev_commitment,
            ciphertext=mev_ciphertext,
        )

        response = await subtensor.sign_and_send_extrinsic(
            wallet=wallet,
            sign_with=sign_with,
            call=extrinsic_call,
            nonce=current_nonce,
            period=period,
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
                "signed_extrinsic_hash": f"0x{signed_extrinsic.extrinsic_hash.hex()}",
            }
            if wait_for_revealed_execution:
                triggered_events = await response.extrinsic_receipt.triggered_events
                event = get_event_data_by_event_name(
                    events=triggered_events,
                    event_name="mevShield.EncryptedSubmitted",
                )

                if event is None:
                    return ExtrinsicResponse.from_exception(
                        raise_error=raise_error,
                        error=RuntimeError("EncryptedSubmitted event not found."),
                    )

                shield_id = event["attributes"]["id"]

                response.mev_extrinsic = await wait_for_extrinsic_by_hash(
                    subtensor=subtensor,
                    extrinsic_hash=f"0x{signed_extrinsic.extrinsic_hash.hex()}",
                    shield_id=shield_id,
                    submit_block_hash=response.extrinsic_receipt.block_hash,
                    timeout_blocks=blocks_for_revealed_execution,
                )
                if response.mev_extrinsic is None:
                    return ExtrinsicResponse.from_exception(
                        raise_error=raise_error,
                        error=RuntimeError(
                            "Failed to find outcome of the shield extrinsic (The protected extrinsic wasn't decrypted)."
                        ),
                    )

                if not await response.mev_extrinsic.is_success:
                    response.message = format_error_message(
                        await response.mev_extrinsic.error_message
                    )
                    response.error = RuntimeError(response.message)
                    response.success = False
                    if raise_error:
                        raise response.error
                else:
                    logging.debug(
                        "[green]Encrypted extrinsic submitted successfully.[/green]"
                    )
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
