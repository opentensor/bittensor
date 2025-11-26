"""Module provides sync MEV Shield extrinsics."""

from scalecodec.utils.ss58 import ss58_decode
import hashlib
import struct
from typing import TYPE_CHECKING, Optional

import bittensor_drand
from bittensor_drand import encrypt_mlkem768

from bittensor.core.extrinsics.pallets import MevShield
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet, Keypair
    from scalecodec.types import GenericCall


def get_mev_commitment_and_ciphertext(
    call: "GenericCall",
    signer_keypair: "Keypair",
    genesis_hash: str,
    nonce: int,
    ml_kem_768_public_key: bytes,
) -> tuple[str, bytes, bytes, bytes]:
    # Create payload_core: signer (32B) + nonce (u32 LE) + SCALE(call)
    decoded_ss58 = ss58_decode(signer_keypair.ss58_address)
    decoded_ss58_cut = (
        decoded_ss58[2:] if decoded_ss58.startswith("0x") else decoded_ss58
    )
    signer_bytes = bytes.fromhex(decoded_ss58_cut)  # 32 bytes

    # Ensure nonce is u32 (as in Rust)
    nonce_u32 = nonce & 0xFFFFFFFF
    nonce_bytes = struct.pack("<I", nonce_u32)

    scale_call_bytes = bytes(call.data.data)  # SCALE encoded call
    mev_shield_version = bittensor_drand.mlkem_kdf_id()

    # Fix genesis_hash processing
    genesis_hash_clean = (
        genesis_hash[2:] if genesis_hash.startswith("0x") else genesis_hash
    )
    genesis_hash_bytes = bytes.fromhex(genesis_hash_clean)

    payload_core = signer_bytes + nonce_bytes + scale_call_bytes

    # Sign payload: coldkey.sign(b"mev-shield:v1" + genesis_hash + payload_core)
    message_to_sign = (
        b"mev-shield:" + mev_shield_version + genesis_hash_bytes + payload_core
    )

    signature = signer_keypair.sign(message_to_sign)

    # Create plaintext: payload_core + b"\x01" + signature
    plaintext = payload_core + b"\x01" + signature

    # Getting ciphertext (encrypting plaintext using ML-KEM-768)
    ciphertext = encrypt_mlkem768(ml_kem_768_public_key, plaintext)

    # Compute commitment: blake2_256(payload_core)
    commitment_hash = hashlib.blake2b(payload_core, digest_size=32).digest()
    commitment_hex = "0x" + commitment_hash.hex()

    return commitment_hex, ciphertext, payload_core, signature


def submit_encrypted_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    call: "GenericCall",
    signer_keypair: Optional["Keypair"] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> ExtrinsicResponse:
    """
    Submits an encrypted extrinsic to the MEV Shield pallet.

    This function encrypts a call using ML-KEM-768 + XChaCha20Poly1305 and submits it to the MevShield pallet. The
    extrinsic remains encrypted in the transaction pool until it is included in a block and decrypted by validators.

    Parameters:
        subtensor: The Subtensor client instance used for blockchain interaction.
        wallet: The wallet used to sign the extrinsic (must be unlocked, coldkey will be used for signing).
        call: The GenericCall object to encrypt and submit.
        signer_keypair: The Keypair object used for signing the inner call.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        ValueError: If NextKey is not available in storage or encryption fails.
        SubstrateRequestException: If the extrinsic fails to be submitted or included.

    Note:
        The encryption uses the public key from NextKey storage, which rotates every block. The payload structure is:
        payload_core = signer_bytes (32B) + nonce (u32 LE, 4B) + SCALE(call)
        plaintext = payload_core + b"\\x01" + signature (64B for sr25519)
        commitment = blake2_256(payload_core)
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(wallet, raise_error)
        ).success:
            return unlocked

        # Use wallet.coldkey as default signer if signer_keypair is not provided
        if signer_keypair is None:
            signer_keypair = wallet.coldkey

        ml_kem_768_public_key = subtensor.get_mev_shield_next_key()
        if ml_kem_768_public_key is None:
            return ExtrinsicResponse.from_exception(
                raise_error=raise_error,
                error=ValueError("MEV Shield NextKey not available in storage."),
            )

        genesis_hash = subtensor.get_block_hash(block=0)
        nonce = subtensor.substrate.get_account_nonce(signer_keypair.ss58_address)

        mev_commitment, mev_ciphertext, payload_core, signature = (
            get_mev_commitment_and_ciphertext(
                call=call,
                signer_keypair=signer_keypair,
                genesis_hash=genesis_hash,
                nonce=nonce,
                ml_kem_768_public_key=ml_kem_768_public_key,
            )
        )

        extrinsic_call = MevShield(subtensor).submit_encrypted(
            commitment=mev_commitment,
            ciphertext=mev_ciphertext,
        )

        response = subtensor.sign_and_send_extrinsic(
            wallet=wallet,
            call=extrinsic_call,
            sign_with="hotkey",
            use_nonce=False,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if response.success:
            logging.debug("[green]Encrypted extrinsic submitted successfully.[/green]")
            response.data = {
                "commitment": mev_commitment,
                "ciphertext": mev_ciphertext,
                "nonce": nonce,
                "payload_core": payload_core,
                "signature": signature,
                "submitting_id": extrinsic_call.call_hash,
            }
        else:
            logging.error(f"[red]{response.message}[/red]")

        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
