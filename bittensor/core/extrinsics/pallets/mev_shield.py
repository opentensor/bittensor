from dataclasses import dataclass

from .base import CallBuilder as _BasePallet, Call


@dataclass
class MevShield(_BasePallet):
    """Factory class for creating GenericCall objects for MevShield pallet functions.

    This class provides methods to create GenericCall instances for all MevShield pallet extrinsics.

    Works with both sync (Subtensor) and async (AsyncSubtensor) instances. For async operations, pass an AsyncSubtensor
    instance and await the result.

    Example:
        # Sync usage
        call = MevShield(subtensor).submit_encrypted(
            commitment="0x1234...",
            ciphertext=b"encrypted_data..."
        )
        response = subtensor.sign_and_send_extrinsic(call=call, ...)

        # Async usage
        call = await MevShield(async_subtensor).submit_encrypted(
            commitment="0x1234...",
            ciphertext=b"encrypted_data..."
        )
        response = await async_subtensor.sign_and_send_extrinsic(call=call, ...)
    """

    def submit_encrypted(
        self,
        commitment: str,
        ciphertext: bytes,
    ) -> Call:
        """Returns GenericCall instance for MevShield function submit_encrypted.

        This function submits an encrypted extrinsic to the MEV Shield pallet. The extrinsic remains encrypted in the
        transaction pool until it is included in a block and decrypted by validators.

        Parameters:
            commitment: The blake2_256 hash of the payload_core (signer + nonce + SCALE(call)). Must be a hex string
                with "0x" prefix.
            ciphertext: The encrypted blob containing the payload and signature.
                Format: [u16 kem_len LE][kem_ct][nonce24][aead_ct]
                Maximum size: 8192 bytes.

        Returns:
            GenericCall instance ready for extrinsic submission.

        Note:
            The commitment is used to verify the ciphertext's content at decryption time. The ciphertext is encrypted
            using ML-KEM-768 + XChaCha20Poly1305 with the public key from the NextKey storage item, which rotates every
            block.
        """
        return self.create_composed_call(
            commitment=commitment,
            ciphertext=ciphertext,
        )
