from dataclasses import dataclass

from scalecodec import GenericCall

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

    def announce_next_key(
        self,
        public_key: bytes,
        epoch: int,
    ) -> Call:
        """Returns GenericCall instance for MevShield function announce_next_key.

        This function allows Aura validators to announce the next ML-KEM-768 public key that will be used for encryption
        in the next block. The key will be rotated from NextKey to CurrentKey at the beginning of the next block.

        Parameters:
            public_key: The ML-KEM-768 public key as bytes. Must be exactly 1184 bytes for ML-KEM-768.
            epoch: The epoch number associated with this key.

        Returns:
            GenericCall instance ready for extrinsic submission.

        Note:
            Only current Aura validators can call this function. The public_key length must be exactly 1184 bytes
            (MAX_KYBER768_PK_LENGTH). This is enforced by the pallet.
        """
        return self.create_composed_call(
            public_key=public_key,
            epoch=epoch,
        )

    def execute_revealed(
        self,
        id: str,
        signer: str,
        nonce: int,
        call: "GenericCall",
        signature: bytes,
    ) -> Call:
        """Returns GenericCall instance for MevShield function execute_revealed.

        This function is executed by the block author (validator) to execute a decrypted extrinsic. It verifies the
        commitment, signature, and nonce before dispatching the inner call.

        Parameters:
            id: The submission ID (hash of signer + commitment + ciphertext). Must be a hex string with "0x" prefix.
            signer: The SS58 address of the account that originally submitted the encrypted extrinsic.
            nonce: The nonce that was used when creating the payload_core.
            call: The GenericCall object to execute (the decrypted call).
            signature: The signature bytes (MultiSignature format) that was included in the encrypted payload.

        Returns:
            GenericCall instance ready for extrinsic submission.

        Note:
            This is an unsigned extrinsic that can only be called by the block author. It verifies:
            1. Commitment matches the stored submission
            2. Signature is valid over "mev-shield:v1" + genesis_hash + payload_core
            3. Nonce matches the current account nonce
            4. Then dispatches the inner call from the signer
        """
        return self.create_composed_call(
            id=id,
            signer=signer,
            nonce=nonce,
            call=call,
            signature=signature,
        )

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
