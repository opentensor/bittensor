from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes

import bittensor.bittensor_pb2 as bittensor_pb2


class Crypto():

    @staticmethod
    def digest(synapse: bittensor_pb2.Synapse) -> bytes:
        """ Returns the synapse Sha256 hash digest of the Synapse contents.

        Args:
            synapse (bittensor_pb2.Synapse): Synapse to hash.

        Returns:
            bytes: Sha256 Hash digest.
        """
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(synapse.neuron_key.encode('utf-8'))
        digest.update(synapse.synapse_key.encode('utf-8'))
        digest.update(synapse.address.encode('utf-8'))
        digest.update(str(synapse.port).encode('utf-8'))
        digest.update(synapse.block_hash.encode('utf-8'))
        digest.update(bytes(synapse.nounce))
        digest = digest.finalize()
        return digest

    @staticmethod
    def generate_private_ed25519() -> Ed25519PrivateKey:
        """ Creates and returns a new Ed25519PrivateKey

        Returns:
            Ed25519PrivateKey: Private key of type Ed25519privateKey.
        """
        return Ed25519PrivateKey.generate()

    @staticmethod
    def public_key_from_private(
            private_key: Ed25519PrivateKey) -> Ed25519PublicKey:
        """ Returns the corresponding Ed25519PublicKey for a Ed25519PrivateKey

        Args:
            private_key (Ed25519PrivateKey): Private key.

        Returns:
            Ed25519PublicKey: Public key.
        """
        return private_key.public_key()

    @staticmethod
    def public_key_to_bytes(public_key: Ed25519PublicKey) -> bytes:
        """Returns the raw hex encoded bytes of a Ed25519PublicKey.

        Args:
            public_key (Ed25519PublicKey): Public key to encode.

        Returns:
            bytes: Byte encoding.
        """
        return public_key.public_bytes(serialization.Encoding.Raw,
                                       serialization.PublicFormat.Raw)

    @staticmethod
    def public_key_to_string(public_key: Ed25519PublicKey) -> str:
        """Returns the string encoding for a Ed25519PublicKey

        Args:
            public_key (Ed25519PublicKey): Key to encode as string.

        Returns:
            str: String encoding.
        """
        return '0x%s' % Crypto.public_key_to_bytes(public_key).hex()
