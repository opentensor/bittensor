from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
import requests
import binascii

import bittensor.bittensor_pb2 as bittensor_pb2


class Crypto():

    @staticmethod
    def count_zeros(to_count: bytes) -> int:
        """ Counts the number of prepended zeros from a bytes string encoded as hex.

        Args:
            to_count (bytes): Bytes to count zeros

        Returns:
            int: Number of prepending zeros.
        """
        hex_string = binascii.hexlify(to_count).decode("ascii")
        return len(hex_string) - len(hex_string.lstrip('0'))

    @staticmethod
    def lastest_block_hash() -> str:
        """ Returns the string representation of the hash of latest bitcoin block.

        Returns:
            str: Latest bitcoin block hash.
        """
        url = 'https://blockchain.info/latestblock'
        last_block_hash = requests.get(url).json()['hash']
        return last_block_hash

    @staticmethod
    def check_signature(synapse: bittensor_pb2.Synapse) -> bool:
        """Checks that the synapse is properly signed by the neuron.

        Args:
            synapse (bittensor_pb2.Synapse): Synapse to check.

        Returns:
            bool: True is signed properly otherwise false
        """
        digest = Crypto.digest(synapse)
        public_key = Crypto.public_key_from_string(synapse.neuron_key)
        try:
            public_key.verify(synapse.signature, digest)
        except:
            return False
        return True

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
        digest.update(synapse.port.encode('utf-8'))
        digest.update(synapse.block_hash.encode('utf-8'))
        digest.update(bytes(synapse.nounce))
        digest = digest.finalize()
        return digest

    @staticmethod
    def difficulty(digest: bytes) -> int:
        hex_string = digest.hex()
        return len(hex_string) - len(hex_string.lstrip('0'))

    @staticmethod
    def fill_proof_of_work(synapse: bittensor_pb2.Synapse,
                           difficulty: int) -> bittensor_pb2.Synapse:
        """ Fills the synapse.proof_of_work with a pow digest with difficulty.

        Args:
            synapse (bittensor_pb2.Synapse): Synapse to fill.
            difficulty (int): Sha256 POW difficulty.

        Returns:
            bittensor_pb2.Synapse: Synapse with filled proof of work and nounce.
        """
        nounce = 0
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(synapse.neuron_key.encode('utf-8'))
        digest.update(synapse.synapse_key.encode('utf-8'))
        digest.update(synapse.address.encode('utf-8'))
        digest.update(synapse.port.encode('utf-8'))
        digest.update(synapse.block_hash.encode('utf-8'))
        synapse_digest = digest.finalize()
        while True:
            proof_of_work = hashes.Hash(hashes.SHA256(),
                                        backend=default_backend())
            proof_of_work.update(synapse_digest)
            proof_of_work.update(bytes(nounce))
            proof_of_work = proof_of_work.finalize()
            if Crypto.count_zeros(proof_of_work) >= difficulty:
                break
            else:
                nounce = nounce + 1
        # fill digest.
        synapse.nounce = nounce
        synapse.proof_of_work = proof_of_work
        return synapse

    @staticmethod
    def sign_synapse(private_key: Ed25519PrivateKey,
                     synapse: bittensor_pb2.Synapse) -> bittensor_pb2.Synapse:
        """ Signs the passed bittensor_pb2.Synapse contents and appends the signature.

        Args:
            private_key (Ed25519PrivateKey): Private key to use for signature.
            synapse (bittensor_pb2.Synapse): Synapse to sign.

        Returns:
            bittensor_pb2.Synapse: Synapse with signature.
        """
        digest = Crypto.digest(synapse)
        signature = private_key.sign(digest)  # to create.
        synapse.signature = signature
        return synapse

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

    @staticmethod
    def public_bytes_from_string(public_string: str) -> bytes:
        """Returns the byte encoding from a ed25519 public key as a string. 

        Args:
            public_string (str): Ed25519 public key string.

        Returns:
            bytes: Bytes encoding.
        """
        result = bytes.fromhex(public_string[2:])
        return result

    @staticmethod
    def public_key_from_bytes(public_bytes: bytes) -> Ed25519PublicKey:
        """ Returns the Ed25519PublicKey from the passed bytes.
        Args:
            public_bytes (bytes): ed25519 public key as bytes. 

        Returns:
            Ed25519PublicKey: Public key object.
        """
        result = Identity.public_bytes_from_string(string)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(result)
        return public_key

    @staticmethod
    def public_key_from_string(public_string: str) -> Ed25519PublicKey:
        """ Returns the Ed25519PublicKey from passed string.

        Args:
            public_string (str): ed25519 string encoded.

        Returns:
            Ed25519PrivateKey: Public key object.
        """
        result = Crypto.public_bytes_from_string(public_string)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(result)
        return public_key

    @staticmethod
    def sign_bytes(private_key: Ed25519PrivateKey, digest: bytes) -> bytes:
        """ Signs bytest with passed Ed25519PrivateKey.

        Args:
            private_key (Ed25519PrivateKey): Signing Key.
            digest (bytes): Bytes to sign.

        Returns:
            bytes: Signature.
        """
        return private_key.sign(digest)

    @staticmethod
    def verify_synapse(synapse: bittensor_pb2.Synapse) -> bool:
        """ Verifies the synapse contents

        Args:
            synapse (bittensor_pb2.Synapse): Synapse to verify

        Returns:
            bool: True is synapse is properly signed.
        """
        public_key = Crypto.public_key_from_string(synapse.neuron_key)
        digest = Crypto.digest(synapse)
        return Crypto.verify(public_key, synapse.signature, digest)

    @staticmethod
    def verify(public_key: Ed25519PublicKey, signature: bytes,
               data: bytes) -> bool:
        """ Verifies that the passed public key was used to create the signature for passed data.

        Args:
            public_key (Ed25519PublicKey): Public key to use for verification. 
            signature (bytes): Signature to verify
            data (bytes): Signed data. 

        Returns:
            bool: True if signature is correct.
        """
        try:
            public_key.verify(signature, data)
        except:
            return False
        return True
