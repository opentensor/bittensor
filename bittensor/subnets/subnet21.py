# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 philanthrope

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import json
import time
import base64
import Crypto
import typing
import pydantic
import binascii
import bittensor as bt

from Crypto.Cipher import AES
from Crypto.PublicKey import ECC
from nacl import pwhash, secret
from nacl.encoding import HexEncoder
from nacl.utils import EncryptedMessage

## Encrpytion setup ##

NACL_SALT = b"\x13q\x83\xdf\xf1Z\t\xbc\x9c\x90\xb5Q\x879\xe9\xb1"


def setup_CRS(curve="P-256"):
    """
    Generate a pair of random points to serve as a Common Reference String (CRS) for elliptic curve operations.

    The CRS is essential for various cryptographic protocols that rely on a shared reference
    between parties, typically for the purpose of ensuring consistent cryptographic operations.

    Parameters:
    - curve (str, optional): Name of the elliptic curve to use; defaults to "P-256".

    Returns:
    - tuple(ECC.EccPoint, ECC.EccPoint): A 2-tuple of ECC.EccPoint instances representing the base points (g, h).

    Raises:
    - ValueError: If the specified elliptic curve name is not recognized.
    """
    curve_obj = ECC.generate(curve=curve)
    g = curve_obj.pointQ  # Base point
    h = ECC.generate(curve=curve).pointQ  # Another random point
    return g, h


def ecc_point_to_hex(point):
    """
    Convert an elliptic curve point to a hexadecimal string.

    This encoding is typically used for compact representation or for preparing the data
    to be transmitted over protocols that may not support binary data.

    Parameters:
    - point (ECC.EccPoint): An ECC point to convert.

    Returns:
    - str: Hexadecimal string representing the elliptic curve point.

    Raises:
    - AttributeError: If the input is not a valid ECC point with accessible x and y coordinates.
    """
    point_str = "{},{}".format(point.x, point.y)
    return binascii.hexlify(point_str.encode()).decode()


def encrypt_aes(filename: typing.Union[bytes, str], key: bytes) -> bytes:
    """
    Encrypt the data in the given filename using AES-GCM.

    Parameters:
    - filename: str or bytes. If str, it's considered as a file name. If bytes, as the data itself.
    - key: bytes. 16-byte (128-bit), 24-byte (192-bit), or 32-byte (256-bit) secret key.

    Returns:
    - cipher_text: bytes. The encrypted data.
    - nonce: bytes. The nonce used for the GCM mode.
    - tag: bytes. The tag for authentication.
    """

    # If filename is a string, treat it as a file name and read the data
    if isinstance(filename, str):
        with open(filename, "rb") as file:
            data = file.read()
    else:
        data = filename

    # Initialize AES-GCM cipher
    cipher = AES.new(key, AES.MODE_GCM)

    # Encrypt the data
    cipher_text, tag = cipher.encrypt_and_digest(data)

    return cipher_text, cipher.nonce, tag


def decrypt_aes(cipher_text: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
    """
    Decrypt the data using AES-GCM.

    Parameters:
    - cipher_text: bytes. The encrypted data.
    - key: bytes. The secret key used for decryption.
    - nonce: bytes. The nonce used in the GCM mode for encryption.
    - tag: bytes. The tag for authentication.

    Returns:
    - data: bytes. The decrypted data.
    """

    # Initialize AES-GCM cipher with the given key and nonce
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

    # Decrypt the data and verify the tag
    try:
        data = cipher.decrypt_and_verify(cipher_text, tag)
    except ValueError:
        # This is raised if the tag does not match
        raise ValueError("Incorrect decryption key or corrupted data.")

    return data


def encrypt_data_with_wallet(data: bytes, wallet) -> bytes:
    """
    Encrypts the given data using a symmetric key derived from the wallet's coldkey public key.

    Args:
        data (bytes): Data to be encrypted.
        wallet (bt.wallet): Bittensor wallet object containing the coldkey.

    Returns:
        bytes: Encrypted data.

    This function generates a symmetric key using the public key of the wallet's coldkey.
    The generated key is used to encrypt the data using the NaCl secret box (XSalsa20-Poly1305).
    The function is intended for encrypting arbitrary data securely using wallet-based keys.
    """
    # Derive symmetric key from wallet's coldkey
    password = wallet.coldkey.private_key.hex()
    password_bytes = bytes(password, "utf-8")
    kdf = pwhash.argon2i.kdf
    key = kdf(
        secret.SecretBox.KEY_SIZE,
        password_bytes,
        NACL_SALT,
        opslimit=pwhash.argon2i.OPSLIMIT_SENSITIVE,
        memlimit=pwhash.argon2i.MEMLIMIT_SENSITIVE,
    )

    # Encrypt the data
    box = secret.SecretBox(key)
    encrypted = box.encrypt(data)
    return encrypted


def decrypt_data_with_coldkey_private_key(
    encrypted_data: bytes, private_key: typing.Union[str, bytes]
) -> bytes:
    """
    Decrypts the given encrypted data using a symmetric key derived from the wallet's coldkey public key.

    Args:
        encrypted_data (bytes): Data to be decrypted.
        private_key (bytes): The bittensor wallet private key (password) to decrypt the AES payload.

    Returns:
        bytes: Decrypted data.

    Similar to the encryption function, this function derives a symmetric key from the wallet's coldkey public key.
    It then uses this key to decrypt the given encrypted data. The function is primarily used for decrypting data
    that was previously encrypted by the `encrypt_data_with_wallet` function.
    """
    password_bytes = (
        bytes(private_key, "utf-8") if isinstance(private_key, str) else private_key
    )

    kdf = pwhash.argon2i.kdf
    key = kdf(
        secret.SecretBox.KEY_SIZE,
        password_bytes,
        NACL_SALT,
        opslimit=pwhash.argon2i.OPSLIMIT_SENSITIVE,
        memlimit=pwhash.argon2i.MEMLIMIT_SENSITIVE,
    )

    box = secret.SecretBox(key)
    decrypted = box.decrypt(encrypted_data)
    return decrypted


def decrypt_data_with_wallet(encrypted_data: bytes, wallet) -> bytes:
    """
    Decrypts the given encrypted data using a symmetric key derived from the wallet's coldkey public key.

    Args:
        encrypted_data (bytes): Data to be decrypted.
        wallet (bt.wallet): Bittensor wallet object containing the coldkey.

    Returns:
        bytes: Decrypted data.

    Similar to the encryption function, this function derives a symmetric key from the wallet's coldkey public key.
    It then uses this key to decrypt the given encrypted data. The function is primarily used for decrypting data
    that was previously encrypted by the `encrypt_data_with_wallet` function.
    """
    # Derive symmetric key from wallet's coldkey
    password = wallet.coldkey.private_key.hex()
    password_bytes = bytes(password, "utf-8")
    kdf = pwhash.argon2i.kdf
    key = kdf(
        secret.SecretBox.KEY_SIZE,
        password_bytes,
        NACL_SALT,
        opslimit=pwhash.argon2i.OPSLIMIT_SENSITIVE,
        memlimit=pwhash.argon2i.MEMLIMIT_SENSITIVE,
    )

    # Decrypt the data
    box = secret.SecretBox(key)
    decrypted = box.decrypt(encrypted_data)
    return decrypted


def encrypt_data_with_aes_and_serialize(
    data: bytes, wallet: "bittensor.wallet"
) -> typing.Tuple[bytes, bytes]:
    """
    Decrypts the given encrypted data using a symmetric key derived from the wallet's coldkey public key.

    Args:
        encrypted_data (bytes): Data to be decrypted.
        wallet (bt.wallet): Bittensor wallet object containing the coldkey.

    Returns:
        bytes: Decrypted data.

    Similar to the encryption function, this function derives a symmetric key from the wallet's coldkey public key.
    It then uses this key to decrypt the given encrypted data. The function is primarily used for decrypting data
    that was previously encrypted by the `encrypt_data_with_wallet` function.
    """
    # Generate a random AES key
    aes_key = Crypto.Random.get_random_bytes(32)  # AES key for 256-bit encryption

    # Create AES cipher
    cipher = AES.new(aes_key, AES.MODE_GCM)
    nonce = cipher.nonce

    # Encrypt the data
    encrypted_data, tag = cipher.encrypt_and_digest(data)

    # Serialize AES key, nonce, and tag
    aes_info = {
        "aes_key": aes_key.hex(),  # Convert bytes to hex string for serialization
        "nonce": nonce.hex(),
        "tag": tag.hex(),
    }
    aes_info_str = json.dumps(aes_info)

    encrypted_msg: EncryptedMessage = encrypt_data_with_wallet(
        aes_info_str.encode(), wallet
    )  # Encrypt the serialized JSON string

    return encrypted_data, serialize_nacl_encrypted_message(encrypted_msg)


encrypt_data = encrypt_data_with_aes_and_serialize


def serialize_nacl_encrypted_message(encrypted_message: EncryptedMessage) -> str:
    """
    Serializes an EncryptedMessage object to a JSON string.

    Args:
        encrypted_message (EncryptedMessage): The EncryptedMessage object to serialize.

    Returns:
        str: A JSON string representing the serialized object.

    This function takes an EncryptedMessage object, extracts its nonce and ciphertext,
    and encodes them into a hex format. It then constructs a dictionary with these
    values and serializes the dictionary into a JSON string.
    """
    data = {
        "nonce": HexEncoder.encode(encrypted_message.nonce).decode("utf-8"),
        "ciphertext": HexEncoder.encode(encrypted_message.ciphertext).decode("utf-8"),
    }
    return json.dumps(data)


## Storage protocol ##


# Basically setup for a given piece of data
class Store(bt.Synapse):
    # Data to store
    encrypted_data: str  # base64 encoded string of encrypted data (bytes)

    # Setup parameters
    curve: str  # e.g. P-256
    g: str  # base point   (hex string representation)
    h: str  # random point (hex string representation)

    seed: typing.Union[
        str, int, bytes
    ]  # random seed (bytes stored as hex) for the commitment

    # Return signature of received data
    randomness: typing.Optional[int] = None
    commitment: typing.Optional[str] = None
    signature: typing.Optional[bytes] = None
    commitment_hash: typing.Optional[str] = None  # includes seed

    required_hash_fields: typing.List[str] = pydantic.Field(
        [
            "curve",
            "g",
            "h",
            "seed",
            "randomness",
            "commitment",
            "signature",
            "commitment_hash",
        ],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    def __str__(self):
        return (
            f"Store(encrypted_data={self.encrypted_data[:12]}, "
            f"curve={self.curve}, "
            f"g={self.g}, "
            f"h={self.h}, "
            f"seed={str(self.seed)[:12]}, "
            f"randomness={str(self.randomness)[:12]}, "
            f"commitment={str(self.commitment)[:12]}, "
            f"commitment_hash={str(self.commitment_hash)[:12]})"
            f"axon={self.axon.dict()}, "
            f"dendrite={self.dendrite.dict()}"
        )


class StoreUser(bt.Synapse):
    # Data to store
    encrypted_data: str  # base64 encoded string of encrypted data (bytes)
    encryption_payload: str  # encrypted json serialized bytestring of encryption params

    data_hash: typing.Optional[str] = None  # Miner storage lookup key

    required_hash_fields: typing.List[str] = pydantic.Field(
        ["encrypted_data", "encryption_payload"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


class Retrieve(bt.Synapse):
    # Where to find the data
    data_hash: str  # Miner storage lookup key
    seed: str  # New random seed to hash the data with

    # Fetched data and proof
    data: typing.Optional[str] = None
    commitment_hash: typing.Optional[str] = None
    commitment_proof: typing.Optional[str] = None

    required_hash_fields: typing.List[str] = pydantic.Field(
        ["data", "data_hash", "seed", "commtiment_proof", "commitment_hash"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    def __str__(self):
        return (
            f"Retrieve(data_hash={str(self.data_hash[:12])}, "
            f"seed={str(self.seed[:12])}, "
            f"data={str(self.data[:12])}, "
            f"commitment_hash={str(self.commitment_hash[:12])}, "
            f"commitment_proof={str(self.commitment_proof[:12])})"
            f"axon={self.axon.dict()}, "
            f"dendrite={self.dendrite.dict()}"
        )


class RetrieveUser(bt.Synapse):
    # Where to find the data
    data_hash: str  # Miner storage lookup key

    # Fetched data to return along with AES payload in base64 encoding
    encrypted_data: typing.Optional[str] = None
    encryption_payload: typing.Optional[str] = None

    required_hash_fields: typing.List[str] = pydantic.Field(
        ["data_hash"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


def ensure_bytes(data: typing.Union[str, bytes]):
    """Ensure that the given data is in bytes."""
    if isinstance(data, str):
        return bytes(data, "utf-8")
    return data


def setup_synapse(protocol: str, *args, **kwargs):
    """Setup a synapse for a given piece of data."""
    if protocol.lower() == "store":
        return setup_store_synapse(*args, **kwargs)
    elif protocol.lower() == "retrieve":
        return setup_retrieve_synapse(*args, **kwargs)
    elif protocol.lower() == "storeuser":
        return setup_store_user_synapse(*args, **kwargs)
    elif protocol.lower() == "retrieveuser":
        return setup_retrieve_user_synapse(*args, **kwargs)
    else:
        raise ValueError(
            f"Function {protocol} not currently supported. Please open a PR if you are the subnet owner."
        )


def setup_store_synapse(
    data: bytes,
    wallet: "bittensor.wallet",
    curve: str = "P-256",
    noencrypt: bool = False,
) -> Store:
    """Setup a synapse for a given piece of data."""

    data = ensure_bytes(data)

    # Setup CRS for this round of validation
    g, h = setup_CRS(curve=curve)

    # Encrypt the data according to protocol specifications using bt wallet
    if not noencrypt:
        encrypted_data, encryption_payload = encrypt_data(
            data,
            wallet,
        )
    else:
        encrypted_data = data
        encryption_payload = "{}"

    # Encode to base64 for transport.
    b64_encrypted_data = base64.b64encode(encrypted_data).decode("utf-8")

    # Generate random seed
    seed = Crypto.Random.get_random_bytes(32)

    # Generate synapse
    return Store(
        encrypted_data=b64_encrypted_data,
        encryption_payload=encryption_payload,
        curve=curve,
        g=ecc_point_to_hex(g),
        h=ecc_point_to_hex(h),
        seed=seed.hex(),
    )


def setup_store_user_synapse(
    data: typing.Union[str, bytes],
    wallet: "bittensor.wallet",
    noencrypt: bool = False,
):
    """Setup a synapse for storing a given piece of user data."""
    # Ensure data is in bytes
    data = ensure_bytes(data)

    # Unlock the wallet and encrypt the data
    if not noencrypt:
        encrypted_data, encryption_payload = encrypt_data(
            data,
            wallet,
        )
    else:
        encrypted_data = data
        encryption_payload = "{}"

    encoded_data = base64.b64encode(encrypted_data)
    return StoreUser(
        encrypted_data=encoded_data,
        encryption_payload=encryption_payload,
    )


def setup_retrieve_synapse(
    data_hash: str,
) -> Retrieve:
    """Setup a synapse for a given piece of data."""
    return Retrieve(
        data_hash=data_hash,
        seed=Crypto.Random.get_random_bytes(32).hex(),
    )


def setup_retrieve_user_synapse(
    data_hash: str,
):
    """Setup a synapse for a given piece of data."""
    return RetrieveUser(data_hash=data_hash)
