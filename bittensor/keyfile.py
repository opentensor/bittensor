# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import base64
import json
import stat
import getpass
import bittensor
from typing import Optional
from pathlib import Path

from ansible_vault import Vault
from ansible.parsing.vault import AnsibleVaultError
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from password_strength import PasswordPolicy
from substrateinterface.utils.ss58 import ss58_encode
from termcolor import colored


def serialized_keypair_to_keyfile_data(keypair: "bittensor.Keypair") -> bytes:
    """Serializes keypair object into keyfile data.
    Args:
        keypair (bittensor.Keypair): The keypair object to be serialized.
    Returns:
        data (bytes): Serialized keypair data.
    """
    json_data = {
        "accountId": "0x" + keypair.public_key.hex() if keypair.public_key else None,
        "publicKey": "0x" + keypair.public_key.hex() if keypair.public_key else None,
        "secretPhrase": keypair.mnemonic if keypair.mnemonic else None,
        "secretSeed": "0x"
        + (
            keypair.seed_hex
            if isinstance(keypair.seed_hex, str)
            else keypair.seed_hex.hex()
        )
        if keypair.seed_hex
        else None,
        "ss58Address": keypair.ss58_address if keypair.ss58_address else None,
    }
    data = json.dumps(json_data).encode()
    return data


def deserialize_keypair_from_keyfile_data(keyfile_data: bytes) -> "bittensor.Keypair":
    """Deserializes Keypair object from passed keyfile data.
    Args:
        keyfile_data (bytes): The keyfile data as bytes to be loaded.
    Returns:
        keypair (bittensor.Keypair): The Keypair loaded from bytes.
    Raises:
        KeyFileError: Raised if the passed bytes cannot construct a keypair object.
    """
    keyfile_data = keyfile_data.decode()
    try:
        keyfile_dict = dict(json.loads(keyfile_data))
    except:
        string_value = str(keyfile_data)
        if string_value[:2] == "0x":
            string_value = ss58_encode(string_value)
            keyfile_dict = {
                "accountId": None,
                "publicKey": None,
                "secretPhrase": None,
                "secretSeed": None,
                "ss58Address": string_value,
            }
        else:
            raise bittensor.KeyFileError(
                "Keypair could not be created from keyfile data: {}".format(
                    string_value
                )
            )

    if "secretSeed" in keyfile_dict and keyfile_dict["secretSeed"] is not None:
        return bittensor.Keypair.create_from_seed(keyfile_dict["secretSeed"])

    if "secretPhrase" in keyfile_dict and keyfile_dict["secretPhrase"] is not None:
        return bittensor.Keypair.create_from_mnemonic(
            mnemonic=keyfile_dict["secretPhrase"]
        )

    if "ss58Address" in keyfile_dict and keyfile_dict["ss58Address"] is not None:
        return bittensor.Keypair(ss58_address=keyfile_dict["ss58Address"])

    else:
        raise bittensor.KeyFileError(
            "Keypair could not be created from keyfile data: {}".format(keyfile_dict)
        )


def validate_password(password: str) -> bool:
    """Validates the password against a password policy.
    Args:
        password (str): The password to verify.
    Returns:
        valid (bool): True if the password meets validity requirements.
    """
    policy = PasswordPolicy.from_names(strength=0.20, entropybits=10, length=6)
    if not password:
        return False
    tested_pass = policy.password(password)
    result = tested_pass.test()
    if len(result) > 0:
        print(
            colored(
                "Password not strong enough. Try increasing the length of the password or the password complexity"
            )
        )
        return False
    password_verification = getpass.getpass("Retype your password: ")
    if password != password_verification:
        print("Passwords do not match")
        return False
    return True


def ask_password_to_encrypt() -> str:
    """Prompts the user to enter a password for key encryption.
    Returns:
        password (str): The valid password entered by the user.
    """
    valid = False
    while not valid:
        password = getpass.getpass("Specify password for key encryption: ")
        valid = validate_password(password)
    return password


def keyfile_data_is_encrypted_ansible(keyfile_data: bytes) -> bool:
    """Returns true if the keyfile data is ansible encrypted.
    Args:
        keyfile_data (bytes): The bytes to validate.
    Returns:
        is_ansible (bool): True if the data is ansible encrypted.
    """
    return keyfile_data[:14] == b"$ANSIBLE_VAULT"


def keyfile_data_is_encrypted_legacy(keyfile_data: bytes) -> bool:
    """Returns true if the keyfile data is legacy encrypted.
    Args:
        keyfile_data (bytes): The bytes to validate.
    Returns:
        is_legacy (bool): True if the data is legacy encrypted.
    """
    return keyfile_data[:6] == b"gAAAAA"


def keyfile_data_is_encrypted(keyfile_data: bytes) -> bool:
    """Returns true if the keyfile data is encrypted.
    Args:
        keyfile_data (bytes): The bytes to validate.
    Returns:
        is_encrypted (bool): True if the data is encrypted.
    """
    return keyfile_data_is_encrypted_ansible(
        keyfile_data
    ) or keyfile_data_is_encrypted_legacy(keyfile_data)


def encrypt_keyfile_data(keyfile_data: bytes, password: str = None) -> bytes:
    """Encrypts the passed keyfile data using ansible vault.
    Args:
        keyfile_data (bytes): The bytes to encrypt.
        password (str, optional): The password used to encrypt the data. If None, asks for user input.
    Returns:
        encrypted_data (bytes): The encrypted data.
    """
    password = ask_password_to_encrypt() if password is None else password
    console = bittensor.__console__
    with console.status(":locked_with_key: Encrypting key..."):
        vault = Vault(password)
    return vault.vault.encrypt(keyfile_data)


def get_coldkey_password_from_environment(coldkey_name: str) -> Optional[str]:
    """Retrieves the cold key password from the environment variables.
    Args:
        coldkey_name (str): The name of the cold key.
    Returns:
        password (str): The password retrieved from the environment variables, or None if not found.
    """
    for env_var in os.environ:
        if env_var.upper().startswith("BT_COLD_PW_") and env_var.upper().endswith(
            coldkey_name.upper()
        ):
            return os.getenv(env_var)

    return None


def decrypt_keyfile_data(
    keyfile_data: bytes, password: str = None, coldkey_name: Optional[str] = None
) -> bytes:
    """Decrypts the passed keyfile data using ansible vault.
    Args:
        keyfile_data (bytes): The bytes to decrypt.
        password (str, optional): The password used to decrypt the data. If None, asks for user input.
        coldkey_name (str, optional): The name of the cold key. If provided, retrieves the password from environment variables.
    Returns:
        decrypted_data (bytes): The decrypted data.
    Raises:
        KeyFileError: Raised if the file is corrupted or if the password is incorrect.
    """
    if coldkey_name is not None and password is None:
        password = get_coldkey_password_from_environment(coldkey_name)

    try:
        password = (
            getpass.getpass("Enter password to unlock key: ")
            if password is None
            else password
        )
        console = bittensor.__console__
        with console.status(":key: Decrypting key..."):
            # Ansible decrypt.
            if keyfile_data_is_encrypted_ansible(keyfile_data):
                vault = Vault(password)
                try:
                    decrypted_keyfile_data = vault.load(keyfile_data)
                except AnsibleVaultError:
                    raise bittensor.KeyFileError("Invalid password")
            # Legacy decrypt.
            elif keyfile_data_is_encrypted_legacy(keyfile_data):
                __SALT = (
                    b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"
                )
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    salt=__SALT,
                    length=32,
                    iterations=10000000,
                    backend=default_backend(),
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                cipher_suite = Fernet(key)
                decrypted_keyfile_data = cipher_suite.decrypt(keyfile_data)
            # Unknown.
            else:
                raise bittensor.KeyFileError(
                    "keyfile data: {} is corrupt".format(keyfile_data)
                )

    except (InvalidSignature, InvalidKey, InvalidToken):
        raise bittensor.KeyFileError("Invalid password")

    if not isinstance(decrypted_keyfile_data, bytes):
        decrypted_keyfile_data = json.dumps(decrypted_keyfile_data).encode()
    return decrypted_keyfile_data


class keyfile:
    """Defines an interface for a substrate interface keypair stored on device."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        self.name = Path(self.path).parent.stem

    def __str__(self):
        if not self.exists_on_device():
            return "keyfile (empty, {})>".format(self.path)
        if self.is_encrypted():
            return "keyfile (encrypted, {})>".format(self.path)
        else:
            return "keyfile (decrypted, {})>".format(self.path)

    def __repr__(self):
        return self.__str__()

    @property
    def keypair(self) -> "bittensor.Keypair":
        """Returns the keypair from path, decrypts data if the file is encrypted.
        Returns:
            keypair (bittensor.Keypair): The keypair stored under the path.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, writable, corrupted, or if the password is incorrect.
        """
        return self.get_keypair()

    @property
    def data(self) -> bytes:
        """Returns the keyfile data under path.
        Returns:
            keyfile_data (bytes): The keyfile data stored under the path.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    @property
    def keyfile_data(self) -> bytes:
        """Returns the keyfile data under path.
        Returns:
            keyfile_data (bytes): The keyfile data stored under the path.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    def set_keypair(
        self,
        keypair: "bittensor.Keypair",
        encrypt: bool = True,
        overwrite: bool = False,
        password: str = None,
    ):
        """Writes the keypair to the file and optionally encrypts data.
        Args:
            keypair (bittensor.Keypair): The keypair to store under the path.
            encrypt (bool, optional): If True, encrypts the file under the path. Default is True.
            overwrite (bool, optional): If True, forces overwrite of the current file. Default is False.
            password (str, optional): The password used to encrypt the file. If None, asks for user input.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, writable, or if the password is incorrect.
        """
        self.make_dirs()
        keyfile_data = serialized_keypair_to_keyfile_data(keypair)
        if encrypt:
            keyfile_data = encrypt_keyfile_data(keyfile_data, password)
        self._write_keyfile_data_to_file(keyfile_data, overwrite=overwrite)

    def get_keypair(self, password: str = None) -> "bittensor.Keypair":
        """Returns the keypair from the path, decrypts data if the file is encrypted.
        Args:
            password (str, optional): The password used to decrypt the file. If None, asks for user input.
        Returns:
            keypair (bittensor.Keypair): The keypair stored under the path.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, writable, corrupted, or if the password is incorrect.
        """
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted(keyfile_data):
            keyfile_data = decrypt_keyfile_data(
                keyfile_data, password, coldkey_name=self.name
            )
        return deserialize_keypair_from_keyfile_data(keyfile_data)

    def make_dirs(self):
        """Creates directories for the path if they do not exist."""
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def exists_on_device(self) -> bool:
        """Returns True if the file exists on the device.
        Returns:
            on_device (bool): True if the file is on the device.
        """
        if not os.path.isfile(self.path):
            return False
        return True

    def is_readable(self) -> bool:
        """Returns True if the file under path is readable.
        Returns:
            readable (bool): True if the file is readable.
        """
        if not self.exists_on_device():
            return False
        if not os.access(self.path, os.R_OK):
            return False
        return True

    def is_writable(self) -> bool:
        """Returns True if the file under path is writable.
        Returns:
            writable (bool): True if the file is writable.
        """
        if os.access(self.path, os.W_OK):
            return True
        return False

    def is_encrypted(self) -> bool:
        """Returns True if the file under path is encrypted.
        Returns:
            encrypted (bool): True if the file is encrypted.
        """
        if not self.exists_on_device():
            return False
        if not self.is_readable():
            return False
        return keyfile_data_is_encrypted(self._read_keyfile_data_from_file())

    def _may_overwrite(self) -> bool:
        """Asks the user if it's okay to overwrite the file.
        Returns:
            may_overwrite (bool): True if the user allows overwriting the file.
        """
        choice = input("File {} already exists. Overwrite? (y/N) ".format(self.path))
        return choice == "y"

    def encrypt(self, password: str = None):
        """Encrypts the file under the path.
        Args:
            password (str, optional): The password for encryption. If None, asks for user input.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, or writable.
        """
        if not self.exists_on_device():
            raise bittensor.KeyFileError(
                "Keyfile at: {} does not exist".format(self.path)
            )
        if not self.is_readable():
            raise bittensor.KeyFileError(
                "Keyfile at: {} is not readable".format(self.path)
            )
        if not self.is_writable():
            raise bittensor.KeyFileError(
                "Keyfile at: {} is not writable".format(self.path)
            )
        keyfile_data = self._read_keyfile_data_from_file()
        if not keyfile_data_is_encrypted(keyfile_data):
            as_keypair = deserialize_keypair_from_keyfile_data(keyfile_data)
            keyfile_data = serialized_keypair_to_keyfile_data(as_keypair)
            keyfile_data = encrypt_keyfile_data(keyfile_data, password)
        self._write_keyfile_data_to_file(keyfile_data, overwrite=True)

    def decrypt(self, password: str = None):
        """Decrypts the file under the path.
        Args:
            password (str, optional): The password for decryption. If None, asks for user input.
        Raises:
            KeyFileError: Raised if the file does not exist, is not readable, writable, corrupted, or if the password is incorrect.
        """
        if not self.exists_on_device():
            raise bittensor.KeyFileError(
                "Keyfile at: {} does not exist".format(self.path)
            )
        if not self.is_readable():
            raise bittensor.KeyFileError(
                "Keyfile at: {} is not readable".format(self.path)
            )
        if not self.is_writable():
            raise bittensor.KeyFileError(
                "Keyfile at: {} is not writable".format(self.path)
            )
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted(keyfile_data):
            keyfile_data = decrypt_keyfile_data(
                keyfile_data, password, coldkey_name=self.name
            )
        as_keypair = deserialize_keypair_from_keyfile_data(keyfile_data)
        keyfile_data = serialized_keypair_to_keyfile_data(as_keypair)
        self._write_keyfile_data_to_file(keyfile_data, overwrite=True)

    def _read_keyfile_data_from_file(self) -> bytes:
        """Reads the keyfile data from the file.
        Returns:
            keyfile_data (bytes): The keyfile data stored under the path.
        Raises:
            KeyFileError: Raised if the file does not exist or is not readable.
        """
        if not self.exists_on_device():
            raise bittensor.KeyFileError(
                "Keyfile at: {} does not exist".format(self.path)
            )
        if not self.is_readable():
            raise bittensor.KeyFileError(
                "Keyfile at: {} is not readable".format(self.path)
            )
        with open(self.path, "rb") as file:
            data = file.read()
        return data

    def _write_keyfile_data_to_file(self, keyfile_data: bytes, overwrite: bool = False):
        """Writes the keyfile data to the file.
        Args:
            keyfile_data (bytes): The byte data to store under the path.
            overwrite (bool, optional): If True, overwrites the data without asking for permission from the user. Default is False.
        Raises:
            KeyFileError: Raised if the file is not writable or the user responds No to the overwrite prompt.
        """
        # Check overwrite.
        if self.exists_on_device() and not overwrite:
            if not self._may_overwrite():
                raise bittensor.KeyFileError(
                    "Keyfile at: {} is not writable".format(self.path)
                )
        with open(self.path, "wb") as keyfile:
            keyfile.write(keyfile_data)
        # Set file permissions.
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)


class Mockkeyfile:
    """
    The Mockkeyfile is a mock object representing a keyfile that does not exist on the device.
    It is designed for use in testing scenarios and simulations where actual filesystem operations are not required.
    The keypair stored in the Mockkeyfile is treated as non-encrypted and the data is stored as a serialized string.
    """

    def __init__(self, path: str):
        """
        Initializes a Mockkeyfile object.

        Args:
            path (str): The path of the mock keyfile.
        """
        self.path = path
        self._mock_keypair = None
        self._mock_data = None

    def __str__(self):
        """
        Returns a string representation of the Mockkeyfile. The representation will indicate if the
        keyfile is empty, encrypted, or decrypted.

        Returns:
            str: The string representation of the Mockkeyfile.
        """
        return f"Mockkeyfile({self.path})"

    def __repr__(self):
        """
        Returns a string representation of the Mockkeyfile, same as __str__().

        Returns:
            str: The string representation of the Mockkeyfile.
        """
        return self.__str__()

    @property
    def keypair(self):
        """
        Returns the mock keypair stored in the keyfile.

        Returns:
            bittensor.Keypair: The mock keypair.
        """
        return self._mock_keypair

    @property
    def data(self):
        """
        Returns the serialized keypair data stored in the keyfile.

        Returns:
            bytes: The serialized keypair data.
        """
        return self._mock_data

    def set_keypair(self, keypair, encrypt=True, overwrite=False, password=None):
        """
        Sets the mock keypair in the keyfile. The `encrypt` and `overwrite` parameters are ignored.

        Args:
            keypair (bittensor.Keypair): The mock keypair to be set.
            encrypt (bool, optional): Ignored in this context. Defaults to True.
            overwrite (bool, optional): Ignored in this context. Defaults to False.
            password (str, optional): Ignored in this context. Defaults to None.
        """
        self._mock_keypair = keypair
        self._mock_data = None  # You may need to serialize the keypair here

    def get_keypair(self, password=None):
        """
        Returns the mock keypair stored in the keyfile. The `password` parameter is ignored.

        Args:
            password (str, optional): Ignored in this context. Defaults to None.

        Returns:
            bittensor.Keypair: The mock keypair stored in the keyfile.
        """
        return self._mock_keypair

    def make_dirs(self):
        """
        Creates the directories for the mock keyfile. Does nothing in this class,
        since no actual filesystem operations are needed.
        """
        pass

    def exists_on_device(self):
        """
        Returns True indicating that the mock keyfile exists on the device (although
        it is not created on the actual file system).

        Returns:
            bool: Always returns True for Mockkeyfile.
        """
        return True

    def is_readable(self):
        """
        Returns True indicating that the mock keyfile is readable (although it is not
        read from the actual file system).

        Returns:
            bool: Always returns True for Mockkeyfile.
        """
        return True

    def is_writable(self):
        """
        Returns True indicating that the mock keyfile is writable (although it is not
        written to the actual file system).

        Returns:
            bool: Always returns True for Mockkeyfile.
        """
        return True

    def is_encrypted(self):
        """
        Returns False indicating that the mock keyfile is not encrypted.

        Returns:
            bool: Always returns False for Mockkeyfile.
        """
        return False

    def encrypt(self, password=None):
        """
        Raises a ValueError since encryption is not supported for the mock keyfile.

        Args:
            password (str, optional): Ignored in this context. Defaults to None.

        Raises:
            ValueError: Always raises this exception for Mockkeyfile.
        """
        raise ValueError("Cannot encrypt a Mockkeyfile")

    def decrypt(self, password=None):
        """
        Returns without doing anything since the mock keyfile is not encrypted.

        Args:
            password (str, optional): Ignored in this context. Defaults to None.
        """
        pass
