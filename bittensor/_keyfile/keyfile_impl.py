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

class KeyFileError(Exception):
    """ Error thrown when the keyfile is corrupt, non-writable, nno-readable or the password used to decrypt is invalid.
    """

def serialized_keypair_to_keyfile_data( keypair: 'bittensor.Keypair' ):
    """ Serializes keypair object into keyfile data.
        Args:
            password ( str, required ):
                password to verify.
        Returns:
            valid ( bool ):
                True if the password meets validity requirements.
    """
    json_data = {
        'accountId': "0x" + keypair.public_key.hex() if keypair.public_key != None else None,
        'publicKey': "0x" + keypair.public_key.hex()  if keypair.public_key != None else None,
        'secretPhrase': keypair.mnemonic if keypair.mnemonic != None else None,
        'secretSeed': "0x" + \
            # If bytes -> str
            ( keypair.seed_hex if isinstance(keypair.seed_hex, str) else keypair.seed_hex.hex() ) 
                # If None -> None
                if keypair.seed_hex != None else None,
        'ss58Address': keypair.ss58_address if keypair.ss58_address != None else None
    }
    data = json.dumps( json_data ).encode()
    return data

def deserialize_keypair_from_keyfile_data( keyfile_data:bytes ) -> 'bittensor.Keypair':
    """ Deserializes Keypair object from passed keyfile data.
        Args:
            keyfile_data ( bytest, required ):
                Keyfile data as bytes to be loaded.
        Returns:
            keypair (bittensor.Keypair):
                Keypair loaded from bytes.
        Raises:
            KeyFileError:
                Raised if the passed bytest cannot construct a keypair object.
    """
    # Decode from json.
    keyfile_data = keyfile_data.decode()
    try:
        keyfile_dict = dict(json.loads( keyfile_data ))
    except:
        string_value = str(keyfile_data)
        if string_value[:2] == "0x":
            string_value = ss58_encode( string_value )
            keyfile_dict = {
                'accountId': None,
                'publicKey': None,
                'secretPhrase': None,
                'secretSeed': None,
                'ss58Address': string_value
            }
        else:
            raise KeyFileError('Keypair could not be created from keyfile data: {}'.format( string_value ))

    if "secretSeed" in keyfile_dict and keyfile_dict['secretSeed'] != None:
        return bittensor.Keypair.create_from_seed(keyfile_dict['secretSeed'])

    if "secretPhrase" in keyfile_dict and keyfile_dict['secretPhrase'] != None:
        return bittensor.Keypair.create_from_mnemonic(mnemonic=keyfile_dict['secretPhrase'])

    if "ss58Address" in keyfile_dict and keyfile_dict['ss58Address'] != None:
        return bittensor.Keypair( ss58_address = keyfile_dict['ss58Address'] )

    else:
        raise KeyFileError('Keypair could not be created from keyfile data: {}'.format( keyfile_dict ))

def validate_password( password:str ) -> bool:
    """ Validates the password again a password policy.
        Args:
            password ( str, required ):
                password to verify.
        Returns:
            valid ( bool ):
                True if the password meets validity requirements.
    """
    policy = PasswordPolicy.from_names(
        strength=0.20,
        entropybits=10,
        length=6,
    )
    if not password:
        return False
    tested_pass = policy.password(password)
    result = tested_pass.test()
    if len(result) > 0:
        print(colored('Password not strong enough. Try increasing the length of the password or the password complexity'))
        return False
    password_verification = getpass.getpass("Retype your password: ")
    if password != password_verification:
        print("Passwords do not match")
        return False
    return True

def ask_password_to_encrypt() -> str:
    """ Password from user prompt.
        Returns:
            password (str):
                Valid password from user prompt.
    """
    valid = False
    while not valid:
        password = getpass.getpass("Specify password for key encryption: ")
        valid = validate_password(password)
    return password

def keyfile_data_is_encrypted_ansible( keyfile_data:bytes ) -> bool:
    """ Returns true if the keyfile data is ansible encrypted.
        Args:
            keyfile_data ( bytes, required ):
                Bytes to validate
        Returns:
            is_ansible (bool):
                True if data is ansible encrypted.
    """
    return keyfile_data[:14] == b'$ANSIBLE_VAULT'

def keyfile_data_is_encrypted_legacy( keyfile_data:bytes ) -> bool:
    """ Returns true if the keyfile data is legacy encrypted.
        Args:
            keyfile_data ( bytes, required ):
                Bytes to validate
        Returns:
            is_legacy (bool):
                True if data is legacy encrypted.
    """
    return keyfile_data[:6] == b"gAAAAA"

def keyfile_data_is_encrypted( keyfile_data:bytes ) -> bool:
    """ Returns true if the keyfile data is encrypted.
        Args:
            keyfile_data ( bytes, required ):
                Bytes to validate
        Returns:
            is_encrypted (bool):
                True if data is encrypted.
    """
    return keyfile_data_is_encrypted_ansible( keyfile_data ) or keyfile_data_is_encrypted_legacy( keyfile_data )

def encrypt_keyfile_data ( keyfile_data:bytes, password: str = None ) -> bytes:
    """ Encrypts passed keyfile data using ansible vault.
        Args:
            keyfile_data ( bytes, required ):
                Bytes to validate
            password ( bool, optional ):
                It set, uses this password to encrypt data.
        Returns:
            encrytped_data (bytes):
                Ansible encrypted data.
    """
    password = ask_password_to_encrypt() if password == None else password
    console = bittensor.__console__;             
    with console.status(":locked_with_key: Encrypting key..."):
        vault = Vault( password )
    return vault.vault.encrypt ( keyfile_data )


def get_coldkey_password_from_environment(coldkey_name: str) -> Optional[str]:

    for env_var in os.environ:
        if (
            env_var.upper().startswith("BT_COLD_PW_")
            and env_var.upper().endswith(coldkey_name.upper())
        ):
            return os.getenv(env_var)

    return None


def decrypt_keyfile_data(keyfile_data: bytes, password: str = None, coldkey_name: Optional[str] = None) -> bytes:
    """ Decrypts passed keyfile data using ansible vault.
        Args:
            keyfile_data ( bytes, required ):
                Bytes to validate
            password ( bool, optional ):
                It set, uses this password to decrypt data.
        Returns:
            decrypted_data (bytes):
                Decrypted data.
         Raises:
            KeyFileError:
                Raised if the file is corrupted or if the password is incorrect.
    """
    if coldkey_name is not None and password is None:
        password = get_coldkey_password_from_environment(coldkey_name)

    try:
        password = getpass.getpass("Enter password to unlock key: ") if password is None else password
        console = bittensor.__console__;             
        with console.status(":key: Decrypting key..."):
            # Ansible decrypt.
            if keyfile_data_is_encrypted_ansible( keyfile_data ):
                vault = Vault( password )
                try:
                    decrypted_keyfile_data = vault.load( keyfile_data )
                except AnsibleVaultError:
                    raise KeyFileError('Invalid password')
            # Legacy decrypt.
            elif keyfile_data_is_encrypted_legacy( keyfile_data ):
                __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                cipher_suite = Fernet(key)
                decrypted_keyfile_data = cipher_suite.decrypt( keyfile_data )   
            # Unknown.
            else: 
                raise KeyFileError( "Keyfile data: {} is corrupt".format( keyfile_data ))

    except (InvalidSignature, InvalidKey, InvalidToken):
        raise KeyFileError('Invalid password')

    if not isinstance(decrypted_keyfile_data, bytes):
        decrypted_keyfile_data = json.dumps( decrypted_keyfile_data ).encode()
    return decrypted_keyfile_data

class Keyfile( object ):
    """ Defines an interface for a subtrate interface keypair stored on device.
    """
    def __init__( self, path: str ):
        self.path = os.path.expanduser(path)
        self.name = Path(self.path).parent.stem

    def __str__(self):
        if not self.exists_on_device():
            return "Keyfile (empty, {})>".format( self.path )
        if self.is_encrypted():
            return "Keyfile (encrypted, {})>".format( self.path )
        else:
            return "Keyfile (decrypted, {})>".format( self.path )

    def __repr__(self):
        return self.__str__()

    @property
    def keypair( self ) -> 'bittensor.Keypair':
        """ Returns the keypair from path, decrypts data if the file is encrypted.
            Args:
                password ( str, optional ):
                    Optional password used to decrypt file. If None, asks for user input.
            Returns:
                keypair (bittensor.Keypair):
                    Keypair stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        return self.get_keypair()

    @property
    def data( self ) -> bytes:
        """ Returns keyfile data under path.
            Returns:
                keyfile_data (bytes):   
                    Keyfile data stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    @property
    def keyfile_data( self ) -> bytes:
        """ Returns keyfile data under path.
            Returns:
                keyfile_data (bytes):   
                    Keyfile data stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        return self._read_keyfile_data_from_file()

    def set_keypair ( self, keypair: 'bittensor.Keypair', encrypt: bool = True, overwrite: bool = False, password:str = None):
        """ Writes the keypair to the file and optional encrypts data.
            Args:
                keypair (bittensor.Keypair):
                    Keypair to store under path.
                encrypt ( bool, optional, default = True ):
                    If True, encrypts file under path.
                overwrite ( bool, optional, default = True ):
                    If True, forces overwrite of current file.
                password ( str, optional ):
                    Optional password used to encrypt file. If None, asks for user input.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, or writable.
        """
        self.make_dirs()
        keyfile_data = serialized_keypair_to_keyfile_data( keypair )
        if encrypt:
            keyfile_data = encrypt_keyfile_data( keyfile_data, password )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = overwrite )

    def get_keypair(self, password: str = None) -> 'bittensor.Keypair':
        """ Returns the keypair from path, decrypts data if the file is encrypted.
            Args:
                password ( str, optional ):
                    Optional password used to decrypt file. If None, asks for user input.
            Returns:
                keypair (bittensor.Keypair):
                    Keypair stored under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = decrypt_keyfile_data(keyfile_data, password, coldkey_name=self.name)
        return deserialize_keypair_from_keyfile_data( keyfile_data )

    def make_dirs( self ):
        """ Makes directories for path.
        """
        directory = os.path.dirname( self.path )
        if not os.path.exists( directory ):
            os.makedirs( directory ) 

    def exists_on_device( self ) -> bool:
        """ Returns true if the file exists on the device.
            Returns:
                on_device (bool):
                    True if the file is on device.
        """
        if not os.path.isfile( self.path ):
            return False
        return True

    def is_readable( self ) -> bool:
        """ Returns true if the file under path is readable.
            Returns:
                readable (bool):
                    True if the file is readable.
        """
        if not self.exists_on_device():
            return False
        if not os.access( self.path , os.R_OK ):
            return False
        return True

    def is_writable( self ) -> bool:
        """ Returns true if the file under path is writable.
            Returns:
                writable (bool):
                    True if the file is writable.
        """
        if os.access(self.path, os.W_OK):
            return True
        return False

    def is_encrypted ( self ) -> bool:
        """ Returns true if the file under path is encrypted.
            Returns:
                encrypted (bool):
                    True if the file is encrypted.
        """
        if not self.exists_on_device():
            return False
        if not self.is_readable():
            return False
        return keyfile_data_is_encrypted( self._read_keyfile_data_from_file() )

    def _may_overwrite ( self ) -> bool:
        choice = input("File {} already exists. Overwrite ? (y/N) ".format( self.path ))
        return choice == 'y'

    def encrypt( self, password: str = None):
        """ Encrypts file under path.
            Args:
                password: (str, optional):
                    Optional password for encryption. Otherwise asks for user input.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable.
        """
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.path ))
        if not self.is_writable():
            raise KeyFileError( "Keyfile at: {} is not writeable".format( self.path ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if not keyfile_data_is_encrypted( keyfile_data ):
            as_keypair = deserialize_keypair_from_keyfile_data( keyfile_data )
            keyfile_data = serialized_keypair_to_keyfile_data( as_keypair )
            keyfile_data = encrypt_keyfile_data( keyfile_data, password )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def decrypt( self, password: str = None):
        """ Decrypts file under path.
            Args:
                password: (str, optional):
                    Optional password for decryption. Otherwise asks for user input.
            Raises:
                KeyFileError:
                    Raised if the file does not exists, is not readable, writable 
                    corrupted, or if the password is incorrect.
        """
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.path ))
        if not self.is_writable():
            raise KeyFileError( "No write access for {}".format( self.path ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = decrypt_keyfile_data(keyfile_data, password, coldkey_name=self.name)
        as_keypair = deserialize_keypair_from_keyfile_data( keyfile_data )
        keyfile_data = serialized_keypair_to_keyfile_data( as_keypair )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def _read_keyfile_data_from_file ( self ) -> bytes:
        """ Reads keyfile data from path.
            Returns:
                keyfile_data: (bytes, required):
                    Keyfile data sotred under path.
            Raises:
                KeyFileError:
                    Raised if the file does not exists or is not readable.
        """
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.path ))
        if not self.is_readable():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.path ))
        with open( self.path , 'rb') as file:
            data = file.read()
        return data

    def _write_keyfile_data_to_file ( self, keyfile_data:bytes, overwrite: bool = False ):
        """ Writes the keyfile data to path, if overwrite is true, forces operation without asking.
            Args:
                keyfile_data: (bytes, required):
                    Byte data to store under path.
                overwrite (bool, optional):
                    If True, overwrites data without asking for overwrite permissions from the user.
            Raises:
                KeyFileError:
                    Raised if the file is not writable or the user returns No to overwrite prompt.
        """
        # Check overwrite.
        if self.exists_on_device() and not overwrite:
            if not self._may_overwrite():
                raise KeyFileError( "Keyfile at: {} is not writeable".format( self.path ) ) 
        with open(self.path, "wb") as keyfile:
            keyfile.write( keyfile_data )
        # Set file permissions.
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)


class MockKeyfile( object ):
    """ Defines an interface to a mocked keyfile object (nothing is created on device) keypair is treated as non encrypted and the data is just the string version.
    """
    def __init__( self, path: str ):
        self.path = os.path.expanduser(path)
        self._mock_keypair = bittensor.Keypair.create_from_mnemonic( mnemonic = 'arrive produce someone view end scout bargain coil slight festival excess struggle' )
        self._mock_data = serialized_keypair_to_keyfile_data( self._mock_keypair )

    def __str__(self):
        if not self.exists_on_device():
            return "Keyfile (empty, {})>".format( self.path )
        if self.is_encrypted():
            return "Keyfile (encrypted, {})>".format( self.path )
        else:
            return "Keyfile (decrypted, {})>".format( self.path )

    def __repr__(self):
        return self.__str__()

    @property
    def keypair( self ) -> 'bittensor.Keypair':
        return self._mock_keypair

    @property
    def data( self ) -> bytes:
        return bytes(self._mock_data)

    @property
    def keyfile_data( self ) -> bytes:
        return bytes( self._mock_data) 

    def set_keypair ( self, keypair: 'bittensor.Keypair', encrypt: bool = True, overwrite: bool = False, password:str = None):
        self._mock_keypair = keypair
        self._mock_data = serialized_keypair_to_keyfile_data( self._mock_keypair )

    def get_keypair(self, password: str = None) -> 'bittensor.Keypair':
        return self._mock_keypair

    def make_dirs( self ):
        return

    def exists_on_device( self ) -> bool:
        return True

    def is_readable( self ) -> bool:
        return True

    def is_writable( self ) -> bool:
        return True

    def is_encrypted ( self ) -> bool:
        return False

    def encrypt( self, password: str = None):
        raise ValueError('Cannot encrypt a mock keyfile')

    def decrypt( self, password: str = None):
        return










        