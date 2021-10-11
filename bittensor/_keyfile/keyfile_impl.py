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

from ansible_vault import Vault
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from password_strength import PasswordPolicy
from substrateinterface.utils.ss58 import ss58_encode
from termcolor import colored
from substrateinterface import Keypair

class CryptoKeyError(Exception):
    """ Exception for invalid signature, key, token, password, etc 
        Overwrite the built-in CryptoKeyError
    """

class KeyFileError(Exception):
    """ Overwrite the built-in CryptoKeyError
    """

def load_keypair_from_data( keyfile_data:bytes ) -> 'bittensor.Keypair':

    try:
        keyfile_dict = json.loads( keyfile_data.decode() )
    except json.decoder.JSONDecodeError:
        # This is a legacy coldkey pub.
        string_value = str(keyfile_data.decode())
        if string_value[:2] == "0x":
            string_value = ss58_encode( string_value )
        keyfile_dict = {
            'accountId': None,
            'publicKey': None,
            'secretPhrase': None,
            'secretSeed': None,
            'ss58Address': string_value
        }

    if "secretSeed" in keyfile_dict and keyfile_dict['secretSeed'] != None:
        return Keypair.create_from_seed(keyfile_dict['secretSeed'])

    if "secretPhrase" in keyfile_dict and keyfile_dict['secretPhrase'] != None:
        return Keypair.create_from_mnemonic(mnemonic=keyfile_dict['secretPhrase'])

    if "ss58Address" in keyfile_dict and keyfile_dict['ss58Address'] != None:
        return Keypair( ss58_address = keyfile_dict['ss58Address'] )

    else:
        raise CryptoKeyError('Keypair could not be created from keyfile data: {}'.format( keyfile_dict ))

def validate_password( password ):
    """ The policy to validate the strength of password
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

def keypair_to_keyfile_data( keypair ):
    """ Convert the keypair to dictionary with accountId, publicKey, secretPhrase, secretSeed, and ss58Address  
    """
    json_data = {
        'accountId': keypair.public_key if keypair.public_key != None else None,
        'publicKey': keypair.public_key if keypair.public_key != None else None,
        'secretPhrase': keypair.mnemonic if keypair.mnemonic != None else None,
        'secretSeed': "0x" + keypair.seed_hex if keypair.seed_hex != None else None,
        'ss58Address': keypair.ss58_address if keypair.ss58_address != None else None
    }
    return json.dumps( json_data ).encode()

def ask_password_to_encrypt():
    valid = False
    while not valid:
        password = getpass.getpass("Specify password for key encryption: ")
        valid = validate_password(password)
    return password

def keyfile_data_is_encrypted_ansible( keyfile_data:bytes ) -> bool:
    return keyfile_data[:14] == b'$ANSIBLE_VAULT'

def keyfile_data_is_encrypted_legacy( keyfile_data:bytes ) -> bool:
    return keyfile_data[:6] == b"gAAAAA"

def keyfile_data_is_encrypted( keyfile_data:bytes ) -> bool:
    return keyfile_data_is_encrypted_ansible( keyfile_data ) or keyfile_data_is_encrypted_legacy( keyfile_data )

def encrypt_keyfile_data ( keyfile_data:bytes ) -> bytes:
    password = ask_password_to_encrypt()
    vault = Vault( password )
    return vault.vault.encrypt ( keyfile_data )

def decrypt_keyfile_data( keyfile_data: bytes ) -> bytes:

    try:
        # Ansible decrypt.
        if keyfile_data_is_encrypted_ansible( keyfile_data ):
            password = getpass.getpass("Enter password to unlock key: ")
            vault = Vault( password )
            decrypted_keyfile_data = vault.load( keyfile_data )

        # Legacy decrypt.
        elif keyfile_data_is_encrypted_legacy( keyfile_data ):
            password = getpass.getpass("Enter password to unlock key: ")
            __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            cipher_suite = Fernet(key)
            decrypted_keyfile_data = cipher_suite.decrypt( keyfile_data )   

        else: 
            raise KeyFileError( "Keyfile data: {} is corrupt".format( keyfile_data ))

    except (InvalidSignature, InvalidKey, InvalidToken):
        raise CryptoKeyError('Invalid password')

    if not isinstance(decrypted_keyfile_data, bytes):
        decrypted_keyfile_data = json.dumps( decrypted_keyfile_data ).encode()
    return decrypted_keyfile_data

class Keyfile( object ):
    """ Defines an interface for a subtrate interface keypair stored on device.
    """
    def __init__( self, path: str ):
        self.fullpath = os.path.expanduser(path)

    def __str__(self):
        if not self.exists_on_device():
            return "Keyfile<(empty:{})>".format( self.fullpath )
        if self.is_encrypted():
            return "Keyfile<(encrypted:{})>".format( self.fullpath )
        else:
            return "Keyfile<(decrypted:{})>".format( self.fullpath )

    def __repr__(self):
        return self.__str__()

    @property
    def keypair( self ) -> 'Keypair':
        return self.get_keypair()

    @property
    def keyfile_data( self ) -> bytes:
        return self._read_keyfile_data_from_file()

    def set_keypair ( self, keypair: 'Keypair', encrypt: bool = True, overwrite: bool = False ):
        # Create dirs.
        directory = os.path.dirname( self.fullpath )
        if not os.path.exists( directory ):
            os.makedirs( directory ) 
        keyfile_data = keypair_to_keyfile_data( keypair )
        if encrypt:
            keyfile_data = encrypt_keyfile_data( keyfile_data )
        self._write_keyfile_data_to_file( keyfile_data, overwrite )

    def get_keypair(self) -> 'Keypair':
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = decrypt_keyfile_data( keyfile_data )
        return load_keypair_from_data( keyfile_data )

    def exists_on_device( self ) -> bool:
        if not os.path.isfile( self.fullpath ):
            return False
        return True

    def is_readable( self ) -> bool:
        if not self.exists_on_device():
            return False
        if not os.access( self.fullpath , os.R_OK ):
            return False
        return True

    def is_writable( self ) -> bool:
        if os.access(self.fullpath, os.W_OK):
            return True
        return False

    def is_encrypted ( self ) -> bool:
        if not self.exists_on_device():
            return False
        if not self.is_readable():
            return False
        return keyfile_data_is_encrypted( self._read_keyfile_data_from_file() )

    def _may_overwrite ( self ) -> bool:
        choice = input("File {} already exists. Overwrite ? (y/N) ".format( self.fullpath ))
        return choice == 'y'

    def encrypt( self ):
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.fullpath ))
        if not self.is_readable():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.fullpath ))
        if not self.is_writable():
            raise KeyFileError( "Keyfile at: {} is not writeable".format( self.fullpath ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if not keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = encrypt_keyfile_data( keyfile_data )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def decrypt( self ):
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.fullpath ))
        if not self.is_readable():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.fullpath ))
        if not self.is_writable():
            raise KeyFileError( "No write access for {}".format( self.fullpath ) ) 
        keyfile_data = self._read_keyfile_data_from_file()
        if keyfile_data_is_encrypted( keyfile_data ):
            keyfile_data = decrypt_keyfile_data( keyfile_data )
        self._write_keyfile_data_to_file( keyfile_data, overwrite = True )

    def _read_keyfile_data_from_file ( self ) -> bytes:
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not a file".format( self.fullpath ))
        if not self.exists_on_device():
            raise KeyFileError( "Keyfile at: {} is not readable".format( self.fullpath ))
        with open( self.fullpath , 'rb') as file:
            data = file.read()
        return data

    def _write_keyfile_data_to_file ( self, keyfile_data:bytes, overwrite: bool = False ):
        # Check overwrite.
        if self.exists_on_device() and not overwrite and not self._may_overwrite():
            raise KeyFileError( "Keyfile at: {} is not writeable".format( self.fullpath ) ) 
        with open(self.fullpath, "wb") as keyfile:
            keyfile.write( keyfile_data )
        # Set file permissions.
        os.chmod(self.fullpath, stat.S_IRUSR | stat.S_IWUSR)










        