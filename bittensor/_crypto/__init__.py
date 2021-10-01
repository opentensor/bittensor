""" Functions for encryption and decryption of data with password
"""
import os
import stat
import base64
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger
from ansible_vault import Vault

class CryptoKeyError(Exception):
    """ Exception for invalid signature, key, token, password, etc 
        Overwrite the built-in CryptoKeyError
    """

__SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"

def encrypt_to_file(data, password, full_path):
    """ Encrypt the data with password
    """
    vault = Vault(password)
    vault.dump( data, open( full_path, 'w') )
    return 

def decrypt_keypair(password, full_path):
    """ Decrypt the data with password
    """
    vault = Vault(password)
    return vault.load(open(full_path).read())

def __generate_key(password):
    """ Get key from password
    """
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def is_encrypted(file):
    """ Check if data was encrypted
    """
    with open( file , 'rb') as f:
        data = f.read()
        return data[0:14] == b'$ANSIBLE_VAULT'

def decrypt_file(password, full_path):
    """ Decrypt the data with password
        With error handling
    """
    try:
        return decrypt_keypair(password, full_path)
    except (InvalidSignature, InvalidKey, InvalidToken) as key_error:
        raise CryptoKeyError from key_error
