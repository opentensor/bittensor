from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger
import base64

class KeyError(Exception):
    pass

__SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"

def encrypt(data, password):
    key = __generate_key(password)
    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(data)

def decrypt_keypair(data, password):
    key = __generate_key(password)
    cipher_suite = Fernet(key)
    return cipher_suite.decrypt(data)

def __generate_key(password):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def is_encrypted(data):
    return data[:6] == b"gAAAAA"

def decrypt_data(password, data):
    try:
        return decrypt_keypair(data, password)
    except (InvalidSignature, InvalidKey, InvalidToken):
        raise KeyError