import bittensor
from unittest.mock import MagicMock
import os
import shutil
import unittest.mock as mock

# Init dirs.
if os.path.exists('/tmp/pytest'):
    shutil.rmtree('/tmp/pytest')

def test_create():
    keyfile = bittensor.keyfile (path = '/tmp/pytest/keyfile' )
    alice = bittensor.Keypair.create_from_uri ('/Alice')
    keyfile.set_keypair(alice, encrypt=True, overwrite=True, password = 'thisisafakepassword')
    assert keyfile.is_readable()
    assert keyfile.is_writable()
    assert keyfile.is_encrypted()
    keyfile.decrypt( password = 'thisisafakepassword' )
    assert not keyfile.is_encrypted()
    keyfile.encrypt( password = 'thisisafakepassword' )
    assert keyfile.is_encrypted()
    keyfile.decrypt( password = 'thisisafakepassword' )
    assert not keyfile.is_encrypted()
    keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == alice.ss58_address
    keyfile.get_keypair( password = 'thisisafakepassword' ).mnemonic == alice.mnemonic
    keyfile.get_keypair( password = 'thisisafakepassword' ).seed_hex == alice.seed_hex
    keyfile.get_keypair( password = 'thisisafakepassword' ).private_key == alice.private_key
    keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == alice.public_key
    bob = bittensor.Keypair.create_from_uri ('/Bob')
    keyfile.set_keypair(bob, encrypt=True, overwrite=True, password = 'thisisafakepassword')
    keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == bob.ss58_address
    keyfile.get_keypair( password = 'thisisafakepassword' ).mnemonic == bob.mnemonic
    keyfile.get_keypair( password = 'thisisafakepassword' ).seed_hex == bob.seed_hex
    keyfile.get_keypair( password = 'thisisafakepassword' ).private_key == bob.private_key
    keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == bob.public_key

def test_legacy_coldkey():
    keyfile = bittensor.keyfile (path = '/tmp/pytest/coldlegacy_keyfile' )
    keyfile.make_dirs()
    keyfile_data = b'0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f'
    with open('/tmp/pytest/coldlegacy_keyfile', "wb") as keyfile_obj:
        keyfile_obj.write( keyfile_data )
    assert keyfile.keyfile_data == keyfile_data
    keyfile.encrypt( password = 'this is the fake password' )
    keyfile.decrypt( password = 'this is the fake password' )
    keypair_bytes = b'{"accountId": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "publicKey": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "secretPhrase": null, "secretSeed": null, "ss58Address": "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"}'
    assert keyfile.keyfile_data == keypair_bytes
    assert keyfile.get_keypair().ss58_address == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    assert keyfile.get_keypair().public_key == "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"

def test_validate_password():
    from bittensor._keyfile.keyfile_impl import validate_password
    assert validate_password(None) == False
    assert validate_password('passw0rd') == False
    assert validate_password('123456789') == False
    with mock.patch('getpass.getpass',return_value='biTTensor'):
        assert validate_password('biTTensor') == True
    with mock.patch('getpass.getpass',return_value='biTTenso'):
        assert validate_password('biTTensor') == False

def test_decrypt_keyfile_data_legacy():
    import base64
    from bittensor._keyfile.keyfile_impl import decrypt_keyfile_data
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"
    
    def __generate_key(password):
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    key = __generate_key('fakepasssword238947239')
    cipher_suite = Fernet(key)
    cipher_suite.encrypt('encrypt me!')

    decrypt_keyfile_data( keyfile_data: bytes, password: str = None)

test_legacy_coldkey()