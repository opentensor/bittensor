# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

import os
import shutil
import time
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import unittest
import unittest.mock as mock

import pytest

import bittensor


class TestKeyFiles(unittest.TestCase):

    def setUp(self) -> None:
        self.root_path = f"/tmp/pytest{time.time()}"
        os.makedirs(self.root_path)

        self.create_keyfile()

    def tearDown(self) -> None:
        shutil.rmtree(self.root_path)

    def create_keyfile(self):
        keyfile = bittensor.keyfile(path=os.path.join(self.root_path, "keyfile"))

        mnemonic = bittensor.Keypair.generate_mnemonic(12)
        alice = bittensor.Keypair.create_from_mnemonic(mnemonic)
        keyfile.set_keypair(alice, encrypt=True, overwrite=True, password='thisisafakepassword')

        bob = bittensor.Keypair.create_from_uri('/Bob')
        keyfile.set_keypair(bob, encrypt=True, overwrite=True, password='thisisafakepassword')

        return keyfile

    def test_create(self):
        keyfile = bittensor.keyfile(path=os.path.join(self.root_path, "keyfile"))

        mnemonic = bittensor.Keypair.generate_mnemonic( 12 )
        alice = bittensor.Keypair.create_from_mnemonic(mnemonic)
        keyfile.set_keypair(alice, encrypt=True, overwrite=True, password = 'thisisafakepassword')
        assert keyfile.is_readable()
        assert keyfile.is_writable()
        assert keyfile.is_encrypted()
        keyfile.decrypt( password = 'thisisafakepassword' )
        assert not keyfile.is_encrypted()
        keyfile.encrypt( password = 'thisisafakepassword' )
        assert keyfile.is_encrypted()
        str(keyfile)
        keyfile.decrypt( password = 'thisisafakepassword' )
        assert not keyfile.is_encrypted()
        str(keyfile)

        assert keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == alice.ss58_address
        assert keyfile.get_keypair( password = 'thisisafakepassword' ).private_key == alice.private_key
        assert keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == alice.public_key

        bob = bittensor.Keypair.create_from_uri ('/Bob')
        keyfile.set_keypair(bob, encrypt=True, overwrite=True, password = 'thisisafakepassword')
        assert keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == bob.ss58_address
        assert keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == bob.public_key

        repr(keyfile)

    def test_legacy_coldkey(self):
        legacy_filename = os.path.join(self.root_path, "coldlegacy_keyfile")
        keyfile = bittensor.keyfile (path = legacy_filename)
        keyfile.make_dirs()
        keyfile_data = b'0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f'
        with open(legacy_filename, "wb") as keyfile_obj:
            keyfile_obj.write( keyfile_data )
        assert keyfile.keyfile_data == keyfile_data
        keyfile.encrypt( password = 'this is the fake password' )
        keyfile.decrypt( password = 'this is the fake password' )
        keypair_bytes = b'{"accountId": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "publicKey": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "secretPhrase": null, "secretSeed": null, "ss58Address": "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"}'
        assert keyfile.keyfile_data == keypair_bytes
        assert keyfile.get_keypair().ss58_address == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        assert "0x" + keyfile.get_keypair().public_key.hex() == "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"

    def test_validate_password(self):
        from bittensor._keyfile.keyfile_impl import validate_password
        assert validate_password(None) == False
        assert validate_password('passw0rd') == False
        assert validate_password('123456789') == False
        with mock.patch('getpass.getpass',return_value='biTTensor'):
            assert validate_password('biTTensor') == True
        with mock.patch('getpass.getpass',return_value='biTTenso'):
            assert validate_password('biTTensor') == False

    def test_decrypt_keyfile_data_legacy(self):
        import base64

        from cryptography.fernet import Fernet
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        from bittensor._keyfile.keyfile_impl import decrypt_keyfile_data

        __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"

        def __generate_key(password):
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=__SALT, length=32, iterations=10000000, backend=default_backend())
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key

        pw = 'fakepasssword238947239'
        data = b'encrypt me!'
        key = __generate_key(pw)
        cipher_suite = Fernet(key)
        encrypted_data = cipher_suite.encrypt(data)

        decrypted_data = decrypt_keyfile_data( encrypted_data, pw)
        assert decrypted_data == data

    def test_user_interface(self):
        from bittensor._keyfile.keyfile_impl import ask_password_to_encrypt

        with mock.patch('getpass.getpass', side_effect = ['pass', 'password', 'asdury3294y', 'asdury3294y']):
            assert ask_password_to_encrypt() == 'asdury3294y'

    def test_overwriting(self):
        from bittensor._keyfile.keyfile_impl import KeyFileError

        keyfile = bittensor.keyfile (path = os.path.join(self.root_path, "keyfile"))
        alice = bittensor.Keypair.create_from_uri ('/Alice')
        keyfile.set_keypair(alice, encrypt=True, overwrite=True, password = 'thisisafakepassword')
        bob = bittensor.Keypair.create_from_uri ('/Bob')

        with pytest.raises(KeyFileError) as pytest_wrapped_e:
            with mock.patch('builtins.input', return_value = 'n'):
                keyfile.set_keypair(bob, encrypt=True, overwrite=False, password = 'thisisafakepassword')

    def test_keyfile_mock(self):
        file = bittensor.keyfile( _mock = True )
        assert file.exists_on_device()
        assert not file.is_encrypted()
        assert file.is_readable()
        assert file.data
        assert file.keypair
        file.set_keypair( keypair = bittensor.Keypair.create_from_mnemonic( mnemonic = bittensor.Keypair.generate_mnemonic() ))

    def test_keyfile_mock_func(self):
        file = bittensor.keyfile.mock()
