# The MIT License (MIT)
# Copyright © 2022 Yuma Rao

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

import unittest
from unittest.mock import patch, MagicMock
import pytest
import bittensor
from ansible_vault import Vault
import random
import getpass
from rich.prompt import Confirm
import json
from bittensor._keyfile.keyfile_impl import encrypt_keyfile_data, decrypt_keyfile_data, keyfile_data_is_encrypted, keyfile_data_is_encrypted_nacl, keyfile_data_is_encrypted_ansible
class TestWalletUpdate(unittest.TestCase):
    def setUp(self):
        test_id = random.randint(0, 10000)
        self.wallet = bittensor.wallet(name = f'updated_wallet_{test_id}')
        self.empty_wallet = bittensor.wallet(name = f'empty_wallet_{test_id}')
        self.legacy_wallet = bittensor.wallet(name = f'legacy_wallet_{test_id}')
        self.create_wallet()
    
    def legacy_encrypt_keyfile_data ( keyfile_data:bytes, password: str = None ) -> bytes:
        password = 'ansible_password' if password == None else password
        console = bittensor.__console__;             
        with console.status(":locked_with_key: Encrypting key..."):
            vault = Vault( password )
        return vault.vault.encrypt ( keyfile_data )
    
    def create_wallet(self):
        # create an nacl wallet
        with patch.object(bittensor._keyfile.keyfile_impl, 'ask_password_to_encrypt', return_value = 'nacl_password'):
            self.wallet.create()
            assert 'NaCl' in str(self.wallet.coldkey_file)
        
        # create a legacy ansible wallet
        with patch.object(bittensor._keyfile.keyfile_impl, 'encrypt_keyfile_data', new = TestWalletUpdate.legacy_encrypt_keyfile_data):
            self.legacy_wallet.create()
            assert 'Ansible' in str(self.legacy_wallet.coldkey_file)

    def test_encrypt_and_decrypt(self):
        json_data = {
            'address': 'This is the address.',
            'id': 'This is the id.',
            'key': 'This is the key.',
        }
        message = json.dumps( json_data ).encode()
        
        # encrypt and decrypt with nacl
        encrypted_message = encrypt_keyfile_data(message, 'password')
        decrypted_message = decrypt_keyfile_data(encrypted_message, 'password')
        assert decrypted_message == message
        print(message, decrypted_message)
        assert keyfile_data_is_encrypted(encrypted_message)
        assert not keyfile_data_is_encrypted(decrypted_message)
        assert not keyfile_data_is_encrypted(decrypted_message)
        assert keyfile_data_is_encrypted_nacl(encrypted_message)

        # encrypt and decrypt with legacy ansible
        encrypted_message = TestWalletUpdate.legacy_encrypt_keyfile_data(message, 'password')
        decrypted_message = decrypt_keyfile_data(encrypted_message, 'password')
        assert decrypted_message == message
        print(message, decrypted_message)
        assert keyfile_data_is_encrypted(encrypted_message)
        assert not keyfile_data_is_encrypted(decrypted_message)
        assert not keyfile_data_is_encrypted_nacl(decrypted_message)
        assert keyfile_data_is_encrypted_ansible(encrypted_message)

    def test_check_and_update_encryption_not_updated(self):
        # test the checking with no rewriting needs to be done.
        with patch('bittensor._keyfile.keyfile_impl.encrypt_keyfile_data') as encrypt:
            assert self.wallet.coldkey_file.check_and_update_encryption()
            assert not self.wallet.hotkey_file.check_and_update_encryption()
            assert not self.empty_wallet.coldkey_file.check_and_update_encryption()
            assert not self.legacy_wallet.coldkey_file.check_and_update_encryption(no_prompt=True)
            assert not encrypt.called

    def test_check_and_update_excryption(self):        
        # get old keyfile data 
        old_keyfile_data = self.legacy_wallet.coldkey_file._read_keyfile_data_from_file()
        old_decrypted_keyfile_data = decrypt_keyfile_data(old_keyfile_data, 'ansible_password')
        old_path = self.legacy_wallet.coldkey_file.path
        
        # update legacy_wallet from ansible to nacl
        with patch('getpass.getpass', return_value = 'ansible_password'), patch.object(Confirm, 'ask', return_value = True):
            self.legacy_wallet.coldkey_file.check_and_update_encryption()
        
        # get new keyfile data 
        new_keyfile_data = self.legacy_wallet.coldkey_file._read_keyfile_data_from_file()
        new_decrypted_keyfile_data = decrypt_keyfile_data(new_keyfile_data, 'ansible_password')
        new_path = self.legacy_wallet.coldkey_file.path

        assert bittensor._keyfile.keyfile_impl.keyfile_data_is_encrypted_ansible(old_keyfile_data)
        assert bittensor._keyfile.keyfile_impl.keyfile_data_is_encrypted_nacl(new_keyfile_data)
        assert old_decrypted_keyfile_data == new_decrypted_keyfile_data
        assert new_path == old_path


class TestWallet(unittest.TestCase):
    def setUp(self):
        self.mock_wallet = bittensor.wallet( _mock = True )

    def test_regen_coldkeypub_from_ss58_addr(self):
        ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub( ss58_address=ss58_address )

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual(keypair.ss58_address, ss58_address)

        ss58_address_bad = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zx" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkeypub(ss58_address=ss58_address_bad)

    def test_regen_coldkeypub_from_hex_pubkey_str(self):
        pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub(public_key=pubkey_str)

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual('0x' + keypair.public_key.hex(), pubkey_str)

        pubkey_str_bad = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkeypub(ss58_address=pubkey_str_bad)

    def test_regen_coldkeypub_from_hex_pubkey_bytes(self):
        pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        pubkey_bytes = bytes.fromhex(pubkey_str[2:]) # Remove 0x from beginning
        with patch.object(self.mock_wallet, 'set_coldkeypub') as mock_set_coldkeypub:
            self.mock_wallet.regenerate_coldkeypub(public_key=pubkey_bytes)

            mock_set_coldkeypub.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
            self.assertEqual(keypair.public_key, pubkey_bytes)

    def test_regen_coldkeypub_no_pubkey(self):
        with pytest.raises(ValueError):
            # Must provide either public_key or ss58_address
            self.mock_wallet.regenerate_coldkeypub(ss58_address=None, public_key=None)

    def test_regen_coldkey_from_hex_seed_str(self):
        ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
        seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
        with patch.object(self.mock_wallet, 'set_coldkey') as mock_set_coldkey:
            self.mock_wallet.regenerate_coldkey(seed=seed_str)

            mock_set_coldkey.assert_called_once()
            keypair: bittensor.Keypair = mock_set_coldkey.call_args_list[0][0][0]
            self.assertRegexpMatches(keypair.seed_hex if isinstance(keypair.seed_hex, str) else keypair.seed_hex.hex(), rf'(0x|){seed_str[2:]}')
            self.assertEqual(keypair.ss58_address, ss58_addr) # Check that the ss58 address is correct

        seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_coldkey(seed=seed_str_bad)

    def test_regen_hotkey_from_hex_seed_str(self):
        ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
        seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
        with patch.object(self.mock_wallet, 'set_hotkey') as mock_set_hotkey:
            self.mock_wallet.regenerate_hotkey(seed=seed_str)

            mock_set_hotkey.assert_called_once()
            keypair: bittensor.Keypair = mock_set_hotkey.call_args_list[0][0][0]
            self.assertRegexpMatches(keypair.seed_hex if isinstance(keypair.seed_hex, str) else keypair.seed_hex.hex(), rf'(0x|){seed_str[2:]}')
            self.assertEqual(keypair.ss58_address, ss58_addr) # Check that the ss58 address is correct

        seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763" # 1 character short
        with pytest.raises(ValueError):
            self.mock_wallet.regenerate_hotkey(seed=seed_str_bad)

class TestWalletReregister(unittest.TestCase):
    def test_wallet_reregister_use_cuda_flag_none(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = None # don't set the argument, but do specify the flag
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], None) # should be None when no argument, but flag set

    def test_wallet_reregister_use_cuda_flag_true(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = True
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], True) # should be default when no argument

    def test_wallet_reregister_use_cuda_flag_false(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        config.subtensor.register.cuda.use_cuda = False
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], False) # should be default when no argument

    def test_wallet_reregister_use_cuda_flag_not_specified_false(self):
        config = bittensor.Config()
        config.wallet = bittensor.Config()
        config.wallet.reregister = True

        config.subtensor = bittensor.Config()
        config.subtensor.register = bittensor.Config()
        config.subtensor.register.cuda = bittensor.Config()
        #config.subtensor.register.cuda.use_cuda # don't specify the flag
        config.subtensor.register.cuda.dev_id = 0
        # No need to specify the other config options as they are default to None

        mock_wallet = bittensor.wallet.mock()
        mock_wallet.is_registered = MagicMock(return_value=False)
        mock_wallet.config = config

        class MockException(Exception):
            pass

        def exit_early(*args, **kwargs):
            raise MockException('exit_early')

        with patch('bittensor.Subtensor.register', side_effect=exit_early) as mock_register:
            # Should be able to set without argument
            with pytest.raises(MockException):
                mock_wallet.reregister( netuid = -1 )

            call_args = mock_register.call_args
            _, kwargs = call_args

            mock_register.assert_called_once()
            self.assertEqual(kwargs['cuda'], False) # should be False when no flag was set


if __name__ == '__main__':
    unittest.main()