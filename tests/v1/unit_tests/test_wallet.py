# The MIT License (MIT)
# Copyright © 2022 Opentensor Foundation

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

import json
import time
import pytest
import random
import re
import bittensor
from bittensor.errors import KeyFileError
from rich.prompt import Confirm
from ansible_vault import Vault
from unittest.mock import patch


def legacy_encrypt_keyfile_data(keyfile_data: bytes, password: str = None) -> bytes:
    console = bittensor.__console__
    with console.status(":locked_with_key: Encrypting key..."):
        vault = Vault(password)
    return vault.vault.encrypt(keyfile_data)


def create_wallet(default_updated_password):
    # create an nacl wallet
    wallet = bittensor.wallet(
        name=f"mock-{str(time.time())}",
        path="/tmp/tests_wallets/do_not_use",
    )
    with patch.object(
        bittensor,
        "ask_password_to_encrypt",
        return_value=default_updated_password,
    ):
        wallet.create()
        assert "NaCl" in str(wallet.coldkey_file)

    return wallet


def create_legacy_wallet(default_legacy_password=None, legacy_password=None):
    def _legacy_encrypt_keyfile_data(*args, **kwargs):
        args = {
            k: v
            for k, v in zip(
                legacy_encrypt_keyfile_data.__code__.co_varnames[: len(args)],
                args,
            )
        }
        kwargs = {**args, **kwargs}
        kwargs["password"] = legacy_password
        return legacy_encrypt_keyfile_data(**kwargs)

    legacy_wallet = bittensor.wallet(
        name=f"mock-legacy-{str(time.time())}",
        path="/tmp/tests_wallets/do_not_use",
    )
    legacy_password = (
        default_legacy_password if legacy_password == None else legacy_password
    )

    # create a legacy ansible wallet
    with patch.object(
        bittensor,
        "encrypt_keyfile_data",
        new=_legacy_encrypt_keyfile_data,
        # new = TestWalletUpdate.legacy_encrypt_keyfile_data,
    ):
        legacy_wallet.create()
        assert "Ansible" in str(legacy_wallet.coldkey_file)

    return legacy_wallet


@pytest.fixture
def wallet_update_setup():
    # Setup the default passwords and wallets
    default_updated_password = "nacl_password"
    default_legacy_password = "ansible_password"
    empty_wallet = bittensor.wallet(
        name=f"mock-empty-{str(time.time())}",
        path="/tmp/tests_wallets/do_not_use",
    )
    legacy_wallet = create_legacy_wallet(
        default_legacy_password=default_legacy_password
    )
    wallet = create_wallet(default_updated_password)

    return {
        "default_updated_password": default_updated_password,
        "default_legacy_password": default_legacy_password,
        "empty_wallet": empty_wallet,
        "legacy_wallet": legacy_wallet,
        "wallet": wallet,
    }


def test_encrypt_and_decrypt():
    """Test message can be encrypted and decrypted successfully with ansible/nacl."""
    json_data = {
        "address": "This is the address.",
        "id": "This is the id.",
        "key": "This is the key.",
    }
    message = json.dumps(json_data).encode()

    # encrypt and decrypt with nacl
    encrypted_message = bittensor.encrypt_keyfile_data(message, "password")
    decrypted_message = bittensor.decrypt_keyfile_data(encrypted_message, "password")
    assert decrypted_message == message
    assert bittensor.keyfile_data_is_encrypted(encrypted_message)
    assert not bittensor.keyfile_data_is_encrypted(decrypted_message)
    assert not bittensor.keyfile_data_is_encrypted_ansible(decrypted_message)
    assert bittensor.keyfile_data_is_encrypted_nacl(encrypted_message)

    # encrypt and decrypt with legacy ansible
    encrypted_message = legacy_encrypt_keyfile_data(message, "password")
    decrypted_message = bittensor.decrypt_keyfile_data(encrypted_message, "password")
    assert decrypted_message == message
    assert bittensor.keyfile_data_is_encrypted(encrypted_message)
    assert not bittensor.keyfile_data_is_encrypted(decrypted_message)
    assert not bittensor.keyfile_data_is_encrypted_nacl(decrypted_message)
    assert bittensor.keyfile_data_is_encrypted_ansible(encrypted_message)


def test_check_and_update_encryption_not_updated(wallet_update_setup):
    """Test for a few cases where wallet should not be updated.
    1. When the wallet is already updated.
    2. When it is the hotkey.
    3. When the wallet is empty.
    4. When the wallet is legacy but no prompt to ask for password.
    5. When the password is wrong.
    """
    wallet = wallet_update_setup["wallet"]
    empty_wallet = wallet_update_setup["empty_wallet"]
    legacy_wallet = wallet_update_setup["legacy_wallet"]
    default_legacy_password = wallet_update_setup["default_legacy_password"]
    # test the checking with no rewriting needs to be done.
    with patch("bittensor.encrypt_keyfile_data") as encrypt:
        # self.wallet is already the most updated with nacl encryption.
        assert wallet.coldkey_file.check_and_update_encryption()

        # hotkey_file is not encrypted, thus do not need to be updated.
        assert not wallet.hotkey_file.check_and_update_encryption()

        # empty_wallet has not been created, thus do not need to be updated.
        assert not empty_wallet.coldkey_file.check_and_update_encryption()

        # legacy wallet cannot be updated without asking for password form prompt.
        assert not legacy_wallet.coldkey_file.check_and_update_encryption(
            no_prompt=True
        )

        # Wrong password
        legacy_wallet = create_legacy_wallet(
            default_legacy_password=default_legacy_password
        )
        with patch("getpass.getpass", return_value="wrong_password"), patch.object(
            Confirm, "ask", return_value=False
        ):
            assert not legacy_wallet.coldkey_file.check_and_update_encryption()

        # no renewal has been done in this test.
        assert not encrypt.called


def test_check_and_update_excryption(wallet_update_setup, legacy_wallet=None):
    """Test for the alignment of the updated VS old wallet.
    1. Same coldkey_file data.
    2. Same coldkey path.
    3. Same hotkey_file data.
    4. Same hotkey path.
    5. same password.

    Read the updated wallet in 2 ways.
    1. Directly as the output of check_and_update_encryption()
    2. Read from file using the same coldkey and hotkey name
    """
    default_legacy_password = wallet_update_setup["default_legacy_password"]

    def check_new_coldkey_file(keyfile):
        new_keyfile_data = keyfile._read_keyfile_data_from_file()
        new_decrypted_keyfile_data = bittensor.decrypt_keyfile_data(
            new_keyfile_data, legacy_password
        )
        new_path = legacy_wallet.coldkey_file.path

        assert old_coldkey_file_data != None
        assert new_keyfile_data != None
        assert not old_coldkey_file_data == new_keyfile_data
        assert bittensor.keyfile_data_is_encrypted_ansible(old_coldkey_file_data)
        assert bittensor.keyfile_data_is_encrypted_nacl(new_keyfile_data)
        assert not bittensor.keyfile_data_is_encrypted_nacl(old_coldkey_file_data)
        assert not bittensor.keyfile_data_is_encrypted_ansible(new_keyfile_data)
        assert old_decrypted_coldkey_file_data == new_decrypted_keyfile_data
        assert new_path == old_coldkey_path

    def check_new_hotkey_file(keyfile):
        new_keyfile_data = keyfile._read_keyfile_data_from_file()
        new_path = legacy_wallet.hotkey_file.path

        assert old_hotkey_file_data == new_keyfile_data
        assert new_path == old_hotkey_path
        assert not bittensor.keyfile_data_is_encrypted(new_keyfile_data)

    if legacy_wallet == None:
        legacy_password = f"PASSword-{random.randint(0, 10000)}"
        legacy_wallet = create_legacy_wallet(legacy_password=legacy_password)

    else:
        legacy_password = default_legacy_password

    # get old cold keyfile data
    old_coldkey_file_data = legacy_wallet.coldkey_file._read_keyfile_data_from_file()
    old_decrypted_coldkey_file_data = bittensor.decrypt_keyfile_data(
        old_coldkey_file_data, legacy_password
    )
    old_coldkey_path = legacy_wallet.coldkey_file.path

    # get old hot keyfile data
    old_hotkey_file_data = legacy_wallet.hotkey_file._read_keyfile_data_from_file()
    old_hotkey_path = legacy_wallet.hotkey_file.path

    # update legacy_wallet from ansible to nacl
    with patch("getpass.getpass", return_value=legacy_password), patch.object(
        Confirm, "ask", return_value=True
    ):
        legacy_wallet.coldkey_file.check_and_update_encryption()

    # get new keyfile data from the same legacy wallet
    check_new_coldkey_file(legacy_wallet.coldkey_file)
    check_new_hotkey_file(legacy_wallet.hotkey_file)

    # get new keyfile data from wallet name
    updated_legacy_wallet = bittensor.wallet(
        name=legacy_wallet.name,
        hotkey=legacy_wallet.hotkey_str,
        path="/tmp/tests_wallets/do_not_use",
    )
    check_new_coldkey_file(updated_legacy_wallet.coldkey_file)
    check_new_hotkey_file(updated_legacy_wallet.hotkey_file)

    # def test_password_retain(self):
    # [tick] test the same password works
    # [tick] try to read using the same hotkey/coldkey name
    # [tick] test the same keyfile data could be retained
    # [tick] test what if a wrong password was inserted
    # [no need] try to read from the new file path
    # [tick] test the old and new encrypted is not the same
    # [tick] test that the hotkeys are not affected


@pytest.fixture
def mock_wallet():
    wallet = bittensor.wallet(
        name=f"mock-{str(time.time())}",
        hotkey=f"mock-{str(time.time())}",
        path="/tmp/tests_wallets/do_not_use",
    )
    wallet.create_new_coldkey(use_password=False, overwrite=True, suppress=True)
    wallet.create_new_hotkey(use_password=False, overwrite=True, suppress=True)

    return wallet


def test_regen_coldkeypub_from_ss58_addr(mock_wallet):
    """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from an SS58 address.
    It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's SS58 address matches the input SS58 address.
    It also tests the behavior when an invalid SS58 address is provided, raising a `ValueError` as expected.
    """
    ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    with patch.object(mock_wallet, "set_coldkeypub") as mock_set_coldkeypub:
        mock_wallet.regenerate_coldkeypub(
            ss58_address=ss58_address, overwrite=True, suppress=True
        )

        mock_set_coldkeypub.assert_called_once()
        keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
        assert keypair.ss58_address == ss58_address

    ss58_address_bad = (
        "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zx"  # 1 character short
    )
    with pytest.raises(ValueError):
        mock_wallet.regenerate_coldkeypub(
            ss58_address=ss58_address_bad, overwrite=True, suppress=True
        )


def test_regen_coldkeypub_from_hex_pubkey_str(mock_wallet):
    """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from a hex public key string.
    It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's public key matches the input public key.
    It also tests the behavior when an invalid public key string is provided, raising a `ValueError` as expected.
    """
    pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
    with patch.object(mock_wallet, "set_coldkeypub") as mock_set_coldkeypub:
        mock_wallet.regenerate_coldkeypub(
            public_key=pubkey_str, overwrite=True, suppress=True
        )

        mock_set_coldkeypub.assert_called_once()
        keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
        assert "0x" + keypair.public_key.hex() == pubkey_str

    pubkey_str_bad = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512"  # 1 character short
    with pytest.raises(ValueError):
        mock_wallet.regenerate_coldkeypub(
            ss58_address=pubkey_str_bad, overwrite=True, suppress=True
        )


def test_regen_coldkeypub_from_hex_pubkey_bytes(mock_wallet):
    """Test the `regenerate_coldkeypub` method of the wallet class, which regenerates the cold key pair from a hex public key byte string.
    It checks whether the `set_coldkeypub` method is called with the expected arguments, and verifies that the generated key pair's public key matches the input public key.
    """
    pubkey_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
    pubkey_bytes = bytes.fromhex(pubkey_str[2:])  # Remove 0x from beginning
    with patch.object(mock_wallet, "set_coldkeypub") as mock_set_coldkeypub:
        mock_wallet.regenerate_coldkeypub(
            public_key=pubkey_bytes, overwrite=True, suppress=True
        )

        mock_set_coldkeypub.assert_called_once()
        keypair: bittensor.Keypair = mock_set_coldkeypub.call_args_list[0][0][0]
        assert keypair.public_key == pubkey_bytes


def test_regen_coldkeypub_no_pubkey(mock_wallet):
    """Test the `regenerate_coldkeypub` method of the wallet class when no public key is provided.
    It verifies that a `ValueError` is raised when neither a public key nor an SS58 address is provided.
    """
    with pytest.raises(ValueError):
        # Must provide either public_key or ss58_address
        mock_wallet.regenerate_coldkeypub(
            ss58_address=None, public_key=None, overwrite=True, suppress=True
        )


def test_regen_coldkey_from_hex_seed_str(mock_wallet):
    """Test the `regenerate_coldkey` method of the wallet class, which regenerates the cold key pair from a hex seed string.
    It checks whether the `set_coldkey` method is called with the expected arguments, and verifies that the generated key pair's seed and SS58 address match the input seed and the expected SS58 address.
    It also tests the behavior when an invalid seed string is provided, raising a `ValueError` as expected.
    """
    ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
    seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
    with patch.object(mock_wallet, "set_coldkey") as mock_set_coldkey:
        mock_wallet.regenerate_coldkey(seed=seed_str, overwrite=True, suppress=True)

        mock_set_coldkey.assert_called_once()
        keypair: bittensor.Keypair = mock_set_coldkey.call_args_list[0][0][0]
        seed_hex = (
            keypair.seed_hex
            if isinstance(keypair.seed_hex, str)
            else keypair.seed_hex.hex()
        )

        assert re.match(
            rf"(0x|){seed_str[2:]}", seed_hex
        ), "The seed_hex does not match the expected pattern"
        assert (
            keypair.ss58_address == ss58_addr
        )  # Check that the ss58 address is correct

    seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763"  # 1 character short
    with pytest.raises(ValueError):
        mock_wallet.regenerate_coldkey(seed=seed_str_bad, overwrite=True, suppress=True)


def test_regen_hotkey_from_hex_seed_str(mock_wallet):
    """Test the `regenerate_coldkey` method of the wallet class, which regenerates the cold key pair from a hex seed string.
    It checks whether the `set_coldkey` method is called with the expected arguments, and verifies that the generated key pair's seed and SS58 address match the input seed and the expected SS58 address.
    It also tests the behavior when an invalid seed string is provided, raising a `ValueError` as expected.
    """
    ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
    seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"
    with patch.object(mock_wallet, "set_hotkey") as mock_set_hotkey:
        mock_wallet.regenerate_hotkey(seed=seed_str, overwrite=True, suppress=True)

        mock_set_hotkey.assert_called_once()
        keypair: bittensor.Keypair = mock_set_hotkey.call_args_list[0][0][0]

        seed_hex = (
            keypair.seed_hex
            if isinstance(keypair.seed_hex, str)
            else keypair.seed_hex.hex()
        )

        pattern = rf"(0x|){seed_str[2:]}"
        assert re.match(
            pattern, seed_hex
        ), f"The seed_hex '{seed_hex}' does not match the expected pattern '{pattern}'"
        assert (
            keypair.ss58_address == ss58_addr
        )  # Check that the ss58 address is correct

    seed_str_bad = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f763"  # 1 character short
    with pytest.raises(ValueError):
        mock_wallet.regenerate_hotkey(seed=seed_str_bad, overwrite=True, suppress=True)


@pytest.mark.parametrize(
    "mnemonic, expected_exception",
    [
        # Input is in a string format
        (
            "fiscal prevent noise record smile believe quote front weasel book axis legal",
            None,
        ),
        # Input is in a list format (acquired by encapsulating mnemonic arg in a string "" in the cli)
        (
            [
                "fiscal prevent noise record smile believe quote front weasel book axis legal"
            ],
            None,
        ),
        # Input is in a full list format (aquired by pasting mnemonic arg simply w/o quotes in cli)
        (
            [
                "fiscal",
                "prevent",
                "noise",
                "record",
                "smile",
                "believe",
                "quote",
                "front",
                "weasel",
                "book",
                "axis",
                "legal",
            ],
            None,
        ),
        # Incomplete mnemonic
        ("word1 word2 word3", ValueError),
        # No mnemonic added
        (None, ValueError),
    ],
    ids=[
        "string-format",
        "list-format-thru-string",
        "list-format",
        "incomplete-mnemonic",
        "no-mnemonic",
    ],
)
def test_regen_coldkey_mnemonic(mock_wallet, mnemonic, expected_exception):
    """Test the `regenerate_coldkey` method of the wallet class, which regenerates the cold key pair from a mnemonic.
    We test different input formats of mnemonics and check if the function works as expected.
    """
    with patch.object(mock_wallet, "set_coldkey") as mock_set_coldkey, patch.object(
        mock_wallet, "set_coldkeypub"
    ) as mock_set_coldkeypub:
        if expected_exception:
            with pytest.raises(expected_exception):
                mock_wallet.regenerate_coldkey(
                    mnemonic=mnemonic, overwrite=True, suppress=True
                )
        else:
            mock_wallet.regenerate_coldkey(mnemonic=mnemonic)
            mock_set_coldkey.assert_called_once()
            mock_set_coldkeypub.assert_called_once()


@pytest.mark.parametrize(
    "overwrite, user_input, expected_exception",
    [
        (True, None, None),  # Test with overwrite=True, no user input needed
        (False, "n", True),  # Test with overwrite=False and user says no, KeyFileError
        (False, "y", None),  # Test with overwrite=False and user says yes
    ],
)
def test_regen_coldkey_overwrite_functionality(
    mock_wallet, overwrite, user_input, expected_exception
):
    """Test the `regenerate_coldkey` method of the wallet class, emphasizing on the overwrite functionality"""
    ss58_addr = "5D5cwd8DX6ij7nouVcoxDuWtJfiR1BnzCkiBVTt7DU8ft5Ta"
    seed_str = "0x659c024d5be809000d0d93fe378cfde020846150b01c49a201fc2a02041f7636"

    with patch.object(mock_wallet, "set_coldkey") as mock_set_coldkey, patch(
        "builtins.input", return_value=user_input
    ):
        if expected_exception:
            with pytest.raises(KeyFileError):
                mock_wallet.regenerate_coldkey(
                    seed=seed_str, overwrite=overwrite, suppress=True
                )
        else:
            mock_wallet.regenerate_coldkey(
                seed=seed_str, overwrite=overwrite, suppress=True
            )
            mock_set_coldkey.assert_called_once()
            keypair = mock_set_coldkey.call_args_list[0][0][0]
            seed_hex = (
                keypair.seed_hex
                if isinstance(keypair.seed_hex, str)
                else keypair.seed_hex.hex()
            )
            assert re.match(
                rf"(0x|){seed_str[2:]}", seed_hex
            ), "The seed_hex does not match the expected pattern"
            assert (
                keypair.ss58_address == ss58_addr
            ), "The SS58 address does not match the expected address"
