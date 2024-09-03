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

import os
import json
import time
import pytest
import shutil
import bittensor
import unittest.mock as mock
from scalecodec import ScaleBytes
from substrateinterface import Keypair, KeypairType
from substrateinterface.constants import DEV_PHRASE
from substrateinterface.exceptions import ConfigurationError
from bip39 import bip39_validate

from bittensor import get_coldkey_password_from_environment


def test_generate_mnemonic():
    """
    Test the generation of a mnemonic and its validation.
    """
    mnemonic = Keypair.generate_mnemonic()
    assert bip39_validate(mnemonic) == True


def test_invalid_mnemonic():
    """
    Test the validation of an invalid mnemonic.
    """
    mnemonic = "This is an invalid mnemonic"
    assert bip39_validate(mnemonic) == False


def test_create_sr25519_keypair():
    """
    Test the creation of a sr25519 keypair from a mnemonic and verify the SS58 address.
    """
    mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
    keypair = Keypair.create_from_mnemonic(mnemonic, ss58_format=0)
    assert keypair.ss58_address == "16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2"


def test_only_provide_ss58_address():
    """
    Test the creation of a keypair with only the SS58 address provided.
    """
    keypair = Keypair(ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2")

    assert (
        f"0x{keypair.public_key.hex()}"
        == "0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446"
    )


def test_only_provide_public_key():
    """
    Test the creation of a keypair with only the public key provided.
    """
    keypair = Keypair(
        public_key="0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446",
        ss58_format=0,
    )

    assert keypair.ss58_address == "16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2"


def test_provide_no_ss58_address_and_public_key():
    """
    Test the creation of a keypair without providing SS58 address and public key.
    """
    with pytest.raises(ValueError):
        Keypair()


def test_incorrect_private_key_length_sr25519():
    """
    Test the creation of a keypair with an incorrect private key length for sr25519.
    """
    with pytest.raises(ValueError):
        Keypair(
            private_key="0x23",
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
        )


def test_incorrect_public_key():
    """
    Test the creation of a keypair with an incorrect public key.
    """
    with pytest.raises(ValueError):
        Keypair(public_key="0x23")


def test_sign_and_verify():
    """
    Test the signing and verification of a message using a keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    signature = keypair.sign("Test1231223123123")
    assert keypair.verify("Test1231223123123", signature) == True


def test_sign_and_verify_hex_data():
    """
    Test the signing and verification of hex data using a keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    signature = keypair.sign("0x1234")
    assert keypair.verify("0x1234", signature) == True


def test_sign_and_verify_scale_bytes():
    """
    Test the signing and verification of ScaleBytes data using a keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    data = ScaleBytes("0x1234")
    signature = keypair.sign(data)
    assert keypair.verify(data, signature) == True


def test_sign_missing_private_key():
    """
    Test signing a message with a keypair that is missing the private key.
    """
    keypair = Keypair(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    with pytest.raises(ConfigurationError):
        keypair.sign("0x1234")


def test_sign_unsupported_crypto_type():
    """
    Test signing a message with an unsupported crypto type.
    """
    keypair = Keypair.create_from_private_key(
        ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
        private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
        crypto_type=3,
    )
    with pytest.raises(ConfigurationError):
        keypair.sign("0x1234")


def test_verify_unsupported_crypto_type():
    """
    Test verifying a signature with an unsupported crypto type.
    """
    keypair = Keypair.create_from_private_key(
        ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
        private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
        crypto_type=3,
    )
    with pytest.raises(ConfigurationError):
        keypair.verify("0x1234", "0x1234")


def test_sign_and_verify_incorrect_signature():
    """
    Test verifying an incorrect signature for a signed message.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    signature = "0x4c291bfb0bb9c1274e86d4b666d13b2ac99a0bacc04a4846fb8ea50bda114677f83c1f164af58fc184451e5140cc8160c4de626163b11451d3bbb208a1889f8a"
    assert keypair.verify("Test1231223123123", signature) == False


def test_sign_and_verify_invalid_signature():
    """
    Test verifying an invalid signature format for a signed message.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    signature = "Test"
    with pytest.raises(TypeError):
        keypair.verify("Test1231223123123", signature)


def test_sign_and_verify_invalid_message():
    """
    Test verifying a signature against an incorrect message.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    signature = keypair.sign("Test1231223123123")
    assert keypair.verify("OtherMessage", signature) == False


def test_create_ed25519_keypair():
    """
    Test the creation of an ed25519 keypair from a mnemonic and verify the SS58 address.
    """
    mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
    keypair = Keypair.create_from_mnemonic(
        mnemonic, ss58_format=0, crypto_type=KeypairType.ED25519
    )
    assert keypair.ss58_address == "16dYRUXznyhvWHS1ktUENGfNAEjCawyDzHRtN9AdFnJRc38h"


def test_sign_and_verify_ed25519():
    """
    Test the signing and verification of a message using an ed25519 keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic, crypto_type=KeypairType.ED25519)
    signature = keypair.sign("Test1231223123123")
    assert keypair.verify("Test1231223123123", signature) == True


def test_sign_and_verify_invalid_signature_ed25519():
    """
    Test verifying an incorrect signature for a message signed with an ed25519 keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic, crypto_type=KeypairType.ED25519)
    signature = "0x4c291bfb0bb9c1274e86d4b666d13b2ac99a0bacc04a4846fb8ea50bda114677f83c1f164af58fc184451e5140cc8160c4de626163b11451d3bbb208a1889f8a"
    assert keypair.verify("Test1231223123123", signature) == False


def test_unsupport_crypto_type():
    """
    Test creating a keypair with an unsupported crypto type.
    """
    with pytest.raises(ValueError):
        Keypair.create_from_seed(
            seed_hex="0xda3cf5b1e9144931?a0f0db65664aab662673b099415a7f8121b7245fb0be4143",
            crypto_type=2,
        )


def test_create_keypair_from_private_key():
    """
    Test creating a keypair from a private key and verify the public key.
    """
    keypair = Keypair.create_from_private_key(
        ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
        private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
    )
    assert (
        f"0x{keypair.public_key.hex()}"
        == "0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446"
    )


def test_hdkd_hard_path():
    """
    Test hierarchical deterministic key derivation with a hard derivation path.
    """
    mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
    derivation_address = "5FEiH8iuDUw271xbqWTWuB6WrDjv5dnCeDX1CyHubAniXDNN"
    derivation_path = "//Alice"
    derived_keypair = Keypair.create_from_uri(mnemonic + derivation_path)
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_soft_path():
    """
    Test hierarchical deterministic key derivation with a soft derivation path.
    """
    derivation_address = "5GNXbA46ma5dg19GXdiKi5JH3mnkZ8Yea3bBtZAvj7t99P9i"
    mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
    derived_keypair = Keypair.create_from_uri(f"{mnemonic}/Alice")
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_default_to_dev_mnemonic():
    """
    Test hierarchical deterministic key derivation with a default development mnemonic.
    """
    derivation_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    derivation_path = "//Alice"
    derived_keypair = Keypair.create_from_uri(derivation_path)
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_nested_hard_soft_path():
    """
    Test hierarchical deterministic key derivation with a nested hard and soft derivation path.
    """
    derivation_address = "5CJGwWiKXSE16WJaxBdPZhWqUYkotgenLUALv7ZvqQ4TXeqf"
    derivation_path = "//Bob/test"
    derived_keypair = Keypair.create_from_uri(derivation_path)
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_nested_soft_hard_path():
    """
    Test hierarchical deterministic key derivation with a nested soft and hard derivation path.
    """
    derivation_address = "5Cwc8tShrshDJUp1P1M21dKUTcYQpV9GcfSa4hUBNmMdV3Cx"
    derivation_path = "/Bob//test"
    derived_keypair = Keypair.create_from_uri(derivation_path)
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_path_gt_32_bytes():
    """
    Test hierarchical deterministic key derivation with a derivation path longer than 32 bytes.
    """
    derivation_address = "5GR5pfZeNs1uQiSWVxZaQiZou3wdZiX894eqgvfNfHbEh7W2"
    derivation_path = "//PathNameLongerThan32BytesWhichShouldBeHashed"
    derived_keypair = Keypair.create_from_uri(derivation_path)
    assert derivation_address == derived_keypair.ss58_address


def test_hdkd_unsupported_password():
    """
    Test hierarchical deterministic key derivation with an unsupported password.
    """

    with pytest.raises(NotImplementedError):
        Keypair.create_from_uri(f"{DEV_PHRASE}///test")


def create_keyfile(root_path):
    """
    Creates a keyfile object with two keypairs: alice and bob.

    Args:
        root_path (str): The root path for the keyfile.

    Returns:
        bittensor.keyfile: The created keyfile object.
    """
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))

    mnemonic = bittensor.Keypair.generate_mnemonic(12)
    alice = bittensor.Keypair.create_from_mnemonic(mnemonic)
    keyfile.set_keypair(
        alice, encrypt=True, overwrite=True, password="thisisafakepassword"
    )

    bob = bittensor.Keypair.create_from_uri("/Bob")
    keyfile.set_keypair(
        bob, encrypt=True, overwrite=True, password="thisisafakepassword"
    )

    return keyfile


@pytest.fixture(scope="session")
def keyfile_setup_teardown():
    root_path = f"/tmp/pytest{time.time()}"
    os.makedirs(root_path, exist_ok=True)

    create_keyfile(root_path)

    yield root_path

    shutil.rmtree(root_path)


def test_create(keyfile_setup_teardown):
    """
    Test case for creating a keyfile and performing various operations on it.
    """
    root_path = keyfile_setup_teardown
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))

    mnemonic = bittensor.Keypair.generate_mnemonic(12)
    alice = bittensor.Keypair.create_from_mnemonic(mnemonic)
    keyfile.set_keypair(
        alice, encrypt=True, overwrite=True, password="thisisafakepassword"
    )
    assert keyfile.is_readable()
    assert keyfile.is_writable()
    assert keyfile.is_encrypted()
    keyfile.decrypt(password="thisisafakepassword")
    assert not keyfile.is_encrypted()
    keyfile.encrypt(password="thisisafakepassword")
    assert keyfile.is_encrypted()
    str(keyfile)
    keyfile.decrypt(password="thisisafakepassword")
    assert not keyfile.is_encrypted()
    str(keyfile)

    assert (
        keyfile.get_keypair(password="thisisafakepassword").ss58_address
        == alice.ss58_address
    )
    assert (
        keyfile.get_keypair(password="thisisafakepassword").private_key
        == alice.private_key
    )
    assert (
        keyfile.get_keypair(password="thisisafakepassword").public_key
        == alice.public_key
    )

    bob = bittensor.Keypair.create_from_uri("/Bob")
    keyfile.set_keypair(
        bob, encrypt=True, overwrite=True, password="thisisafakepassword"
    )
    assert (
        keyfile.get_keypair(password="thisisafakepassword").ss58_address
        == bob.ss58_address
    )
    assert (
        keyfile.get_keypair(password="thisisafakepassword").public_key == bob.public_key
    )

    repr(keyfile)


def test_legacy_coldkey(keyfile_setup_teardown):
    """
    Test case for legacy cold keyfile.
    """
    root_path = keyfile_setup_teardown
    legacy_filename = os.path.join(root_path, "coldlegacy_keyfile")
    keyfile = bittensor.keyfile(path=legacy_filename)
    keyfile.make_dirs()
    keyfile_data = b"0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
    with open(legacy_filename, "wb") as keyfile_obj:
        keyfile_obj.write(keyfile_data)
    assert keyfile.keyfile_data == keyfile_data
    keyfile.encrypt(password="this is the fake password")
    keyfile.decrypt(password="this is the fake password")
    expected_decryption = {
        "accountId": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f",
        "publicKey": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f",
        "privateKey": None,
        "secretPhrase": None,
        "secretSeed": None,
        "ss58Address": "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm",
    }
    for key, value in expected_decryption.items():
        value_str = f'"{value}"' if value is not None else "null"
        assert f'"{key}": {value_str}'.encode() in keyfile.keyfile_data

    assert (
        keyfile.get_keypair().ss58_address
        == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    )
    assert (
        f"0x{keyfile.get_keypair().public_key.hex()}"
        == "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
    )


def test_validate_password():
    """
    Test case for the validate_password function.

    This function tests the behavior of the validate_password function from the bittensor.keyfile module.
    It checks various scenarios to ensure that the function correctly validates passwords.
    """
    from bittensor.keyfile import validate_password

    assert validate_password(None) == False
    assert validate_password("passw0rd") == False
    assert validate_password("123456789") == False
    with mock.patch("getpass.getpass", return_value="biTTensor"):
        assert validate_password("biTTensor") == True
    with mock.patch("getpass.getpass", return_value="biTTenso"):
        assert validate_password("biTTensor") == False


def test_decrypt_keyfile_data_legacy():
    """
    Test case for decrypting legacy keyfile data.

    This test case verifies that the `decrypt_keyfile_data` function correctly decrypts
    encrypted data using a legacy encryption scheme.

    The test generates a key using a password and encrypts a sample data. Then, it decrypts
    the encrypted data using the same password and asserts that the decrypted data matches
    the original data.
    """
    import base64

    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    from bittensor.keyfile import decrypt_keyfile_data

    __SALT = b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins"

    def __generate_key(password):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            salt=__SALT,
            length=32,
            iterations=10000000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    pw = "fakepasssword238947239"
    data = b"encrypt me!"
    key = __generate_key(pw)
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(data)

    decrypted_data = decrypt_keyfile_data(encrypted_data, pw)
    assert decrypted_data == data


def test_user_interface():
    """
    Test the user interface for asking password to encrypt.

    This test case uses the `ask_password_to_encrypt` function from the `bittensor.keyfile` module.
    It mocks the `getpass.getpass` function to simulate user input of passwords.
    The expected result is that the `ask_password_to_encrypt` function returns the correct password.
    """
    from bittensor.keyfile import ask_password_to_encrypt

    with mock.patch(
        "getpass.getpass",
        side_effect=["pass", "password", "asdury3294y", "asdury3294y"],
    ):
        assert ask_password_to_encrypt() == "asdury3294y"


def test_overwriting(keyfile_setup_teardown):
    """
    Test case for overwriting a keypair in the keyfile.
    """
    root_path = keyfile_setup_teardown
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))
    alice = bittensor.Keypair.create_from_uri("/Alice")
    keyfile.set_keypair(
        alice, encrypt=True, overwrite=True, password="thisisafakepassword"
    )
    bob = bittensor.Keypair.create_from_uri("/Bob")

    with pytest.raises(bittensor.KeyFileError) as pytest_wrapped_e:
        with mock.patch("builtins.input", return_value="n"):
            keyfile.set_keypair(
                bob, encrypt=True, overwrite=False, password="thisisafakepassword"
            )


def test_serialized_keypair_to_keyfile_data(keyfile_setup_teardown):
    """
    Test case for serializing a keypair to keyfile data.

    This test case verifies that the `serialized_keypair_to_keyfile_data` function correctly
    serializes a keypair to keyfile data. It then deserializes the keyfile data and asserts
    that the deserialized keypair matches the original keypair.
    """
    from bittensor.keyfile import serialized_keypair_to_keyfile_data

    root_path = keyfile_setup_teardown
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))

    mnemonic = bittensor.Keypair.generate_mnemonic(12)
    keypair = bittensor.Keypair.create_from_mnemonic(mnemonic)

    keyfile.set_keypair(
        keypair, encrypt=True, overwrite=True, password="thisisafakepassword"
    )
    keypair_data = serialized_keypair_to_keyfile_data(keypair)
    decoded_keypair_data = json.loads(keypair_data.decode())

    assert decoded_keypair_data["secretPhrase"] == keypair.mnemonic
    assert decoded_keypair_data["ss58Address"] == keypair.ss58_address
    assert decoded_keypair_data["publicKey"] == f"0x{keypair.public_key.hex()}"
    assert decoded_keypair_data["accountId"] == f"0x{keypair.public_key.hex()}"


def test_deserialize_keypair_from_keyfile_data(keyfile_setup_teardown):
    """
    Test case for deserializing a keypair from keyfile data.

    This test case verifies that the `deserialize_keypair_from_keyfile_data` function correctly
    deserializes keyfile data to a keypair. It first serializes a keypair to keyfile data and
    then deserializes the keyfile data to a keypair. It then asserts that the deserialized keypair
    matches the original keypair.
    """
    from bittensor.keyfile import serialized_keypair_to_keyfile_data
    from bittensor.keyfile import deserialize_keypair_from_keyfile_data

    root_path = keyfile_setup_teardown
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))

    mnemonic = bittensor.Keypair.generate_mnemonic(12)
    keypair = bittensor.Keypair.create_from_mnemonic(mnemonic)

    keyfile.set_keypair(
        keypair, encrypt=True, overwrite=True, password="thisisafakepassword"
    )
    keypair_data = serialized_keypair_to_keyfile_data(keypair)
    deserialized_keypair = deserialize_keypair_from_keyfile_data(keypair_data)

    assert deserialized_keypair.ss58_address == keypair.ss58_address
    assert deserialized_keypair.public_key == keypair.public_key
    assert deserialized_keypair.private_key == keypair.private_key


def test_get_coldkey_password_from_environment(monkeypatch):
    password_by_wallet = {
        "WALLET": "password",
        "my_wallet": "password2",
        "my-wallet": "password2",
    }

    monkeypatch.setenv("bt_cold_pw_wallet", password_by_wallet["WALLET"])
    monkeypatch.setenv("BT_COLD_PW_My_Wallet", password_by_wallet["my_wallet"])

    for wallet, password in password_by_wallet.items():
        assert get_coldkey_password_from_environment(wallet) == password

    assert get_coldkey_password_from_environment("non_existent_wallet") is None


def test_keyfile_error_incorrect_password(keyfile_setup_teardown):
    """
    Test case for attempting to decrypt a keyfile with an incorrect password.
    """
    root_path = keyfile_setup_teardown
    keyfile = bittensor.keyfile(path=os.path.join(root_path, "keyfile"))

    # Ensure the keyfile is encrypted
    assert keyfile.is_encrypted()

    # Attempt to decrypt with an incorrect password
    with pytest.raises(bittensor.KeyFileError) as excinfo:
        keyfile.get_keypair(password="incorrect_password")

    assert "Invalid password" in str(excinfo.value)
