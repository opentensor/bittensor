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
import time
import pytest
import shutil
import unittest
import bittensor
import unittest.mock as mock
from scalecodec import ScaleBytes
from substrateinterface import Keypair, KeypairType
from substrateinterface.constants import DEV_PHRASE
from substrateinterface.exceptions import ConfigurationError
from bip39 import bip39_validate


class KeyPairTestCase(unittest.TestCase):
    """
    Test case for the KeyPair class.
    """

    def test_generate_mnemonic(self):
        """
        Test the generation of a mnemonic and its validation.
        """
        mnemonic = Keypair.generate_mnemonic()
        self.assertTrue(bip39_validate(mnemonic))

    def test_invalid_mnemonic(self):
        """
        Test the validation of an invalid mnemonic.
        """
        mnemonic = "This is an invalid mnemonic"
        self.assertFalse(bip39_validate(mnemonic))

    def test_create_sr25519_keypair(self):
        """
        Test the creation of a sr25519 keypair from a mnemonic and verify the SS58 address.
        """
        mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
        keypair = Keypair.create_from_mnemonic(mnemonic, ss58_format=0)
        self.assertEqual(
            keypair.ss58_address, "16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2"
        )

    def test_only_provide_ss58_address(self):
        """
        Test the creation of a keypair with only the SS58 address provided.
        """
        keypair = Keypair(
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2"
        )
        self.assertEqual(
            "0x" + keypair.public_key.hex(),
            "0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446",
        )

    def test_only_provide_public_key(self):
        """
        Test the creation of a keypair with only the public key provided.
        """
        keypair = Keypair(
            public_key="0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446",
            ss58_format=0,
        )
        self.assertEqual(
            keypair.ss58_address, "16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2"
        )

    def test_provide_no_ss58_address_and_public_key(self):
        """
        Test the creation of a keypair without providing SS58 address and public key.
        """
        self.assertRaises(ValueError, Keypair)

    def test_incorrect_private_key_length_sr25519(self):
        """
        Test the creation of a keypair with an incorrect private key length for sr25519.
        """
        self.assertRaises(
            ValueError,
            Keypair,
            private_key="0x23",
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
        )

    def test_incorrect_public_key(self):
        """
        Test the creation of a keypair with an incorrect public key.
        """
        self.assertRaises(ValueError, Keypair, public_key="0x23")

    def test_sign_and_verify(self):
        """
        Test the signing and verification of a message using a keypair.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        signature = keypair.sign("Test1231223123123")
        self.assertTrue(keypair.verify("Test1231223123123", signature))

    def test_sign_and_verify_hex_data(self):
        """
        Test the signing and verification of hex data using a keypair.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        signature = keypair.sign("0x1234")
        self.assertTrue(keypair.verify("0x1234", signature))

    def test_sign_and_verify_scale_bytes(self):
        """
        Test the signing and verification of ScaleBytes data using a keypair.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        data = ScaleBytes("0x1234")
        signature = keypair.sign(data)
        self.assertTrue(keypair.verify(data, signature))

    def test_sign_missing_private_key(self):
        """
        Test signing a message with a keypair that is missing the private key.
        """
        keypair = Keypair(
            ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        self.assertRaises(ConfigurationError, keypair.sign, "0x1234")

    def test_sign_unsupported_crypto_type(self):
        """
        Test signing a message with an unsupported crypto type.
        """
        keypair = Keypair.create_from_private_key(
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
            private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
            crypto_type=3,
        )
        self.assertRaises(ConfigurationError, keypair.sign, "0x1234")

    def test_verify_unsupported_crypto_type(self):
        """
        Test verifying a signature with an unsupported crypto type.
        """
        keypair = Keypair.create_from_private_key(
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
            private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
            crypto_type=3,
        )
        self.assertRaises(ConfigurationError, keypair.verify, "0x1234", "0x1234")

    def test_sign_and_verify_incorrect_signature(self):
        """
        Test verifying an incorrect signature for a signed message.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        signature = "0x4c291bfb0bb9c1274e86d4b666d13b2ac99a0bacc04a4846fb8ea50bda114677f83c1f164af58fc184451e5140cc8160c4de626163b11451d3bbb208a1889f8a"
        self.assertFalse(keypair.verify("Test1231223123123", signature))

    def test_sign_and_verify_invalid_signature(self):
        """
        Test verifying an invalid signature format for a signed message.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        signature = "Test"
        self.assertRaises(TypeError, keypair.verify, "Test1231223123123", signature)

    def test_sign_and_verify_invalid_message(self):
        """
        Test verifying a signature against an incorrect message.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        signature = keypair.sign("Test1231223123123")
        self.assertFalse(keypair.verify("OtherMessage", signature))

    def test_create_ed25519_keypair(self):
        """
        Test the creation of an ed25519 keypair from a mnemonic and verify the SS58 address.
        """
        mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
        keypair = Keypair.create_from_mnemonic(
            mnemonic, ss58_format=0, crypto_type=KeypairType.ED25519
        )
        self.assertEqual(
            keypair.ss58_address, "16dYRUXznyhvWHS1ktUENGfNAEjCawyDzHRtN9AdFnJRc38h"
        )

    def test_sign_and_verify_ed25519(self):
        """
        Test the signing and verification of a message using an ed25519 keypair.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(
            mnemonic, crypto_type=KeypairType.ED25519
        )
        signature = keypair.sign("Test1231223123123")
        self.assertTrue(keypair.verify("Test1231223123123", signature))

    def test_sign_and_verify_invalid_signature_ed25519(self):
        """
        Test verifying an incorrect signature for a message signed with an ed25519 keypair.
        """
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(
            mnemonic, crypto_type=KeypairType.ED25519
        )
        signature = "0x4c291bfb0bb9c1274e86d4b666d13b2ac99a0bacc04a4846fb8ea50bda114677f83c1f164af58fc184451e5140cc8160c4de626163b11451d3bbb208a1889f8a"
        self.assertFalse(keypair.verify("Test1231223123123", signature))

    def test_unsupport_crypto_type(self):
        """
        Test creating a keypair with an unsupported crypto type.
        """
        self.assertRaises(
            ValueError,
            Keypair.create_from_seed,
            seed_hex="0xda3cf5b1e9144931?a0f0db65664aab662673b099415a7f8121b7245fb0be4143",
            crypto_type=2,
        )

    def test_create_keypair_from_private_key(self):
        """
        Test creating a keypair from a private key and verify the public key.
        """
        keypair = Keypair.create_from_private_key(
            ss58_address="16ADqpMa4yzfmWs3nuTSMhfZ2ckeGtvqhPWCNqECEGDcGgU2",
            private_key="0x1f1995bdf3a17b60626a26cfe6f564b337d46056b7a1281b64c649d592ccda0a9cffd34d9fb01cae1fba61aeed184c817442a2186d5172416729a4b54dd4b84e",
        )
        self.assertEqual(
            "0x" + keypair.public_key.hex(),
            "0xe4359ad3e2716c539a1d663ebd0a51bdc5c98a12e663bb4c4402db47828c9446",
        )

    def test_hdkd_hard_path(self):
        """
        Test hierarchical deterministic key derivation with a hard derivation path.
        """
        mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
        derivation_address = "5FEiH8iuDUw271xbqWTWuB6WrDjv5dnCeDX1CyHubAniXDNN"
        derivation_path = "//Alice"
        derived_keypair = Keypair.create_from_uri(mnemonic + derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_soft_path(self):
        """
        Test hierarchical deterministic key derivation with a soft derivation path.
        """
        mnemonic = "old leopard transfer rib spatial phone calm indicate online fire caution review"
        derivation_address = "5GNXbA46ma5dg19GXdiKi5JH3mnkZ8Yea3bBtZAvj7t99P9i"
        derivation_path = "/Alice"
        derived_keypair = Keypair.create_from_uri(mnemonic + derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_default_to_dev_mnemonic(self):
        """
        Test hierarchical deterministic key derivation with a default development mnemonic.
        """
        derivation_address = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        derivation_path = "//Alice"
        derived_keypair = Keypair.create_from_uri(derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_nested_hard_soft_path(self):
        """
        Test hierarchical deterministic key derivation with a nested hard and soft derivation path.
        """
        derivation_address = "5CJGwWiKXSE16WJaxBdPZhWqUYkotgenLUALv7ZvqQ4TXeqf"
        derivation_path = "//Bob/test"
        derived_keypair = Keypair.create_from_uri(derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_nested_soft_hard_path(self):
        """
        Test hierarchical deterministic key derivation with a nested soft and hard derivation path.
        """
        derivation_address = "5Cwc8tShrshDJUp1P1M21dKUTcYQpV9GcfSa4hUBNmMdV3Cx"
        derivation_path = "/Bob//test"
        derived_keypair = Keypair.create_from_uri(derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_path_gt_32_bytes(self):
        """
        Test hierarchical deterministic key derivation with a derivation path longer than 32 bytes.
        """
        derivation_address = "5GR5pfZeNs1uQiSWVxZaQiZou3wdZiX894eqgvfNfHbEh7W2"
        derivation_path = "//PathNameLongerThan32BytesWhichShouldBeHashed"
        derived_keypair = Keypair.create_from_uri(derivation_path)
        self.assertEqual(derivation_address, derived_keypair.ss58_address)

    def test_hdkd_unsupported_password(self):
        """
        Test hierarchical deterministic key derivation with an unsupported password.
        """
        self.assertRaises(
            NotImplementedError, Keypair.create_from_uri, DEV_PHRASE + "///test"
        )


class TestKeyFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = f"/tmp/pytest{time.time()}"
        os.makedirs(self.root_path, exist_ok=True)

        self.create_keyfile()

    def tearDown(self) -> None:
        shutil.rmtree(self.root_path)

    def create_keyfile(self):
        keyfile = bittensor.keyfile(path=os.path.join(self.root_path, "keyfile"))

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

    def test_create(self):
        keyfile = bittensor.keyfile(path=os.path.join(self.root_path, "keyfile"))

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
            keyfile.get_keypair(password="thisisafakepassword").public_key
            == bob.public_key
        )

        repr(keyfile)

    def test_legacy_coldkey(self):
        legacy_filename = os.path.join(self.root_path, "coldlegacy_keyfile")
        keyfile = bittensor.keyfile(path=legacy_filename)
        keyfile.make_dirs()
        keyfile_data = (
            b"0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        )
        with open(legacy_filename, "wb") as keyfile_obj:
            keyfile_obj.write(keyfile_data)
        assert keyfile.keyfile_data == keyfile_data
        keyfile.encrypt(password="this is the fake password")
        keyfile.decrypt(password="this is the fake password")
        keypair_bytes = b'{"accountId": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "publicKey": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "secretPhrase": null, "secretSeed": null, "ss58Address": "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"}'
        assert keyfile.keyfile_data == keypair_bytes
        assert (
            keyfile.get_keypair().ss58_address
            == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        )
        assert (
            "0x" + keyfile.get_keypair().public_key.hex()
            == "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
        )

    def test_validate_password(self):
        from bittensor.keyfile import validate_password

        assert validate_password(None) == False
        assert validate_password("passw0rd") == False
        assert validate_password("123456789") == False
        with mock.patch("getpass.getpass", return_value="biTTensor"):
            assert validate_password("biTTensor") == True
        with mock.patch("getpass.getpass", return_value="biTTenso"):
            assert validate_password("biTTensor") == False

    def test_decrypt_keyfile_data_legacy(self):
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

    def test_user_interface(self):
        from bittensor.keyfile import ask_password_to_encrypt

        with mock.patch(
            "getpass.getpass",
            side_effect=["pass", "password", "asdury3294y", "asdury3294y"],
        ):
            assert ask_password_to_encrypt() == "asdury3294y"

    def test_overwriting(self):
        keyfile = bittensor.keyfile(path=os.path.join(self.root_path, "keyfile"))
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
