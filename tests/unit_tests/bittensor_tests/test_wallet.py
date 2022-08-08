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
from unittest.mock import patch
import pytest
import bittensor

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
