# The MIT License (MIT)

# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies

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
import bittensor
from typing import Optional
from Crypto.Hash import keccak

from .keyfile_mock import MockKeyfile


class MockWallet(bittensor.wallet):
    """
    Mocked Version of the bittensor wallet class, meant to be used for testing
    """

    def __init__(self, **kwargs):
        r"""Init bittensor wallet object containing a hot and coldkey.
        Args:
            _mock (required=True, default=False):
                If true creates a mock wallet with random keys.
        """
        super().__init__(**kwargs)
        # For mocking.
        self._is_mock = True
        self._mocked_coldkey_keyfile = None
        self._mocked_hotkey_keyfile = None

    @property
    def hotkey_file(self) -> "bittensor.keyfile":
        if self._is_mock:
            if self._mocked_hotkey_keyfile == None:
                self._mocked_hotkey_keyfile = MockKeyfile(path="MockedHotkey")
            return self._mocked_hotkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
            return bittensor.keyfile(path=hotkey_path)

    @property
    def coldkey_file(self) -> "bittensor.keyfile":
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = MockKeyfile(path="MockedColdkey")
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkey_path = os.path.join(wallet_path, "coldkey")
            return bittensor.keyfile(path=coldkey_path)

    @property
    def coldkeypub_file(self) -> "bittensor.keyfile":
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = MockKeyfile(path="MockedColdkeyPub")
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
            return bittensor.keyfile(path=coldkeypub_path)


def get_mock_wallet(
    coldkey: "bittensor.Keypair" = None, hotkey: "bittensor.Keypair" = None
):
    wallet = MockWallet(name="mock_wallet", hotkey="mock", path="/tmp/mock_wallet")

    if not coldkey:
        coldkey = bittensor.Keypair.create_from_mnemonic(
            bittensor.Keypair.generate_mnemonic()
        )
    if not hotkey:
        hotkey = bittensor.Keypair.create_from_mnemonic(
            bittensor.Keypair.generate_mnemonic()
        )

    wallet.set_coldkey(coldkey, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=True)
    wallet.set_hotkey(hotkey, encrypt=False, overwrite=True)

    return wallet


def get_mock_keypair(uid: int, test_name: Optional[str] = None) -> bittensor.Keypair:
    """
    Returns a mock keypair from a uid and optional test_name.
    If test_name is not provided, the uid is the only seed.
    If test_name is provided, the uid is hashed with the test_name to create a unique seed for the test.
    """
    if test_name is not None:
        hashed_test_name: bytes = keccak.new(
            digest_bits=256, data=test_name.encode("utf-8")
        ).digest()
        hashed_test_name_as_int: int = int.from_bytes(
            hashed_test_name, byteorder="big", signed=False
        )
        uid = uid + hashed_test_name_as_int

    return bittensor.Keypair.create_from_seed(
        seed_hex=int.to_bytes(uid, 32, "big", signed=False),
        ss58_format=bittensor.__ss58_format__,
    )


def get_mock_hotkey(uid: int) -> str:
    return get_mock_keypair(uid).ss58_address


def get_mock_coldkey(uid: int) -> str:
    return get_mock_keypair(uid).ss58_address
