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
import bittensor_wallet

from .keyfile_mock import MockKeyfile

class MockWallet(bittensor_wallet.Wallet):
    """
    Mocked Version of the bittensor wallet class, meant to be used for testing
    """
    def __init__(
        self,
        **kwargs,
    ):
        r""" Init bittensor wallet object containing a hot and coldkey.
            Args:
                _mock (required=True, default=False):
                    If true creates a mock wallet with random keys.
        """
        super().__init__(**kwargs)
        # For mocking.
        self._is_mock = True
        self._mocked_coldkey_keyfile = None
        self._mocked_hotkey_keyfile = None

        print("---- MOCKED WALLET INITIALIZED- ---")

    @property
    def hotkey_file(self) -> 'bittensor_wallet.Keyfile':
        if self._is_mock:
            if self._mocked_hotkey_keyfile == None:
                self._mocked_hotkey_keyfile = MockKeyfile(path='MockedHotkey')
            return self._mocked_hotkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
            return bittensor.keyfile( path = hotkey_path )

    @property
    def coldkey_file(self) -> 'bittensor_wallet.Keyfile':
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = MockKeyfile(path='MockedColdkey')
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkey_path = os.path.join(wallet_path, "coldkey")
            return bittensor.keyfile( path = coldkey_path )

    @property
    def coldkeypub_file(self) -> 'bittensor_wallet.Keyfile':
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = MockKeyfile(path='MockedColdkeyPub')
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
            return bittensor_wallet.Keyfile( path = coldkeypub_path )