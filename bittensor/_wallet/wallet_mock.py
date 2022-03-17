
from . import wallet_impl
import os
import bittensor

class Wallet_mock(wallet_impl.Wallet):
    """
    Mocked Version of the bittensor wallet class, meant to be used for testing
    """
    def __init__( 
        self,
        _mock:bool,
        **kwargs,
    ):
        r""" Init bittensor wallet object containing a hot and coldkey.
            Args:
                _mock (required=True, default=False):
                    If true creates a mock wallet with random keys.
        """
        super().__init__(**kwargs)
        # For mocking.
        self._is_mock = _mock
        self._mocked_coldkey_keyfile = None
        self._mocked_hotkey_keyfile = None

        print("---- MOCKED WALLET INITIALIZED- ---")

    @property
    def hotkey_file(self) -> 'bittensor.Keyfile':
        if self._is_mock:
            if self._mocked_hotkey_keyfile == None:
                self._mocked_hotkey_keyfile = bittensor.keyfile(path='MockedHotkey', _mock = True)
            return self._mocked_hotkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
            return bittensor.keyfile( path = hotkey_path )

    @property
    def coldkey_file(self) -> 'bittensor.Keyfile':
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = bittensor.keyfile(path='MockedColdkey', _mock = True)
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkey_path = os.path.join(wallet_path, "coldkey")
            return bittensor.keyfile( path = coldkey_path )

    @property
    def coldkeypub_file(self) -> 'bittensor.Keyfile':
        if self._is_mock:
            if self._mocked_coldkey_keyfile == None:
                self._mocked_coldkey_keyfile = bittensor.keyfile(path='MockedColdkeyPub', _mock = True)
            return self._mocked_coldkey_keyfile
        else:
            wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
            coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
            return bittensor.Keyfile( path = coldkeypub_path )