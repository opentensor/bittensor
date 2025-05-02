from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Delegates:
    """Class for managing delegate operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.is_hotkey_delegate = subtensor.is_hotkey_delegate
        self.get_delegate_by_hotkey = subtensor.get_delegate_by_hotkey
        self.set_delegate_take = subtensor.set_delegate_take
        self.get_delegate_identities = subtensor.get_delegate_identities
        self.get_delegate_take = subtensor.get_delegate_take
        self.get_delegated = subtensor.get_delegated
        self.get_delegates = subtensor.get_delegates
