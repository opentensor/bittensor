from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Commitments:
    """Class for managing any commitment operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.commit_reveal_enabled = subtensor.commit_reveal_enabled
        self.get_all_commitments = subtensor.get_all_commitments
        self.get_all_revealed_commitments = subtensor.get_all_revealed_commitments
        self.get_commitment = subtensor.get_commitment
        self.get_current_weight_commit_info = subtensor.get_current_weight_commit_info
        self.get_revealed_commitment = subtensor.get_revealed_commitment
        self.get_revealed_commitment_by_hotkey = (
            subtensor.get_revealed_commitment_by_hotkey
        )
        self.set_commitment = subtensor.set_commitment
        self.set_reveal_commitment = subtensor.set_reveal_commitment
