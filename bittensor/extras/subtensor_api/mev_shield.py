from typing import Union

from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor
from bittensor.core.subtensor import Subtensor as _Subtensor


class MevShield:
    """Class for managing MEV Shield operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        # Storage queries
        self.get_mev_shield_current_key = subtensor.get_mev_shield_current_key
        self.get_mev_shield_next_key = subtensor.get_mev_shield_next_key
        self.get_mev_shield_submission = subtensor.get_mev_shield_submission
        self.get_mev_shield_submissions = subtensor.get_mev_shield_submissions

        # Extrinsics
        self.mev_submit_encrypted = subtensor.mev_submit_encrypted
