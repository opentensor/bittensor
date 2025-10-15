from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Crowdloans:
    """Class for managing any Crowdloans operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.get_crowdloan_constants = subtensor.get_crowdloan_constants
        self.get_crowdloan_contributions = subtensor.get_crowdloan_contributions
        self.get_crowdloan_by_id = subtensor.get_crowdloan_by_id
        self.get_crowdloan_next_id = subtensor.get_crowdloan_next_id
        self.get_crowdloans = subtensor.get_crowdloans
