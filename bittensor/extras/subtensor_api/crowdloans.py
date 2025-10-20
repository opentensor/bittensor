from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Crowdloans:
    """Class for managing any Crowdloans operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.contribute_crowdloan = subtensor.contribute_crowdloan
        self.create_crowdloan = subtensor.create_crowdloan
        self.dissolve_crowdloan = subtensor.dissolve_crowdloan
        self.finalize_crowdloan = subtensor.finalize_crowdloan
        self.get_crowdloan_constants = subtensor.get_crowdloan_constants
        self.get_crowdloan_contributions = subtensor.get_crowdloan_contributions
        self.get_crowdloan_by_id = subtensor.get_crowdloan_by_id
        self.get_crowdloan_next_id = subtensor.get_crowdloan_next_id
        self.get_crowdloans = subtensor.get_crowdloans
        self.refund_crowdloan = subtensor.refund_crowdloan
        self.update_cap_crowdloan = subtensor.update_cap_crowdloan
        self.update_end_crowdloan = subtensor.update_end_crowdloan
        self.update_min_contribution_crowdloan = (
            subtensor.update_min_contribution_crowdloan
        )
        self.withdraw_crowdloan = subtensor.withdraw_crowdloan
