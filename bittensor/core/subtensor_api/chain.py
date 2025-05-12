from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Chain:
    """Class for managing chain state operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.get_block_hash = subtensor.get_block_hash
        self.get_current_block = subtensor.get_current_block
        self.get_delegate_identities = subtensor.get_delegate_identities
        self.get_existential_deposit = subtensor.get_existential_deposit
        self.get_minimum_required_stake = subtensor.get_minimum_required_stake
        self.get_vote_data = subtensor.get_vote_data
        self.get_timestamp = subtensor.get_timestamp
        self.is_fast_blocks = subtensor.is_fast_blocks
        self.last_drand_round = subtensor.last_drand_round
        self.state_call = subtensor.state_call
        self.tx_rate_limit = subtensor.tx_rate_limit
