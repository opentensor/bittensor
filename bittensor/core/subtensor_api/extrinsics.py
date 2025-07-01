from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Extrinsics:
    """Class for managing extrinsic operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.add_liquidity = subtensor.add_liquidity
        self.add_stake = subtensor.add_stake
        self.add_stake_multiple = subtensor.add_stake_multiple
        self.burned_register = subtensor.burned_register
        self.commit_weights = subtensor.commit_weights
        self.modify_liquidity = subtensor.modify_liquidity
        self.move_stake = subtensor.move_stake
        self.register = subtensor.register
        self.register_subnet = subtensor.register_subnet
        self.remove_liquidity = subtensor.remove_liquidity
        self.reveal_weights = subtensor.reveal_weights
        self.root_register = subtensor.root_register
        self.root_set_weights = subtensor.root_set_weights
        self.root_set_pending_childkey_cooldown = (
            subtensor.root_set_pending_childkey_cooldown
        )
        self.set_children = subtensor.set_children
        self.set_subnet_identity = subtensor.set_subnet_identity
        self.set_weights = subtensor.set_weights
        self.serve_axon = subtensor.serve_axon
        self.start_call = subtensor.start_call
        self.swap_stake = subtensor.swap_stake
        self.toggle_user_liquidity = subtensor.toggle_user_liquidity
        self.transfer = subtensor.transfer
        self.transfer_stake = subtensor.transfer_stake
        self.unstake = subtensor.unstake
        self.unstake_multiple = subtensor.unstake_multiple
