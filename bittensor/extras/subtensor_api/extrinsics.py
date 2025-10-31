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
        self.claim_root = subtensor.claim_root
        self.commit_weights = subtensor.commit_weights
        self.contribute_crowdloan = subtensor.contribute_crowdloan
        self.create_crowdloan = subtensor.create_crowdloan
        self.dissolve_crowdloan = subtensor.dissolve_crowdloan
        self.finalize_crowdloan = subtensor.finalize_crowdloan
        self.get_extrinsic_fee = subtensor.get_extrinsic_fee
        self.modify_liquidity = subtensor.modify_liquidity
        self.move_stake = subtensor.move_stake
        self.refund_crowdloan = subtensor.refund_crowdloan
        self.register = subtensor.register
        self.register_subnet = subtensor.register_subnet
        self.remove_liquidity = subtensor.remove_liquidity
        self.reveal_weights = subtensor.reveal_weights
        self.root_register = subtensor.root_register
        self.root_set_pending_childkey_cooldown = (
            subtensor.root_set_pending_childkey_cooldown
        )
        self.set_children = subtensor.set_children
        self.set_subnet_identity = subtensor.set_subnet_identity
        self.set_weights = subtensor.set_weights
        self.serve_axon = subtensor.serve_axon
        self.set_commitment = subtensor.set_commitment
        self.set_root_claim_type = subtensor.set_root_claim_type
        self.start_call = subtensor.start_call
        self.swap_stake = subtensor.swap_stake
        self.toggle_user_liquidity = subtensor.toggle_user_liquidity
        self.transfer = subtensor.transfer
        self.transfer_stake = subtensor.transfer_stake
        self.unstake = subtensor.unstake
        self.unstake_all = subtensor.unstake_all
        self.unstake_multiple = subtensor.unstake_multiple
        self.update_cap_crowdloan = subtensor.update_cap_crowdloan
        self.update_end_crowdloan = subtensor.update_end_crowdloan
        self.update_min_contribution_crowdloan = (
            subtensor.update_min_contribution_crowdloan
        )
        self.validate_extrinsic_params = subtensor.validate_extrinsic_params
        self.withdraw_crowdloan = subtensor.withdraw_crowdloan
