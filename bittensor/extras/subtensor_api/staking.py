from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Staking:
    """Class for managing staking operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.add_stake = subtensor.add_stake
        self.add_stake_burn = subtensor.add_stake_burn
        self.add_stake_multiple = subtensor.add_stake_multiple
        self.claim_root = subtensor.claim_root
        self.get_auto_stakes = subtensor.get_auto_stakes
        self.get_coldkey_lock = subtensor.get_coldkey_lock
        self.get_hotkey_conviction = subtensor.get_hotkey_conviction
        self.get_hotkey_stake = subtensor.get_hotkey_stake
        self.get_minimum_required_stake = subtensor.get_minimum_required_stake
        self.get_most_convicted_hotkey_on_subnet = (
            subtensor.get_most_convicted_hotkey_on_subnet
        )
        self.get_root_alpha_dividends_per_subnet = (
            subtensor.get_root_alpha_dividends_per_subnet
        )
        self.get_root_claim_type = subtensor.get_root_claim_type
        self.get_root_claimable_all_rates = subtensor.get_root_claimable_all_rates
        self.get_root_claimable_rate = subtensor.get_root_claimable_rate
        self.get_root_claimable_stake = subtensor.get_root_claimable_stake
        self.get_root_claimed = subtensor.get_root_claimed
        self.get_stake = subtensor.get_stake
        self.get_stake_add_fee = subtensor.get_stake_add_fee
        self.get_stake_for_coldkey_and_hotkey = (
            subtensor.get_stake_for_coldkey_and_hotkey
        )
        self.get_stake_info_for_coldkey = subtensor.get_stake_info_for_coldkey
        self.get_stake_info_for_coldkeys = subtensor.get_stake_info_for_coldkeys
        self.get_stake_lock = subtensor.get_stake_lock
        self.get_stake_locks = subtensor.get_stake_locks
        self.get_stake_movement_fee = subtensor.get_stake_movement_fee
        self.get_stake_weight = subtensor.get_stake_weight
        self.get_staking_hotkeys = subtensor.get_staking_hotkeys
        self.get_unstake_fee = subtensor.get_unstake_fee
        self.is_perpetual_lock = subtensor.is_perpetual_lock
        self.lock_stake = subtensor.lock_stake
        self.move_lock = subtensor.move_lock
        self.move_stake = subtensor.move_stake
        self.set_auto_stake = subtensor.set_auto_stake
        self.set_perpetual_lock = subtensor.set_perpetual_lock
        self.set_root_claim_type = subtensor.set_root_claim_type
        self.sim_swap = subtensor.sim_swap
        self.swap_stake = subtensor.swap_stake
        self.transfer_stake = subtensor.transfer_stake
        self.unstake = subtensor.unstake
        self.unstake_all = subtensor.unstake_all
        self.unstake_multiple = subtensor.unstake_multiple
