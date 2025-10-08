from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Wallets:
    """Class for managing coldkey, hotkey, wallet operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.does_hotkey_exist = subtensor.does_hotkey_exist
        self.filter_netuids_by_registered_hotkeys = (
            subtensor.filter_netuids_by_registered_hotkeys
        )
        self.is_hotkey_registered_any = subtensor.is_hotkey_registered_any
        self.is_hotkey_registered = subtensor.is_hotkey_registered
        self.is_hotkey_registered_on_subnet = subtensor.is_hotkey_registered_on_subnet
        self.is_hotkey_delegate = subtensor.is_hotkey_delegate
        self.get_balance = subtensor.get_balance
        self.get_balances = subtensor.get_balances
        self.get_children = subtensor.get_children
        self.get_children_pending = subtensor.get_children_pending
        self.get_delegate_by_hotkey = subtensor.get_delegate_by_hotkey
        self.get_delegate_take = subtensor.get_delegate_take
        self.get_delegated = subtensor.get_delegated
        self.get_hotkey_owner = subtensor.get_hotkey_owner
        self.get_hotkey_stake = subtensor.get_hotkey_stake
        self.get_minimum_required_stake = subtensor.get_minimum_required_stake
        self.get_netuids_for_hotkey = subtensor.get_netuids_for_hotkey
        self.get_owned_hotkeys = subtensor.get_owned_hotkeys
        self.get_parents = subtensor.get_parents
        self.get_stake = subtensor.get_stake
        self.get_stake_add_fee = subtensor.get_stake_add_fee
        self.get_stake_for_coldkey_and_hotkey = (
            subtensor.get_stake_for_coldkey_and_hotkey
        )
        self.get_stake_for_hotkey = subtensor.get_stake_for_hotkey
        self.get_stake_info_for_coldkey = subtensor.get_stake_info_for_coldkey
        self.get_stake_movement_fee = subtensor.get_stake_movement_fee
        self.get_transfer_fee = subtensor.get_transfer_fee
        self.get_unstake_fee = subtensor.get_unstake_fee
        self.set_children = subtensor.set_children
        self.transfer = subtensor.transfer
