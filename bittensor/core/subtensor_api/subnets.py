from typing import Union

from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor
from bittensor.core.subtensor import Subtensor as _Subtensor


class Subnets:
    """Class for managing subnet operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.all_subnets = subtensor.all_subnets
        self.blocks_since_last_step = subtensor.blocks_since_last_step
        self.blocks_since_last_update = subtensor.blocks_since_last_update
        self.bonds = subtensor.bonds
        self.difficulty = subtensor.difficulty
        self.get_all_subnets_info = subtensor.get_all_subnets_info
        self.get_children = subtensor.get_children
        self.get_children_pending = subtensor.get_children_pending
        self.get_current_weight_commit_info = subtensor.get_current_weight_commit_info
        self.get_hyperparameter = subtensor.get_hyperparameter
        self.get_neuron_for_pubkey_and_subnet = (
            subtensor.get_neuron_for_pubkey_and_subnet
        )
        self.get_next_epoch_start_block = subtensor.get_next_epoch_start_block
        self.get_subnet_burn_cost = subtensor.get_subnet_burn_cost
        self.get_subnet_hyperparameters = subtensor.get_subnet_hyperparameters
        self.get_subnet_info = subtensor.get_subnet_info
        self.get_subnet_owner_hotkey = subtensor.get_subnet_owner_hotkey
        self.get_subnet_reveal_period_epochs = subtensor.get_subnet_reveal_period_epochs
        self.get_subnet_validator_permits = subtensor.get_subnet_validator_permits
        self.get_subnets = subtensor.get_subnets
        self.get_total_subnets = subtensor.get_total_subnets
        self.get_uid_for_hotkey_on_subnet = subtensor.get_uid_for_hotkey_on_subnet
        self.immunity_period = subtensor.immunity_period
        self.is_hotkey_registered_on_subnet = subtensor.is_hotkey_registered_on_subnet
        self.is_subnet_active = subtensor.is_subnet_active
        self.max_weight_limit = subtensor.max_weight_limit
        self.min_allowed_weights = subtensor.min_allowed_weights
        self.recycle = subtensor.recycle
        self.register_subnet = subtensor.register_subnet
        self.set_subnet_identity = subtensor.set_subnet_identity
        self.subnet = subtensor.subnet
        self.subnet_exists = subtensor.subnet_exists
        self.subnetwork_n = subtensor.subnetwork_n
        self.tempo = subtensor.tempo
        self.weights_rate_limit = subtensor.weights_rate_limit
        self.weights = subtensor.weights
