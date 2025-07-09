from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittensor.core.subtensor_api import SubtensorApi


def add_legacy_methods(subtensor: "SubtensorApi"):
    """If SubtensorApi get `subtensor_fields=True` arguments, then all classic Subtensor fields added to root level."""
    subtensor.add_liquidity = subtensor._subtensor.add_liquidity
    subtensor.add_stake = subtensor._subtensor.add_stake
    subtensor.add_stake_multiple = subtensor._subtensor.add_stake_multiple
    subtensor.all_subnets = subtensor._subtensor.all_subnets
    subtensor.blocks_since_last_step = subtensor._subtensor.blocks_since_last_step
    subtensor.blocks_since_last_update = subtensor._subtensor.blocks_since_last_update
    subtensor.bonds = subtensor._subtensor.bonds
    subtensor.burned_register = subtensor._subtensor.burned_register
    subtensor.chain_endpoint = subtensor._subtensor.chain_endpoint
    subtensor.commit = subtensor._subtensor.commit
    subtensor.commit_reveal_enabled = subtensor._subtensor.commit_reveal_enabled
    subtensor.commit_weights = subtensor._subtensor.commit_weights
    subtensor.determine_block_hash = subtensor._subtensor.determine_block_hash
    subtensor.difficulty = subtensor._subtensor.difficulty
    subtensor.does_hotkey_exist = subtensor._subtensor.does_hotkey_exist
    subtensor.encode_params = subtensor._subtensor.encode_params
    subtensor.filter_netuids_by_registered_hotkeys = (
        subtensor._subtensor.filter_netuids_by_registered_hotkeys
    )
    subtensor.get_all_commitments = subtensor._subtensor.get_all_commitments
    subtensor.get_all_metagraphs_info = subtensor._subtensor.get_all_metagraphs_info
    subtensor.get_all_neuron_certificates = (
        subtensor._subtensor.get_all_neuron_certificates
    )
    subtensor.get_all_revealed_commitments = (
        subtensor._subtensor.get_all_revealed_commitments
    )
    subtensor.get_all_subnets_info = subtensor._subtensor.get_all_subnets_info
    subtensor.get_balance = subtensor._subtensor.get_balance
    subtensor.get_balances = subtensor._subtensor.get_balances
    subtensor.get_block_hash = subtensor._subtensor.get_block_hash
    subtensor.get_parents = subtensor._subtensor.get_parents
    subtensor.get_children = subtensor._subtensor.get_children
    subtensor.get_children_pending = subtensor._subtensor.get_children_pending
    subtensor.get_commitment = subtensor._subtensor.get_commitment
    subtensor.get_current_block = subtensor._subtensor.get_current_block
    subtensor.get_last_commitment_bonds_reset_block = (
        subtensor._subtensor.get_last_commitment_bonds_reset_block
    )
    subtensor.get_current_weight_commit_info = (
        subtensor._subtensor.get_current_weight_commit_info
    )
    subtensor.get_delegate_by_hotkey = subtensor._subtensor.get_delegate_by_hotkey
    subtensor.get_delegate_identities = subtensor._subtensor.get_delegate_identities
    subtensor.get_delegate_take = subtensor._subtensor.get_delegate_take
    subtensor.get_delegated = subtensor._subtensor.get_delegated
    subtensor.get_delegates = subtensor._subtensor.get_delegates
    subtensor.get_existential_deposit = subtensor._subtensor.get_existential_deposit
    subtensor.get_hotkey_owner = subtensor._subtensor.get_hotkey_owner
    subtensor.get_hotkey_stake = subtensor._subtensor.get_hotkey_stake
    subtensor.get_hyperparameter = subtensor._subtensor.get_hyperparameter
    subtensor.get_liquidity_list = subtensor._subtensor.get_liquidity_list
    subtensor.get_metagraph_info = subtensor._subtensor.get_metagraph_info
    subtensor.get_minimum_required_stake = (
        subtensor._subtensor.get_minimum_required_stake
    )
    subtensor.get_netuids_for_hotkey = subtensor._subtensor.get_netuids_for_hotkey
    subtensor.get_neuron_certificate = subtensor._subtensor.get_neuron_certificate
    subtensor.get_neuron_for_pubkey_and_subnet = (
        subtensor._subtensor.get_neuron_for_pubkey_and_subnet
    )
    subtensor.get_next_epoch_start_block = (
        subtensor._subtensor.get_next_epoch_start_block
    )
    subtensor.get_owned_hotkeys = subtensor._subtensor.get_owned_hotkeys
    subtensor.get_revealed_commitment = subtensor._subtensor.get_revealed_commitment
    subtensor.get_revealed_commitment_by_hotkey = (
        subtensor._subtensor.get_revealed_commitment_by_hotkey
    )
    subtensor.get_stake = subtensor._subtensor.get_stake
    subtensor.get_stake_add_fee = subtensor._subtensor.get_stake_add_fee
    subtensor.get_stake_for_coldkey = subtensor._subtensor.get_stake_for_coldkey
    subtensor.get_stake_for_coldkey_and_hotkey = (
        subtensor._subtensor.get_stake_for_coldkey_and_hotkey
    )
    subtensor.get_stake_for_hotkey = subtensor._subtensor.get_stake_for_hotkey
    subtensor.get_stake_info_for_coldkey = (
        subtensor._subtensor.get_stake_info_for_coldkey
    )
    subtensor.get_stake_movement_fee = subtensor._subtensor.get_stake_movement_fee
    subtensor.get_subnet_burn_cost = subtensor._subtensor.get_subnet_burn_cost
    subtensor.get_subnet_hyperparameters = (
        subtensor._subtensor.get_subnet_hyperparameters
    )
    subtensor.get_subnet_info = subtensor._subtensor.get_subnet_info
    subtensor.get_subnet_price = subtensor._subtensor.get_subnet_price
    subtensor.get_subnet_prices = subtensor._subtensor.get_subnet_prices
    subtensor.get_subnet_owner_hotkey = subtensor._subtensor.get_subnet_owner_hotkey
    subtensor.get_subnet_reveal_period_epochs = (
        subtensor._subtensor.get_subnet_reveal_period_epochs
    )
    subtensor.get_subnet_validator_permits = (
        subtensor._subtensor.get_subnet_validator_permits
    )
    subtensor.get_subnets = subtensor._subtensor.get_subnets
    subtensor.get_timestamp = subtensor._subtensor.get_timestamp
    subtensor.get_total_subnets = subtensor._subtensor.get_total_subnets
    subtensor.get_transfer_fee = subtensor._subtensor.get_transfer_fee
    subtensor.get_uid_for_hotkey_on_subnet = (
        subtensor._subtensor.get_uid_for_hotkey_on_subnet
    )
    subtensor.get_unstake_fee = subtensor._subtensor.get_unstake_fee
    subtensor.get_vote_data = subtensor._subtensor.get_vote_data
    subtensor.immunity_period = subtensor._subtensor.immunity_period
    subtensor.is_fast_blocks = subtensor._subtensor.is_fast_blocks
    subtensor.is_hotkey_delegate = subtensor._subtensor.is_hotkey_delegate
    subtensor.is_hotkey_registered = subtensor._subtensor.is_hotkey_registered
    subtensor.is_hotkey_registered_any = subtensor._subtensor.is_hotkey_registered_any
    subtensor.is_hotkey_registered_on_subnet = (
        subtensor._subtensor.is_hotkey_registered_on_subnet
    )
    subtensor.is_subnet_active = subtensor._subtensor.is_subnet_active
    subtensor.last_drand_round = subtensor._subtensor.last_drand_round
    subtensor.log_verbose = subtensor._subtensor.log_verbose
    subtensor.max_weight_limit = subtensor._subtensor.max_weight_limit
    subtensor.metagraph = subtensor._subtensor.metagraph
    subtensor.min_allowed_weights = subtensor._subtensor.min_allowed_weights
    subtensor.modify_liquidity = subtensor._subtensor.modify_liquidity
    subtensor.move_stake = subtensor._subtensor.move_stake
    subtensor.network = subtensor._subtensor.network
    subtensor.neurons = subtensor._subtensor.neurons
    subtensor.neuron_for_uid = subtensor._subtensor.neuron_for_uid
    subtensor.neurons_lite = subtensor._subtensor.neurons_lite
    subtensor.query_constant = subtensor._subtensor.query_constant
    subtensor.query_identity = subtensor._subtensor.query_identity
    subtensor.query_map = subtensor._subtensor.query_map
    subtensor.query_map_subtensor = subtensor._subtensor.query_map_subtensor
    subtensor.query_module = subtensor._subtensor.query_module
    subtensor.query_runtime_api = subtensor._subtensor.query_runtime_api
    subtensor.query_subtensor = subtensor._subtensor.query_subtensor
    subtensor.recycle = subtensor._subtensor.recycle
    subtensor.remove_liquidity = subtensor._subtensor.remove_liquidity
    subtensor.register = subtensor._subtensor.register
    subtensor.register_subnet = subtensor._subtensor.register_subnet
    subtensor.reveal_weights = subtensor._subtensor.reveal_weights
    subtensor.root_register = subtensor._subtensor.root_register
    subtensor.root_set_pending_childkey_cooldown = (
        subtensor._subtensor.root_set_pending_childkey_cooldown
    )
    subtensor.root_set_weights = subtensor._subtensor.root_set_weights
    subtensor.serve_axon = subtensor._subtensor.serve_axon
    subtensor.set_children = subtensor._subtensor.set_children
    subtensor.set_commitment = subtensor._subtensor.set_commitment
    subtensor.set_delegate_take = subtensor._subtensor.set_delegate_take
    subtensor.set_reveal_commitment = subtensor._subtensor.set_reveal_commitment
    subtensor.set_subnet_identity = subtensor._subtensor.set_subnet_identity
    subtensor.set_weights = subtensor._subtensor.set_weights
    subtensor.setup_config = subtensor._subtensor.setup_config
    subtensor.sign_and_send_extrinsic = subtensor._subtensor.sign_and_send_extrinsic
    subtensor.start_call = subtensor._subtensor.start_call
    subtensor.state_call = subtensor._subtensor.state_call
    subtensor.subnet = subtensor._subtensor.subnet
    subtensor.subnet_exists = subtensor._subtensor.subnet_exists
    subtensor.subnetwork_n = subtensor._subtensor.subnetwork_n
    subtensor.substrate = subtensor._subtensor.substrate
    subtensor.swap_stake = subtensor._subtensor.swap_stake
    subtensor.tempo = subtensor._subtensor.tempo
    subtensor.toggle_user_liquidity = subtensor._subtensor.toggle_user_liquidity
    subtensor.transfer = subtensor._subtensor.transfer
    subtensor.transfer_stake = subtensor._subtensor.transfer_stake
    subtensor.tx_rate_limit = subtensor._subtensor.tx_rate_limit
    subtensor.unstake = subtensor._subtensor.unstake
    subtensor.unstake_all = subtensor._subtensor.unstake_all
    subtensor.unstake_multiple = subtensor._subtensor.unstake_multiple
    subtensor.wait_for_block = subtensor._subtensor.wait_for_block
    subtensor.weights = subtensor._subtensor.weights
    subtensor.weights_rate_limit = subtensor._subtensor.weights_rate_limit
