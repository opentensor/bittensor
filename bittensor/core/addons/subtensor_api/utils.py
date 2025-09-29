from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittensor.core.addons import SubtensorApi


def add_legacy_methods(subtensor: "SubtensorApi"):
    """If SubtensorApi get `subtensor_fields=True` arguments, then all classic Subtensor fields added to root level."""
    subtensor.add_liquidity = subtensor.inner_subtensor.add_liquidity
    subtensor.add_stake = subtensor.inner_subtensor.add_stake
    subtensor.add_stake_multiple = subtensor.inner_subtensor.add_stake_multiple
    subtensor.all_subnets = subtensor.inner_subtensor.all_subnets
    subtensor.blocks_since_last_step = subtensor.inner_subtensor.blocks_since_last_step
    subtensor.blocks_since_last_update = (
        subtensor.inner_subtensor.blocks_since_last_update
    )
    subtensor.bonds = subtensor.inner_subtensor.bonds
    subtensor.burned_register = subtensor.inner_subtensor.burned_register
    subtensor.chain_endpoint = subtensor.inner_subtensor.chain_endpoint
    subtensor.commit_reveal_enabled = subtensor.inner_subtensor.commit_reveal_enabled
    subtensor.commit_weights = subtensor.inner_subtensor.commit_weights
    subtensor.determine_block_hash = subtensor.inner_subtensor.determine_block_hash
    subtensor.difficulty = subtensor.inner_subtensor.difficulty
    subtensor.does_hotkey_exist = subtensor.inner_subtensor.does_hotkey_exist
    subtensor.encode_params = subtensor.inner_subtensor.encode_params
    subtensor.filter_netuids_by_registered_hotkeys = (
        subtensor.inner_subtensor.filter_netuids_by_registered_hotkeys
    )
    subtensor.get_admin_freeze_window = (
        subtensor.inner_subtensor.get_admin_freeze_window
    )
    subtensor.get_all_commitments = subtensor.inner_subtensor.get_all_commitments
    subtensor.get_all_metagraphs_info = (
        subtensor.inner_subtensor.get_all_metagraphs_info
    )
    subtensor.get_all_neuron_certificates = (
        subtensor.inner_subtensor.get_all_neuron_certificates
    )
    subtensor.get_all_revealed_commitments = (
        subtensor.inner_subtensor.get_all_revealed_commitments
    )
    subtensor.get_all_subnets_info = subtensor.inner_subtensor.get_all_subnets_info
    subtensor.get_balance = subtensor.inner_subtensor.get_balance
    subtensor.get_balances = subtensor.inner_subtensor.get_balances
    subtensor.get_block_hash = subtensor.inner_subtensor.get_block_hash
    subtensor.get_children = subtensor.inner_subtensor.get_children
    subtensor.get_children_pending = subtensor.inner_subtensor.get_children_pending
    subtensor.get_commitment = subtensor.inner_subtensor.get_commitment
    subtensor.get_current_block = subtensor.inner_subtensor.get_current_block
    subtensor.get_delegate_by_hotkey = subtensor.inner_subtensor.get_delegate_by_hotkey
    subtensor.get_delegate_identities = (
        subtensor.inner_subtensor.get_delegate_identities
    )
    subtensor.get_delegate_take = subtensor.inner_subtensor.get_delegate_take
    subtensor.get_delegated = subtensor.inner_subtensor.get_delegated
    subtensor.get_delegates = subtensor.inner_subtensor.get_delegates
    subtensor.get_existential_deposit = (
        subtensor.inner_subtensor.get_existential_deposit
    )
    subtensor.get_hotkey_owner = subtensor.inner_subtensor.get_hotkey_owner
    subtensor.get_hotkey_stake = subtensor.inner_subtensor.get_hotkey_stake
    subtensor.get_hyperparameter = subtensor.inner_subtensor.get_hyperparameter
    subtensor.get_last_commitment_bonds_reset_block = (
        subtensor.inner_subtensor.get_last_commitment_bonds_reset_block
    )
    subtensor.get_liquidity_list = subtensor.inner_subtensor.get_liquidity_list
    subtensor.get_mechanism_count = subtensor.inner_subtensor.get_mechanism_count
    subtensor.get_mechanism_emission_split = (
        subtensor.inner_subtensor.get_mechanism_emission_split
    )
    subtensor.get_metagraph_info = subtensor.inner_subtensor.get_metagraph_info
    subtensor.get_minimum_required_stake = (
        subtensor.inner_subtensor.get_minimum_required_stake
    )
    subtensor.get_netuids_for_hotkey = subtensor.inner_subtensor.get_netuids_for_hotkey
    subtensor.get_neuron_certificate = subtensor.inner_subtensor.get_neuron_certificate
    subtensor.get_neuron_for_pubkey_and_subnet = (
        subtensor.inner_subtensor.get_neuron_for_pubkey_and_subnet
    )
    subtensor.get_next_epoch_start_block = (
        subtensor.inner_subtensor.get_next_epoch_start_block
    )
    subtensor.get_owned_hotkeys = subtensor.inner_subtensor.get_owned_hotkeys
    subtensor.get_parents = subtensor.inner_subtensor.get_parents
    subtensor.get_revealed_commitment = (
        subtensor.inner_subtensor.get_revealed_commitment
    )
    subtensor.get_revealed_commitment_by_hotkey = (
        subtensor.inner_subtensor.get_revealed_commitment_by_hotkey
    )
    subtensor.get_stake = subtensor.inner_subtensor.get_stake
    subtensor.get_stake_add_fee = subtensor.inner_subtensor.get_stake_add_fee
    subtensor.get_stake_for_coldkey_and_hotkey = (
        subtensor.inner_subtensor.get_stake_for_coldkey_and_hotkey
    )
    subtensor.get_stake_for_hotkey = subtensor.inner_subtensor.get_stake_for_hotkey
    subtensor.get_stake_info_for_coldkey = (
        subtensor.inner_subtensor.get_stake_info_for_coldkey
    )
    subtensor.get_stake_movement_fee = subtensor.inner_subtensor.get_stake_movement_fee
    subtensor.get_stake_operations_fee = (
        subtensor.inner_subtensor.get_stake_operations_fee
    )
    subtensor.get_stake_weight = subtensor.inner_subtensor.get_stake_weight
    subtensor.get_subnet_burn_cost = subtensor.inner_subtensor.get_subnet_burn_cost
    subtensor.get_subnet_hyperparameters = (
        subtensor.inner_subtensor.get_subnet_hyperparameters
    )
    subtensor.get_subnet_info = subtensor.inner_subtensor.get_subnet_info
    subtensor.get_subnet_owner_hotkey = (
        subtensor.inner_subtensor.get_subnet_owner_hotkey
    )
    subtensor.get_subnet_price = subtensor.inner_subtensor.get_subnet_price
    subtensor.get_subnet_prices = subtensor.inner_subtensor.get_subnet_prices
    subtensor.get_subnet_reveal_period_epochs = (
        subtensor.inner_subtensor.get_subnet_reveal_period_epochs
    )
    subtensor.get_subnet_validator_permits = (
        subtensor.inner_subtensor.get_subnet_validator_permits
    )
    subtensor.get_all_subnets_netuid = subtensor.inner_subtensor.get_all_subnets_netuid
    subtensor.get_timelocked_weight_commits = (
        subtensor.inner_subtensor.get_timelocked_weight_commits
    )
    subtensor.get_timestamp = subtensor.inner_subtensor.get_timestamp
    subtensor.get_total_subnets = subtensor.inner_subtensor.get_total_subnets
    subtensor.get_transfer_fee = subtensor.inner_subtensor.get_transfer_fee
    subtensor.get_uid_for_hotkey_on_subnet = (
        subtensor.inner_subtensor.get_uid_for_hotkey_on_subnet
    )
    subtensor.get_unstake_fee = subtensor.inner_subtensor.get_unstake_fee
    subtensor.get_vote_data = subtensor.inner_subtensor.get_vote_data
    subtensor.immunity_period = subtensor.inner_subtensor.immunity_period
    subtensor.is_fast_blocks = subtensor.inner_subtensor.is_fast_blocks
    subtensor.is_hotkey_delegate = subtensor.inner_subtensor.is_hotkey_delegate
    subtensor.is_hotkey_registered = subtensor.inner_subtensor.is_hotkey_registered
    subtensor.is_hotkey_registered_any = (
        subtensor.inner_subtensor.is_hotkey_registered_any
    )
    subtensor.is_hotkey_registered_on_subnet = (
        subtensor.inner_subtensor.is_hotkey_registered_on_subnet
    )
    subtensor.is_in_admin_freeze_window = (
        subtensor.inner_subtensor.is_in_admin_freeze_window
    )
    subtensor.is_subnet_active = subtensor.inner_subtensor.is_subnet_active
    subtensor.last_drand_round = subtensor.inner_subtensor.last_drand_round
    subtensor.log_verbose = subtensor.inner_subtensor.log_verbose
    subtensor.max_weight_limit = subtensor.inner_subtensor.max_weight_limit
    subtensor.metagraph = subtensor.inner_subtensor.metagraph
    subtensor.min_allowed_weights = subtensor.inner_subtensor.min_allowed_weights
    subtensor.modify_liquidity = subtensor.inner_subtensor.modify_liquidity
    subtensor.move_stake = subtensor.inner_subtensor.move_stake
    subtensor.neuron_for_uid = subtensor.inner_subtensor.neuron_for_uid
    subtensor.neurons = subtensor.inner_subtensor.neurons
    subtensor.neurons_lite = subtensor.inner_subtensor.neurons_lite
    subtensor.network = subtensor.inner_subtensor.network
    subtensor.query_constant = subtensor.inner_subtensor.query_constant
    subtensor.query_identity = subtensor.inner_subtensor.query_identity
    subtensor.query_map = subtensor.inner_subtensor.query_map
    subtensor.query_map_subtensor = subtensor.inner_subtensor.query_map_subtensor
    subtensor.query_module = subtensor.inner_subtensor.query_module
    subtensor.query_runtime_api = subtensor.inner_subtensor.query_runtime_api
    subtensor.query_subtensor = subtensor.inner_subtensor.query_subtensor
    subtensor.recycle = subtensor.inner_subtensor.recycle
    subtensor.register = subtensor.inner_subtensor.register
    subtensor.register_subnet = subtensor.inner_subtensor.register_subnet
    subtensor.remove_liquidity = subtensor.inner_subtensor.remove_liquidity
    subtensor.reveal_weights = subtensor.inner_subtensor.reveal_weights
    subtensor.root_register = subtensor.inner_subtensor.root_register
    subtensor.root_set_pending_childkey_cooldown = (
        subtensor.inner_subtensor.root_set_pending_childkey_cooldown
    )
    subtensor.serve_axon = subtensor.inner_subtensor.serve_axon
    subtensor.set_children = subtensor.inner_subtensor.set_children
    subtensor.set_commitment = subtensor.inner_subtensor.set_commitment
    subtensor.set_delegate_take = subtensor.inner_subtensor.set_delegate_take
    subtensor.set_reveal_commitment = subtensor.inner_subtensor.set_reveal_commitment
    subtensor.set_subnet_identity = subtensor.inner_subtensor.set_subnet_identity
    subtensor.set_weights = subtensor.inner_subtensor.set_weights
    subtensor.setup_config = subtensor.inner_subtensor.setup_config
    subtensor.sign_and_send_extrinsic = (
        subtensor.inner_subtensor.sign_and_send_extrinsic
    )
    subtensor.start_call = subtensor.inner_subtensor.start_call
    subtensor.state_call = subtensor.inner_subtensor.state_call
    subtensor.subnet = subtensor.inner_subtensor.subnet
    subtensor.subnet_exists = subtensor.inner_subtensor.subnet_exists
    subtensor.subnetwork_n = subtensor.inner_subtensor.subnetwork_n
    subtensor.substrate = subtensor.inner_subtensor.substrate
    subtensor.swap_stake = subtensor.inner_subtensor.swap_stake
    subtensor.tempo = subtensor.inner_subtensor.tempo
    subtensor.toggle_user_liquidity = subtensor.inner_subtensor.toggle_user_liquidity
    subtensor.transfer = subtensor.inner_subtensor.transfer
    subtensor.transfer_stake = subtensor.inner_subtensor.transfer_stake
    subtensor.tx_rate_limit = subtensor.inner_subtensor.tx_rate_limit
    subtensor.unstake = subtensor.inner_subtensor.unstake
    subtensor.unstake_all = subtensor.inner_subtensor.unstake_all
    subtensor.unstake_multiple = subtensor.inner_subtensor.unstake_multiple
    subtensor.wait_for_block = subtensor.inner_subtensor.wait_for_block
    subtensor.weights = subtensor.inner_subtensor.weights
    subtensor.weights_rate_limit = subtensor.inner_subtensor.weights_rate_limit
