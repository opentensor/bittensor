"""
This file is auto-generated. Do not edit manually.

For developers:
- Use the function `recreate_calls_subpackage()` to regenerate this file.
- The command lists are built dynamically from the current Subtensor metadata (`Subtensor.substrate.metadata`).
- Each command is represented as a `namedtuple` with fields:
    * System arguments: wallet, pallet (and `sudo` for sudo calls).
    * Additional arguments: taken from the extrinsic definition (with type hints for reference).
- These namedtuples are intended as convenient templates for building commands in tests and end-to-end scenarios.

Note:
    Any manual changes will be overwritten the next time the generator is run.
    Subtensor spec version: 331
"""

from collections import namedtuple


SUDO_AS = namedtuple(
    "SUDO_AS", ["wallet", "pallet", "sudo", "who", "call"]
)  # args: [who: AccountIdLookupOf<T>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Sudo
SUDO_SET_ACTIVITY_CUTOFF = namedtuple(
    "SUDO_SET_ACTIVITY_CUTOFF",
    ["wallet", "pallet", "sudo", "netuid", "activity_cutoff"],
)  # args: [netuid: NetUid, activity_cutoff: u16]  | Pallet: AdminUtils
SUDO_SET_ADJUSTMENT_ALPHA = namedtuple(
    "SUDO_SET_ADJUSTMENT_ALPHA",
    ["wallet", "pallet", "sudo", "netuid", "adjustment_alpha"],
)  # args: [netuid: NetUid, adjustment_alpha: u64]  | Pallet: AdminUtils
SUDO_SET_ADJUSTMENT_INTERVAL = namedtuple(
    "SUDO_SET_ADJUSTMENT_INTERVAL",
    ["wallet", "pallet", "sudo", "netuid", "adjustment_interval"],
)  # args: [netuid: NetUid, adjustment_interval: u16]  | Pallet: AdminUtils
SUDO_SET_ADMIN_FREEZE_WINDOW = namedtuple(
    "SUDO_SET_ADMIN_FREEZE_WINDOW", ["wallet", "pallet", "sudo", "window"]
)  # args: [window: u16]  | Pallet: AdminUtils
SUDO_SET_ALPHA_SIGMOID_STEEPNESS = namedtuple(
    "SUDO_SET_ALPHA_SIGMOID_STEEPNESS",
    ["wallet", "pallet", "sudo", "netuid", "steepness"],
)  # args: [netuid: NetUid, steepness: i16]  | Pallet: AdminUtils
SUDO_SET_ALPHA_VALUES = namedtuple(
    "SUDO_SET_ALPHA_VALUES",
    ["wallet", "pallet", "sudo", "netuid", "alpha_low", "alpha_high"],
)  # args: [netuid: NetUid, alpha_low: u16, alpha_high: u16]  | Pallet: AdminUtils
SUDO_SET_BONDS_MOVING_AVERAGE = namedtuple(
    "SUDO_SET_BONDS_MOVING_AVERAGE",
    ["wallet", "pallet", "sudo", "netuid", "bonds_moving_average"],
)  # args: [netuid: NetUid, bonds_moving_average: u64]  | Pallet: AdminUtils
SUDO_SET_BONDS_PENALTY = namedtuple(
    "SUDO_SET_BONDS_PENALTY", ["wallet", "pallet", "sudo", "netuid", "bonds_penalty"]
)  # args: [netuid: NetUid, bonds_penalty: u16]  | Pallet: AdminUtils
SUDO_SET_BONDS_RESET_ENABLED = namedtuple(
    "SUDO_SET_BONDS_RESET_ENABLED", ["wallet", "pallet", "sudo", "netuid", "enabled"]
)  # args: [netuid: NetUid, enabled: bool]  | Pallet: AdminUtils
SUDO_SET_CK_BURN = namedtuple(
    "SUDO_SET_CK_BURN", ["wallet", "pallet", "sudo", "burn"]
)  # args: [burn: u64]  | Pallet: AdminUtils
SUDO_SET_COLDKEY_SWAP_SCHEDULE_DURATION = namedtuple(
    "SUDO_SET_COLDKEY_SWAP_SCHEDULE_DURATION", ["wallet", "pallet", "sudo", "duration"]
)  # args: [duration: BlockNumberFor<T>]  | Pallet: AdminUtils
SUDO_SET_COMMIT_REVEAL_VERSION = namedtuple(
    "SUDO_SET_COMMIT_REVEAL_VERSION", ["wallet", "pallet", "sudo", "version"]
)  # args: [version: u16]  | Pallet: AdminUtils
SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED = namedtuple(
    "SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED",
    ["wallet", "pallet", "sudo", "netuid", "enabled"],
)  # args: [netuid: NetUid, enabled: bool]  | Pallet: AdminUtils
SUDO_SET_COMMIT_REVEAL_WEIGHTS_INTERVAL = namedtuple(
    "SUDO_SET_COMMIT_REVEAL_WEIGHTS_INTERVAL",
    ["wallet", "pallet", "sudo", "netuid", "interval"],
)  # args: [netuid: NetUid, interval: u64]  | Pallet: AdminUtils
SUDO_SET_DEFAULT_TAKE = namedtuple(
    "SUDO_SET_DEFAULT_TAKE", ["wallet", "pallet", "sudo", "default_take"]
)  # args: [default_take: u16]  | Pallet: AdminUtils
SUDO_SET_DIFFICULTY = namedtuple(
    "SUDO_SET_DIFFICULTY", ["wallet", "pallet", "sudo", "netuid", "difficulty"]
)  # args: [netuid: NetUid, difficulty: u64]  | Pallet: AdminUtils
SUDO_SET_DISSOLVE_NETWORK_SCHEDULE_DURATION = namedtuple(
    "SUDO_SET_DISSOLVE_NETWORK_SCHEDULE_DURATION",
    ["wallet", "pallet", "sudo", "duration"],
)  # args: [duration: BlockNumberFor<T>]  | Pallet: AdminUtils
SUDO_SET_EMA_PRICE_HALVING_PERIOD = namedtuple(
    "SUDO_SET_EMA_PRICE_HALVING_PERIOD",
    ["wallet", "pallet", "sudo", "netuid", "ema_halving"],
)  # args: [netuid: NetUid, ema_halving: u64]  | Pallet: AdminUtils
SUDO_SET_EVM_CHAIN_ID = namedtuple(
    "SUDO_SET_EVM_CHAIN_ID", ["wallet", "pallet", "sudo", "chain_id"]
)  # args: [chain_id: u64]  | Pallet: AdminUtils
SUDO_SET_IMMUNITY_PERIOD = namedtuple(
    "SUDO_SET_IMMUNITY_PERIOD",
    ["wallet", "pallet", "sudo", "netuid", "immunity_period"],
)  # args: [netuid: NetUid, immunity_period: u16]  | Pallet: AdminUtils
SUDO_SET_KAPPA = namedtuple(
    "SUDO_SET_KAPPA", ["wallet", "pallet", "sudo", "netuid", "kappa"]
)  # args: [netuid: NetUid, kappa: u16]  | Pallet: AdminUtils
SUDO_SET_LIQUID_ALPHA_ENABLED = namedtuple(
    "SUDO_SET_LIQUID_ALPHA_ENABLED", ["wallet", "pallet", "sudo", "netuid", "enabled"]
)  # args: [netuid: NetUid, enabled: bool]  | Pallet: AdminUtils
SUDO_SET_LOCK_REDUCTION_INTERVAL = namedtuple(
    "SUDO_SET_LOCK_REDUCTION_INTERVAL", ["wallet", "pallet", "sudo", "interval"]
)  # args: [interval: u64]  | Pallet: AdminUtils
SUDO_SET_MAX_ALLOWED_UIDS = namedtuple(
    "SUDO_SET_MAX_ALLOWED_UIDS",
    ["wallet", "pallet", "sudo", "netuid", "max_allowed_uids"],
)  # args: [netuid: NetUid, max_allowed_uids: u16]  | Pallet: AdminUtils
SUDO_SET_MAX_ALLOWED_VALIDATORS = namedtuple(
    "SUDO_SET_MAX_ALLOWED_VALIDATORS",
    ["wallet", "pallet", "sudo", "netuid", "max_allowed_validators"],
)  # args: [netuid: NetUid, max_allowed_validators: u16]  | Pallet: AdminUtils
SUDO_SET_MAX_BURN = namedtuple(
    "SUDO_SET_MAX_BURN", ["wallet", "pallet", "sudo", "netuid", "max_burn"]
)  # args: [netuid: NetUid, max_burn: TaoCurrency]  | Pallet: AdminUtils
SUDO_SET_MAX_CHILDKEY_TAKE = namedtuple(
    "SUDO_SET_MAX_CHILDKEY_TAKE", ["wallet", "pallet", "sudo", "take"]
)  # args: [take: u16]  | Pallet: SubtensorModule
SUDO_SET_MAX_DIFFICULTY = namedtuple(
    "SUDO_SET_MAX_DIFFICULTY", ["wallet", "pallet", "sudo", "netuid", "max_difficulty"]
)  # args: [netuid: NetUid, max_difficulty: u64]  | Pallet: AdminUtils
SUDO_SET_MAX_REGISTRATIONS_PER_BLOCK = namedtuple(
    "SUDO_SET_MAX_REGISTRATIONS_PER_BLOCK",
    ["wallet", "pallet", "sudo", "netuid", "max_registrations_per_block"],
)  # args: [netuid: NetUid, max_registrations_per_block: u16]  | Pallet: AdminUtils
SUDO_SET_MECHANISM_COUNT = namedtuple(
    "SUDO_SET_MECHANISM_COUNT",
    ["wallet", "pallet", "sudo", "netuid", "mechanism_count"],
)  # args: [netuid: NetUid, mechanism_count: MechId]  | Pallet: AdminUtils
SUDO_SET_MECHANISM_EMISSION_SPLIT = namedtuple(
    "SUDO_SET_MECHANISM_EMISSION_SPLIT",
    ["wallet", "pallet", "sudo", "netuid", "maybe_split"],
)  # args: [netuid: NetUid, maybe_split: Option<Vec<u16>>]  | Pallet: AdminUtils
SUDO_SET_MIN_ALLOWED_UIDS = namedtuple(
    "SUDO_SET_MIN_ALLOWED_UIDS",
    ["wallet", "pallet", "sudo", "netuid", "min_allowed_uids"],
)  # args: [netuid: NetUid, min_allowed_uids: u16]  | Pallet: AdminUtils
SUDO_SET_MIN_ALLOWED_WEIGHTS = namedtuple(
    "SUDO_SET_MIN_ALLOWED_WEIGHTS",
    ["wallet", "pallet", "sudo", "netuid", "min_allowed_weights"],
)  # args: [netuid: NetUid, min_allowed_weights: u16]  | Pallet: AdminUtils
SUDO_SET_MIN_BURN = namedtuple(
    "SUDO_SET_MIN_BURN", ["wallet", "pallet", "sudo", "netuid", "min_burn"]
)  # args: [netuid: NetUid, min_burn: TaoCurrency]  | Pallet: AdminUtils
SUDO_SET_MIN_CHILDKEY_TAKE = namedtuple(
    "SUDO_SET_MIN_CHILDKEY_TAKE", ["wallet", "pallet", "sudo", "take"]
)  # args: [take: u16]  | Pallet: SubtensorModule
SUDO_SET_MIN_DELEGATE_TAKE = namedtuple(
    "SUDO_SET_MIN_DELEGATE_TAKE", ["wallet", "pallet", "sudo", "take"]
)  # args: [take: u16]  | Pallet: AdminUtils
SUDO_SET_MIN_DIFFICULTY = namedtuple(
    "SUDO_SET_MIN_DIFFICULTY", ["wallet", "pallet", "sudo", "netuid", "min_difficulty"]
)  # args: [netuid: NetUid, min_difficulty: u64]  | Pallet: AdminUtils
SUDO_SET_NETWORK_IMMUNITY_PERIOD = namedtuple(
    "SUDO_SET_NETWORK_IMMUNITY_PERIOD", ["wallet", "pallet", "sudo", "immunity_period"]
)  # args: [immunity_period: u64]  | Pallet: AdminUtils
SUDO_SET_NETWORK_MIN_LOCK_COST = namedtuple(
    "SUDO_SET_NETWORK_MIN_LOCK_COST", ["wallet", "pallet", "sudo", "lock_cost"]
)  # args: [lock_cost: TaoCurrency]  | Pallet: AdminUtils
SUDO_SET_NETWORK_POW_REGISTRATION_ALLOWED = namedtuple(
    "SUDO_SET_NETWORK_POW_REGISTRATION_ALLOWED",
    ["wallet", "pallet", "sudo", "netuid", "registration_allowed"],
)  # args: [netuid: NetUid, registration_allowed: bool]  | Pallet: AdminUtils
SUDO_SET_NETWORK_RATE_LIMIT = namedtuple(
    "SUDO_SET_NETWORK_RATE_LIMIT", ["wallet", "pallet", "sudo", "rate_limit"]
)  # args: [rate_limit: u64]  | Pallet: AdminUtils
SUDO_SET_NETWORK_REGISTRATION_ALLOWED = namedtuple(
    "SUDO_SET_NETWORK_REGISTRATION_ALLOWED",
    ["wallet", "pallet", "sudo", "netuid", "registration_allowed"],
)  # args: [netuid: NetUid, registration_allowed: bool]  | Pallet: AdminUtils
SUDO_SET_NOMINATOR_MIN_REQUIRED_STAKE = namedtuple(
    "SUDO_SET_NOMINATOR_MIN_REQUIRED_STAKE", ["wallet", "pallet", "sudo", "min_stake"]
)  # args: [min_stake: u64]  | Pallet: AdminUtils
SUDO_SET_NUM_ROOT_CLAIMS = namedtuple(
    "SUDO_SET_NUM_ROOT_CLAIMS", ["wallet", "pallet", "sudo", "new_value"]
)  # args: [new_value: u64]  | Pallet: SubtensorModule
SUDO_SET_OWNER_HPARAM_RATE_LIMIT = namedtuple(
    "SUDO_SET_OWNER_HPARAM_RATE_LIMIT", ["wallet", "pallet", "sudo", "epochs"]
)  # args: [epochs: u16]  | Pallet: AdminUtils
SUDO_SET_OWNER_IMMUNE_NEURON_LIMIT = namedtuple(
    "SUDO_SET_OWNER_IMMUNE_NEURON_LIMIT",
    ["wallet", "pallet", "sudo", "netuid", "immune_neurons"],
)  # args: [netuid: NetUid, immune_neurons: u16]  | Pallet: AdminUtils
SUDO_SET_RAO_RECYCLED = namedtuple(
    "SUDO_SET_RAO_RECYCLED", ["wallet", "pallet", "sudo", "netuid", "rao_recycled"]
)  # args: [netuid: NetUid, rao_recycled: TaoCurrency]  | Pallet: AdminUtils
SUDO_SET_RECYCLE_OR_BURN = namedtuple(
    "SUDO_SET_RECYCLE_OR_BURN",
    ["wallet", "pallet", "sudo", "netuid", "recycle_or_burn"],
)  # args: [netuid: NetUid, recycle_or_burn: pallet_subtensor::RecycleOrBurnEnum]  | Pallet: AdminUtils
SUDO_SET_RHO = namedtuple(
    "SUDO_SET_RHO", ["wallet", "pallet", "sudo", "netuid", "rho"]
)  # args: [netuid: NetUid, rho: u16]  | Pallet: AdminUtils
SUDO_SET_ROOT_CLAIM_THRESHOLD = namedtuple(
    "SUDO_SET_ROOT_CLAIM_THRESHOLD", ["wallet", "pallet", "sudo", "netuid", "new_value"]
)  # args: [netuid: NetUid, new_value: u64]  | Pallet: SubtensorModule
SUDO_SET_SERVING_RATE_LIMIT = namedtuple(
    "SUDO_SET_SERVING_RATE_LIMIT",
    ["wallet", "pallet", "sudo", "netuid", "serving_rate_limit"],
)  # args: [netuid: NetUid, serving_rate_limit: u64]  | Pallet: AdminUtils
SUDO_SET_SN_OWNER_HOTKEY = namedtuple(
    "SUDO_SET_SN_OWNER_HOTKEY", ["wallet", "pallet", "sudo", "netuid", "hotkey"]
)  # args: [netuid: NetUid, hotkey: <T as frame_system::Config>::AccountId]  | Pallet: AdminUtils
SUDO_SET_STAKE_THRESHOLD = namedtuple(
    "SUDO_SET_STAKE_THRESHOLD", ["wallet", "pallet", "sudo", "min_stake"]
)  # args: [min_stake: u64]  | Pallet: AdminUtils
SUDO_SET_SUBNET_LIMIT = namedtuple(
    "SUDO_SET_SUBNET_LIMIT", ["wallet", "pallet", "sudo", "max_subnets"]
)  # args: [max_subnets: u16]  | Pallet: AdminUtils
SUDO_SET_SUBNET_MOVING_ALPHA = namedtuple(
    "SUDO_SET_SUBNET_MOVING_ALPHA", ["wallet", "pallet", "sudo", "alpha"]
)  # args: [alpha: I96F32]  | Pallet: AdminUtils
SUDO_SET_SUBNET_OWNER_CUT = namedtuple(
    "SUDO_SET_SUBNET_OWNER_CUT", ["wallet", "pallet", "sudo", "subnet_owner_cut"]
)  # args: [subnet_owner_cut: u16]  | Pallet: AdminUtils
SUDO_SET_SUBNET_OWNER_HOTKEY = namedtuple(
    "SUDO_SET_SUBNET_OWNER_HOTKEY", ["wallet", "pallet", "sudo", "netuid", "hotkey"]
)  # args: [netuid: NetUid, hotkey: <T as frame_system::Config>::AccountId]  | Pallet: AdminUtils
SUDO_SET_SUBTOKEN_ENABLED = namedtuple(
    "SUDO_SET_SUBTOKEN_ENABLED",
    ["wallet", "pallet", "sudo", "netuid", "subtoken_enabled"],
)  # args: [netuid: NetUid, subtoken_enabled: bool]  | Pallet: AdminUtils
SUDO_SET_TARGET_REGISTRATIONS_PER_INTERVAL = namedtuple(
    "SUDO_SET_TARGET_REGISTRATIONS_PER_INTERVAL",
    ["wallet", "pallet", "sudo", "netuid", "target_registrations_per_interval"],
)  # args: [netuid: NetUid, target_registrations_per_interval: u16]  | Pallet: AdminUtils
SUDO_SET_TEMPO = namedtuple(
    "SUDO_SET_TEMPO", ["wallet", "pallet", "sudo", "netuid", "tempo"]
)  # args: [netuid: NetUid, tempo: u16]  | Pallet: AdminUtils
SUDO_SET_TOGGLE_TRANSFER = namedtuple(
    "SUDO_SET_TOGGLE_TRANSFER", ["wallet", "pallet", "sudo", "netuid", "toggle"]
)  # args: [netuid: NetUid, toggle: bool]  | Pallet: AdminUtils
SUDO_SET_TOTAL_ISSUANCE = namedtuple(
    "SUDO_SET_TOTAL_ISSUANCE", ["wallet", "pallet", "sudo", "total_issuance"]
)  # args: [total_issuance: TaoCurrency]  | Pallet: AdminUtils
SUDO_SET_TX_CHILDKEY_TAKE_RATE_LIMIT = namedtuple(
    "SUDO_SET_TX_CHILDKEY_TAKE_RATE_LIMIT",
    ["wallet", "pallet", "sudo", "tx_rate_limit"],
)  # args: [tx_rate_limit: u64]  | Pallet: SubtensorModule
SUDO_SET_TX_DELEGATE_TAKE_RATE_LIMIT = namedtuple(
    "SUDO_SET_TX_DELEGATE_TAKE_RATE_LIMIT",
    ["wallet", "pallet", "sudo", "tx_rate_limit"],
)  # args: [tx_rate_limit: u64]  | Pallet: AdminUtils
SUDO_SET_TX_RATE_LIMIT = namedtuple(
    "SUDO_SET_TX_RATE_LIMIT", ["wallet", "pallet", "sudo", "tx_rate_limit"]
)  # args: [tx_rate_limit: u64]  | Pallet: AdminUtils
SUDO_SET_WEIGHTS_SET_RATE_LIMIT = namedtuple(
    "SUDO_SET_WEIGHTS_SET_RATE_LIMIT",
    ["wallet", "pallet", "sudo", "netuid", "weights_set_rate_limit"],
)  # args: [netuid: NetUid, weights_set_rate_limit: u64]  | Pallet: AdminUtils
SUDO_SET_WEIGHTS_VERSION_KEY = namedtuple(
    "SUDO_SET_WEIGHTS_VERSION_KEY",
    ["wallet", "pallet", "sudo", "netuid", "weights_version_key"],
)  # args: [netuid: NetUid, weights_version_key: u64]  | Pallet: AdminUtils
SUDO_SET_YUMA3_ENABLED = namedtuple(
    "SUDO_SET_YUMA3_ENABLED", ["wallet", "pallet", "sudo", "netuid", "enabled"]
)  # args: [netuid: NetUid, enabled: bool]  | Pallet: AdminUtils
SUDO_TOGGLE_EVM_PRECOMPILE = namedtuple(
    "SUDO_TOGGLE_EVM_PRECOMPILE",
    ["wallet", "pallet", "sudo", "precompile_id", "enabled"],
)  # args: [precompile_id: PrecompileEnum, enabled: bool]  | Pallet: AdminUtils
SUDO_TRIM_TO_MAX_ALLOWED_UIDS = namedtuple(
    "SUDO_TRIM_TO_MAX_ALLOWED_UIDS", ["wallet", "pallet", "sudo", "netuid", "max_n"]
)  # args: [netuid: NetUid, max_n: u16]  | Pallet: AdminUtils
SUDO_UNCHECKED_WEIGHT = namedtuple(
    "SUDO_UNCHECKED_WEIGHT", ["wallet", "pallet", "sudo", "call", "weight"]
)  # args: [call: Box<<T as Config>::RuntimeCall>, weight: Weight]  | Pallet: Sudo
