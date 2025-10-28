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


ADD_LIQUIDITY = namedtuple(
    "ADD_LIQUIDITY",
    ["wallet", "pallet", "hotkey", "netuid", "tick_low", "tick_high", "liquidity"],
)  # args: [hotkey: T::AccountId, netuid: NetUid, tick_low: TickIndex, tick_high: TickIndex, liquidity: u64]  | Pallet: Swap
ADD_PROXY = namedtuple(
    "ADD_PROXY", ["wallet", "pallet", "delegate", "proxy_type", "delay"]
)  # args: [delegate: AccountIdLookupOf<T>, proxy_type: T::ProxyType, delay: BlockNumberFor<T>]  | Pallet: Proxy
ADD_STAKE = namedtuple(
    "ADD_STAKE", ["wallet", "pallet", "hotkey", "netuid", "amount_staked"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, amount_staked: TaoCurrency]  | Pallet: SubtensorModule
ADD_STAKE_LIMIT = namedtuple(
    "ADD_STAKE_LIMIT",
    [
        "wallet",
        "pallet",
        "hotkey",
        "netuid",
        "amount_staked",
        "limit_price",
        "allow_partial",
    ],
)  # args: [hotkey: T::AccountId, netuid: NetUid, amount_staked: TaoCurrency, limit_price: TaoCurrency, allow_partial: bool]  | Pallet: SubtensorModule
ANNOUNCE = namedtuple(
    "ANNOUNCE", ["wallet", "pallet", "real", "call_hash"]
)  # args: [real: AccountIdLookupOf<T>, call_hash: CallHashOf<T>]  | Pallet: Proxy
APPLY_AUTHORIZED_UPGRADE = namedtuple(
    "APPLY_AUTHORIZED_UPGRADE", ["wallet", "pallet", "code"]
)  # args: [code: Vec<u8>]  | Pallet: System
APPROVE_AS_MULTI = namedtuple(
    "APPROVE_AS_MULTI",
    [
        "wallet",
        "pallet",
        "threshold",
        "other_signatories",
        "maybe_timepoint",
        "call_hash",
        "max_weight",
    ],
)  # args: [threshold: u16, other_signatories: Vec<T::AccountId>, maybe_timepoint: Option<Timepoint<BlockNumberFor<T>>>, call_hash: [u8; 32], max_weight: Weight]  | Pallet: Multisig
ASSOCIATE_EVM_KEY = namedtuple(
    "ASSOCIATE_EVM_KEY",
    ["wallet", "pallet", "netuid", "evm_key", "block_number", "signature"],
)  # args: [netuid: NetUid, evm_key: H160, block_number: u64, signature: Signature]  | Pallet: SubtensorModule
AS_DERIVATIVE = namedtuple(
    "AS_DERIVATIVE", ["wallet", "pallet", "index", "call"]
)  # args: [index: u16, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Utility
AS_MULTI = namedtuple(
    "AS_MULTI",
    [
        "wallet",
        "pallet",
        "threshold",
        "other_signatories",
        "maybe_timepoint",
        "call",
        "max_weight",
    ],
)  # args: [threshold: u16, other_signatories: Vec<T::AccountId>, maybe_timepoint: Option<Timepoint<BlockNumberFor<T>>>, call: Box<<T as Config>::RuntimeCall>, max_weight: Weight]  | Pallet: Multisig
AS_MULTI_THRESHOLD_1 = namedtuple(
    "AS_MULTI_THRESHOLD_1", ["wallet", "pallet", "other_signatories", "call"]
)  # args: [other_signatories: Vec<T::AccountId>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Multisig
AUTHORIZE_UPGRADE = namedtuple(
    "AUTHORIZE_UPGRADE", ["wallet", "pallet", "code_hash"]
)  # args: [code_hash: T::Hash]  | Pallet: System
AUTHORIZE_UPGRADE_WITHOUT_CHECKS = namedtuple(
    "AUTHORIZE_UPGRADE_WITHOUT_CHECKS", ["wallet", "pallet", "code_hash"]
)  # args: [code_hash: T::Hash]  | Pallet: System
BATCH = namedtuple(
    "BATCH", ["wallet", "pallet", "calls"]
)  # args: [calls: Vec<<T as Config>::RuntimeCall>]  | Pallet: Utility
BATCH_ALL = namedtuple(
    "BATCH_ALL", ["wallet", "pallet", "calls"]
)  # args: [calls: Vec<<T as Config>::RuntimeCall>]  | Pallet: Utility
BATCH_COMMIT_WEIGHTS = namedtuple(
    "BATCH_COMMIT_WEIGHTS", ["wallet", "pallet", "netuids", "commit_hashes"]
)  # args: [netuids: Vec<Compact<NetUid>>, commit_hashes: Vec<H256>]  | Pallet: SubtensorModule
BATCH_REVEAL_WEIGHTS = namedtuple(
    "BATCH_REVEAL_WEIGHTS",
    [
        "wallet",
        "pallet",
        "netuid",
        "uids_list",
        "values_list",
        "salts_list",
        "version_keys",
    ],
)  # args: [netuid: NetUid, uids_list: Vec<Vec<u16>>, values_list: Vec<Vec<u16>>, salts_list: Vec<Vec<u16>>, version_keys: Vec<u64>]  | Pallet: SubtensorModule
BATCH_SET_WEIGHTS = namedtuple(
    "BATCH_SET_WEIGHTS", ["wallet", "pallet", "netuids", "weights", "version_keys"]
)  # args: [netuids: Vec<Compact<NetUid>>, weights: Vec<Vec<(Compact<u16>, Compact<u16>)>>, version_keys: Vec<Compact<u64>>]  | Pallet: SubtensorModule
BURN = namedtuple(
    "BURN", ["wallet", "pallet", "value", "keep_alive"]
)  # args: [value: T::Balance, keep_alive: bool]  | Pallet: Balances
BURNED_REGISTER = namedtuple(
    "BURNED_REGISTER", ["wallet", "pallet", "netuid", "hotkey"]
)  # args: [netuid: NetUid, hotkey: T::AccountId]  | Pallet: SubtensorModule
BURN_ALPHA = namedtuple(
    "BURN_ALPHA", ["wallet", "pallet", "hotkey", "amount", "netuid"]
)  # args: [hotkey: T::AccountId, amount: AlphaCurrency, netuid: NetUid]  | Pallet: SubtensorModule
CALL = namedtuple(
    "CALL",
    [
        "wallet",
        "pallet",
        "source",
        "target",
        "input",
        "value",
        "gas_limit",
        "max_fee_per_gas",
        "max_priority_fee_per_gas",
        "nonce",
        "access_list",
        "authorization_list",
    ],
)  # args: [source: H160, target: H160, input: Vec<u8>, value: U256, gas_limit: u64, max_fee_per_gas: U256, max_priority_fee_per_gas: Option<U256>, nonce: Option<U256>, access_list: Vec<(H160, Vec<H256>)>, authorization_list: AuthorizationList]  | Pallet: EVM
CANCEL = namedtuple(
    "CANCEL", ["wallet", "pallet", "when", "index"]
)  # args: [when: BlockNumberFor<T>, index: u32]  | Pallet: Scheduler
CANCEL_AS_MULTI = namedtuple(
    "CANCEL_AS_MULTI",
    ["wallet", "pallet", "threshold", "other_signatories", "timepoint", "call_hash"],
)  # args: [threshold: u16, other_signatories: Vec<T::AccountId>, timepoint: Timepoint<BlockNumberFor<T>>, call_hash: [u8; 32]]  | Pallet: Multisig
CANCEL_NAMED = namedtuple(
    "CANCEL_NAMED", ["wallet", "pallet", "id"]
)  # args: [id: TaskName]  | Pallet: Scheduler
CANCEL_RETRY = namedtuple(
    "CANCEL_RETRY", ["wallet", "pallet", "task"]
)  # args: [task: TaskAddress<BlockNumberFor<T>>]  | Pallet: Scheduler
CANCEL_RETRY_NAMED = namedtuple(
    "CANCEL_RETRY_NAMED", ["wallet", "pallet", "id"]
)  # args: [id: TaskName]  | Pallet: Scheduler
CLAIM_ROOT = namedtuple(
    "CLAIM_ROOT",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SubtensorModule
CLEAR_IDENTITY = namedtuple(
    "CLEAR_IDENTITY", ["wallet", "pallet", "identified"]
)  # args: [identified: T::AccountId]  | Pallet: Registry
COMMIT_CRV3_MECHANISM_WEIGHTS = namedtuple(
    "COMMIT_CRV3_MECHANISM_WEIGHTS",
    ["wallet", "pallet", "netuid", "mecid", "commit", "reveal_round"],
)  # args: [netuid: NetUid, mecid: MechId, commit: BoundedVec<u8, ConstU32<MAX_CRV3_COMMIT_SIZE_BYTES>>, reveal_round: u64]  | Pallet: SubtensorModule
COMMIT_MECHANISM_WEIGHTS = namedtuple(
    "COMMIT_MECHANISM_WEIGHTS", ["wallet", "pallet", "netuid", "mecid", "commit_hash"]
)  # args: [netuid: NetUid, mecid: MechId, commit_hash: H256]  | Pallet: SubtensorModule
COMMIT_TIMELOCKED_MECHANISM_WEIGHTS = namedtuple(
    "COMMIT_TIMELOCKED_MECHANISM_WEIGHTS",
    [
        "wallet",
        "pallet",
        "netuid",
        "mecid",
        "commit",
        "reveal_round",
        "commit_reveal_version",
    ],
)  # args: [netuid: NetUid, mecid: MechId, commit: BoundedVec<u8, ConstU32<MAX_CRV3_COMMIT_SIZE_BYTES>>, reveal_round: u64, commit_reveal_version: u16]  | Pallet: SubtensorModule
COMMIT_TIMELOCKED_WEIGHTS = namedtuple(
    "COMMIT_TIMELOCKED_WEIGHTS",
    ["wallet", "pallet", "netuid", "commit", "reveal_round", "commit_reveal_version"],
)  # args: [netuid: NetUid, commit: BoundedVec<u8, ConstU32<MAX_CRV3_COMMIT_SIZE_BYTES>>, reveal_round: u64, commit_reveal_version: u16]  | Pallet: SubtensorModule
COMMIT_WEIGHTS = namedtuple(
    "COMMIT_WEIGHTS", ["wallet", "pallet", "netuid", "commit_hash"]
)  # args: [netuid: NetUid, commit_hash: H256]  | Pallet: SubtensorModule
CONTRIBUTE = namedtuple(
    "CONTRIBUTE", ["wallet", "pallet", "crowdloan_id", "amount"]
)  # args: [crowdloan_id: CrowdloanId, amount: BalanceOf<T>]  | Pallet: Crowdloan
CREATE = namedtuple(
    "CREATE",
    [
        "wallet",
        "pallet",
        "deposit",
        "min_contribution",
        "cap",
        "end",
        "call",
        "target_address",
    ],
)  # args: [deposit: BalanceOf<T>, min_contribution: BalanceOf<T>, cap: BalanceOf<T>, end: BlockNumberFor<T>, call: Option<Box<<T as Config>::RuntimeCall>>, target_address: Option<T::AccountId>]  | Pallet: Crowdloan
CREATE = namedtuple(
    "CREATE",
    [
        "wallet",
        "pallet",
        "source",
        "init",
        "value",
        "gas_limit",
        "max_fee_per_gas",
        "max_priority_fee_per_gas",
        "nonce",
        "access_list",
        "authorization_list",
    ],
)  # args: [source: H160, init: Vec<u8>, value: U256, gas_limit: u64, max_fee_per_gas: U256, max_priority_fee_per_gas: Option<U256>, nonce: Option<U256>, access_list: Vec<(H160, Vec<H256>)>, authorization_list: AuthorizationList]  | Pallet: EVM
CREATE2 = namedtuple(
    "CREATE2",
    [
        "wallet",
        "pallet",
        "source",
        "init",
        "salt",
        "value",
        "gas_limit",
        "max_fee_per_gas",
        "max_priority_fee_per_gas",
        "nonce",
        "access_list",
        "authorization_list",
    ],
)  # args: [source: H160, init: Vec<u8>, salt: H256, value: U256, gas_limit: u64, max_fee_per_gas: U256, max_priority_fee_per_gas: Option<U256>, nonce: Option<U256>, access_list: Vec<(H160, Vec<H256>)>, authorization_list: AuthorizationList]  | Pallet: EVM
CREATE_PURE = namedtuple(
    "CREATE_PURE", ["wallet", "pallet", "proxy_type", "delay", "index"]
)  # args: [proxy_type: T::ProxyType, delay: BlockNumberFor<T>, index: u16]  | Pallet: Proxy
DECREASE_TAKE = namedtuple(
    "DECREASE_TAKE", ["wallet", "pallet", "hotkey", "take"]
)  # args: [hotkey: T::AccountId, take: u16]  | Pallet: SubtensorModule
DISABLE_WHITELIST = namedtuple(
    "DISABLE_WHITELIST", ["wallet", "pallet", "disabled"]
)  # args: [disabled: bool]  | Pallet: EVM
DISPATCH_AS = namedtuple(
    "DISPATCH_AS", ["wallet", "pallet", "as_origin", "call"]
)  # args: [as_origin: Box<T::PalletsOrigin>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Utility
DISPATCH_AS_FALLIBLE = namedtuple(
    "DISPATCH_AS_FALLIBLE", ["wallet", "pallet", "as_origin", "call"]
)  # args: [as_origin: Box<T::PalletsOrigin>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Utility
DISSOLVE = namedtuple(
    "DISSOLVE", ["wallet", "pallet", "crowdloan_id"]
)  # args: [crowdloan_id: CrowdloanId]  | Pallet: Crowdloan
DISSOLVE_NETWORK = namedtuple(
    "DISSOLVE_NETWORK", ["wallet", "pallet", "coldkey", "netuid"]
)  # args: [coldkey: T::AccountId, netuid: NetUid]  | Pallet: SubtensorModule
ENSURE_UPDATED = namedtuple(
    "ENSURE_UPDATED", ["wallet", "pallet", "hashes"]
)  # args: [hashes: Vec<T::Hash>]  | Pallet: Preimage
ENTER = namedtuple(
    "ENTER",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SafeMode
EXTEND = namedtuple(
    "EXTEND",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SafeMode
FAUCET = namedtuple(
    "FAUCET", ["wallet", "pallet", "block_number", "nonce", "work"]
)  # args: [block_number: u64, nonce: u64, work: Vec<u8>]  | Pallet: SubtensorModule
FINALIZE = namedtuple(
    "FINALIZE", ["wallet", "pallet", "crowdloan_id"]
)  # args: [crowdloan_id: CrowdloanId]  | Pallet: Crowdloan
FORCE_ADJUST_TOTAL_ISSUANCE = namedtuple(
    "FORCE_ADJUST_TOTAL_ISSUANCE", ["wallet", "pallet", "direction", "delta"]
)  # args: [direction: AdjustmentDirection, delta: T::Balance]  | Pallet: Balances
FORCE_BATCH = namedtuple(
    "FORCE_BATCH", ["wallet", "pallet", "calls"]
)  # args: [calls: Vec<<T as Config>::RuntimeCall>]  | Pallet: Utility
FORCE_ENTER = namedtuple(
    "FORCE_ENTER",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SafeMode
FORCE_EXIT = namedtuple(
    "FORCE_EXIT",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SafeMode
FORCE_EXTEND = namedtuple(
    "FORCE_EXTEND",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: SafeMode
FORCE_RELEASE_DEPOSIT = namedtuple(
    "FORCE_RELEASE_DEPOSIT", ["wallet", "pallet", "account", "block"]
)  # args: [account: T::AccountId, block: BlockNumberFor<T>]  | Pallet: SafeMode
FORCE_SET_BALANCE = namedtuple(
    "FORCE_SET_BALANCE", ["wallet", "pallet", "who", "new_free"]
)  # args: [who: AccountIdLookupOf<T>, new_free: T::Balance]  | Pallet: Balances
FORCE_SLASH_DEPOSIT = namedtuple(
    "FORCE_SLASH_DEPOSIT", ["wallet", "pallet", "account", "block"]
)  # args: [account: T::AccountId, block: BlockNumberFor<T>]  | Pallet: SafeMode
FORCE_TRANSFER = namedtuple(
    "FORCE_TRANSFER", ["wallet", "pallet", "source", "dest", "value"]
)  # args: [source: AccountIdLookupOf<T>, dest: AccountIdLookupOf<T>, value: T::Balance]  | Pallet: Balances
FORCE_UNRESERVE = namedtuple(
    "FORCE_UNRESERVE", ["wallet", "pallet", "who", "amount"]
)  # args: [who: AccountIdLookupOf<T>, amount: T::Balance]  | Pallet: Balances
IF_ELSE = namedtuple(
    "IF_ELSE", ["wallet", "pallet", "main", "fallback"]
)  # args: [main: Box<<T as Config>::RuntimeCall>, fallback: Box<<T as Config>::RuntimeCall>]  | Pallet: Utility
INCREASE_TAKE = namedtuple(
    "INCREASE_TAKE", ["wallet", "pallet", "hotkey", "take"]
)  # args: [hotkey: T::AccountId, take: u16]  | Pallet: SubtensorModule
KILL_PREFIX = namedtuple(
    "KILL_PREFIX", ["wallet", "pallet", "prefix", "subkeys"]
)  # args: [prefix: Key, subkeys: u32]  | Pallet: System
KILL_PURE = namedtuple(
    "KILL_PURE",
    ["wallet", "pallet", "spawner", "proxy_type", "index", "height", "ext_index"],
)  # args: [spawner: AccountIdLookupOf<T>, proxy_type: T::ProxyType, index: u16, height: BlockNumberFor<T>, ext_index: u32]  | Pallet: Proxy
KILL_STORAGE = namedtuple(
    "KILL_STORAGE", ["wallet", "pallet", "keys"]
)  # args: [keys: Vec<Key>]  | Pallet: System
MODIFY_POSITION = namedtuple(
    "MODIFY_POSITION",
    ["wallet", "pallet", "hotkey", "netuid", "position_id", "liquidity_delta"],
)  # args: [hotkey: T::AccountId, netuid: NetUid, position_id: PositionId, liquidity_delta: i64]  | Pallet: Swap
MOVE_STAKE = namedtuple(
    "MOVE_STAKE",
    [
        "wallet",
        "pallet",
        "origin_hotkey",
        "destination_hotkey",
        "origin_netuid",
        "destination_netuid",
        "alpha_amount",
    ],
)  # args: [origin_hotkey: T::AccountId, destination_hotkey: T::AccountId, origin_netuid: NetUid, destination_netuid: NetUid, alpha_amount: AlphaCurrency]  | Pallet: SubtensorModule
NOTE_PREIMAGE = namedtuple(
    "NOTE_PREIMAGE", ["wallet", "pallet", "bytes"]
)  # args: [bytes: Vec<u8>]  | Pallet: Preimage
NOTE_STALLED = namedtuple(
    "NOTE_STALLED", ["wallet", "pallet", "delay", "best_finalized_block_number"]
)  # args: [delay: BlockNumberFor<T>, best_finalized_block_number: BlockNumberFor<T>]  | Pallet: Grandpa
POKE_DEPOSIT = namedtuple(
    "POKE_DEPOSIT", ["wallet", "pallet", "threshold", "other_signatories", "call_hash"]
)  # args: [threshold: u16, other_signatories: Vec<T::AccountId>, call_hash: [u8; 32]]  | Pallet: Multisig
POKE_DEPOSIT = namedtuple(
    "POKE_DEPOSIT",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: Proxy
PROXY = namedtuple(
    "PROXY", ["wallet", "pallet", "real", "force_proxy_type", "call"]
)  # args: [real: AccountIdLookupOf<T>, force_proxy_type: Option<T::ProxyType>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Proxy
PROXY_ANNOUNCED = namedtuple(
    "PROXY_ANNOUNCED",
    ["wallet", "pallet", "delegate", "real", "force_proxy_type", "call"],
)  # args: [delegate: AccountIdLookupOf<T>, real: AccountIdLookupOf<T>, force_proxy_type: Option<T::ProxyType>, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Proxy
RECYCLE_ALPHA = namedtuple(
    "RECYCLE_ALPHA", ["wallet", "pallet", "hotkey", "amount", "netuid"]
)  # args: [hotkey: T::AccountId, amount: AlphaCurrency, netuid: NetUid]  | Pallet: SubtensorModule
REFUND = namedtuple(
    "REFUND", ["wallet", "pallet", "crowdloan_id"]
)  # args: [crowdloan_id: CrowdloanId]  | Pallet: Crowdloan
REGISTER = namedtuple(
    "REGISTER",
    [
        "wallet",
        "pallet",
        "netuid",
        "block_number",
        "nonce",
        "work",
        "hotkey",
        "coldkey",
    ],
)  # args: [netuid: NetUid, block_number: u64, nonce: u64, work: Vec<u8>, hotkey: T::AccountId, coldkey: T::AccountId]  | Pallet: SubtensorModule
REGISTER_LEASED_NETWORK = namedtuple(
    "REGISTER_LEASED_NETWORK", ["wallet", "pallet", "emissions_share", "end_block"]
)  # args: [emissions_share: Percent, end_block: Option<BlockNumberFor<T>>]  | Pallet: SubtensorModule
REGISTER_NETWORK = namedtuple(
    "REGISTER_NETWORK", ["wallet", "pallet", "hotkey"]
)  # args: [hotkey: T::AccountId]  | Pallet: SubtensorModule
REGISTER_NETWORK_WITH_IDENTITY = namedtuple(
    "REGISTER_NETWORK_WITH_IDENTITY", ["wallet", "pallet", "hotkey", "identity"]
)  # args: [hotkey: T::AccountId, identity: Option<SubnetIdentityOfV3>]  | Pallet: SubtensorModule
REJECT_ANNOUNCEMENT = namedtuple(
    "REJECT_ANNOUNCEMENT", ["wallet", "pallet", "delegate", "call_hash"]
)  # args: [delegate: AccountIdLookupOf<T>, call_hash: CallHashOf<T>]  | Pallet: Proxy
RELEASE_DEPOSIT = namedtuple(
    "RELEASE_DEPOSIT", ["wallet", "pallet", "account", "block"]
)  # args: [account: T::AccountId, block: BlockNumberFor<T>]  | Pallet: SafeMode
REMARK = namedtuple(
    "REMARK", ["wallet", "pallet", "remark"]
)  # args: [remark: Vec<u8>]  | Pallet: System
REMARK_WITH_EVENT = namedtuple(
    "REMARK_WITH_EVENT", ["wallet", "pallet", "remark"]
)  # args: [remark: Vec<u8>]  | Pallet: System
REMOVE_ANNOUNCEMENT = namedtuple(
    "REMOVE_ANNOUNCEMENT", ["wallet", "pallet", "real", "call_hash"]
)  # args: [real: AccountIdLookupOf<T>, call_hash: CallHashOf<T>]  | Pallet: Proxy
REMOVE_KEY = namedtuple(
    "REMOVE_KEY",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: Sudo
REMOVE_LIQUIDITY = namedtuple(
    "REMOVE_LIQUIDITY", ["wallet", "pallet", "hotkey", "netuid", "position_id"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, position_id: PositionId]  | Pallet: Swap
REMOVE_PROXIES = namedtuple(
    "REMOVE_PROXIES",
    [
        "wallet",
        "pallet",
    ],
)  # args: []  | Pallet: Proxy
REMOVE_PROXY = namedtuple(
    "REMOVE_PROXY", ["wallet", "pallet", "delegate", "proxy_type", "delay"]
)  # args: [delegate: AccountIdLookupOf<T>, proxy_type: T::ProxyType, delay: BlockNumberFor<T>]  | Pallet: Proxy
REMOVE_STAKE = namedtuple(
    "REMOVE_STAKE", ["wallet", "pallet", "hotkey", "netuid", "amount_unstaked"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, amount_unstaked: AlphaCurrency]  | Pallet: SubtensorModule
REMOVE_STAKE_FULL_LIMIT = namedtuple(
    "REMOVE_STAKE_FULL_LIMIT", ["wallet", "pallet", "hotkey", "netuid", "limit_price"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, limit_price: Option<TaoCurrency>]  | Pallet: SubtensorModule
REMOVE_STAKE_LIMIT = namedtuple(
    "REMOVE_STAKE_LIMIT",
    [
        "wallet",
        "pallet",
        "hotkey",
        "netuid",
        "amount_unstaked",
        "limit_price",
        "allow_partial",
    ],
)  # args: [hotkey: T::AccountId, netuid: NetUid, amount_unstaked: AlphaCurrency, limit_price: TaoCurrency, allow_partial: bool]  | Pallet: SubtensorModule
REPORT_EQUIVOCATION = namedtuple(
    "REPORT_EQUIVOCATION", ["wallet", "pallet", "equivocation_proof", "key_owner_proof"]
)  # args: [equivocation_proof: Box<EquivocationProof<T::Hash, BlockNumberFor<T>>>, key_owner_proof: T::KeyOwnerProof]  | Pallet: Grandpa
REPORT_EQUIVOCATION_UNSIGNED = namedtuple(
    "REPORT_EQUIVOCATION_UNSIGNED",
    ["wallet", "pallet", "equivocation_proof", "key_owner_proof"],
)  # args: [equivocation_proof: Box<EquivocationProof<T::Hash, BlockNumberFor<T>>>, key_owner_proof: T::KeyOwnerProof]  | Pallet: Grandpa
REQUEST_PREIMAGE = namedtuple(
    "REQUEST_PREIMAGE", ["wallet", "pallet", "hash"]
)  # args: [hash: T::Hash]  | Pallet: Preimage
REVEAL_MECHANISM_WEIGHTS = namedtuple(
    "REVEAL_MECHANISM_WEIGHTS",
    ["wallet", "pallet", "netuid", "mecid", "uids", "values", "salt", "version_key"],
)  # args: [netuid: NetUid, mecid: MechId, uids: Vec<u16>, values: Vec<u16>, salt: Vec<u16>, version_key: u64]  | Pallet: SubtensorModule
REVEAL_WEIGHTS = namedtuple(
    "REVEAL_WEIGHTS",
    ["wallet", "pallet", "netuid", "uids", "values", "salt", "version_key"],
)  # args: [netuid: NetUid, uids: Vec<u16>, values: Vec<u16>, salt: Vec<u16>, version_key: u64]  | Pallet: SubtensorModule
ROOT_DISSOLVE_NETWORK = namedtuple(
    "ROOT_DISSOLVE_NETWORK", ["wallet", "pallet", "netuid"]
)  # args: [netuid: NetUid]  | Pallet: SubtensorModule
ROOT_REGISTER = namedtuple(
    "ROOT_REGISTER", ["wallet", "pallet", "hotkey"]
)  # args: [hotkey: T::AccountId]  | Pallet: SubtensorModule
SCHEDULE = namedtuple(
    "SCHEDULE", ["wallet", "pallet", "when", "maybe_periodic", "priority", "call"]
)  # args: [when: BlockNumberFor<T>, maybe_periodic: Option<schedule::Period<BlockNumberFor<T>>>, priority: schedule::Priority, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Scheduler
SCHEDULE_AFTER = namedtuple(
    "SCHEDULE_AFTER",
    ["wallet", "pallet", "after", "maybe_periodic", "priority", "call"],
)  # args: [after: BlockNumberFor<T>, maybe_periodic: Option<schedule::Period<BlockNumberFor<T>>>, priority: schedule::Priority, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Scheduler
SCHEDULE_GRANDPA_CHANGE = namedtuple(
    "SCHEDULE_GRANDPA_CHANGE",
    ["wallet", "pallet", "next_authorities", "in_blocks", "forced"],
)  # args: [next_authorities: AuthorityList, in_blocks: BlockNumberFor<T>, forced: Option<BlockNumberFor<T>>]  | Pallet: AdminUtils
SCHEDULE_NAMED = namedtuple(
    "SCHEDULE_NAMED",
    ["wallet", "pallet", "id", "when", "maybe_periodic", "priority", "call"],
)  # args: [id: TaskName, when: BlockNumberFor<T>, maybe_periodic: Option<schedule::Period<BlockNumberFor<T>>>, priority: schedule::Priority, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Scheduler
SCHEDULE_NAMED_AFTER = namedtuple(
    "SCHEDULE_NAMED_AFTER",
    ["wallet", "pallet", "id", "after", "maybe_periodic", "priority", "call"],
)  # args: [id: TaskName, after: BlockNumberFor<T>, maybe_periodic: Option<schedule::Period<BlockNumberFor<T>>>, priority: schedule::Priority, call: Box<<T as Config>::RuntimeCall>]  | Pallet: Scheduler
SCHEDULE_SWAP_COLDKEY = namedtuple(
    "SCHEDULE_SWAP_COLDKEY", ["wallet", "pallet", "new_coldkey"]
)  # args: [new_coldkey: T::AccountId]  | Pallet: SubtensorModule
SERVE_AXON = namedtuple(
    "SERVE_AXON",
    [
        "wallet",
        "pallet",
        "netuid",
        "version",
        "ip",
        "port",
        "ip_type",
        "protocol",
        "placeholder1",
        "placeholder2",
    ],
)  # args: [netuid: NetUid, version: u32, ip: u128, port: u16, ip_type: u8, protocol: u8, placeholder1: u8, placeholder2: u8]  | Pallet: SubtensorModule
SERVE_AXON_TLS = namedtuple(
    "SERVE_AXON_TLS",
    [
        "wallet",
        "pallet",
        "netuid",
        "version",
        "ip",
        "port",
        "ip_type",
        "protocol",
        "placeholder1",
        "placeholder2",
        "certificate",
    ],
)  # args: [netuid: NetUid, version: u32, ip: u128, port: u16, ip_type: u8, protocol: u8, placeholder1: u8, placeholder2: u8, certificate: Vec<u8>]  | Pallet: SubtensorModule
SERVE_PROMETHEUS = namedtuple(
    "SERVE_PROMETHEUS",
    ["wallet", "pallet", "netuid", "version", "ip", "port", "ip_type"],
)  # args: [netuid: NetUid, version: u32, ip: u128, port: u16, ip_type: u8]  | Pallet: SubtensorModule
SET = namedtuple(
    "SET", ["wallet", "pallet", "now"]
)  # args: [now: T::Moment]  | Pallet: Timestamp
SET_BASE_FEE_PER_GAS = namedtuple(
    "SET_BASE_FEE_PER_GAS", ["wallet", "pallet", "fee"]
)  # args: [fee: U256]  | Pallet: BaseFee
SET_BEACON_CONFIG = namedtuple(
    "SET_BEACON_CONFIG", ["wallet", "pallet", "config_payload", "signature"]
)  # args: [config_payload: BeaconConfigurationPayload<T::Public, BlockNumberFor<T>>, signature: Option<T::Signature>]  | Pallet: Drand
SET_CHILDKEY_TAKE = namedtuple(
    "SET_CHILDKEY_TAKE", ["wallet", "pallet", "hotkey", "netuid", "take"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, take: u16]  | Pallet: SubtensorModule
SET_CHILDREN = namedtuple(
    "SET_CHILDREN", ["wallet", "pallet", "hotkey", "netuid", "children"]
)  # args: [hotkey: T::AccountId, netuid: NetUid, children: Vec<(u64, T::AccountId)>]  | Pallet: SubtensorModule
SET_CODE = namedtuple(
    "SET_CODE", ["wallet", "pallet", "code"]
)  # args: [code: Vec<u8>]  | Pallet: System
SET_CODE_WITHOUT_CHECKS = namedtuple(
    "SET_CODE_WITHOUT_CHECKS", ["wallet", "pallet", "code"]
)  # args: [code: Vec<u8>]  | Pallet: System
SET_COLDKEY_AUTO_STAKE_HOTKEY = namedtuple(
    "SET_COLDKEY_AUTO_STAKE_HOTKEY", ["wallet", "pallet", "netuid", "hotkey"]
)  # args: [netuid: NetUid, hotkey: T::AccountId]  | Pallet: SubtensorModule
SET_COMMITMENT = namedtuple(
    "SET_COMMITMENT", ["wallet", "pallet", "netuid", "info"]
)  # args: [netuid: NetUid, info: Box<CommitmentInfo<T::MaxFields>>]  | Pallet: Commitments
SET_ELASTICITY = namedtuple(
    "SET_ELASTICITY", ["wallet", "pallet", "elasticity"]
)  # args: [elasticity: Permill]  | Pallet: BaseFee
SET_FEE_RATE = namedtuple(
    "SET_FEE_RATE", ["wallet", "pallet", "netuid", "rate"]
)  # args: [netuid: NetUid, rate: u16]  | Pallet: Swap
SET_HEAP_PAGES = namedtuple(
    "SET_HEAP_PAGES", ["wallet", "pallet", "pages"]
)  # args: [pages: u64]  | Pallet: System
SET_IDENTITY = namedtuple(
    "SET_IDENTITY", ["wallet", "pallet", "identified", "info"]
)  # args: [identified: T::AccountId, info: Box<IdentityInfo<T::MaxAdditionalFields>>]  | Pallet: Registry
SET_IDENTITY = namedtuple(
    "SET_IDENTITY",
    [
        "wallet",
        "pallet",
        "name",
        "url",
        "github_repo",
        "image",
        "discord",
        "description",
        "additional",
    ],
)  # args: [name: Vec<u8>, url: Vec<u8>, github_repo: Vec<u8>, image: Vec<u8>, discord: Vec<u8>, description: Vec<u8>, additional: Vec<u8>]  | Pallet: SubtensorModule
SET_KEY = namedtuple(
    "SET_KEY", ["wallet", "pallet", "new"]
)  # args: [new: AccountIdLookupOf<T>]  | Pallet: Sudo
SET_MAX_SPACE = namedtuple(
    "SET_MAX_SPACE", ["wallet", "pallet", "new_limit"]
)  # args: [new_limit: u32]  | Pallet: Commitments
SET_MECHANISM_WEIGHTS = namedtuple(
    "SET_MECHANISM_WEIGHTS",
    ["wallet", "pallet", "netuid", "mecid", "dests", "weights", "version_key"],
)  # args: [netuid: NetUid, mecid: MechId, dests: Vec<u16>, weights: Vec<u16>, version_key: u64]  | Pallet: SubtensorModule
SET_OLDEST_STORED_ROUND = namedtuple(
    "SET_OLDEST_STORED_ROUND", ["wallet", "pallet", "oldest_round"]
)  # args: [oldest_round: u64]  | Pallet: Drand
SET_PENDING_CHILDKEY_COOLDOWN = namedtuple(
    "SET_PENDING_CHILDKEY_COOLDOWN", ["wallet", "pallet", "cooldown"]
)  # args: [cooldown: u64]  | Pallet: SubtensorModule
SET_RETRY = namedtuple(
    "SET_RETRY", ["wallet", "pallet", "task", "retries", "period"]
)  # args: [task: TaskAddress<BlockNumberFor<T>>, retries: u8, period: BlockNumberFor<T>]  | Pallet: Scheduler
SET_RETRY_NAMED = namedtuple(
    "SET_RETRY_NAMED", ["wallet", "pallet", "id", "retries", "period"]
)  # args: [id: TaskName, retries: u8, period: BlockNumberFor<T>]  | Pallet: Scheduler
SET_ROOT_CLAIM_TYPE = namedtuple(
    "SET_ROOT_CLAIM_TYPE", ["wallet", "pallet", "new_root_claim_type"]
)  # args: [new_root_claim_type: RootClaimTypeEnum]  | Pallet: SubtensorModule
SET_STORAGE = namedtuple(
    "SET_STORAGE", ["wallet", "pallet", "items"]
)  # args: [items: Vec<KeyValue>]  | Pallet: System
SET_SUBNET_IDENTITY = namedtuple(
    "SET_SUBNET_IDENTITY",
    [
        "wallet",
        "pallet",
        "netuid",
        "subnet_name",
        "github_repo",
        "subnet_contact",
        "subnet_url",
        "discord",
        "description",
        "logo_url",
        "additional",
    ],
)  # args: [netuid: NetUid, subnet_name: Vec<u8>, github_repo: Vec<u8>, subnet_contact: Vec<u8>, subnet_url: Vec<u8>, discord: Vec<u8>, description: Vec<u8>, logo_url: Vec<u8>, additional: Vec<u8>]  | Pallet: SubtensorModule
SET_WEIGHTS = namedtuple(
    "SET_WEIGHTS", ["wallet", "pallet", "netuid", "dests", "weights", "version_key"]
)  # args: [netuid: NetUid, dests: Vec<u16>, weights: Vec<u16>, version_key: u64]  | Pallet: SubtensorModule
SET_WHITELIST = namedtuple(
    "SET_WHITELIST", ["wallet", "pallet", "new"]
)  # args: [new: Vec<H160>]  | Pallet: EVM
START_CALL = namedtuple(
    "START_CALL", ["wallet", "pallet", "netuid"]
)  # args: [netuid: NetUid]  | Pallet: SubtensorModule
SUDO = namedtuple(
    "SUDO", ["wallet", "pallet", "call"]
)  # args: [call: Box<<T as Config>::RuntimeCall>]  | Pallet: Sudo
SWAP_AUTHORITIES = namedtuple(
    "SWAP_AUTHORITIES", ["wallet", "pallet", "new_authorities"]
)  # args: [new_authorities: BoundedVec<<T as Config>::AuthorityId, T::MaxAuthorities>]  | Pallet: AdminUtils
SWAP_COLDKEY = namedtuple(
    "SWAP_COLDKEY", ["wallet", "pallet", "old_coldkey", "new_coldkey", "swap_cost"]
)  # args: [old_coldkey: T::AccountId, new_coldkey: T::AccountId, swap_cost: TaoCurrency]  | Pallet: SubtensorModule
SWAP_HOTKEY = namedtuple(
    "SWAP_HOTKEY", ["wallet", "pallet", "hotkey", "new_hotkey", "netuid"]
)  # args: [hotkey: T::AccountId, new_hotkey: T::AccountId, netuid: Option<NetUid>]  | Pallet: SubtensorModule
SWAP_STAKE = namedtuple(
    "SWAP_STAKE",
    [
        "wallet",
        "pallet",
        "hotkey",
        "origin_netuid",
        "destination_netuid",
        "alpha_amount",
    ],
)  # args: [hotkey: T::AccountId, origin_netuid: NetUid, destination_netuid: NetUid, alpha_amount: AlphaCurrency]  | Pallet: SubtensorModule
SWAP_STAKE_LIMIT = namedtuple(
    "SWAP_STAKE_LIMIT",
    [
        "wallet",
        "pallet",
        "hotkey",
        "origin_netuid",
        "destination_netuid",
        "alpha_amount",
        "limit_price",
        "allow_partial",
    ],
)  # args: [hotkey: T::AccountId, origin_netuid: NetUid, destination_netuid: NetUid, alpha_amount: AlphaCurrency, limit_price: TaoCurrency, allow_partial: bool]  | Pallet: SubtensorModule
TERMINATE_LEASE = namedtuple(
    "TERMINATE_LEASE", ["wallet", "pallet", "lease_id", "hotkey"]
)  # args: [lease_id: LeaseId, hotkey: T::AccountId]  | Pallet: SubtensorModule
TOGGLE_USER_LIQUIDITY = namedtuple(
    "TOGGLE_USER_LIQUIDITY", ["wallet", "pallet", "netuid", "enable"]
)  # args: [netuid: NetUid, enable: bool]  | Pallet: Swap
TRANSACT = namedtuple(
    "TRANSACT", ["wallet", "pallet", "transaction"]
)  # args: [transaction: Transaction]  | Pallet: Ethereum
TRANSFER_ALL = namedtuple(
    "TRANSFER_ALL", ["wallet", "pallet", "dest", "keep_alive"]
)  # args: [dest: AccountIdLookupOf<T>, keep_alive: bool]  | Pallet: Balances
TRANSFER_ALLOW_DEATH = namedtuple(
    "TRANSFER_ALLOW_DEATH", ["wallet", "pallet", "dest", "value"]
)  # args: [dest: AccountIdLookupOf<T>, value: T::Balance]  | Pallet: Balances
TRANSFER_KEEP_ALIVE = namedtuple(
    "TRANSFER_KEEP_ALIVE", ["wallet", "pallet", "dest", "value"]
)  # args: [dest: AccountIdLookupOf<T>, value: T::Balance]  | Pallet: Balances
TRANSFER_STAKE = namedtuple(
    "TRANSFER_STAKE",
    [
        "wallet",
        "pallet",
        "destination_coldkey",
        "hotkey",
        "origin_netuid",
        "destination_netuid",
        "alpha_amount",
    ],
)  # args: [destination_coldkey: T::AccountId, hotkey: T::AccountId, origin_netuid: NetUid, destination_netuid: NetUid, alpha_amount: AlphaCurrency]  | Pallet: SubtensorModule
TRY_ASSOCIATE_HOTKEY = namedtuple(
    "TRY_ASSOCIATE_HOTKEY", ["wallet", "pallet", "hotkey"]
)  # args: [hotkey: T::AccountId]  | Pallet: SubtensorModule
UNNOTE_PREIMAGE = namedtuple(
    "UNNOTE_PREIMAGE", ["wallet", "pallet", "hash"]
)  # args: [hash: T::Hash]  | Pallet: Preimage
UNREQUEST_PREIMAGE = namedtuple(
    "UNREQUEST_PREIMAGE", ["wallet", "pallet", "hash"]
)  # args: [hash: T::Hash]  | Pallet: Preimage
UNSTAKE_ALL = namedtuple(
    "UNSTAKE_ALL", ["wallet", "pallet", "hotkey"]
)  # args: [hotkey: T::AccountId]  | Pallet: SubtensorModule
UNSTAKE_ALL_ALPHA = namedtuple(
    "UNSTAKE_ALL_ALPHA", ["wallet", "pallet", "hotkey"]
)  # args: [hotkey: T::AccountId]  | Pallet: SubtensorModule
UPDATE_CAP = namedtuple(
    "UPDATE_CAP", ["wallet", "pallet", "crowdloan_id", "new_cap"]
)  # args: [crowdloan_id: CrowdloanId, new_cap: BalanceOf<T>]  | Pallet: Crowdloan
UPDATE_END = namedtuple(
    "UPDATE_END", ["wallet", "pallet", "crowdloan_id", "new_end"]
)  # args: [crowdloan_id: CrowdloanId, new_end: BlockNumberFor<T>]  | Pallet: Crowdloan
UPDATE_MIN_CONTRIBUTION = namedtuple(
    "UPDATE_MIN_CONTRIBUTION",
    ["wallet", "pallet", "crowdloan_id", "new_min_contribution"],
)  # args: [crowdloan_id: CrowdloanId, new_min_contribution: BalanceOf<T>]  | Pallet: Crowdloan
UPDATE_SYMBOL = namedtuple(
    "UPDATE_SYMBOL", ["wallet", "pallet", "netuid", "symbol"]
)  # args: [netuid: NetUid, symbol: Vec<u8>]  | Pallet: SubtensorModule
UPGRADE_ACCOUNTS = namedtuple(
    "UPGRADE_ACCOUNTS", ["wallet", "pallet", "who"]
)  # args: [who: Vec<T::AccountId>]  | Pallet: Balances
WITHDRAW = namedtuple(
    "WITHDRAW", ["wallet", "pallet", "address", "value"]
)  # args: [address: H160, value: BalanceOf<T>]  | Pallet: EVM
WITHDRAW = namedtuple(
    "WITHDRAW", ["wallet", "pallet", "crowdloan_id"]
)  # args: [crowdloan_id: CrowdloanId]  | Pallet: Crowdloan
WITH_WEIGHT = namedtuple(
    "WITH_WEIGHT", ["wallet", "pallet", "call", "weight"]
)  # args: [call: Box<<T as Config>::RuntimeCall>, weight: Weight]  | Pallet: Utility
WRITE_PULSE = namedtuple(
    "WRITE_PULSE", ["wallet", "pallet", "pulses_payload", "signature"]
)  # args: [pulses_payload: PulsesPayload<T::Public, BlockNumberFor<T>>, signature: Option<T::Signature>]  | Pallet: Drand
