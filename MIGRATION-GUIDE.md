---
title: "Bittensor 10.0 Migration Guide"
---

# Bittensor 10.0 Migration Guide

This page documents breaking changes and new features for the Bittensor Python SDK `v10.0`. This is a major release with significant refactoring, standardization, and new functionality.

## Executive Summary

Bittensor SDK v10.0 is a **major breaking release** with significant improvements to consistency, type safety, and functionality. Key changes include:

**Breaking Changes:**
- **Python 3.10+ required** - Python 3.9 no longer supported ([details](#python-version-support))
- **ExtrinsicResponse return type** - All blockchain transaction functions now return structured `ExtrinsicResponse` objects instead of `bool` or tuples ([details](#extrinsicresponse-return-type))
- **Strict Balance type checking** - All amount parameters require `Balance` objects. ([details](#balance-handling))
- **Parameter renames** - Consistent `_ss58` suffix for all address parameters (e.g., `hotkey` → `hotkey_ss58`) ([details](#parameter-renames))
- **Method renames and removals** - Several methods renamed or removed for consistency ([details](#breaking-changes-method-renames))

**New Features:**
- **Multiple Incentive Mechanisms** - Full SDK support for running multiple evaluation mechanisms per subnet with independent weight matrices and emissions ([details](#multiple-incentive-mechanisms-support))
- **Transaction simulation** - `sim_swap()` calculates exact token yields without executing transactions ([details](#simulate-token-swaps))
- **Fee estimation** - `get_extrinsic_fee()` estimates blockchain transaction costs before submission ([details](#estimate-transaction-fees))
- **BlockInfo class** - Rich blockchain block information objects ([details](#blockinfo-class))

**Major Improvements:**
- Standardized parameter ordering across all functions ([details](#standardized-parameters))
- Centralized extrinsic parameters in `bittensor.core.extrinsics.params` ([details](#extrinsic-parameters-package))
- Enhanced metagraph support for mechanism-specific queries ([details](#metagraph-changes))
- Environment variable updates for clarity ([details](#environment-variables))

**Migration Required:**
See the [Migration Checklist](#migration-checklist) for step-by-step upgrade instructions.

---


## Python Version Support

**Python 3.9 is no longer supported.** The SDK now requires **Python 3.10 or higher**.



## New Features
### Structured Extrinsic Responses (ExtrinsicResponse)

`ExtrinsicResponse` provides rich, structured data for both outgoing requests and incoming on-chain results. While it is a breaking change in return types, it primarily unlocks better development, testing, and debugging workflows by standardizing success flags, messages, fees, receipts, and operation-specific data in one object.

- **Purpose**: Enables more accurate, predictable code paths and easier assertions in tests
- **What you get**: Success flag, human-readable message, network fee, application-level swap fee(s), inclusion/finalization receipts, and operation-specific data
- **Where it applies**: Returned by all functions that submit extrinsics

See details: [ExtrinsicResponse Return Type](#extrinsicresponse-return-type)




### Multiple Incentive Mechanisms Support

Full SDK support for **multiple incentive mechanisms within subnets** is now implemented, a major new Subtensor blockchain feature.

Previously referred to as "sub-subnets" during development, this feature allows subnet creators to run multiple independent evaluation mechanisms within a single subnet, each with separate weight matrices, bond pools, and emission distributions. This is a significant architectural change that enables more sophisticated subnet designs. See [Multiple Incentive Mechanisms Within Subnets](../subnets/understanding-multiple-mech-subnets) for a complete overview.


**Key Concepts:**

- **`mechid` (Mechanism ID)**: An integer identifying which mechanism within a subnet (0 for the first mechanism, 1 for the second, etc.).
- **Default behavior**: All methods default to `mechid=0`, so existing single-mechanism subnets work unchanged
- **Backward compatible**: Subnets with only one mechanism (the default) don't need code changes


#### Setting Mechanism Weights:

Validators must set weights independently for each mechanism in a subnet:

```python
# Set weights for a specific mechanism (mechid)
response = subtensor.set_weights(
    wallet,
    netuid=1,
    uids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2],
    mechid=0  # Mechanism ID (default: 0)
)

# For subnets with multiple mechanisms, set weights for each:
mechanism1_response = subtensor.set_weights(wallet, netuid=1, uids, weights1, mechid=0)
mechanism2_response = subtensor.set_weights(wallet, netuid=1, uids, weights2, mechid=1)
```

#### Querying Mechanism-Specific Data on the Metagraph:

All metagraph queries now accept a `mechid` parameter:

```python
# Get metagraph for specific mechanism
metagraph = subtensor.metagraph(netuid=1, mechid=0)

# Get weights for specific mechanism
weights = subtensor.weights(netuid=1, mechid=0)

# Get bonds for specific mechanism
bonds = subtensor.bonds(netuid=1, mechid=0)

# Get timelocked weight commits for a mechanism
commits = subtensor.get_timelocked_weight_commits(netuid=1, mechid=0)
```

###  Simulate Token Swaps

`sim_swap()` calculates the **exact token yields** for stake or unstake operations at a given block, without actually executing the transaction.

```python
# Simulate adding stake (TAO → Alpha) to see exact Alpha received
result = subtensor.sim_swap(
    origin_netuid=0,        # 0 = TAO (root)
    destination_netuid=1,   # Target subnet
    amount=tao(100.0),
)

print(f"TAO amount: {result.tao_amount}")
print(f"Alpha received: {result.alpha_amount}")
print(f"TAO fee: {result.tao_fee}")
print(f"Alpha fee: {result.alpha_fee}")

# Simulate unstaking (Alpha → TAO)
result = subtensor.sim_swap(
    origin_netuid=1,        # Source subnet
    destination_netuid=0,   # 0 = TAO (root)
    amount=tao(100.0),
)
```

### Build Extrinsic Calls

Compose an extrinsic call without submitting it to the blockchain with `compose_call`. Useful for fee estimation and transaction preparation.

```python
from bittensor.core.extrinsics.params import StakingParams

# Compose a call for later submission or fee estimation
call = subtensor.compose_call(
    call_module="SubtensorModule",
    call_function="add_stake",
    call_params=StakingParams.add_stake(
        netuid=1,
        hotkey_ss58=hotkey_ss58,
        amount=amount
    )
)

# Use the composed call to estimate fees (requires keypair)
fee = subtensor.get_extrinsic_fee(call, wallet.coldkeypub)
```




### SimSwap Fee Calculation Methods

The SDK now provides dedicated methods for calculating swap-based fees for staking operations. These methods use the new `sim_swap()` functionality (see [New Subtensor Methods](#new-subtensor-methods)) to query the Subtensor blockchain and return precise fee calculations:

```python
# Get fee for adding stake (staking operation)
fee = subtensor.get_stake_add_fee(amount, netuid)

# Get fee for moving stake between subnets
fee = subtensor.get_stake_movement_fee(origin_netuid, destination_netuid, amount)

# Get fee for removing stake (unstaking operation)
fee = subtensor.get_unstake_fee(netuid, amount)

# All methods return Balance objects representing the fee in TAO or Alpha
```

**Note:** These are **application-level swap fees** (0.05% of transacted liquidity), separate from **blockchain transaction fees** (weight-based). See [Transaction Fees in Bittensor](../learn/fees) for details on both fee types.


### Verbose Logging Control

The `Subtensor` class now supports verbose logging to help debug interactions with the Subtensor blockchain:

```python
# Set verbose mode for trace-level logging
subtensor = Subtensor(network="test", log_verbose=True)
# Automatically sets btlogging to TRACE level for detailed blockchain interaction logs
```

### BlockInfo Class

New [`BlockInfo`](pathname:///python-api/html/autoapi/bittensor/core/types/index.html) class provides rich information about blockchain blocks, including metadata and links to block explorers.

```python
from bittensor.core.types import BlockInfo

# Get block information by block number
block = subtensor.get_block_info(block=12345)

# Or get block information by block hash
block = subtensor.get_block_info(block_hash="0x1234...")

# Access block data
print(f"Block number: {block.number}")
print(f"Block hash: {block.hash}")
print(f"Timestamp: {block.timestamp}")
print(f"Number of extrinsics: {len(block.extrinsics)}")
print(f"View on explorer: {block.explorer}")

# Access raw header data
print(f"Parent hash: {block.header['parentHash']}")

# Iterate through extrinsics
for extrinsic in block.extrinsics:
    print(f"Extrinsic: {extrinsic}")
```

**BlockInfo attributes:**
- **`number`** (int): The block number
- **`hash`** (str): The block hash
- **`timestamp`** (Optional[int]): Unix timestamp when the block was created (from `Timestamp.Now` extrinsic)
- **`header`** (dict): Raw block header data from the node RPC, including parent hash, state root, and other metadata
- **`extrinsics`** (list): List of all extrinsics (transactions) included in the block
- **`explorer`** (str): Direct link to view the block on [tao.app/](https://tao.app) block explorer

**Use cases:**
- Inspect transaction history and block contents
- Debug blockchain interactions
- Verify transaction inclusion in specific blocks
- Access detailed block metadata for analysis
- Link users to block explorer for visual inspection

### Hex <--> SS58 Conversion

New utility function for converting hex addresses to SS58 format:

```python
from bittensor.utils import hex_to_ss58

ss58_address = hex_to_ss58(hex_string)

from bittensor.utils import ss58_to_hex

hex_string = ss58_to_hex(ss58_address)
```

**Note:** `hex_to_ss58` is an alias for `ss58_encode` from scalecodec. Similarly, `ss58_to_hex` is an alias for `ss58_decode`.

### Development Test Framework

New developer testing utilities provide helpers and fixtures for rapid local testing of SDK integrations and extrinsic flows. Use cases: Simulate common workflows, stub chain interactions, and write predictable tests around `ExtrinsicResponse` without a full node.

Learn more: [`bittensor/extras/dev_framework`](https://github.com/opentensor/bittensor/tree/SDKv10/bittensor/extras/dev_framework)


### Estimate Transaction Fees

Query the estimated fee for submitting an extrinsic to the Subtensor blockchain before actually sending it, with ``get_extrinsic_fee()`.

```python
# Estimate the fee for a transfer extrinsic
extrinsic = subtensor.compose_call(
    call_module="Balances",
    call_function="transfer_keep_alive",
    call_params={
        "dest": destination_address,
        "value": amount.rao
    }
)
fee = subtensor.get_extrinsic_fee(extrinsic, wallet.coldkeypub)
print(f"Estimated fee: {fee}")  # Returns Balance object
```

**Use cases:**
- Check if wallet has sufficient balance before submitting transactions
- Display estimated costs to users
- Optimize transaction batching based on fee estimates

See also: [Transaction Fees in Bittensor](../learn/fees) for complete fee information.


### Parameter Validation

Validate extrinsic parameters before submission to catch errors early, with `validate_extrinsic_params`.

```python
# Validate parameters match the extrinsic schema
params = {
    "netuid": 1,
    "hotkey": hotkey_ss58,
    "amount_staked": amount.rao
}

try:
    # Returns validated/corrected params or raises an exception
    validated_params = subtensor.validate_extrinsic_params(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params=params
    )
    print(f"Parameters validated: {validated_params}")
except Exception as e:
    print(f"Invalid parameters: {e}")
```

**Note:** This method returns the validated parameters (potentially with corrections) or raises an exception if validation fails. It does not return a boolean.

### Query Commitment Data

Get commitment metadata for a hotkey on a subnet with `get_commitment_metadata`. Previously in `bittensor.core.extrinsic.serving`, now a `Subtensor` method.

```python
# Query commitment metadata
metadata = subtensor.get_commitment_metadata(netuid=1, hotkey_ss58="5D...")
```

### Query Bonds Reset

Get the last bonds reset block for a subnet with `get_last_bonds_reset`. Previously in `bittensor.core.extrinsic.serving`, now a `Subtensor` method.

```python
# Query when bonds were last reset (requires both netuid and hotkey_ss58)
last_reset_block = subtensor.get_last_bonds_reset(netuid=1, hotkey_ss58="5D...")
```

### Historical Block Data Query

The `blocks_since_last_update` method has been improved and can now be used to query historical data from archive nodes.

```python
# Query blocks since last update with archive node support
blocks = subtensor.blocks_since_last_update(netuid=1, uid=0)
```



## Breaking Changes: Method Renames

### Subnet Methods

```python
# ❌ Old:
netuids = subtensor.get_subnets()

# ✅ New:
netuids = subtensor.get_all_subnets_netuid()
```

### Stake Methods

```python
# ❌ Old (removed - duplicate functionality):
stake_info = subtensor.get_stake_for_coldkey(coldkey)

# ✅ Use instead:
stake_info = subtensor.get_stake_info_for_coldkey(coldkey_ss58)
```

### Commitment Methods

```python
# ❌ Old:
subtensor.commit(wallet, netuid, data)

# ✅ New:
subtensor.set_commitment(wallet, netuid, data)

```

### Serving Methods

Functions moved from `bittensor.core.extrinsic.serving` to `Subtensor` methods:

```python
# ❌ Old:
from bittensor.core.extrinsic.serving import get_metadata, get_last_bonds_reset
metadata = get_metadata(subtensor, netuid, hotkey)
last_reset = get_last_bonds_reset(subtensor, netuid)

# ✅ New:
metadata = subtensor.get_commitment_metadata(netuid, hotkey_ss58)
last_reset = subtensor.get_last_bonds_reset(netuid, hotkey_ss58)
```

### Timelocked Weight Commits

```python
# ❌ Old (deprecated):
commits = subtensor.get_current_weight_commit_info(netuid, mechid)
commits_v2 = subtensor.get_current_weight_commit_info_v2(netuid, mechid)

# ✅ Use:
commits = subtensor.get_timelocked_weight_commits(netuid, mechid)
```

## Breaking Changes: Parameter Changes

### Consistent Parameter Ordering

Many methods now follow the standard order: `subtensor`, `netuid`, `hotkey_ss58`, ...

```python
# ❌ Old:
subtensor.set_children(wallet, netuid, hotkey, children, proportions)
subtensor.move_stake(wallet, origin_hotkey, dest_hotkey, amount)

# ✅ New:
subtensor.set_children(wallet, netuid, hotkey_ss58, children)
subtensor.move_stake(wallet, origin_netuid, origin_hotkey_ss58, destination_netuid, destination_hotkey_ss58, amount)
```

**Methods with reordered parameters:**
- `add_stake_multiple`
- `set_children`
- `move_stake`
- `query_subtensor`
- `query_module`
- `query_map_subtensor`
- `query_map`
- `root_set_pending_childkey_cooldown`

### Block Parameter Standardization

```python
# ❌ Old:
subnets = subtensor.all_subnets(block_number=12345)

# ✅ New:
subnets = subtensor.all_subnets(block=12345)
```

All `block_number` and `block_id` parameters are now consistently named `block`.

### Get Metagraph Info Fields Parameter

```python
# ❌ Old:
info = subtensor.get_metagraph_info(netuid, field_indices=[0, 1, 2])

# ✅ New:
info = subtensor.get_metagraph_info(netuid, selected_indices=[0, 1, 2])
```

### Mock Parameter

```python
# ❌ Old:
subtensor = Subtensor(network="local", _mock=True)

# ✅ New:
subtensor = Subtensor(network="local", mock=True)
```

The `_mock` parameter is now public as `mock` and moved to the last position in the parameter list.

### Async Methods Parity

All async methods now have the same parameters as sync methods:

```python
# New parameters added to async versions:
async_subtensor.get_subnet_validator_permits(netuid, block_hash=None, reuse_block=None)
async_subtensor.get_subnet_owner_hotkey(netuid, block_hash=None, reuse_block=None)
```

### Removed `reuse_block` from Sync Methods

```python
# ❌ Old:
hotkeys = subtensor.get_owned_hotkeys(coldkey, reuse_block=True)

# ✅ New:
hotkeys = subtensor.get_owned_hotkeys(coldkey_ss58)
```

The `reuse_block` parameter has been removed from sync methods for consistency. It's still available in async methods where appropriate.

### Transfer Fee Parameter Rename

```python
# ❌ Old:
fee = subtensor.get_transfer_fee(wallet, destination, value)

# ✅ New:
fee = subtensor.get_transfer_fee(wallet, destination, amount)
```

The `value` parameter has been renamed to `amount` for consistency with other amount parameters across the SDK.

## Breaking Changes:Removed Methods

### Duplicate References

```python
# ❌ Removed (was just a reference):
subtensor.get_stake_info_for_coldkey = subtensor.get_stake_for_coldkey

# ✅ Use the canonical name:
subtensor.get_stake_info_for_coldkey(coldkey_ss58)
```

### DefaultConfig Class Removed

```python
# ❌ Removed:
from bittensor.core.config import DefaultConfig

# ✅ Use Config directly:
from bittensor.core.config import Config
```

The `DefaultConfig` class has been removed as it was unused in the codebase.

### DelegateInfo Attribute Removed

```python
# ❌ Removed:
delegate_info.total_daily_return  # No longer calculated or provided

# This attribute has been removed from both DelegateInfo and DelegateInfoLite classes
```

The `total_daily_return` attribute has been removed as it was not providing accurate information and should not be relied upon.

## Extrinsic Calling Changes

All functions that submit extrinsics to the Subtensor blockchain (both standalone functions and `Subtensor` class methods) have been standardized for consistency.

### Standardized Parameters

All functions that submit extrinsics now follow a **consistent parameter ordering** and include standard flags:

```python
# Standard parameter order:
extrinsic_function(
    subtensor,
    # ... extrinsic-specific parameters (wallet, netuid, hotkey_ss58, amount, etc.) ...
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
)
```

1. **Extrinsic-specific arguments come first** (e.g., `wallet`, `hotkey_ss58`, `netuid`, `amount`)
2. **Optional general flags come last** (e.g., `period`, `raise_error`, `wait_for_inclusion`, `wait_for_finalization`)
3. **Default behavior changed**: `wait_for_inclusion` and `wait_for_finalization` now default to `True`

This ensures the SDK function response correctly reflects the blockchain transaction outcome.

:::note
When `raise_error=False`, extrinsic functions do not raise exceptions; all error information is captured inside the returned `ExtrinsicResponse` object. Set `raise_error=True` if you prefer exceptions to be raised directly for error cases.
:::

### ExtrinsicResponse Return Type

All SDK functions that submit extrinsics to the blockchain now return an `ExtrinsicResponse` object instead of `bool` or tuples.

- **`success`**: Primary indicator - `True` if transaction succeeded, `False` otherwise
- **`message`**: User-friendly status (e.g., "Success", "Insufficient balance")
- **`extrinsic_fee`**: Network fee paid to validators (similar to gas fees) - see [Transaction Fees](../learn/fees)
- **`transaction_tao_fee`**: Application-level fee charged in TAO for swap-based staking operations (0.05% of transacted liquidity when applicable)
- **`transaction_alpha_fee`**: Application-level fee charged in Alpha (subnet token) for swap-based staking operations (0.05% of transacted liquidity when applicable)
- **`extrinsic_receipt`**: Contains block number, hash, events, and execution details when `wait_for_inclusion=True`
- **`data`**: Operation-specific results:
  - Registration: `{"uid": int}` - assigned neuron UID
  - Commit weights: `{"reveal_round": int}` - round for revealing
  - Metadata: `{"encrypted": bytes, "reveal_round": int}`
  - Stake operations: balance information
- **`error`**: Python exception for programmatic error handling when `raise_error=False`


See [source code](https://github.com/opentensor/bittensor/blob/main/bittensor/core/types.py#L290-L484).

### Parameter Renames

Many parameters in functions that submit blockchain transactions have been renamed for consistency across the SDK:

#### Hotkey and Coldkey Parameters

**All SS58 address parameters now use the `_ss58` suffix:**

```python
# ❌ Old:
subtensor.move_stake(wallet, origin_hotkey, destination_hotkey, amount)

# ✅ New:
subtensor.move_stake(wallet, origin_netuid, origin_hotkey_ss58, destination_netuid, destination_hotkey_ss58, amount)
```

**Affected methods:**
- `hotkey` → `hotkey_ss58` (all methods)
- `hotkey_ss58_address` → `hotkey_ss58` (all methods)
- `coldkey` → `coldkey_ss58` (all methods)
- `coldkey_ss58_address` → `coldkey_ss58` (all methods)
- `hotkeypub` → `hotkeypub_ss58` (where applicable)
- `coldkeypub` → `coldkeypub_ss58` (where applicable)

#### Transfer Extrinsic

```python
# ❌ Old:
subtensor.transfer(wallet, dest, amount)

# ✅ New:
subtensor.transfer(wallet, destination_ss58, amount)
```

#### Staking Parameters

```python
# ❌ Old:
subtensor.unstake(wallet, hotkey, amount, safe_staking=True)
subtensor.swap_stake(wallet, from_hotkey, to_hotkey, amount, safe_staking=True)

# ✅ New:
subtensor.unstake(wallet, netuid, hotkey_ss58, amount, safe_unstaking=True)

# For swapping stake between subnets (same hotkey, different subnets):
subtensor.swap_stake(
    wallet,
    hotkey_ss58="5D...",
    origin_netuid=1,
    destination_netuid=2,
    amount=amount,
    safe_swapping=True
)

# For moving stake between hotkeys (use move_stake instead):
subtensor.move_stake(
    wallet,
    origin_netuid=1,
    origin_hotkey_ss58="5D...1",
    destination_netuid=1,
    destination_hotkey_ss58="5D...2",
    amount=amount
)
```

#### Required Parameters

Several previously optional parameters are now **required**:

```python
# ❌ Old:
subtensor.add_stake(wallet, hotkey_ss58=None, amount=None)

# ✅ New:
subtensor.add_stake(
    wallet,
    netuid: int,          # Now required
    hotkey_ss58: str,     # Now required
    amount: Balance       # Now required
)
```

**Affected functions:**
- `add_stake_extrinsic`: `netuid`, `hotkey_ss58`, `amount` now required
- `add_stake_multiple_extrinsic`: `amounts` now required
- `unstake_extrinsic`: `netuid`, `hotkey_ss58`, `amount` now required
- `unstake_multiple_extrinsic`: `amounts` now required

### Removed Functions

#### `unstake_all` parameter removed from `unstake_extrinsic`

```python
# ❌ Old:
subtensor.unstake(wallet, hotkey, amount=None, unstake_all=True)

# ✅ New: Use dedicated method
subtensor.unstake_all(wallet, netuid, hotkey_ss58)
```

#### Internal `_do*` methods removed

All internal helper methods have been consolidated into the main functions that submit extrinsics:

- `_do_commit_reveal_v3` → merged into `commit_timelocked_weights_extrinsic`
- `_do_commit_weights` → merged into `commit_weights_extrinsic`
- `_do_reveal_weights` → merged into `reveal_weights_extrinsic`
- `_do_set_weights` → merged into `set_weights_extrinsic`
- `_do_burned_register` → merged into `burned_register_extrinsic`
- `_do_pow_register` → merged into `register_extrinsic`
- `_do_set_root_weights` → merged into `set_root_weights_extrinsic`
- `_do_transfer` → merged into `transfer_extrinsic`

#### Deprecated root weights method removed

```python
# ❌ Removed:
subtensor.set_root_weights_extrinsic(...)

# This was obsolete and has been removed entirely
```

#### Commit-Reveal v3 (CRv3) removed

All CRv3-related logic and extrinsics have been removed as CRv3 is no longer supported on the chain.

```python
# ❌ Removed:
commit_reveal_extrinsic(...)  # Old non-mechanism version

# ✅ Use:
commit_timelocked_weights_extrinsic(..., commit_reveal_version=4)
```

### Merged Functions

#### `increase_take` and `decrease_take` merged into `set_delegate_take`

```python
# ❌ Old:
subtensor.increase_take(wallet, hotkey_ss58, take)
subtensor.decrease_take(wallet, hotkey_ss58, take)

# ✅ New:
# Automatically determines whether to increase or decrease based on current vs new take
subtensor.set_delegate_take(
    wallet,
    hotkey_ss58="5D...",
    take=0.18  # 18% as a float between 0 and 1
)

# ✅ New (low-level extrinsic with explicit action):
# The extrinsic requires an explicit action parameter with a strict type:
# action: Literal["increase_take", "decrease_take"]
from bittensor.core.extrinsics.take import set_take_extrinsic

# Example: increase take
response_inc = set_take_extrinsic(
    subtensor,
    wallet=wallet,
    hotkey_ss58="5D...",
    take=0.20,
    action="increase_take",
)

# Example: decrease take
response_dec = set_take_extrinsic(
    subtensor,
    wallet=wallet,
    hotkey_ss58="5D...",
    take=0.15,
    action="decrease_take",
)
```


**Note:** The method automatically calls `increase_take` or `decrease_take` internally based on whether the new take is higher or lower than the current take.

#### Mechanism-specific weight functions consolidated

Non-mechanism versions removed, mechanism versions renamed and moved:

```python
# Old paths (removed):
from bittensor.core.extrinsics.mechanism import (
    commit_timelocked_mechanism_weights_extrinsic,
    commit_mechanism_weights_extrinsic,
    reveal_mechanism_weights_extrinsic,
)

# ✅ New paths:
from bittensor.core.extrinsics.weights import (
    commit_timelocked_weights_extrinsic,
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
    set_weights_extrinsic,
)
```




## Balance Handling

### Stricter Type Checking

**Balance operations now raise errors instead of warnings** for type mismatches.

#### 1) Invalid balance type (`BalanceTypeError`)

Raised when a method expecting a `Balance` receives a non-`Balance` value.

```python
from bittensor.utils.balance import Balance, tao, rao
from bittensor.core.errors import BalanceTypeError

# ❌ Old (pre-10.0): would warn but continue
amount = 1.0  # float
# subtensor.transfer(wallet, dest, amount)

# ✅ New: must pass a Balance object (choose your preferred constructor)
subtensor.transfer(wallet, dest, tao(1.0))
subtensor.transfer(wallet, dest, rao(1_000_000_000))
subtensor.transfer(wallet, dest, Balance.from_tao(1.0))
subtensor.transfer(wallet, dest, Balance.from_rao(1_000_000_000))
```

#### 2) Mismatched balance units (`BalanceUnitMismatchError`)

Raised when performing operations on two `Balance` objects that represent different units (e.g., TAO vs Alpha from a subnet).

```python
from bittensor.utils.balance import Balance
from bittensor.core.errors import BalanceUnitMismatchError

balance_tao = Balance.from_tao(1.0)
balance_alpha = Balance.from_rao(1_000_000_000)  # 1 tao = 1x10^9 rao

# ❌ Will raise BalanceUnitMismatchError: mixing units in arithmetic
_ = balance_tao + balance_alpha

# ❌ Will raise BalanceUnitMismatchError: comparing mismatched units
_ = balance_tao > balance_alpha

# ✅ Ensure units match before operations
_ = balance_tao + Balance.from_tao(0.5)
```

### Function Renames

```python
# ❌ Old:
from bittensor.utils.balance import check_balance, check_and_convert_to_balance

amount = check_and_convert_to_balance(value)

# ✅ New:
from bittensor.utils.balance import check_balance_amount

amount = check_balance_amount(value)
```

### All Amount Parameters Require Balance Objects

```python
# ❌ Old:
subtensor.transfer(wallet, destination, 1.0)  # float accepted
subtensor.add_stake(wallet, hotkey, 5)  # int accepted

# ✅ New:
from bittensor.utils.balance import tao, rao

subtensor.transfer(wallet, destination, tao(1.0))  # convenience helper
subtensor.add_stake(wallet, netuid, hotkey_ss58, tao(5.0))
# or
subtensor.transfer(wallet, destination, rao(1000000000))
```

**Affected methods:**
- `transfer`
- `add_stake`
- `add_stake_multiple`
- `unstake`
- `unstake_multiple`
- `move_stake`
- `swap_stake`
- `transfer_stake`
- `get_transfer_fee`
- `get_stake_add_fee`
- `get_stake_movement_fee`
- `get_unstake_fee`



## Import Changes

### Removed Backwards Compatibility Aliases

The following lowercase aliases have been **removed** from `bittensor`:

```python
# ❌ Old (removed):
from bittensor import subtensor, wallet, config, axon, dendrite
from bittensor import keyfile, metagraph, synapse, async_subtensor

# ✅ New (use PascalCase):
from bittensor import Subtensor, Wallet, Config, Axon, Dendrite
from bittensor import Keyfile, Metagraph, Synapse, AsyncSubtensor
```

### Removed Direct Subpackage Links

```python
# ❌ Old (removed):
from bittensor.mock import MockSubtensor
from bittensor.extrinsics import transfer_extrinsic

# ✅ New (use top-level convenience imports or full paths):
from bittensor import mock, extrinsics
mock_sub = mock.MockSubtensor()
response = extrinsics.transfer.transfer_extrinsic(...)

# Or use full paths:
from bittensor.utils.mock import MockSubtensor
from bittensor.core.extrinsics import transfer_extrinsic
```

### New Import Convenience

```python
# ✅ New convenience imports:
from bittensor import extrinsics
from bittensor import mock
from bittensor import get_async_subtensor

# Use them:
response = extrinsics.transfer.transfer_extrinsic(...)
mock_sub = mock.MockSubtensor()
async_sub = get_async_subtensor(network="test")
```

### Module Reorganization

Several modules have been moved to new subpackages:

```python
# ❌ Old:
from bittensor.core.subtensor_api import SubtensorAPI
from bittensor.core.timelock import TimelockManager

# ✅ New:
from bittensor.extras.subtensor_api import SubtensorAPI
from bittensor.extras import timelock

# Or use top-level convenience import:
from bittensor import SubtensorApi, timelock
```

The `bittensor.extras` package now hosts optional extensions like `SubtensorApi`, `timelock` and `dev_framework`.

### Extrinsic Parameters Package

Parameters for functions that submit extrinsics are now centralized in a dedicated package, `bittensor.core.extrinsics.params`.

This makes it easier to discover available parameters and ensures consistency across sync/async implementations of blockchain transaction functions.


## Environment Variables

### Renamed Variables

```python
# ❌ Old:
BT_CHAIN_ENDPOINT=ws://127.0.0.1:9945
BT_NETWORK=local

# ✅ New:
BT_SUBTENSOR_CHAIN_ENDPOINT=ws://127.0.0.1:9945
BT_SUBTENSOR_NETWORK=local
```

**All renamed environment variables:**
- `BT_CHAIN_ENDPOINT` → `BT_SUBTENSOR_CHAIN_ENDPOINT`
- `BT_NETWORK` → `BT_SUBTENSOR_NETWORK`

### Disabling CLI Argument Parsing

If your script uses the SDK and receives unwanted `--config` or other CLI parameters:

```python
# Set environment variable to disable config processing:
BT_NO_PARSE_CLI_ARGS=1  # or: true, yes, on

# In code:
import os
os.environ['BT_NO_PARSE_CLI_ARGS'] = '1'

from bittensor import Subtensor
# CLI args will no longer be processed
```

:::tip
When `BT_NO_PARSE_CLI_ARGS` is set, the SDK skips CLI parsing entirely and falls back to default configuration values defined in `bittensor.core.settings.DEFAULTS` for all configuration options across the SDK. This is useful when embedding the SDK in applications that manage their own configuration.
:::


## Metagraph Changes

This section covers changes to metagraph-related functionality in both the `Subtensor` and `Metagraph` classes, particularly around the new multiple incentive mechanisms feature.

### Mechid Parameter Ordering

With the introduction of multiple incentive mechanisms per subnet, many methods now accept a `mechid` (mechanism ID) parameter to specify which mechanism to query:

```python
# Query methods with mechid support (parameter order: netuid, mechid=0, block=None)
bonds = subtensor.bonds(netuid=1, mechid=0)
weights = subtensor.weights(netuid=1, mechid=0)
metagraph = subtensor.metagraph(netuid=1, mechid=0)
commits = subtensor.get_timelocked_weight_commits(netuid=1, mechid=0)
```

**Methods with mechid parameter:**
- `bonds(netuid, mechid=0, block=None)`
- `weights(netuid, mechid=0, block=None)`
- `metagraph(netuid, mechid=0, lite=True, block=None)`
- `get_metagraph_info(netuid, mechid=0, ...)`
- `get_timelocked_weight_commits(netuid, mechid=0, block=None)`
- `commit_weights(wallet, netuid, uids, weights, mechid=0, ...)`
- `reveal_weights(wallet, netuid, uids, weights, mechid=0, ...)`
- `set_weights(wallet, netuid, uids, weights, mechid=0, ...)`

### MetagraphInfo Changes

The `MetagraphInfo` class now requires a `mechid` parameter to support multiple incentive mechanisms:

```python
# mechid is now required in MetagraphInfo
from bittensor.core.chain_data.metagraph_info import MetagraphInfo

info = MetagraphInfo(
    netuid=1,
    mechid=0,  # Now required (mechanism ID)
    # ... other fields
)
```

**Note:** `mechid=0` refers to the first (or only) incentive mechanism in a subnet. Subnets can have multiple mechanisms (currently limited to 2), each with their own independent weight matrices and emissions. See [Multiple Incentive Mechanisms](../subnets/understanding-multiple-mech-subnets) for details.

### Async Metagraph Initialization

The async `AsyncMetagraph.sync` method no longer terminates the subtensor instance after use, improving resource management and allowing for reuse of connections in async contexts.


## Migration Checklist

1. **Update Python version** to 3.10 or higher
2. **Update bittensor package**: `pip install bittensor>=10.0.0` (until the public release, you should install the latest available release candidate).
3. **Update all imports** to use PascalCase class names (`Subtensor`, `Wallet`, etc.), also known as the CapWords convention.
4. **Replace all amount parameters** with `Balance` objects.`
5. **Update all functions that submit extrinsics** to handle `ExtrinsicResponse` return type instead of `bool` or tuples[bool, str]
6. **Rename parameters** according to the standardization (`hotkey` → `hotkey_ss58`, `dest` → `destination`, etc.)
7. **Update environment variables** (`BT_CHAIN_ENDPOINT` → `BT_SUBTENSOR_CHAIN_ENDPOINT`, `BT_NETWORK` → `BT_SUBTENSOR_NETWORK`)
8. **Review removed methods** and replace with alternatives (see [Removed Methods](#removed-methods))
9. **Update parameter order** for affected methods (see [Parameter Changes](#parameter-changes))
10. **Add `mechid` parameter** to weight-setting code if working with multiple mechanisms
11. **Test thoroughly** with your specific use case, especially blockchain transactions and balance handling