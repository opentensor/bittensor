# Bittensor SDK v10.0.0 - Complete Renaming Reference

## Environment Variables

| Old Name | New Name | Description |
|----------|----------|-------------|
| `BT_CHAIN_ENDPOINT` | `BT_SUBTENSOR_CHAIN_ENDPOINT` | Chain endpoint environment variable |
| `BT_NETWORK` | `BT_SUBTENSOR_NETWORK` | Network environment variable |

## Extrinsic Parameters

### Transfer Functions
| Old Parameter | New Parameter | Context |
|---------------|---------------|---------|
| `dest` | `destination` | `transfer_extrinsic`, `subtensor.transfer` |

### Stake/Unstake Functions
| Old Parameter | New Parameter | Context                                                             |
|---------------|---------------|---------------------------------------------------------------------|
| `safe_staking` | `safe_unstaking` | `unstake_extrinsic`, `subtensor.unstake`                            |
| `safe_staking` | `safe_swapping` | `swap_stake_extrinsic`, `subtensor.swap_stake`                      |
| `origin_hotkey` | `origin_hotkey_ss58` | `move_stake_extrinsic`, `subtensor.move_stake`                      |
| `destination_hotkey` | `destination_hotkey_ss58` | `move_stake_extrinsic`, `subtensor.move_stake`                      |
| `dest` | `destination_ss58` | `subtensor.transfer`, `transfer_extrinsic`  and other related calls |

### General Parameter Standardization
| Old Parameter | New Parameter | Context                                                 |
|---------------|---------------|---------------------------------------------------------|
| `hotkey` | `hotkey_ss58` | All methods expecting SS58 addresses                    |
| `coldkey` | `coldkey_ss58` | All methods expecting SS58 addresses                    |
| `hotkey_ss58_address` | `hotkey_ss58` | All methods                                             |
| `coldkey_ss58_address` | `coldkey_ss58` | All methods                                             |
| `block_number` | `block` | `subtensor.all_subnets` and other methods               |
| `destination` | `destination_ss58` | `subtensor.get_transfer_fee`  and other related methods |
 

### Removed and deleted extrinsics:
- All extrinsics with the `_do_*` prefix have been removed. All logic has now been moved to their main extrinsics.
- `bittensor.core.extrinsics.commit_weights` module renamed to `bittensor.core.extrinsics.weights` (consistent naming with async module)
- `set_weights_extrinsic` moved to `bittensor.core.extrinsics.weights`
- `set_root_weights_extrinsic` removed.

## Subtensor Methods

### Method Renames
| Old Method | New Method | Notes |
|------------|------------|-------|
| `subtensor.get_subnets()` | `subtensor.get_all_subnets_netuid()` | More descriptive name |
| `subtensor.get_stake_for_coldkey()` | `subtensor.get_stake_info_for_coldkey()` | Removed duplicate method |
| `subtensor.comit()` | `subtensor.set_commitment()` | Fixed typo, clearer name |

### Parameter Changes
| Method | Old Parameter                     | New Parameter                                   |
|--------|-----------------------------------|-------------------------------------------------|
| `subtensor.get_transfer_fee()` | `value`                           | `amount`                                        |
| `subtensor.get_metagraph_info()` | `field_indices`                   | `selected_indices`                              |
| All query methods | Various inconsistent orders       | Standardized parameter order                    |
| (async) `get_subnet_validator_permits` and `get_subnet_owner_hotkey` | no `block_hash` and `reuse_block` | added `block_hash` and `reuse_block` parameters |

## Extrinsic Functions

### Function Renames and Moves
| Old Function/Location                                 | New Function/Location                                          |
|-------------------------------------------------------|----------------------------------------------------------------|
| `commit_timelocked_mechanism_weights_extrinsic`       | `commit_timelocked_weights_extrinsic`                          |
| `commit_mechanism_weights_extrinsic`                  | `commit_weights_extrinsic`                                     |
| `reveal_mechanism_weights_extrinsic`                  | `reveal_weights_extrinsic`                                     |
| `set_mechanism_weights_extrinsic`                     | `set_weights_extrinsic`                                        |
| `publish_metadata`  | `publish_metadata_extrinsic` |
| `bittensor.core.extrinsics.mechanism.*`               | `bittensor.core.extrinsics.weights.*`                          |
| `bittensor.core.extrinsics.asyncex.mechanism.*`       | `bittensor.core.extrinsics.asyncex.weights.*`                  |

### Merged Functions
| Old Functions | New Unified Function |
|---------------|---------------------|
| `increase_take_extrinsic`, `decrease_take_extrinsic` | `set_take_extrinsic` with `action` parameter |

## Balance Utilities

| Old Function/logic             | New Function/logic                   | Change Type |
|--------------------------------|--------------------------------------|-------------|
| `check_and_convert_to_balance` | `check_balance_amount`               | Function rename |
| Warning messages               | `BalanceTypeError` exception         | Error handling upgrade |
| Warning messages               | `BalanceUnitMismatchError` exception | Error handling upgrade |

## Package Structure

### Module Moves
| Old Location | New Location                          | Notes |
|--------------|---------------------------------------|-------|
| `bittensor.core.subtensor_api` | `bittensor.core.extras.subtensor_api` | Experimental features |
| `bittensor.core.timelock` | `bittensor.core.extras.timelock`      | Experimental features |
| `bittensor.core.extrinsics.serving.get_metadata` | `subtensor.get_commitment_metadata`   | Method integration |
| `bittensor.core.extrinsics.serving.get_last_bonds_reset` | `subtensor.get_last_bonds_reset`      | Method integration |

### Easy Imports Cleanup
| Old Import                              | New Import | Change                         |
|-----------------------------------------|------------|--------------------------------|
| `from bittensor import async_subtensor` | `from bittensor import AsyncSubtensor` | Direct class import            |
| `from bittensor import axon`            | `from bittensor import Axon` | Direct class import            |
| `from bittensor import config`          | `from bittensor import Config` | Direct class import            |
| `from bittensor import dendrite`        | `from bittensor import Dendrite` | Direct class import            |
| `from bittensor import keyfile`         | `from bittensor import Keyfile` | Direct class import            |
| `from bittensor import metagraph`       | `from bittensor import Metagraph` | Direct class import            |
| `from bittensor import wallet`          | `from bittensor import Wallet` | Direct class import            |
| `from bittensor import subtensor`       | `from bittensor import Subtensor` | Direct class import            |
| `from bittensor import synapse`         | `from bittensor import Synapse` | Direct class import            |
| -                                       | `from bittenosor import extrinsics` | Direct subpackage import       |
| -                                       | `from bittensor import mock` | Direct subpackage import            |
| -                                       | `from bittensor import get_async_subtensor` | Direct factory function import |
NOTE: **References to classes with non-capitalize names are no longer available in Bittensor SDK**

## Mechanisms related changes:
In the next subtensor methods got updated the parameters order with new parameter `mechid`:
  - `bonds`
  - `get_metagraph_info`
  - `get_timelocked_weight_commits`
  - `metagraph`
  - `weights`
  - `commit_weights`
  - `reveal_weights`
  - `set_weights`
  - `bittensor.core.chain_data.metagraph_info.MetagraphInfo` got new attribute `mechid`

## `_mock` parameter in Subtensor classes

| Old Parameter | New Parameter | Context                       |
|---------------|---------------|-------------------------------|
| `_mock` | `mock` | `Async/Subtensor` constructor |
| Inconsistent parameter order | Standardized order | All async methods             |

## Response Types

| Old Return Type | New Return Type | Context |
|-----------------|-----------------|---------|
| `bool` | `ExtrinsicResponse` | All extrinsics |
| `tuple` (bool, str) | `ExtrinsicResponse` | All extrinsics |
| Mixed types | Consistent `ExtrinsicResponse` | All extrinsics |

## Removed Items

### Removed Methods
- `subtensor.get_current_weight_commit_info()`
- `subtensor.get_current_weight_commit_info_v2()`
- `subtensor.set_root_weights_extrinsic()`
- `subtensor.get_stake_for_coldkey()` (use `get_stake_info_for_coldkey` instead)

### Removed Parameters
- `unstake_all` from `unstake_extrinsic` (use `unstake_all_extrinsic` instead)
- `old_balance` from async stake functions

### Removed logic and support
- All `CRv3` related extrinsics and logic.
- `Python 3.9` support.
- `bittensor.utils.version.version_checking`.
- `bittensor.core.config.DefaultConfig`.

## Key Pattern Changes

### Parameter Order Standardization
All extrinsics and related subtensor methods now follow this pattern:
```python
wallet: Wallet,
...
extrinsic specific parameters,
...
period: Optional[int] = None,
raise_error: bool = False, 
wait_for_inclusion: bool = True,
wait_for_finalization: bool = True
```

### SS58 Address Naming Convention
- `hotkey`, `coldkey` = Keypair objects
- `hotkey_ss58`, `coldkey_ss58` = SS58 address strings

### Error Handling
- Balance warnings → Explicit exceptions (`BalanceUnitMismatchError` and `BalanceTypeError`).
- Private function `bittensor.utils.balance._check_currencies` raises `BalanceUnitMismatchError` error instead of deprecated warning message. This function is used inside the Balance class to check if units match during various mathematical and logical operations.
- Extrinsic `bool` returns → Structured `ExtrinsicResponse` objects.
- Extrinsic `tuple[bool, str]` returns → Structured `ExtrinsicResponse` objects.