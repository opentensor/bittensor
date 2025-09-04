# Plan

## Extrinsics and related
1. ✅ Standardize parameter order across all extrinsics and related calls. Pass extrinsic-specific arguments first (e.g., wallet, hotkey, netuid, amount), followed by optional general flags (e.g., wait_for_inclusion, wait_for_finalization)
    <details>
        <summary>Example</summary>

    ```py
    def swap_stake_extrinsic(
        subtensor: "Subtensor",
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Optional[Balance] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        period: Optional[int] = None,
        raise_error: bool = True,
    ) -> bool:
    ```
    it will be 
    ```py
    def swap_stake_extrinsic(
        subtensor: "Subtensor",
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Optional[Balance] = None,
        rate_tolerance: float = 0.005,
        allow_partial_stake: bool = False,
        safe_staking: bool = False,
        period: Optional[int] = None,
        raise_error: bool = True,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
    ```
    </details>
  
2. Unify extrinsic return values by introducing an ExtrinsicResponse class. Extrinsics currently return either a boolean or a tuple. 

    Purpose:
    - Ease of processing
    - This class should contain success, message, and optionally data and logs. (to save all logs during the extrinsic)

3. ✅ Set `wait_for_inclusion` and `wait_for_finalization` to `True` by default in extrinsics and their related calls. Then we will guarantee the correct/expected extrinsic call response is consistent with the chain response. If the user changes those values, then it is the user's responsibility.
4. ✅ Make the internal logic of extrinsics the same. There are extrinsics that are slightly different in implementation.

5. Since SDK is not a responsible tool, try to remove all calculations inside extrinsics that do not affect the result, but are only used in logging. Actually, this should be applied not to extrinsics only but for all codebase.

6. Remove `unstake_all` parameter from `unstake_extrinsic` since we have `unstake_all_extrinsic`which is calles another subtensor function.

7. `unstake` and `unstake_multiple` extrinsics should have `safe_unstaking` parameters instead of `safe_staking`.

8. ✅ Remove `_do*` extrinsic calls and combine them with extrinsic logic.

9. `subtensor.get_transfer_fee` calls extrinsic inside the subtensor module. Actually the method could be updated by using `bittensor.core.extrinsics.utils.get_extrinsic_fee`.

## Subtensor
1. In the synchronous Subtensor class, the `get_owned_hotkeys` method includes a `reuse_block` parameter that is inconsistent with other methods. Either remove this parameter from `get_owned_hotkeys`, or add it to all other methods that directly call self.substrate.* to maintain a consistent interface.
2. In all methods where we `get_stake_operations_fee` is called, remove unused arguments. Consider combining all methods using `get_stake_operations_fee` into one common one. 
3. Delete deprecated `get_current_weight_commit_info` and `get_current_weight_commit_info_v2`. Rename `get_timelocked_weight_commits` to get_current_weight_commit_info.
4. Remove references like `get_stake_info_for_coldkey = get_stake_for_coldkey`.
5. Reconsider some methods naming across the entire subtensor module.
6. Add `hotkey_ss58` parameter to `get_liquidity_list` method. One wallet can have many HKs. Currently, the mentioned method uses default HK only.

## Metagraph
1. Remove verbose archival node warnings for blocks older than 300. Some users complained about many messages for them.
2. Reconsider entire metagraph module logic.

## Balance
1. In `bittensor.utils.balance._check_currencies` raise the error instead of `warnings.warn`.
2. In `bittensor.utils.balance.check_and_convert_to_balance` raise the error instead of `warnings.warn`. 
This may seem like a harsh decision at first, but ultimately we will push the community to use Balance and there will be fewer errors in their calculations. Confusion with TAO and Alpha in calculations and display/printing/logging will be eliminated.

## Common things
1. Reduce the amount of logging.info or transfer part of logging.info to logging.debug

2. To be consistent across all SDK regarding local environment variables name:
remove `BT_CHAIN_ENDPOINT` (settings.py :line 124) and use `BT_SUBTENSOR_CHAIN_ENDPOINT` instead of that.
rename this variable in documentation.

3. Move `bittensor.utils.get_transfer_fn_params` to `bittensor.core.extrinsics.utils`.

4. Common refactoring (improve type annotations, etc)

5. Rename `non-/fast-blocks` to `non-/fast-runtime` in related places to be consistent with subtensor repo. Related with testing, subtensor scripts, documentation.

6. To be consistent throughout the SDK:
`hotkey`, `coldkey`, `hotkeypub`, and `coldkeypub` are keypairs
`hotkey_ss58`, `coldkey_ss58`, `hotkeypub_ss58`, and `coldkeypub_ss58` are SS58 addresses of keypair.

7. Replace `Arguments` with `Parameters`. Matches Python rules. Improve docstrings for writing MСP server.

8. Remove all type annotations for parameters in docstrings.

9. Remove all logic related to CRv3 as it will be removed from the chain next week.

10. Revise `bittensor/utils/easy_imports.py` module to remove deprecated backwards compatibility objects. Use this module as a functionality for exporting existing objects to the package root to keep __init__.py minimal and simple.

11. Remove `bittensor.utils.version.version_checking`

12. Find and process all `TODOs` across the entire code base. If in doubt, discuss each one with the team separately. SDK has 29 TODOs.
13. ✅ The SDK is dropping support for `Python 3.9` starting with this release.
14. Remove `Default is` and `Default to` in docstrings bc parameters enough.

## New features
1. Add `bittensor.utils.hex_to_ss58` function. SDK still doesn't have it. (Probably inner import `from scalecodec import ss58_encode, ss58_decode`) 
2. Implement Crowdloan logic.
3. “Implement Sub-subnets / Metagraph Changes?” (implementation unsure) Maciej Kula idea, requires mode details.

## Testing
1. When running tests via Docker, ensure no lingering processes occupy required ports before launch.

2. Improve failed test reporting from GH Actions to the Docker channel (e.g., clearer messages, formatting).

3. Write a configurable test harness class for tests that will accept arguments and immediately:
create a subnet
activate a subnet (if the argument is passed as True)
register neurons (use wallets as arguments)
set the necessary hyperparameters (tempo, etc. if the argument are passed)
Will greatly simplify tests.

4. Add an async test versions. This will help us greatly improve the asynchronous implementation of Subtensors and Extrinsics.


## Implementation

To implement the above changes and prepare for the v10 release, the following steps must be taken:

- [x] Create a new branch named SDKv10.~~
All breaking changes and refactors should be targeted into this branch to isolate them from staging and maintain backward compatibility during development.
- [ ] Add a `migration.md` document at the root of the repository and use it as a check list. This file will serve as a changelog and technical reference.
It must include:
  - [ ] All change categories (Extrinsics, Subtensor, Metagraph, etc.)
  - [ ] Per-PR breakdown of what was added, removed, renamed, or refactored.
  - [ ] Justifications and migration notes for users (if API behavior changed).

- [ ] Based on the final `migration.md`, develop migration documentation for the community. 
- [ ] Once complete, merge SDKv10 into staging and release version 10.


# Migration guide

- [x] `._do_commit_reveal_v3` logic is included in the main code `.commit_reveal_v3_extrinsic`
- [x] `.commit_reveal_v3_extrinsic` renamed to `.commit_reveal_extrinsic`
- [x] `revecommit_reveal_version` parameter with default value `4` added to `revecommit_reveal_version`
- [x] `._do_commit_weights` logic is included in the main code `.commit_weights_extrinsic`
- [x] `._do_reveal_weights` logic is included in the main code `.reveal_weights_extrinsic`
- [x] `._do_set_weights` logic is included in the main code `.set_weights_extrinsic`
- [x] `set_weights_extrinsic` moved to `bittensor/core/extrinsics/commit_weights.py`
- [x] `bittensor/core/extrinsics/commit_weights.py` module renamed to `bittensor/core/extrinsics/weights.py` (consistent naming with async module)
- [x] `_do_burned_register` logic is included in the main code `.burned_register_extrinsic`
- [x] `_do_pow_register` logic is included in the main code `.register_extrinsic`
- [x] `._do_set_root_weights` logic is included in the main code `.set_root_weights_extrinsic`
- [x] `._do_transfer` logic is included in the main code `.transfer_extrinsic`
- [x] `dest` parameter has been renamed to `destination` in `transfer_extrinsic` function and `subtensor.transfer` method.
- [x] obsolete extrinsic `set_root_weights_extrinsic` removed. Also related subtensor calls `subtensor.set_root_weights_extrinsic` removed too.

# Standardize parameter order is applied for (extrinsics and related calls):

These parameters will now exist in all extrinsics and related calls (default values could be different depends by extrinsic): 

```py
period: Optional[int] = None,
raise_error: bool = False,
wait_for_inclusion: bool = False,
wait_for_finalization: bool = False,
``` 
- [x] `.set_children_extrinsic` and `.root_set_pending_childkey_cooldown_extrinsic`. `subtensor.set_children` and `subtensor.root_set_pending_childkey_cooldown` methods.
- [x] `.commit_reveal_extrinsic` and `subtensor.set_weights`
- [x] `.add_liquidity_extrinsic` and `subtensor.add_liquidity`
- [x] `.modify_liquidity_extrinsic` and `subtensor.modify_liquidity`
- [x] `.remove_liquidity_extrinsic` and `subtensor.remove_liquidity`
- [x] `.toggle_user_liquidity_extrinsic` and `subtensor.toggle_user_liquidity`
- [x] `.transfer_stake_extrinsic` and `subtensor.transfer_stake`
- [x] `.swap_stake_extrinsic` and `subtensor.swap_stake`
- [x] `.move_stake_extrinsic` and `subtensor.move_stake`
- [x] `.move_stake_extrinsic` has renamed parameters:
    - `origin_hotkey` to `origin_hotkey_ss58`
    - `destination_hotkey` to `destination_hotkey_ss58`
- [x] `.burned_register_extrinsic` and `subtensor.burned_register`
- [x] `.register_subnet_extrinsic` and `subtensor.register_subnet`
- [x] `.register_extrinsic` and `subtensor.register`
- [x] `.set_subnet_identity_extrinsic` and `subtensor.set_subnet_identity`
- [x] `.root_register_extrinsic`, `subtensor.burned_register` and `subtensor.root_register`
- [x] `.serve_extrinsic`
- [x] `.serve_axon_extrinsic` and `subtensor.serve_axon`
- [x] alias `subtensor.set_commitment` removed
- [x] `subtensor.comit` renamed to `subtensor.set_commitment`
- [x] `.publish_metadata`, `subtensor.set_commitment` and `subtenor.set_reveal_commitment`
- [x] `.add_stake_extrinsic` and `subtensor.add_stake`
- [x] `.add_stake_multiple_extrinsic` and `subtensor.add_stake_multiple`
- [x] `.start_call_extrinsic` and `subtensor.start_call`
- [x] `.increase_take_extrinsic`, `.decrease_take_extrinsic` and `subtenor.set_reveal_commitment`
- [x] `.transfer_extrinsic` and `subtensor.transfer`
- [x] `.unstake_extrinsic` and `subtensor.unstake`
  - Changes in `unstake_extrinsic`:
    - parameter `netuid: Optional[int]` is now required -> `netuid: int`
    - parameter `hotkey_ss58: Optional[str]` is now required -> `hotkey_ss58: str`
    - parameter `amount: Optional[Balance]` is now required -> `amount: Balance`
    - parameter `unstake_all: bool` removed (use `unstake_all_extrinsic` for unstake all stake)
- [x] `.unstake_all_extrinsic` and `subtensor.unstake_all`
- [x] `.unstake_multiple_extrinsic` and `subtensor.unstake_multiple`
- [x] `.commit_weights_extrinsic` and `subtensor.commit_weights`
