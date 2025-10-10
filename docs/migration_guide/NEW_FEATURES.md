
# New in SDKv10

### New Classes
- `bittensor.core.types.ExtrinsicResponse` - response for all extrinsics and related calls.
- `bittensor.core.types.BlockInfo` - result avalable from subtensor method `Async/Subtensor.get_block_info()`. Provides comprehensive information about a block in the blockchain, includes the link to block explorer on web service.
- `bittensor.core.errors.BalanceTypeError` - rises when expected amount of balance privided with wrong type data.
- `bittensor.core.errors.BalanceUnitMismatchError` - rises when trying to perform logical or mathematical operations with different currencies (different Balance units).

   NOTE: Please read detailed documentation for these classes in their docstrings.

### New sub-packages
- `bittensor.core.extrinsics.params` - contains a strict description of the parameters for extrinsics in a single place.
- `bittensor.extras` - contains all additional implementations for the core logic, but which do not affect the overall operation of the Bittensor. These are most often improvements or extensions.
   - `bittensor.extras.subtensor_api`
   - `bittensor.extras.timelock`
   - `bittensor.extras.dev_framework` (read `bittensor/extras/dev_framework/README.md`)

### New Functions
- `Async/ubtensor.compose_call`
- `Async/ubtensor.validate_extrinsic_params`
- `Async/Subtensor.get_extrinsic_fee`
- `Async/ubtensor.sim_swap`
- `bittensor.utils.hex_to_ss58`
- `bittensor.utils.ss58_to_hex`
   NOTE: Please read detailed documentation for these functions in their docstrings.

### Extrinsics has extra data in response's `data` field:
- `add_stake_extrinsic`
- `add_stake_multiple_extrinsic`
- `burned_register_extrinsic`
- `register_extrinsic`
- `transfer_extrinsic`
- `unstake_extrinsic`
- `unstake_multiple_extrinsic`

### ArgParser issue
- to turn off args parser across SDK, the local env variable `BT_NO_PARSE_CLI_ARGS` should be set to on of the values: `1`, `true`, `yes`, `on`. This solutions is covering the case when SDK is used in external scripts as a dependency and external script expect the same CLI arguments which is parses by SDK's argparser.


### New bihavior

## All SDK extrinsics and their corresponding Subtensor methods now adhere to a unified parameter standard defined in the subpackage `bittensor.core.extrinsics.params`.
This standard ensures consistency across sync, async, and internal implementations.

- Each extrinsic is now constructed internally through the `Async/Subtensor.compose_call()` method, which builds a properly encoded `GenericExtrinsic` object based on current runtime metadata.
- Before any extrinsic is created, its parameters are validated by `Async/Subtensor.validate_extrinsic_params()`.
- This validation layer operates against live on-chain metadata obtained from the Substrate runtime, ensuring that argument names, types, and counts strictly match the expected call definition.
- As a result, invalid or outdated calls are detected before submission, providing clear and deterministic error feedback to the user and preventing false or malformed transactions.
- Even if developers create an extrinsic call themselves without using `bittensor.core.extrinsics.params`, but use the `Subtensor.compose_call()` method, the parameters will be validated.


## Verbose Logging Behavior in Async and Subtensor Instances

When initializing either `AsyncSubtensor` or `Subtensor`, the argument `log_verbose=True` now automatically sets the internal logging level to `TRACE`.
This enables the most detailed output available in the SDK, including low-level RPC calls, parameter validation, and extrinsic composition traces — useful for debugging complex chain interactions.
By default, `log_verbose=False`, which keeps the logging level at Warning level without low-level noise.


## Stake repated transactions fee

Now all stake operations fee is calculated with new `SimSwap` logic via `subtensor.sim_swap` method. Detailed logic you can find in class `bittensor.core.chain_data.sim_swap.SimSwapResult`.


## Refined Logging Levels and Console Utilities

Most `logging.info` calls across the SDK have been downgraded to `logging.debug`.
This change significantly reduces unnecessary log noise during normal operation, allowing developers to view clean, high-signal output by default.

For users who require full visibility into SDK internals — including extrinsic composition, validation, and on-chain responses — enabling the flag `log_verbose=True` automatically raises the logging level to `TRACE`, providing comprehensive diagnostic information.

In cases where fine-grained or context-specific logging is needed regardless of the global logging configuration, developers can use the utilities from `bittensor.utils.btlogging.console.BittensorConsole`, exposed through the `btlogging.console.*` namespace.
These methods allow explicit control over log output even when global logging is disabled, for example:
```py
btlogging.console.info("Subnet registration started.")
btlogging.console.debug(f"Composed call payload: {payload}")
```
This design ensures that all logging remains both minimal by default and fully transparent on demand, depending on the user’s debugging needs.

## Crowdloan implementation
The Crowdloan implementation is expected to be included in the final release of SDKv10.

## Docstring Refactor and Standardization

A large-scale refactor has been applied to docstrings across the entire SDK to improve clarity, consistency, and technical accuracy.
Outdated or redundant descriptions have been removed, and all remaining docstrings were rewritten to follow a unified structure and terminology.

## Resolution of TODOs

All pending TODO items across the entire codebase have been addressed, resolving long-standing sources of technical debt accumulated over previous SDK versions.
Each TODO was either implemented, removed, or replaced with finalized logic after review.
This cleanup improves overall maintainability, eliminates dead code paths, and ensures that every remaining comment in the SDK reflects active, intentional functionality rather than deferred work.
This update makes the inline documentation more transparent, readable, and aligned with current SDK behavior, ensuring developers can quickly understand method purposes, parameter semantics, and expected return types without referring to external sources.

## Testing and Development Enhancements

Beyond general development improvements, significant upgrades were made to the SDK’s testing and CI/CD infrastructure.
The entire pipeline has been refined to include more detailed, scenario-based tests, covering both synchronous and asynchronous functionality.
These improvements enable core developers to deliver updates faster while maintaining a high standard of reliability and regression safety.

Additionally, a new Development Framework has been introduced, available under `bittensor/extras/dev_framework`.
This framework provides a lightweight environment for quickly deploying local subnets, interacting with neurons, and configuring hyperparameters.
It can also be used to connect and interact with the testnet environment, making it a valuable tool for experimentation, feature validation, and integration testing.
Future releases will continue to expand this framework with additional utilities and automated setup capabilities.