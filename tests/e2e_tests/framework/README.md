# E2E Test Framework

## Overview
The E2E test framework provides a unified orchestration layer for subnet operations testing in Bittensor.
It simplifies the creation of end-to-end test scenarios by abstracting low-level Subtensor API calls into declarative steps.

---

## Structure
```
tests/e2e_tests/framework/
├── subnet.py          # Main orchestration class (TestSubnet)
├── utils.py           # Common helpers and validators
├── __init__.py
└── calls/             # Auto-generated extrinsic definitions
    ├── sudo_calls.py
    ├── non_sudo_calls.py
    ├── pallets.py
    └── __init__.py
```

---

## The `TestSubnet` Class

`TestSubnet` provides a high-level API for registering, activating, and configuring subnets, adding neurons,  and 
modifying hyperparameters.

### Key Features
- Supports both single-step and multi-step execution for full flexibility in test composition.
- Supports synchronous and asynchronous execution (`execute_steps`, `async_execute_steps`).
- Maintains a history of all calls through `CALL_RECORD` for detailed introspection.
- Returns a standardized `ExtrinsicResponse` object for each operation — including status, message, receipt, and fees — 
  greatly enhancing debugging capabilities.
- Automatically injects the subnet `netuid` for all commands that use the `NETUID` constant.  
  This ensures all steps executed within a single `TestSubnet` instance are scoped to the subnet registered during that 
  test session.
- Validates pallet and parameter correctness based on live Subtensor metadata.
- Fully compatible with SDK extrinsics such as `sudo_call_extrinsic` and `async_sudo_call_extrinsic`.
- Designed not only for SDK developers but also for the broader Bittensor community — enabling subnet operators, 
  researchers, and contributors to deploy, configure, and validate subnet behavior in reproducible on-chain or simulated
  environments.

---

## Example Usage

### Synchronous Example
```python
def test_subnet_setup(subtensor, alice_wallet):
    from tests.e2e_tests.utils import (
        TestSubnet, 
        NETUID,
        REGISTER_SUBNET,
        ACTIVATE_SUBNET,
        SUDO_SET_TEMPO,
        AdminUtils
    )

    sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, 100),
    ]
    sn.execute_steps(steps)

    assert subtensor.subnets.is_subnet_active(sn.netuid)
```

### Asynchronous Example
```python
import pytest

@pytest.mark.asyncio
async def test_subnet_async(async_subtensor, alice_wallet):
    from tests.e2e_tests.utils import (
        TestSubnet, 
        NETUID,
        REGISTER_SUBNET,
        ACTIVATE_SUBNET,
        SUDO_SET_TEMPO,
        AdminUtils
    )

    sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, 100),
    ]
    await sn.async_execute_steps(steps)

    assert await async_subtensor.subnets.is_subnet_active(sn.netuid)
```

> **Note:**  
> The `NETUID` constant is a dynamic placeholder automatically replaced with the actual `netuid`
> obtained during subnet registration.  
> This ensures all subsequent operations (e.g., `sudo_set_tempo`, `set_weights_set_rate_limit`) apply
> to the correct subnet within the same test instance.

---

## Community and Ecosystem Use

While this framework was originally designed to streamline internal SDK development and testing, it is intentionally 
built as a reusable and transparent orchestration layer.
Community developers, subnet operators, and researchers can leverage it to deploy, configure, and validate subnet 
behavior under real or simulated network conditions.
By exposing the same primitives that the SDK itself uses, the framework enables reproducible experiments, automated 
scenario testing, and regression validation without needing deep familiarity with Subtensor internals.

In other words, it serves both as a developer tool and a community-facing harness for controlled, verifiable subnet 
testing across environments.

## Logging and Debugging
All operations performed through `TestSubnet` are logged with colorized output:
- Subnet registration and activation
- Neuron registration
- Hyperparameter updates
- Extrinsic success or failure

Example console output:
```
Subnet [blue]1[/blue] was registered.
Subnet [blue]1[/blue] was activated.
Hyperparameter [blue]sudo_set_tempo[/blue] was set successfully with params [blue]{'netuid': 1, 'tempo': 20}[/blue].
```

The **`ExtrinsicResponse`** type provides:
- Direct access to `extrinsic_receipt`
- Status and message details
- Fee and inclusion/finalization data
- A unified return contract for all extrinsics

This combination allows tests to become more accurate, expressive, and sensitive to on-chain state.

---

## Extensibility
The framework can be easily extended with new orchestrators.
All of them can reuse the same interface and underlying mechanisms as `TestSubnet`.

---

## Migration Note
All E2E tests now use this framework. The codebase is cleaner, more maintainable, and logging/debugging has been significantly improved.
