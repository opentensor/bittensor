# Testing Guide for Bittensor

Testing is an essential part of software development that ensures the correctness and performance of your code. Bittensor uses a combination of unit tests and integration tests to verify the functionality of its components. This guide will walk you through how to run and write tests for Bittensor.

## Running Tests

Bittensor uses `pytest` for running its tests. To run all tests, navigate to the root directory of the Bittensor repository and run:

```bash
pytest
```

This will automatically discover all test files (those that start with `test_`) and run them.

If you want to run a specific test file, you can specify it directly. For example, to run the tests in `test_wallet.py`, you would use:

```bash
pytest tests/test_wallet.py
```

Similarly, you can run a specific test within a file by appending `::` and the test name. For example:

```bash
pytest tests/test_wallet.py::test_create_new_coldkey
```

## Writing Tests

When writing tests for Bittensor, you should aim to cover both the "happy path" (where everything works as expected) and any potential error conditions. Here's a basic structure for a test file:

```python
import pytest
import bittensor

def test_some_functionality():
    # Setup any necessary objects or state.
    wallet = bittensor.Wallet()

    # Call the function you're testing.
    result = wallet.create_new_coldkey()

    # Assert that the function behaved as expected.
    assert result is not None
```

In this example, we're testing the `create_new_coldkey` function of the `wallet` object. We assert that the result is not `None`, which is the expected behavior.

## Mocking

In some cases, you may need to mock certain functions or objects to isolate the functionality you're testing. Bittensor uses the `unittest.mock` library for this. Here's a simple example from the axon unittest:

```python
import bittensor
import pytest

def test_axon_start(mocker):
    mock_wallet = mocker.Mock(
        spec=bittensor.Wallet,
        coldkey=mocker.Mock(spec=str),
        coldkeypub=mocker.Mock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=mocker.Mock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )
    axon = bittensor.Axon(wallet=mock_wallet, config=bittensor.Config())
    axon.start()
    assert axon.server._state.stage == grpc._server._ServerStage.STARTED
```

In this example, we're mocking the `coldkey`, `coldkeypub` and `hotkey` for a wallet. This allows us to test how the axon code behaves when `bittensor.Wallet()` would normally be called, without actually calling the constructor.

## Test Coverage

It's important to ensure that your tests cover as much of your code as possible. You can use the `pytest-cov` plugin to measure your test coverage. To use it, first install it with pip:

```bash
pip install pytest-cov
```

Then, you can run your tests with coverage like this:

```bash
pytest --cov=bittensor
```

This will output a coverage report showing the percentage of your code that's covered by tests.

Remember, while high test coverage is a good goal, it's also important to write meaningful tests. A test isn't very useful if it doesn't accurately represent the conditions under which your code will run.

## E2E Testing

The Bittensor SDK includes end-to-end (E2E) tests that validate higher-level SDK workflows. All E2E tests live under `tests/e2e_tests/`.

When you add or modify an E2E test module (for example, `tests/e2e_tests/test_some_logic.py`), you must run the affected tests locally before opening a pull request:

```bash
pytest -q tests/e2e_tests/test_some_logic.py
```

Important: E2E tests are implemented in two variants within the same module:
- A synchronous test, e.g. `def test_some_logic(...):` using subtensor
- An asynchronous test, e.g. `@pytest.mark.asyncio` + `async def test_some_logic_async(...):` using async_subtensor

If you change logic covered by an E2E module, ensure that both the sync and async variants pass. You can also run them explicitly:

```bash
pytest -q tests/e2e_tests/test_some_logic.py -k test_some_logic # or test_some_logic_async
```

Async variants must use the `_async` suffix to keep test discovery and review consistent.

To ensure that your changes are tested correctly and that all tests run against the expected tooling versions, we recommend installing the development dependencies provided by the project:

```bash
python -m pip install bittensor[dev]
```

The dev extra includes all required testing and development dependencies pinned to compatible versions. Using this setup helps avoid issues caused by mismatched or missing test-related packages and ensures your local test environment is aligned with CI.

We strongly recommend performing all development and testing inside a pre-created Python virtual environment to avoid dependency conflicts with system or globally installed packages.

## Continuous Integration

Bittensor uses GH Actions for continuous integration. This means that every time you push changes to the repository, all tests are automatically run. If any tests fail, you'll be notified so you can fix the issue before merging your changes.


Remember, tests are an important part of maintaining the health of a codebase. They help catch issues early and make it easier to add new features or refactor existing code. 

## Happy testing!