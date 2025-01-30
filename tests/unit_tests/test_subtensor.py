from bittensor.core.subtensor import Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor


# TODO: It's probably worth adding a test for each corresponding method to check the correctness of the call with arguments


def test_methods_comparable(mocker):
    """Verifies that methods in sync and async Subtensors are comparable."""
    # Preps
    subtensor = Subtensor(_mock=True)
    async_subtensor = AsyncSubtensor(_mock=True)

    # methods which lives in subtensor only
    excluded_subtensor_methods = ["wait_for_block"]

    # methods which lives in async subtensor only
    excluded_async_subtensor_methods = ["initialize"]

    subtensor_methods = [
        m
        for m in dir(subtensor)
        if not m.startswith("_") and m not in excluded_subtensor_methods
    ]

    async_subtensor_methods = [
        m
        for m in dir(async_subtensor)
        if not m.startswith("_") and m not in excluded_async_subtensor_methods
    ]

    # Assertions
    for method in subtensor_methods:
        assert (
            method in async_subtensor_methods
        ), f"`Subtensor.{method}` not in `AsyncSubtensor` class."

    for method in async_subtensor_methods:
        assert (
            method in subtensor_methods
        ), f"`AsyncSubtensor.{method}` not in `Subtensor` class."
