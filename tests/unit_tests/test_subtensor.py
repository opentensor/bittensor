from bittensor.core.subtensor import Subtensor


# TODO: It's probably worth adding a test for each corresponding method to check the correctness of the call with arguments


def test_methods_comparable(mocker):
    """Verifies that methods in sync and async Subtensors are comparable."""
    # Preps
    mocker.patch("bittensor.utils.substrate_interface.AsyncSubstrateInterface")
    subtensor = Subtensor()

    # methods which lives in sync subtensor only
    excluded_subtensor_methods = ["async_subtensor", "event_loop"]
    # methods which lives in async subtensor only
    excluded_async_subtensor_methods = [
        "determine_block_hash",
        "encode_params",
        "get_hyperparameter",
        "sign_and_send_extrinsic",
    ]
    subtensor_methods = [
        m
        for m in dir(subtensor)
        if not m.startswith("_") and m not in excluded_subtensor_methods
    ]

    async_subtensor_methods = [
        m
        for m in dir(subtensor.async_subtensor)
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
