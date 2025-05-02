from bittensor.core.subtensor import Subtensor
from bittensor.core.subtensor_api import SubtensorApi


def test_properties_methods_comparable():
    """Verifies that methods in SubtensorApi and its properties contains all Subtensors methods."""
    # Preps
    subtensor = Subtensor(_mock=True)
    subtensor_api = SubtensorApi(_mock=True)

    subtensor_methods = [m for m in dir(subtensor) if not m.startswith("_")]

    excluded_subtensor_methods = ["commit"]

    subtensor_api_methods = [m for m in dir(subtensor_api) if not m.startswith("_")]
    chain_methods = [m for m in dir(subtensor_api.chain) if not m.startswith("_")]
    commitments_methods = [
        m for m in dir(subtensor_api.commitments) if not m.startswith("_")
    ]
    delegates_methods = [
        m for m in dir(subtensor_api.delegates) if not m.startswith("_")
    ]
    extrinsics_methods = [
        m for m in dir(subtensor_api.extrinsics) if not m.startswith("_")
    ]
    metagraphs_methods = [
        m for m in dir(subtensor_api.metagraphs) if not m.startswith("_")
    ]
    neurons_methods = [m for m in dir(subtensor_api.neurons) if not m.startswith("_")]
    queries_methods = [m for m in dir(subtensor_api.queries) if not m.startswith("_")]
    stakes_methods = [m for m in dir(subtensor_api.stakes) if not m.startswith("_")]
    subnets_methods = [m for m in dir(subtensor_api.subnets) if not m.startswith("_")]
    wallets_methods = [m for m in dir(subtensor_api.wallets) if not m.startswith("_")]

    all_subtensor_api_methods = (
        subtensor_api_methods
        + chain_methods
        + commitments_methods
        + delegates_methods
        + extrinsics_methods
        + metagraphs_methods
        + neurons_methods
        + queries_methods
        + stakes_methods
        + subnets_methods
        + wallets_methods
    )

    # Assertions
    for method in subtensor_methods:
        # skipp excluded methods
        if method in excluded_subtensor_methods:
            continue
        assert method in all_subtensor_api_methods, (
            f"`Subtensor.{method}`is not present in class `SubtensorApi`."
        )


def test__methods_comparable_with_passed_subtensor_fields():
    """Verifies that methods in SubtensorApi contains all Subtensors methods if `subtensor_fields=True` is passed."""
    # Preps
    subtensor = Subtensor(_mock=True)
    subtensor_api = SubtensorApi(_mock=True, subtensor_fields=True)

    subtensor_methods = [m for m in dir(subtensor) if not m.startswith("_")]
    subtensor_api_methods = [m for m in dir(subtensor_api) if not m.startswith("_")]

    excluded_subtensor_methods = ["commit"]

    # Assertions
    for method in subtensor_methods:
        # skipp excluded methods
        if method in excluded_subtensor_methods:
            continue
        assert method in subtensor_api_methods, (
            f"`Subtensor.{method}`is not present in class `SubtensorApi`."
        )
