import pytest

from bittensor.core.addons.subtensor_api import SubtensorApi
from bittensor.core.subtensor import Subtensor


def test_properties_methods_comparable(other_class: "Subtensor" = None):
    """Verifies that methods in SubtensorApi and its properties contains all Subtensors methods."""
    # Preps
    subtensor = (
        other_class(network="latent-lite", _mock=True)
        if other_class
        else Subtensor(network="latent-lite", _mock=True)
    )
    subtensor_api = SubtensorApi(network="latent-lite", mock=True)

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
    stakes_methods = [m for m in dir(subtensor_api.staking) if not m.startswith("_")]
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


def test__methods_comparable_with_passed_legacy_methods(
    other_class: "Subtensor" = None,
):
    """Verifies that methods in SubtensorApi contains all Subtensors methods if `legacy_methods=True` is passed."""
    # Preps
    subtensor = (
        other_class(network="latent-lite", mock=True)
        if other_class
        else Subtensor(network="latent-lite", _mock=True)
    )
    subtensor_api = SubtensorApi(network="latent-lite", mock=True, legacy_methods=True)

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


def test_failed_if_subtensor_has_new_method():
    """Verifies that SubtensorApi fails if Subtensor has a new method."""
    # Preps

    class SubtensorWithNewMethod(Subtensor):
        def return_my_id(self):
            return id(self)

    # Call and assert

    with pytest.raises(AssertionError):
        test_properties_methods_comparable(other_class=SubtensorWithNewMethod)
