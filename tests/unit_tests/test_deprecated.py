import sys


def test_mock_import():
    """
    Tests that `bittensor.mock` can be imported and is the same as `bittensor.utils.mock`.
    """
    import bittensor.mock as redirected_mock
    import bittensor.utils.mock as real_mock

    assert "bittensor.mock" in sys.modules
    assert redirected_mock is real_mock


def test_extrinsics_import():
    """Tests that `bittensor.extrinsics` can be imported and is the same as `bittensor.utils.deprecated.extrinsics`."""
    import bittensor.extrinsics as redirected_extrinsics
    import bittensor.core.extrinsics as real_extrinsics

    assert "bittensor.extrinsics" in sys.modules
    assert redirected_extrinsics is real_extrinsics


def test_object_aliases_are_correctly_mapped():
    """Ensures all object aliases correctly map to their respective classes in Bittensor package."""
    import bittensor

    assert issubclass(bittensor.axon, bittensor.Axon)
    assert issubclass(bittensor.config, bittensor.Config)
    assert issubclass(bittensor.dendrite, bittensor.Dendrite)
    assert issubclass(bittensor.keyfile, bittensor.Keyfile)
    assert issubclass(bittensor.metagraph, bittensor.Metagraph)
    assert issubclass(bittensor.wallet, bittensor.Wallet)
    assert issubclass(bittensor.synapse, bittensor.Synapse)
