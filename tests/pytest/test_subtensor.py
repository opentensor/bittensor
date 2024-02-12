import pytest
from unittest.mock import patch
from bittensor import Subtensor, bittensor  # Replace 'your_module' with the actual module name where Subtensor is defined

# Mock bittensor endpoints for testing
bittensor.__finney_entrypoint__ = "finney.network"
bittensor.__local_entrypoint__ = "http://localhost:9944"
bittensor.__finney_test_entrypoint__ = "test.finney.network"
bittensor.__archive_entrypoint__ = "archive.network"
bittensor.defaults = lambda: None
bittensor.defaults.subtensor = lambda: None
bittensor.defaults.subtensor.network = "default_network"


@pytest.mark.parametrize("network, expected", [
    ("finney", ("finney", "finney.network")),
    ("local", ("local", "http://localhost:9944")),
    ("test", ("test", "test.finney.network")),
    ("archive", ("archive", "archive.network")),
    ("entrypoint-finney.opentensor.ai", ("finney", "finney.network")),
    ("test.finney.opentensor.ai", ("test", "test.finney.network")),
    ("archive.chain.opentensor.ai", ("archive", "archive.network")),
    ("127.0.0.1", ("local", "127.0.0.1")),
    ("localhost", ("local", "localhost")),
    ("unknown", ("unknown", "unknown")),
    (None, (None, None)),
    ("invalid", ("unknown", "invalid")),
])
def test_determine_chain_endpoint_and_network(network, expected):
    assert Subtensor.determine_chain_endpoint_and_network(network) == expected

@pytest.fixture
def mock_config():
    class MockConfig:
        def __init__(self):
            self.subtensor = {
                "chain_endpoint": None,
                "network": None,
                "__is_set": {}
            }
        def get(self, item, default=None):
            return self.subtensor.get(item, default)
    return MockConfig()


@pytest.mark.parametrize("network, config_updates, expected_network, expected_endpoint", [
    ("finney", {}, "finney", "ws://finney.network"),
    (None, {"subtensor": {"network": "test"}}, "test", "ws://test.finney.network"),
    (None, {"subtensor": {"chain_endpoint": "http://localhost:9944"}}, "local", "ws://localhost:9944"),
    (None, {"__is_set": {"subtensor.network": True}, "subtensor": {"network": "archive"}}, "archive", "ws://archive.network"),
])
def test_setup_config(network, config_updates, expected_network, expected_endpoint, mock_config):
    # Apply config updates
    for key, value in config_updates.items():
        if key == "__is_set":
            mock_config.__is_set.update(value)
        else:
            mock_config.subtensor.update(value)

    with patch.object(bittensor.utils.networking, 'get_formatted_ws_endpoint_url', side_effect=lambda x: f"ws://{x}"):
        result = Subtensor.setup_config(network, mock_config)
        assert result == (expected_endpoint, expected_network)
