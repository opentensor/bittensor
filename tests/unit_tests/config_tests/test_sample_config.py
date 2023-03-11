import sys

from bittensor._neuron.text.core_validator import neuron as core_validator
from bittensor._neuron.text.core_server import server as core_server

# TODO: Fix pathing issues in this file so it actually does something.
# These tests were not running on github actions, and most of them just work without reading the config files.


def test_run_core_validator_config():

    PATH = 'sample_configs/template_validator_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = core_validator.config()

    assert config['dataset']['batch_size'] == 10
    assert config['logging']['logging_dir'] == '~/.bittensor/miners'
    assert config['neuron']['clip_gradients'] == 1.0

def test_run_core_server_config():

    PATH = 'tests/unit_tests/config_tests/core_server_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = core_server.config()
    
    assert config['axon']['backward_timeout'] == 20
    assert config['dataset']['data_dir'] == '~/.bittensor/data/'
    assert config['logging']['debug'] == False
    assert config['wandb']['api_key'] == 'default'

if __name__ == "__main__":
    test_run_core_server_config()