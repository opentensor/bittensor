import os, sys

from bittensor._neuron.text.template_miner import neuron as template_miner
from bittensor._neuron.text.template_validator import neuron as template_validator
from bittensor._neuron.text.sgmoe_validator import neuron as sgmoe_validator
from bittensor._neuron.text.template_server import server as template_server
from bittensor._neuron.text.advanced_server import server as advanced_server

def test_run_template_miner_config():

    PATH = '/tests/config_tests/template_miner_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = template_miner.config()

    assert config['axon']['ip'] == '[::]'
    assert config['dataset']['data_dir'] == '~/.bittensor/data/'
    assert config['dendrite']['requires_grad'] == True
    assert config['nucleus']['punishment'] == 0.001

def test_run_template_validator_config():

    PATH = '/tests/config_tests/template_validator_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = template_validator.config()

    assert config['dataset']['batch_size'] == 10
    assert config['dendrite']['requires_grad'] == True
    assert config['logging']['logging_dir'] == '~/.bittensor/miners'
    assert config['neuron']['clip_gradients'] == 1.0


def test_run_sgmoe_validator_config():

    PATH = '/tests/config_tests/sgmoe_validator_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = sgmoe_validator.config()

    assert config['dataset']['batch_size'] == 10
    assert config['dendrite']['requires_grad'] == True
    assert config['logging']['logging_dir'] == '~/.bittensor/miners'
    assert config['neuron']['clip_gradients'] == 1.0


def test_run_template_server_config():

    PATH = '/tests/config_tests/template_server_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = template_server.config()

    assert config['axon']['backward_timeout'] == 20
    assert config['dataset']['data_dir'] == '~/.bittensor/data/'
    assert config['logging']['debug'] == False
    assert config['wandb']['api_key'] == 'default'


def test_run_advanced_server_config():

    PATH = '/tests/config_tests/advanced_server_sample_config.txt'
    sys.argv = [sys.argv[0], '--config', PATH]
    config = advanced_server.config()

    assert config['axon']['backward_timeout'] == 20
    assert config['dataset']['data_dir'] == '~/.bittensor/data/'
    assert config['logging']['debug'] == False
    assert config['neuron']['blacklist']['stake']['backward'] == 100


if __name__ == "__main__":
    test_run_template_miner_config()
    test_run_template_validator_config()
    test_run_sgmoe_validator_config()
    test_run_template_server_config()
    test_run_advanced_server_config()