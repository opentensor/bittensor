from bittensor.config import Config
from bittensor.config import InvalidConfigFile

import bittensor

def test_empty_config():
    c = Config.load()
    assert c.session_settings.axon_port == 8091
    assert c.session_settings.chain_endpoint == '206.189.254.5:12345'
    assert c.session_settings.logdir == 'data/'
    assert c.neuron.neuron_path == '/bittensor/neurons/mnist'
    assert c.neuron.datapath == 'data/'

def test_overwrite_config():
    c1 = Config.load()
    c2 = Config.load('tests/unit_tests/bittensor_tests/config/defaults_overwrite.yaml')
    assert c1.session_settings.axon_port != c2.session_settings.axon_port


def test_overwrite_config_from_passed_yaml():
    passed_yaml = """
        session_settings: 
            axon_port: 8012
    """
    c1 = Config.load(from_yaml = passed_yaml)
    assert c1.session_settings.axon_port == 8012

def test_overwrite_config():
    c1 = Config.load()
    c2 = Config.load('tests/unit_tests/bittensor_tests/config/defaults_overwrite.yaml')
    assert c1.session_settings.remote_ip != c2.session_settings.remote_ip

def test_overwrite_fail_path():
    try:
        Config.load('tests/unit_tests/bittensor_tests/config/defaults_overwrite_fail.yaml')
        assert False
    except FileNotFoundError:
        assert True

if __name__ == "__main__":
    test_empty_config()
    test_overwrite_config()
    test_overwrite_fail_path()
    test_overwrite_config_from_passed_yaml()

