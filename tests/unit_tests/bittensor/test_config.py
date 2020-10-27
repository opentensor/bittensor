from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.config import Config
import bittensor

import argparse

def test_empty_config():
    bittensor.Config()

def test_from_args():
    config = bittensor.Config(
                chain_endpoint = "chain_endpoint",
                axon_port = "8091",
                metagraph_port = "8092",
                metagraph_size = 10000,
                bootstrap = "bootstrap",
                neuron_key = "neuron_key",
                remote_ip = "remote_ip",
                datapath = "datapath"
            )
    assert  config.chain_endpoint == "chain_endpoint"
    assert  config.axon_port == "8091"
    assert  config.metagraph_port == "8092"
    assert  config.metagraph_size == 10000
    assert  config.bootstrap == "bootstrap"
    assert  config.neuron_key == "neuron_key"
    assert  config.remote_ip == "remote_ip"
    assert  config.datapath == "datapath"

def test_defaults():
    config = bittensor.Config()
    assert  config.chain_endpoint == bittensor.Config.__chainendpoint_default__
    assert  config.axon_port == bittensor.Config.__axon_port_default__
    assert  config.metagraph_port == bittensor.Config.__metagraph_port_default__
    assert  config.metagraph_size == bittensor.Config.__metagraph_size_default__
    assert  config.neuron_key == bittensor.Config.__neuron_key_default__
    assert  config.remote_ip == bittensor.Config.__remote_ip_default__
    assert  config.datapath == bittensor.Config.__datapath_default__


def test_type_check():
    # Check that type checking catches the incorrect types.
    try:
        bittensor.Config(axon_port = 8091)
    except:
        return True
    return False



