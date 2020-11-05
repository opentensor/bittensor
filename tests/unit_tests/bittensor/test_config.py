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
    assert  config.chain_endpoint == "http://127.0.0.1:9933"
    assert  config.axon_port == 8091
    assert  config.metagraph_port == 8092
    assert  config.metagraph_size == 10000
    assert  config.remote_ip == None
    assert  config.datapath == "data/"
    assert config.bp_host == None
    assert config.bp_port == None



def test_type_check():
    # Check that type checking catches the incorrect types.
    try:
        bittensor.Config(axon_port = 8091)
    except:
        return True
    return False



