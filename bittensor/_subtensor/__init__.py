# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import argparse
import os

import random
import time
import psutil
import subprocess
from sys import platform   

import bittensor
import copy
from substrateinterface import SubstrateInterface

from . import subtensor_impl
from . import subtensor_mock

from loguru import logger
logger = logger.opt(colors=True)

__type_registery__ = {
    "runtime_id": 2,
    "types": {
        "Balance": "u64",
        "NeuronMetadataOf": {
            "type": "struct",
            "type_mapping": [
                ["version", "u32"],
                ["ip", "u128"], 
                ["port", "u16"], 
                ["ip_type", "u8"], 
                ["uid", "u32"], 
                ["modality", "u8"], 
                ["hotkey", "AccountId"], 
                ["coldkey", "AccountId"], 
                ["active", "u32"],
                ["last_update", "u64"],
                ["priority", "u64"],
                ["stake", "u64"],
                ["rank", "u64"],
                ["trust", "u64"],
                ["consensus", "u64"],
                ["incentive", "u64"],
                ["dividends", "u64"],
                ["emission", "u64"],
                ["bonds", "Vec<(u32, u64)>"],
                ["weights", "Vec<(u32, u32)>"]
            ]
        }
    }
}

GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME = 'node-subtensor'

class subtensor:
    """Factory Class for both bittensor.Subtensor and Mock_Subtensor Classes

    The Subtensor class handles interactions with the substrate subtensor chain.
    By default, the Subtensor class connects to the Nakamoto which serves as the main bittensor network.
    
    """
    
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            network: str = None,
            chain_endpoint: str = None,
            _mock: bool = None,
        ) -> 'bittensor.Subtensor':
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                network (default='local', type=str)
                    The subtensor network flag. The likely choices are:
                            -- local (local running network)
                            -- nakamoto (main network)
                            -- nobunaga (staging network)
                            -- mock (mock network for testing.)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
                _mock (bool, `optional`):
                    Returned object is mocks the underlying chain connection.
        """
        if config == None: config = subtensor.config()
        config = copy.deepcopy( config )

        # Returns a mocked connection with a background chain connection.
        config.subtensor._mock = _mock if _mock != None else config.subtensor._mock
        if config.subtensor._mock == True or network == 'mock' or config.subtensor.get('network', bittensor.defaults.subtensor.network) == 'mock':
            config.subtensor._mock = True
            return subtensor_mock.mock_subtensor.mock()
        
        # Determine config.subtensor.chain_endpoint and config.subtensor.network config.
        # If chain_endpoint is set, we override the network flag, otherwise, the chain_endpoint is assigned by the network.
        # Argument importance: chain_endpoint > network > config.subtensor.chain_endpoint > config.subtensor.network
       
        # Select using chain_endpoint arg.
        if chain_endpoint != None:
            config.subtensor.chain_endpoint = chain_endpoint
            config.subtensor.network = network
            
        # Select using network arg.
        elif network != None:
            config.subtensor.chain_endpoint = subtensor.determine_chain_endpoint( network )
            config.subtensor.network = network
            
        # Select using config.subtensor.chain_endpoint
        elif config.subtensor.chain_endpoint != None:
            config.subtensor.chain_endpoint = config.subtensor.chain_endpoint
            config.subtensor.network = config.subtensor.get('network', bittensor.defaults.subtensor.network)
         
        # Select using config.subtensor.network
        elif config.subtensor.get('network', bittensor.defaults.subtensor.network) != None:
            config.subtensor.chain_endpoint = subtensor.determine_chain_endpoint( config.subtensor.get('network', bittensor.defaults.subtensor.network) )
            config.subtensor.network = config.subtensor.get('network', bittensor.defaults.subtensor.network)
            
        # Fallback to defaults.
        else:
            config.subtensor.chain_endpoint = subtensor.determine_chain_endpoint( bittensor.defaults.subtensor.network )
            config.subtensor.network = bittensor.defaults.subtensor.network
           
        substrate = SubstrateInterface(
            ss58_format = bittensor.__ss58_format__,
            type_registry_preset='substrate-node-template',
            type_registry = __type_registery__,
            url = "ws://{}".format(config.subtensor.chain_endpoint),
            use_remote_preset=True
        )

        subtensor.check_config( config )
        return subtensor_impl.Subtensor( 
            substrate = substrate,
            network = config.subtensor.get('network', bittensor.defaults.subtensor.network),
            chain_endpoint = config.subtensor.chain_endpoint,
        )

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        subtensor.add_args( parser )
        return bittensor.config( parser )

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'subtensor.network', default = bittensor.defaults.subtensor.network, type=str,
                                help='''The subtensor network flag. The likely choices are:
                                        -- nobunaga (staging network)
                                        -- nakamoto (master network)
                                        -- local (local running network)
                                        -- mock (creates a mock connection (for testing))
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--' + prefix_str + 'subtensor.chain_endpoint', default = bittensor.defaults.subtensor.chain_endpoint, type=str, 
                                help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                    ''')       
            parser.add_argument('--' + prefix_str + 'subtensor._mock', action='store_true', help='To turn on subtensor mocking for testing purposes.', default=bittensor.defaults.subtensor._mock)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults ):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.subtensor = bittensor.Config()
        defaults.subtensor.network = os.getenv('BT_SUBTENSOR_NETWORK') if os.getenv('BT_SUBTENSOR_NETWORK') != None else 'nakamoto'
        defaults.subtensor.chain_endpoint = os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') if os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') != None else None
        defaults.subtensor._mock = os.getenv('BT_SUBTENSOR_MOCK') if os.getenv('BT_SUBTENSOR_MOCK') != None else False

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.subtensor
        #assert config.subtensor.network != None

    @staticmethod
    def determine_chain_endpoint(network: str):
        if network == "nakamoto":
            # Main network.
            return bittensor.__nakamoto_entrypoints__[0]
        elif network == "nobunaga": 
            # Staging network.
            return bittensor.__nobunaga_entrypoints__[0]
        elif network == "local":
            # Local chain.
            return bittensor.__local_entrypoints__[0]
        elif network == 'mock':
            return bittensor.__mock_entrypoints__[0]
        else:
            return bittensor.__local_entrypoints__[0]
