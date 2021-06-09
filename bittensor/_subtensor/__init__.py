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
import bittensor
import copy
from typing import List, Tuple
from bittensor._substrate import SubstrateWSInterface

from munch import Munch
from . import subtensor_impl

custom_type_registry = {
        "runtime_id": 2,
        "types": {
            "NeuronMetadataOf": {
                "type": "struct",
                "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"], ["uid", "u64"], ["modality", "u8"], ["hotkey", "AccountId"], ["coldkey", "AccountId"]]
            }
        }
    }

class subtensor:
    """
    Handles interactions with the subtensor chain.
    """

    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            network: str = 'kusanagi',
            chain_endpoint: str = None,
        ) -> 'bittensor.Subtensor':
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                network (default='akira', type=str)
                    The subtensor network flag. The likely choices are:
                            -- akira (staging network)
                            -- kusanagi (testing network)
                            -- exodus (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """
        if config == None: config = subtensor.config().subtensor
        config = copy.deepcopy( config )
        config.network = network if network != None else config.network
        config.chain_endpoint = chain_endpoint if chain_endpoint != None else config.chain_endpoint
        substrate = SubstrateWSInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry = custom_type_registry,
        )
        subtensor.check_config( config )
        return subtensor_impl.Subtensor( 
            substrate = substrate,
            network = config.network,
            chain_endpoint = config.chain_endpoint,
        )

    @staticmethod   
    def config( config: 'bittensor.Config' = None, prefix: str = '', namespace: str = 'subtensor' ) -> 'bittensor.Config':
        if config == None: config = bittensor.config()
        subtensor_config = bittensor.config()
        config[ namespace] = subtensor_config
        if namespace != '': namespace += '.'
        if prefix != '': prefix += '.'
        full_namespace = prefix + namespace
        parser = argparse.ArgumentParser()
        parser.add_argument('--' + full_namespace + 'network', dest = 'network', default='kusanagi', type=str, 
                            help='''The subtensor network flag. The likely choices are:
                                    -- akira (staging network)
                                    -- kusanagi (testing network)
                                    -- exodus (main network)
                                If this option is set it overloads subtensor.chain_endpoint with 
                                an entry point node from that network.
                                ''')
        parser.add_argument('--' + full_namespace + 'chain_endpoint', dest = 'chain_endpoint', default=None, type=str, 
                            help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                ''')
        parser.parse_known_args( namespace = subtensor_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        pass
