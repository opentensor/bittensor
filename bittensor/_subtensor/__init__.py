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
        if config == None: config = subtensor.config()
        config = copy.deepcopy( config )
        config.subtensor.network = network if network != None else config.subtensor.network
        config.subtensor.chain_endpoint = chain_endpoint if chain_endpoint != None else config.subtensor.chain_endpoint
        substrate = SubstrateWSInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry = custom_type_registry,
        )
        subtensor.check_config( config )
        return subtensor_impl.Subtensor( 
            substrate = substrate,
            network = config.subtensor.network,
            chain_endpoint = config.subtensor.chain_endpoint,
        )

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        subtensor.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        try:
            parser.add_argument('--subtensor.network', default='kusanagi', type=str, 
                                help='''The subtensor network flag. The likely choices are:
                                        -- akira (staging network)
                                        -- kusanagi (testing network)
                                        -- exodus (main network)
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--subtensor.chain_endpoint', default=None, type=str, 
                                help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                    ''')       
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.subtensor
