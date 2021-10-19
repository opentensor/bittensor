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

from munch import Munch
import bittensor
import copy
from substrateinterface import SubstrateInterface

from . import subtensor_impl

with_priority = {
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
                    ["trust", "u64"],
                    ["rank", "u64"],
                    ["consensus", "u64"],
                    ["incentive", "u64"],
                    ["inflation", "u64"],
                    ["dividends", "u64"],
                    ["bonds", "Vec<(u32, u64)>"],
                    ["weights", "Vec<(u32, u32)>"]
                ]
            }
        }
    }
without_priority = {
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
                    ["stake", "u64"],
                    ["trust", "u64"],
                    ["rank", "u64"],
                    ["consensus", "u64"],
                    ["incentive", "u64"],
                    ["inflation", "u64"],
                    ["dividends", "u64"],
                    ["bonds", "Vec<(u32, u64)>"],
                    ["weights", "Vec<(u32, u32)>"]
                ]
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
            network: str = None,
            chain_endpoint: str = None,
        ) -> 'bittensor.Subtensor':
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                network (default='akatsuki', type=str)
                    The subtensor network flag. The likely choices are:
                            -- kusanagi (testing network)
                            -- akatsuki (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """
        if config == None: config = subtensor.config()
        config = copy.deepcopy( config )
        if chain_endpoint != None:
            config.subtensor.network = chain_endpoint
            config.subtensor.chain_endpoint = chain_endpoint
        else:
            config.subtensor.network = network if network != None else config.subtensor.network
            config.subtensor.chain_endpoint = chain_endpoint if chain_endpoint != None else subtensor.determine_chain_endpoint(config.subtensor.network)
        if config.subtensor.network == 'nobunaga':
            type_registry = with_priority
        else:
            type_registry = without_priority

        substrate = SubstrateInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry = type_registry,
            url = "ws://{}".format(config.subtensor.chain_endpoint),
            use_remote_preset=True
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
            parser.add_argument('--subtensor.network', default = bittensor.defaults.subtensor.network, type=str, 
                                help='''The subtensor network flag. The likely choices are:
                                        -- kusanagi (testing network)
                                        -- akatsuki (main network)
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--subtensor.chain_endpoint', default = bittensor.defaults.subtensor.network, type=str, 
                                help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                    ''')       
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults ):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.subtensor = bittensor.Config()
        defaults.subtensor.network = os.getenv('BT_SUBTENSOR_NETWORK') if os.getenv('BT_SUBTENSOR_NETWORK') != None else 'akatsuki'
        defaults.subtensor.chain_endpoint = os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') if os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') != None else None

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.subtensor

    @staticmethod
    def determine_chain_endpoint(network: str):
        if network == "akatsuki":
            # Get first network in akatsuki entrypoints, which is main
            return bittensor.__akatsuki_entrypoints__[0]
        elif network == "kusanagi":
            # Kusanagi testnet
            return bittensor.__kusanagi_entrypoints__[0]
        elif network == "local":
            # Kusanagi testnet
            return bittensor.__local_entrypoints__[0]
        elif network == "nobunaga": 
            return bittensor.__nobunaga_entrypoints__[0]
        else:
            return bittensor.__local_entrypoints__[0]
            
