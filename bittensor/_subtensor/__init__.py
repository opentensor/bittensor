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
import copy
import bittensor
from bittensor._substrate import SubstrateWSInterface

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
            config: 'bittensor.Config' = None,
            network: str = None,
            chain_endpoint: str = None,
            namespace: str = ''
        ) -> 'bittensor.Subtensor':
        r""" Initializes a subtensor chain interface.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    metagraph.Metagraph.config()
                network (default='akira', type=str)
                    The subtensor network flag. The likely choices are:
                            -- akira (testing network)
                            -- kusanagi (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the --network flag.
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                namespace (:obj:`str, `optional`): 
                    config namespace.
        """
        if config == None:
            config = bittensor.config.cut_namespace( subtensor.config( namespace ), namespace  ).subtensor
        config.network = network if network != None else config.network
        config.chain_endpoint = chain_endpoint if chain_endpoint != None else config.chain_endpoint
        config = copy.deepcopy( config )
        subtensor.check_config( config )

        # Build substrate.
        substrate = SubstrateWSInterface(
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry = custom_type_registry,
        )

        return subtensor_impl.Subtensor( config, substrate )

    @staticmethod   
    def config(namespace: str = '') -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        subtensor.add_args(parser = parser, namespace = namespace) 
        config = bittensor.config( parser ); 
        return bittensor.config.cut_namespace( config, namespace )

    @staticmethod   
    def print_help(namespace: str = ''):
        parser = argparse.ArgumentParser(); 
        subtensor.add_args( parser, namespace ) 
        parser.print_help()
    
    @staticmethod   
    def add_args(parser: argparse.ArgumentParser, namespace: str = ''):
        if namespace != '':
            namespace = namespace + 'subtensor.'
        else:
            namespace = 'subtensor.'
        parser.add_argument('--' + namespace + 'network', default='kusanagi', type=str, 
                            help='''The subtensor network flag. The likely choices are:
                                    -- akira (testing network)
                                    -- kusanagi (main network)
                                If this option is set it overloads subtensor.chain_endpoint with 
                                an entry point node from that network.
                                ''')
        parser.add_argument('--' + namespace + 'chain_endpoint', default=None, type=str, 
                            help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                ''')
    
        
    @staticmethod   
    def check_config(config: 'bittensor.Config'):
        pass