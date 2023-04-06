# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import os

import bittensor
from loguru import logger
from substrateinterface import SubstrateInterface
from torch.cuda import is_available as is_cuda_available

from bittensor.utils import strtobool_with_default
from .naka_subtensor_impl import Subtensor as Nakamoto_subtensor
from . import subtensor_impl, subtensor_mock

logger = logger.opt(colors=True)

GLOBAL_SUBTENSOR_MOCK_PROCESS_NAME = 'node-subtensor'

class subtensor:
    """Factory Class for both bittensor.Subtensor and Mock_Subtensor Classes

    The Subtensor class handles interactions with the substrate subtensor chain.
    By default, the Subtensor class connects to the Finney which serves as the main bittensor network.
    
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
                            -- finney (main network)
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
            if network != None:
                config.subtensor.network = network
            else:
                config.subtensor.network = config.subtensor.get('network', bittensor.defaults.subtensor.network)
            
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
        
        # make sure it's wss:// or ws://
        # If it's bellagene (parachain testnet) then it has to be wss
        endpoint_url: str = config.subtensor.chain_endpoint
        
        # make sure formatting is good
        endpoint_url = bittensor.utils.networking.get_formatted_ws_endpoint_url(endpoint_url)
        
        

        subtensor.check_config( config )
        network = config.subtensor.get('network', bittensor.defaults.subtensor.network)
        if network == 'nakamoto':
            substrate = SubstrateInterface(
                ss58_format = bittensor.__ss58_format__,
                use_remote_preset=True,
                url = endpoint_url,
            )
            # Use nakamoto-specific subtensor.
            return Nakamoto_subtensor( 
                substrate = substrate,
                network = config.subtensor.get('network', bittensor.defaults.subtensor.network),
                chain_endpoint = config.subtensor.chain_endpoint,
            )
        else:
            substrate = SubstrateInterface(
                ss58_format = bittensor.__ss58_format__,
                use_remote_preset=True,
                url = endpoint_url,
                type_registry=bittensor.__type_registry__
            )
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
                                        -- finney (main network)
                                        -- local (local running network)
                                        -- mock (creates a mock connection (for testing))
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
            parser.add_argument('--' + prefix_str + 'subtensor.chain_endpoint', default = bittensor.defaults.subtensor.chain_endpoint, type=str, 
                                help='''The subtensor endpoint flag. If set, overrides the --network flag.
                                    ''')       
            parser.add_argument('--' + prefix_str + 'subtensor._mock', action='store_true', help='To turn on subtensor mocking for testing purposes.', default=bittensor.defaults.subtensor._mock)
            # registration args. Used for register and re-register and anything that calls register.
            parser.add_argument('--' + prefix_str + 'subtensor.register.num_processes', '-n', dest=prefix_str + 'subtensor.register.num_processes', help="Number of processors to use for registration", type=int, default=bittensor.defaults.subtensor.register.num_processes)
            parser.add_argument('--' + prefix_str + 'subtensor.register.update_interval', '--' + prefix_str + 'subtensor.register.cuda.update_interval', '--' + prefix_str + 'cuda.update_interval', '-u', help="The number of nonces to process before checking for next block during registration", type=int, default=bittensor.defaults.subtensor.register.update_interval)
            parser.add_argument('--' + prefix_str + 'subtensor.register.no_output_in_place', '--' + prefix_str + 'no_output_in_place', dest="subtensor.register.output_in_place", help="Whether to not ouput the registration statistics in-place. Set flag to disable output in-place.", action='store_false', required=False, default=bittensor.defaults.subtensor.register.output_in_place)
            parser.add_argument('--' + prefix_str + 'subtensor.register.verbose', help="Whether to ouput the registration statistics verbosely.", action='store_true', required=False, default=bittensor.defaults.subtensor.register.verbose)
            
            ## Registration args for CUDA registration.
            parser.add_argument( '--' + prefix_str + 'subtensor.register.cuda.use_cuda', '--' + prefix_str + 'cuda', '--' + prefix_str + 'cuda.use_cuda', default=argparse.SUPPRESS, help='''Set flag to use CUDA to register.''', action="store_true", required=False )
            parser.add_argument( '--' + prefix_str + 'subtensor.register.cuda.no_cuda', '--' + prefix_str + 'no_cuda', '--' + prefix_str + 'cuda.no_cuda', dest=prefix_str + 'subtensor.register.cuda.use_cuda', default=argparse.SUPPRESS, help='''Set flag to not use CUDA for registration''', action="store_false", required=False )

            parser.add_argument( '--' + prefix_str + 'subtensor.register.cuda.dev_id', '--' + prefix_str + 'cuda.dev_id',  type=int, nargs='+', default=argparse.SUPPRESS, help='''Set the CUDA device id(s). Goes by the order of speed. (i.e. 0 is the fastest).''', required=False )
            parser.add_argument( '--' + prefix_str + 'subtensor.register.cuda.TPB', '--' + prefix_str + 'cuda.TPB', type=int, default=bittensor.defaults.subtensor.register.cuda.TPB, help='''Set the number of Threads Per Block for CUDA.''', required=False )

            parser.add_argument('--netuid', type=int, help='netuid for subnet to serve this neuron on', default=argparse.SUPPRESS)        
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults ):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.subtensor = bittensor.Config()
        defaults.subtensor.network = os.getenv('BT_SUBTENSOR_NETWORK') if os.getenv('BT_SUBTENSOR_NETWORK') != None else 'finney'
        defaults.subtensor.chain_endpoint = os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') if os.getenv('BT_SUBTENSOR_CHAIN_ENDPOINT') != None else None
        defaults.subtensor._mock = os.getenv('BT_SUBTENSOR_MOCK') if os.getenv('BT_SUBTENSOR_MOCK') != None else False

        defaults.subtensor.register = bittensor.Config()
        defaults.subtensor.register.num_processes = os.getenv('BT_SUBTENSOR_REGISTER_NUM_PROCESSES') if os.getenv('BT_SUBTENSOR_REGISTER_NUM_PROCESSES') != None else None # uses processor count by default within the function
        defaults.subtensor.register.update_interval = os.getenv('BT_SUBTENSOR_REGISTER_UPDATE_INTERVAL') if os.getenv('BT_SUBTENSOR_REGISTER_UPDATE_INTERVAL') != None else 50_000
        defaults.subtensor.register.output_in_place = True
        defaults.subtensor.register.verbose = False

        defaults.subtensor.register.cuda = bittensor.Config()
        defaults.subtensor.register.cuda.dev_id = [0]
        defaults.subtensor.register.cuda.use_cuda = False
        defaults.subtensor.register.cuda.TPB = 256

        

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.subtensor
        #assert config.subtensor.network != None
        if config.subtensor.get('register') and config.subtensor.register.get('cuda'):
            assert all((isinstance(x, int) or isinstance(x, str) and x.isnumeric() ) for x in config.subtensor.register.cuda.get('dev_id', []))

            if config.subtensor.register.cuda.get('use_cuda', bittensor.defaults.subtensor.register.cuda.use_cuda):
                try:
                    import cubit
                except ImportError:
                    raise ImportError('CUDA registration is enabled but cubit is not installed. Please install cubit.')

                if not is_cuda_available():
                    raise RuntimeError('CUDA registration is enabled but no CUDA devices are detected.')


    @staticmethod
    def determine_chain_endpoint(network: str):
        if network == "nakamoto":
            # Main network.
            return bittensor.__nakamoto_entrypoint__
        elif network == "finney": 
            # Kiru Finney stagin network.
            return bittensor.__finney_entrypoint__
        elif network == "nobunaga": 
            # Staging network.
            return bittensor.__nobunaga_entrypoint__
        elif network == "bellagene":
            # Parachain test net
            return bittensor.__bellagene_entrypoint__
        elif network == "local":
            # Local chain.
            return bittensor.__local_entrypoint__
        elif network == 'mock':
            return bittensor.__mock_entrypoint__
        else:
            return bittensor.__local_entrypoint__
