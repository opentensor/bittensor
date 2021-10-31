""" Create and init metagraph, 
which maintains chain state as a torch.nn.Module.
"""
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
from . import metagraph_impl

class metagraph:
    """ create and init metagraph, 
    which maintains chain state as a torch.nn.module.
    """
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
            chain_endpoint: str = None,
        ) -> 'bittensor.Metagraph':
        r""" Creates a new bittensor.Metagraph object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.metagraph.config()
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    bittensor subtensor chain connection.
                network (default='nakamoto', type=str)
                    The subtensor network flag. The likely choices are:
                            -- nobunaga (staging network)
                            -- akatsuki (testing network)
                            -- nakamoto (main network)
                    If this option is set it overloads subtensor.chain_endpoint with 
                    an entry point node from that network.
                chain_endpoint (default=None, type=str)
                    The subtensor endpoint flag. If set, overrides the network argument.
        """      
        if config == None: 
            config = metagraph.config()
        config = copy.deepcopy(config)
        if subtensor == None:
            subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint )
        return metagraph_impl.Metagraph( subtensor = subtensor )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from teh argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        metagraph.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        """ Add specific arguments from parser, 
        which is the identical to subtensor  
        """
        try:
            bittensor.subtensor.add_args( parser )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config,
        which is identical to subtensor
        """
        assert config.subtensor
