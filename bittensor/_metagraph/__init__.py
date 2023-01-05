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
from typing import Optional

import bittensor
from . import metagraph_impl
from . import metagraph_mock

class metagraph:
    """ Factory class for the bittensor.Metagraph class or the MockMetagraph
    The Metagraph object serves as the main storage unit for the chain state. 
    By default, it stores all chain information as a torch.nn.Module which can be
    synced using a subtensor connection.

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> metagraph = bittensor.metagraph()
            >>> metagraph.sync(subtensor=subtensor, netuid=0)
    """
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            network: str = None,
            netuid: Optional[int] = None,
            _mock:bool=None
        ) -> 'bittensor.Metagraph':
        r""" Creates a new bittensor.Metagraph object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.metagraph.config()
                network (default=None, type=str, optional)
                    The subtensor network flag. The likely choices are:
                            -- nobunaga (staging network)
                            -- nakamoto (main network)
                            -- local (local running network)
                    This option allows you to load a metagraph from a local file.
                    If set, overrides config.subtensor.network
                netuid (default=None, type=int)
                    The subnet netuid. If set, overrides config.netuid.
                    This option allows you to load a metagraph from a local file.
                _mock (:obj:`bool`, `optional`):
                    For testing, if true the metagraph returns mocked outputs.
        """      
        if config == None: 
            config = metagraph.config()
        config = copy.deepcopy(config)
        config.metagraph._mock = _mock if _mock != None else config.metagraph._mock
        if config.metagraph._mock:
            return metagraph_mock.MockMetagraph()
        if netuid == None:
            netuid = config.get('netuid', None)
        if network == None:
            network = config.get('subtensor.network', None)
        
        return metagraph_impl.Metagraph( network = network, netuid = netuid )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from teh argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        metagraph.add_args( parser )
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
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Add specific arguments from parser, 
        which is the identical to subtensor  
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'metagraph._mock', action='store_true', help='To turn on metagraph mocking for testing purposes.', default=False)
            bittensor.subtensor.add_args( parser )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
        bittensor.subtensor.add_args( parser, prefix = prefix )

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config,
        which is identical to subtensor
        """
        pass