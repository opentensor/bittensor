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

from os import name
import bittensor
import argparse
import copy

from . import metagraph_impl

class metagraph:

    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
        ) -> 'bittensor.Metagraph':
        r""" Creates a new bittensor.Metagraph object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.metagraph.config()
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    bittensor subtensor chain connection.
        """      
        if config == None: config = metagraph.config()
        config = copy.deepcopy(config)
        config.subtensor.network = network if network != None else config.subtensor.network
        if subtensor == None:
            subtensor = bittensor.subtensor( config = config )
        return metagraph_impl.Metagraph( subtensor = subtensor )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        metagraph.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        try:
            bittensor.subtensor.add_args( parser )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        assert config.subtensor
