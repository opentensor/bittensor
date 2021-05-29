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

import bittensor
import argparse
import copy
from munch import Munch

from . import _metagraph

class metagraph:

    def __new__(
            cls, 
            config: Munch = None
        ):
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`Munch`, `optional`): 
                    bittensor.metagraph.default_config()
        """        
        if config == None:
            config = metagraph.default_config()
        metagraph.check_config( config )
        return _metagraph.Metagraph( config )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        metagraph.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.subtensor.Subtensor.add_args(parser)
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.subtensor.Subtensor.check_config(config)

