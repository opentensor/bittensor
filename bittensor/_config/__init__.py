"""
Create and init the config class, which manages the config of different bittensor modules.
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

import os
from argparse import ArgumentParser

import yaml
from loguru import logger

import bittensor
from . import config_impl

logger = logger.opt(colors=True)
    
class config:
    """
    Create and init the config class, which manages the config of different bittensor modules.
    """
    class InvalidConfigFile(Exception):
        """ In place of YAMLError
        """

    def __new__( cls, parser: ArgumentParser = None):
        if parser == None:
            parser = ArgumentParser()

        # 1. Optionally load defaults if the --config is set.
        try:
            config_file_path = str(os.getcwd()) + '/' + vars(parser.parse_known_args()[0])['config']

        except Exception as e:
            config_file_path = None
            
        if config_file_path != None:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print('Loading config defaults from: {}'.format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        # 2. Continue with loading in params.
        params = parser.parse_known_args()[0]
        _config = config_impl.Config()

        # Splits params on dot syntax i.e neuron.axon_port            
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
            head = _config
            keys = split_keys
            while len(keys) > 1:
                if hasattr(head, keys[0]):
                    head = getattr(head, keys[0])  
                    keys = keys[1:]   
                else:
                    head[keys[0]] = config_impl.Config()
                    head = head[keys[0]] 
                    keys = keys[1:]
            if len(keys) == 1:
                head[keys[0]] = arg_val

        return _config

    @staticmethod
    def full():
        """ From the parser, add arguments to multiple bittensor sub-modules
        """
        parser = ArgumentParser()
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataset.add_args( parser )
        return bittensor.config( parser )

