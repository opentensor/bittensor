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
import yaml
import munch
import bittensor
from argparse import ArgumentParser

from . import config_impl

from loguru import logger
logger = logger.opt(colors=True)
    
class config:

    class InvalidConfigFile(Exception):
        pass

    def __new__( cls, parser: ArgumentParser = None):
        if parser == None:
            parser = ArgumentParser()

        params = parser.parse_known_args()[0]
        config_file = None
        config = config_impl.Config()

        # 3. Splits params on dot syntax i.e neuron.axon_port
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
            
            if len(split_keys) == 1:
                config[arg_key] = arg_val
            else:
                if hasattr(config, split_keys[0]):
                    section = getattr(config, split_keys[0])
                
                    if not hasattr(section, split_keys[1]):
                        head = config
                        for key in split_keys[:-1]:
                            if key not in config:
                                head[key] = config_impl.Config()
                            head = head[key] 
                        head[split_keys[-1]] = arg_val
                else:
                    head = config
                    for key in split_keys[:-1]:
                        if key not in config:
                            head[key] = config_impl.Config()
                        head = head[key] 
                    head[split_keys[-1]] = arg_val

        return config

    @staticmethod
    def full():
        parser = ArgumentParser()
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataloader.add_args( parser )
        return bittensor.config( parser )

    @staticmethod
    def load_from_relative_path(path: str)  -> 'bittensor.Config':
        r""" Loads and returns a Munched config object from a relative path.

            Args:
                path (str, `required`): 
                    Path to config.yaml file. full_path = cwd() + path
    
            Returns:
                config  (:obj:`bittensor.Config` `required`):
                    bittensor.Config object with values from config under path.
        """
        # Load yaml items from relative path.
        path_items = config_impl.Config()
        if path != None:
            path = os.getcwd() + '/' + path
            if not os.path.isfile(path):
                logger.error('CONFIG: cannot find passed configuration file at {}', path)
                raise FileNotFoundError('Cannot find a configuration file at', path)
            with open(path, 'r') as f:
                try:
                    path_items = yaml.safe_load(f)
                    path_items = munch.munchify(path_items)
                    path_items = config_impl.Config( path_items )
                except yaml.YAMLError as exc:
                    logger.error('CONFIG: cannot parse passed configuration file at {}', path)
                    raise config.InvalidConfigFile
        return path_items


