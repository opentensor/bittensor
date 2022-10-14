"""
Create and init the config class, which manages the config of different bittensor modules.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import bittensor
import yaml
from loguru import logger

from . import config_impl

logger = logger.opt(colors=True)
    
class config:
    """
    Create and init the config class, which manages the config of different bittensor modules.
    """
    class InvalidConfigFile(Exception):
        """ In place of YAMLError
        """

    def __new__( cls, parser: ArgumentParser = None, strict: bool = False, args: Optional[List[str]] = None ):
        r""" Translates the passed parser into a nested Bittensor config.
        Args:
            parser (argparse.Parser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.
            args (list of str):
                Command line arguments.
        Returns:
            config (bittensor.Config):
                Nested config object created from parser arguments.
        """
        if parser == None:
            return config_impl.Config()

        # Optionally add config specific arguments
        try:
            parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        except:
            # this can fail if the --config has already been added.
            pass
        try:
            parser.add_argument('--strict',  action='store_true', help='''If flagged, config will check that only exact arguemnts have been set.''', default=False )
        except:
            # this can fail if the --config has already been added.
            pass

        # Get args from argv if not passed in.
        if args == None:
            args = sys.argv[1:]

        # 1.1 Optionally load defaults if the --config is set.
        try:
            config_file_path = str(os.getcwd()) + '/' + vars(parser.parse_known_args(args)[0])['config']
        except Exception as e:
            config_file_path = None

        # Parse args not strict
        params = cls.__parse_args__(args=args, parser=parser, strict=False)

        # 2. Optionally check for --strict, if stict we will parse the args strictly.
        strict = params.strict
                        
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
        params = cls.__parse_args__(args=args, parser=parser, strict=strict)

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
    def __parse_args__( args: List[str], parser: ArgumentParser = None, strict: bool = False) -> Namespace:
        """Parses the passed args use the passed parser.
        Args:
            args (List[str]):
                List of arguments to parse.
            parser (argparse.ArgumentParser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.
        Returns:
            Namespace:
                Namespace object created from parser arguments.
        """
        if not strict:
            params = parser.parse_known_args(args=args)[0]
        else:
            params = parser.parse_args(args=args)

        return params

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
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )

