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
import argparse
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

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        config.add_args( parser )
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
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None  ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument(
                '--' + prefix_str + 'config.file', 
                type=str, 
                help='If set, defaults are overridden by passed file.'
            )
            parser.add_argument(
                '--' + prefix_str + 'config.strict',  
                action = 'store_true', 
                default = bittensor.defaults.config.strict,
                help = '''If flagged, config will check that only exact arguemnts have been set.'''
            )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.config = bittensor.config()
        defaults.config.strict = False

    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config.
        """
        pass

    def __new__( cls, parser: ArgumentParser = None, strict: bool = False ):
        r""" Translates the passed parser into a nested Bittensor config.
        Args:
            parser (argparse.Parser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.
        Returns:
            config (bittensor.Config):
                Nested config object created from parser arguments.
        """
        if parser == None:
            return config_impl.Config()

        # 1.1 Check if --config.file has been set.
        # If the config file is set, we will load defaults from here.
        try:
            # Get config file path from current working directory.
            config_file_path = os.path.expanduser( str(os.getcwd()) + '/' + vars(parser.parse_known_args()[0])['config.file'] )
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print('Loading config defaults from: {}'.format(config_file_path))

                    # Set config items as defaults.
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))
        except Exception as e:
            # Config file not set.
            config_file_path = None

        # 2. Check for --config.strict.
        # Stict parsing means we throw and error if there are command line arguments
        # which the parser does not expect.
        try:
            strict = vars(parser.parse_known_args()[0])['config.strict'] or strict
        except:
            # Strict not set.
            pass
                        
        # 2. Continue with loading in params.
        if not strict:
            params = parser.parse_known_args()[0]
        else:
            params = parser.parse_args()
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
        bittensor.config.add_args( parser )
        return bittensor.config( parser )

