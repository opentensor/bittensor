""" Create and init wandb to logs the interested parameters to weight and biases
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
import copy
import wandb as wb
import bittensor


class wandb:
    """ Create and init wandb to logs the interested parameters to weight and biases
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            api_key: str = None,
            name: str = None,
            project:str = None,
            tags: tuple = None,
            run_group: str = None,
            directory: str = None,
            cold_pubkey: str = None,
            hot_pubkey: str = None,
            root_dir: str = '~/.bittensor/miners/'
        ):
        r""" Initializes a bittensor wandb backend logging object.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.wamdb.config()
                api_key (:obj:`bool`, `optional`):
                    Optionally pass wandb api key for use_wandb.
                name (:obj:`bool`, `optional`):
                    Optionally pass wandb run name for use_wandb.
                project (:obj:`bool`, `optional`):
                    Optionally pass wandb project name for use_wandb.
                tags (:obj:`bool`, `optional`):
                run_group (:obj:`bool`, `optional`):
                directory (:obj:`bool`, `optional`):
                offline (:obj:`bool`, `optional`):
                cold_pubkey (:obj:`bool`, `optional`):
                hot_pubkey (:obj:`bool`, `optional`):
                root_dir (:obj:`bool`, `optional`):
        """
        if config == None: 
            config = wandb.config()
        config = copy.deepcopy( config )
        config.wandb.api_key = api_key if api_key != None else config.wandb.api_key
        config.wandb.name = name if name != None else config.wandb.name
        config.wandb.project = project if project != None else config.wandb.project
        config.wandb.tags = tags if tags != None else config.wandb.tags
        config.wandb.run_group = run_group if run_group != None else config.wandb.run_group
        config.wandb.directory = directory if directory != None else config.wandb.directory
        wandb.check_config( config )

        if config.wandb.api_key != 'default':
            os.environ["WANDB_API_KEY"] = config.wandb.api_key 
        else:
            pass
        os.environ["WANDB_NAME"] = config.wandb.name 
        os.environ["WANDB_PROJECT"] = config.wandb.project if config.wandb.project != 'default' else str(cold_pubkey)[:8]
        os.environ["WANDB_TAGS"] = config.wandb.tags 
        os.environ["WANDB_RUN_GROUP"] = config.wandb.run_group if config.wandb.run_group != 'default' else str(hot_pubkey)[:8]
        os.environ["WANDB_DIR"] = config.wandb.directory if config.wandb.directory != 'default' else root_dir

        wb.init(config = config, config_exclude_keys = ['wandb'])

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accepting specific argument from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'wandb.api_key', type = str, help='''Optionally pass wandb api key for use_wandb''', default='default')
            parser.add_argument('--' + prefix_str + 'wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''', default = bittensor.defaults.wandb.name)
            parser.add_argument('--' + prefix_str + 'wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''', default = bittensor.defaults.wandb.project)
            parser.add_argument('--' + prefix_str + 'wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''', default = bittensor.defaults.wandb.tags)
            parser.add_argument('--' + prefix_str + 'wandb.run_group', type = str, help='''Optionally pass wandb group name for use_wandb''', default = bittensor.defaults.wandb.run_group)
            parser.add_argument('--' + prefix_str + 'wandb.directory', type = str, help='''Optionally pass wandb directory for use_wandb''', default = bittensor.defaults.wandb.directory)
            parser.add_argument('--' + prefix_str + 'wandb.offline', type = bool, help='''Optionally pass wandb offline option for use_wandb''', default = bittensor.defaults.wandb.offline)
            
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.wandb = bittensor.Config()
        defaults.wandb.name = os.getenv('BT_WANDB_NAME') if os.getenv('BT_WANDB_NAME') != None else 'default'
        defaults.wandb.project = os.getenv('BT_WANDB_PROJECT') if os.getenv('BT_WANDB_PROJECT') != None else 'default'
        defaults.wandb.tags = os.getenv('BT_WANDB_TAGS') if os.getenv('BT_WANDB_TAGS') != None else 'default'
        defaults.wandb.run_group = os.getenv('BT_WANDB_RUN_GROUP') if os.getenv('BT_WANDB_RUN_GROUP') != None else 'default'
        defaults.wandb.directory = os.getenv('BT_WANDB_DIRECTORY') if os.getenv('BT_WANDB_DIRECTORY') != None else 'default'
        defaults.wandb.offline = os.getenv('BT_WANDB_OFFLINE') if os.getenv('BT_WANDB_OFFLINE') != None else False
    
    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from argument parser
        Return bittensor.config object
        """
        parser = argparse.ArgumentParser()
        wandb.add_args( parser )
        return bittensor.config( parser )
    
    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Checking config for types
        """
        assert isinstance(config.wandb.api_key, str), 'wandb.api_key must be a string'
        assert isinstance(config.wandb.project, str), 'wandb.project must be a string'
        assert isinstance(config.wandb.name , str), 'wandb.name must be a string'
        assert isinstance(config.wandb.tags , str), 'wandb.tags must be a str'
        assert isinstance(config.wandb.run_group , str), 'wandb.run_group must be a string'
        assert isinstance(config.wandb.directory , str), 'wandb.dir must be a string'
        assert isinstance(config.wandb.offline , bool), 'wandb.offline must be a bool'
