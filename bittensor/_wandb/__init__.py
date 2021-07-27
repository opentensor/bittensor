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
import bittensor
import wandb as wb


class wandb:

    def __new__(
            cls,
            config: 'bittensor.config' = None,
            api_key: str = None,
            name: str = None,
            project:str = None,
            tags: tuple = None,
            run_group: str = None,
            directory: str = None,
            offline: bool = None,
            cold_pubkey: str = None,
            hot_pubkey: str = None,
            root_dir: str = None
        ):
        if config == None: config = wandb.config()
        config = copy.deepcopy( config )
        config.neuron.api_key = api_key if api_key != None else config.neuron.api_key
        config.wandb.name = name if name != None else config.wandb.name
        config.wandb.project = project if project != None else config.wandb.project
        config.wandb.tags = tags if tags != None else config.wandb.tags
        config.wandb.run_group = run_group if run_group != None else config.wandb.run_group
        config.wandb.directory = directory if directory != None else config.wandb.directory
        config.wandb.offline = offline if offline != None else config.wandb.offline
        wandb.check_config( config )

        if config.neuron.api_key != 'default':
            os.environ["WANDB_API_KEY"] = config.neuron.api_key 
        else:
            pass
        os.environ["WANDB_NAME"] = config.wandb.name 
        os.environ["WANDB_PROJECT"] = config.wandb.project if config.wandb.project != 'default' else cold_pubkey
        os.environ["WANDB_TAGS"] = config.wandb.tags 
        os.environ["WANDB_RUN_GROUP"] = config.wandb.run_group if config.wandb.run_group != 'default' else hot_pubkey
        os.environ["WANDB_DIR"] = config.wandb.directory if config.wandb.directory != 'default' else root_dir
        os.environ["WANDB_MODE"] = 'offline' if config.wandb.offline else 'run'

        return wb.init(config = config, config_exclude_keys = ['neuron'], save_code = True)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        try:
            parser.add_argument('--neuron.api_key', type = str, help='''Optionally pass wandb api key for use_wandb''', default='default')
            parser.add_argument('--wandb.name', type=str, help='''Optionally pass wandb run name for use_wandb''', default='default')
            parser.add_argument('--wandb.project', type=str, help='''Optionally pass wandb project name for use_wandb''', default='default')
            parser.add_argument('--wandb.tags', type=str, help='''Optionally pass wandb tags for use_wandb''', default='default')
            parser.add_argument('--wandb.run_group', type = str, help='''Optionally pass wandb group name for use_wandb''', default='default')
            parser.add_argument('--wandb.directory', type = str, help='''Optionally pass wandb directory for use_wandb''', default='default')
            parser.add_argument('--wandb.offline', type = bool, help='''Optionally pass wandb offline option for use_wandb''', default=False)
            
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
    
    @classmethod   
    def config(cls) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        wandb.add_args( parser )
        return bittensor.config( parser )
    
    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        assert isinstance(config.neuron.api_key, str), 'wandb.api_key must be a string'
        assert isinstance(config.wandb.project, str), 'wandb.project must be a string'
        assert isinstance(config.wandb.name , str), 'wandb.name must be a string'
        assert isinstance(config.wandb.tags , str), 'wandb.tags must be a str'
        assert isinstance(config.wandb.run_group , str), 'wandb.run_group must be a string'
        assert isinstance(config.wandb.directory , str), 'wandb.dir must be a string'
        assert isinstance(config.wandb.offline , bool), 'wandb.offline must be a bool'