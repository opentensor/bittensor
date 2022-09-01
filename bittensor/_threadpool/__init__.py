""" Factory method for creating priority threadpool
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
import bittensor
from . import priority_thread_pool_impl

class prioritythreadpool:
    """ Factory method for creating priority threadpool
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            max_workers: int = None,
            maxsize: int = None,
        ):
        r""" Initializes a priority thread pool.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                max_workers (default=10, type=int)
.                   The maximum number of threads in thread pool
                maxsize (default=-1, type=int)
                    The maximum number of tasks in the priority queue
        """        
        if config == None: 
            config = prioritythreadpool.config()
        config = copy.deepcopy( config )
        config.axon.priority.max_workers = max_workers if max_workers != None else config.axon.priority.max_workers
        config.axon.priority.maxsize = maxsize if maxsize != None else config.axon.priority.maxsize

        prioritythreadpool.check_config( config )
        return priority_thread_pool_impl.PriorityThreadPoolExecutor(maxsize = config.axon.priority.maxsize, max_workers = config.axon.priority.max_workers)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'axon.priority.max_workers', type = int, help='''maximum number of threads in thread pool''', default = bittensor.defaults.axon.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.priority.maxsize', type=int, help='''maximum size of tasks in priority queue''', default = bittensor.defaults.axon.priority.maxsize)  
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
        defaults.axon = bittensor.Config()
        defaults.axon.priority = bittensor.Config()
        defaults.axon.priority.max_workers = os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') if os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') != None else 5
        defaults.axon.priority.maxsize = os.getenv('BT_AXON_PRIORITY_MAXSIZE') if os.getenv('BT_AXON_PRIORITY_MAXSIZE') != None else 10
    
    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
            Return: bittensor.config object 
        """
        parser = argparse.ArgumentParser()
        prioritythreadpool.add_args( parser )
        return bittensor.config( parser )
    
    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for threadpool worker number and size
        """
        assert isinstance(config.axon.priority.max_workers, int), 'axon.priority.max_workers must be a int'
        assert isinstance(config.axon.priority.maxsize, int), 'axon.priority.maxsize must be a int'
