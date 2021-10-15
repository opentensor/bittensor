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
        config.threadpool.max_workers = max_workers if max_workers != None else config.threadpool.max_workers
        config.threadpool.maxsize = maxsize if maxsize != None else config.threadpool.maxsize

        prioritythreadpool.check_config( config )

        return priority_thread_pool_impl.PriorityThreadPoolExecutor(maxsize = config.threadpool.maxsize, max_workers = config.threadpool.max_workers)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        """ Accept specific arguments from parser
        """
        try:
            parser.add_argument('--threadpool.max_workers', type = int, help='''maximum number of threads in thread pool''', default = bittensor.defaults.threadpool.max_workers)
            parser.add_argument('--threadpool.maxsize', type=int, help='''maximum size of tasks in priority queue''', default = bittensor.defaults.threadpool.maxsize)
            
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.threadpool = bittensor.Config()
        defaults.threadpool.max_workers = os.getenv('BT_THREADPOOL_MAX_WORKERS') if os.getenv('BT_THREADPOOL_MAX_WORKERS') != None else 10
        defaults.threadpool.maxsize = os.getenv('BT_THREADPOOL_MAXSIZE') if os.getenv('BT_THREADPOOL_MAXSIZE') != None else -1
    
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
        assert isinstance(config.threadpool.max_workers, int), 'threadpool.max_workers must be a int'
        assert isinstance(config.threadpool.maxsize, int), 'threadpool.maxsize must be a int'
