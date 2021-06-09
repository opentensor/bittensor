

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


import argparse
import bittensor
import copy

from munch import Munch
from . import dataloader_impl

class dataloader:

    def __new__(
            cls,
            config: 'bittensor.config' = None,
            block_size: int = None,
            batch_size: int = None,
            max_corpus_size:int = None,
            num_workers: int = None
        ):
        if config == None: config = dataloader.config().dataloader
        config = copy.deepcopy( config )
        config.block_size = block_size if block_size != None else config.block_size
        config.batch_size = batch_size if batch_size != None else config.batch_size
        config.max_corpus_size = max_corpus_size if max_corpus_size != None else config.max_corpus_size
        config.num_workers = num_workers if num_workers != None else config.num_workers
        dataloader.check_config( config )
        return dataloader_impl.GenesisTextDataloader(
            block_size = config.block_size,
            batch_size = config.batch_size,
            max_corpus_size = config.max_corpus_size,
            num_workers = config.num_workers
        )

    @staticmethod   
    def config( config: 'bittensor.Config' = None, prefix: str = '', namespace: str = 'dataloader' ) -> 'bittensor.config':
        if config == None: config = bittensor.config()
        dataloader_config = bittensor.config()
        config[ namespace ] = dataloader_config
        if namespace != '': namespace += '.'
        if prefix != '': prefix += '.'
        full_namespace = prefix + namespace
        parser = argparse.ArgumentParser()
        parser.add_argument('--' + full_namespace + 'batch_size', dest = 'batch_size', default=10, type=int, help='Batch size.')
        parser.add_argument('--' + full_namespace + 'block_size', dest = 'block_size', default=20, type=int, help='Number of text items to pull for each example..')
        parser.add_argument('--' + full_namespace + 'max_corpus_size', dest = 'max_corpus_size', default=1e+6, type=int, help='Maximum amount of data to download from IPFS into memory for training.')
        parser.add_argument('--' + full_namespace + 'num_workers', dest = 'num_workers', default=0, type=int, help='Number of workers for data loader.')
        parser.parse_known_args( namespace = dataloader_config )
        return config

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.batch_size > 0, 'Batch size must be larger than 0'
        assert config.block_size > 0, 'Block size must be larger than 0'
        assert config.max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert config.num_workers >= 0, 'num_workers must be equal to or larger than 0'

