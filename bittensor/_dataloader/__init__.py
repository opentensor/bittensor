

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

from . import dataloader_impl

class dataloader:

    def __new__(
            cls,
            config: 'bittensor.Config '= None,
            block_size: int = None,
            batch_size: int = None,
            max_corpus_size:int = None,
            num_workers: int = None
        ):
        if config == None:
            config = dataloader.default_config()
        config.dataloader.block_size = block_size if block_size != None else config.dataloader.block_size
        config.dataloader.batch_size = batch_size if batch_size != None else config.dataloader.batch_size
        config.dataloader.max_corpus_size = max_corpus_size if max_corpus_size != None else config.dataloader.max_corpus_size
        config.dataloader.num_workers = num_workers if num_workers != None else config.dataloader.num_workers
        config = copy.deepcopy(config)
        dataloader.check_config( config )
        return dataloader_impl.GenesisTextDataloader( config )

    @staticmethod   
    def default_config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        dataloader.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--dataloader.batch_size', default=10, type=int, 
                                help='Batch size..')
        parser.add_argument('--dataloader.block_size', default=20, type=int, 
                                help='Number of text items to pull for each example..')
        parser.add_argument('--dataloader.max_corpus_size', default=1e+6, type=int, 
                                help='Maximum amount of data to download from IPFS into memory for training.')
        parser.add_argument('--dataloader.num_workers', default=0, type=int, 
                                help='Number of workers for data loader.')

    @staticmethod   
    def check_config(config: 'bittensor.Config'):
        assert config.dataloader.batch_size > 0, 'Batch size must be larger than 0'
        assert config.dataloader.block_size > 0, 'Block size must be larger than 0'
        assert config.dataloader.max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert config.dataloader.num_workers >= 0, 'num_workers must be equal to or larger than 0'
