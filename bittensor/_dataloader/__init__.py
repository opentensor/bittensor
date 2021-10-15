""" Create and init the GenesisTextDataloader class, which handles dataloading from ipfs
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

import argparse
import os
import copy
from munch import Munch

import bittensor
from . import dataloader_impl

class dataloader:
    """ Create and init the GenesisTextDataloader class, which handles dataloading from ipfs
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            block_size: int = None,
            batch_size: int = None,
            max_corpus_size:int = None,
            num_workers: int = None,
            dataset: str=None
        ):
        if config == None: 
            config = dataloader.config()
        config = copy.deepcopy( config )
        config.dataloader.block_size = block_size if block_size != None else config.dataloader.block_size
        config.dataloader.batch_size = batch_size if batch_size != None else config.dataloader.batch_size
        config.dataloader.max_corpus_size = max_corpus_size if max_corpus_size != None else config.dataloader.max_corpus_size
        config.dataloader.num_workers = num_workers if num_workers != None else config.dataloader.num_workers
        config.dataloader.dataset = dataset if dataset != None else config.dataloader.dataset
        dataloader.check_config( config )
        return dataloader_impl.GenesisTextDataloader(
            block_size = config.dataloader.block_size,
            batch_size = config.dataloader.batch_size,
            max_corpus_size = config.dataloader.max_corpus_size,
            num_workers = config.dataloader.num_workers,
            dataset = config.dataloader.dataset,
            data_dir = config.dataloader.data_dir
        )

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser 
            Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        dataloader.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        """ Accept specific arguments from parser
        """
        try:
            parser.add_argument('--dataloader.batch_size', type=int, help='Batch size.', default = bittensor.defaults.dataloader.batch_size )
            parser.add_argument('--dataloader.block_size', type=int, help='Number of text items to pull for each example..', default = bittensor.defaults.dataloader.block_size )
            parser.add_argument('--dataloader.max_corpus_size', type=int, help='Maximum amount of data to download from IPFS into memory for training.', default = bittensor.defaults.dataloader.max_corpus_size )
            parser.add_argument('--dataloader.num_workers', type=int, help='Number of workers for data loader.', default = bittensor.defaults.dataloader.num_workers )
            parser.add_argument('--dataloader.dataset',  type=str, help='Which datasets to use (genesis or wikitext)).', default = bittensor.defaults.dataloader.dataset )
            parser.add_argument('--dataloader.data_dir', type=str, help='Where to save and load the data.', default = bittensor.defaults.dataloader.data_dir )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.dataloader = bittensor.Config()
        defaults.dataloader.batch_size = os.getenv('BT_DATALOADER_BATCH_SIZE') if os.getenv('BT_DATALOADER_BATCH_SIZE') != None else 10
        defaults.dataloader.block_size = os.getenv('BT_DATALOADER_BLOCK_SIZE') if os.getenv('BT_DATALOADER_BLOCK_SIZE') != None else 20
        defaults.dataloader.max_corpus_size = os.getenv('BT_DATALOADER_MAX_CORPUS_SIZE') if os.getenv('BT_DATALOADER_MAX_CORPUS_SIZE') != None else 1e+4
        defaults.dataloader.num_workers = os.getenv('BT_DATALOADER_NUM_WORKERS') if os.getenv('BT_DATALOADER_NUM_WORKERS') != None else 0
        defaults.dataloader.dataset = os.getenv('BT_DATALOADER_DATASET') if os.getenv('BT_DATALOADER_DATASET') != None else 'train'
        defaults.dataloader.data_dir = os.getenv('BT_DATALOADER_DATADIR') if os.getenv('BT_DATALOADER_DATADIR') != None else '~/.bittensor/data/'

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config for batch size, block size, corpus size, num_workers and dataset
        """
        assert config.dataloader.batch_size > 0, 'Batch size must be larger than 0'
        assert config.dataloader.block_size > 0, 'Block size must be larger than 0'
        assert config.dataloader.max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert config.dataloader.num_workers >= 0, 'num_workers must be equal to or larger than 0'
        assert config.dataloader.dataset in ['train','test','validation'], 'dataset must be one of the following choices: genesis, wikitext, test, or validation'
