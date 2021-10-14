""" Create and init the GenesisTextDataset class, which handles dataloading from ipfs
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
import copy
from munch import Munch

import bittensor
from . import dataset_impl

class dataset:
    """ Create and init the GenesisTextDataset class, which handles dataloading from ipfs
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            block_size: int = None,
            batch_size: int = None,
            max_corpus_size:int = None,
            num_workers: int = None,
            dataset_name: str=None,
            save_dataset: bool=None
        ):
        if config == None: 
            config = dataset.config()
        config = copy.deepcopy( config )
        config.dataset.block_size = block_size if block_size != None else config.dataset.block_size
        config.dataset.batch_size = batch_size if batch_size != None else config.dataset.batch_size
        config.dataset.max_corpus_size = max_corpus_size if max_corpus_size != None else config.dataset.max_corpus_size
        config.dataset.num_workers = num_workers if num_workers != None else config.dataset.num_workers
        config.dataset.dataset_name = dataset_name if dataset_name != None else config.dataset.dataset_name
        config.dataset.save_dataset = save_dataset if save_dataset != None else config.dataset.save_dataset
        dataset.check_config( config )
        return dataset_impl.GenesisTextDataset(
            block_size = config.dataset.block_size,
            batch_size = config.dataset.batch_size,
            max_corpus_size = config.dataset.max_corpus_size,
            num_workers = config.dataset.num_workers,
            dataset_name = config.dataset.dataset_name,
            data_dir = config.dataset.data_dir,
            save_dataset = config.dataset.save_dataset
        )

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser 
            Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        dataset.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        """ Accept specific arguments from parser
        """
        try:
            parser.add_argument('--dataset.batch_size', default=10, type=int, help='Batch size.')
            parser.add_argument('--dataset.block_size', default=20, type=int, help='Number of text items to pull for each example..')
            parser.add_argument('--dataset.max_corpus_size', default=1e+6, type=int, help='Maximum amount of data to download from IPFS into memory for training.')
            parser.add_argument('--dataset.num_workers', default=0, type=int, help='Number of workers for data loader.')
            parser.add_argument('--dataset.dataset_name', default='train', type=str, help='Which datasets to use (train/test/validation)).')
            parser.add_argument('--dataset.data_dir', default='~/.bittensor/data/', type=str, help='Where to save and load the data.')
            parser.add_argument('--dataset.save_dataset', default=False, type=bool, help='Save the downloaded dataset or not.')
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass


    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config for batch size, block size, corpus size, num_workers and dataset
        """
        assert config.dataset.batch_size > 0, 'Batch size must be larger than 0'
        assert config.dataset.block_size > 0, 'Block size must be larger than 0'
        assert config.dataset.max_corpus_size > 0, 'max_corpus_size must be larger than 0'
        assert config.dataset.num_workers >= 0, 'num_workers must be equal to or larger than 0'
        assert config.dataset.dataset_name in ['train','test','validation'], 'dataset_name must be one of the following choices: train, test, or validation'
        assert isinstance(config.dataset.save_dataset, bool) , 'save_dataset must be True/False only'
