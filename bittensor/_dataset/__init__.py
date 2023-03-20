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
import os
import copy
from typing import Union
import warnings

import bittensor
from . import dataset_impl
from . import dataset_mock

class dataset:
    """ Factory class for the GenesisTextDataset class or the mocked GenesisTextDataset
    The GenesisTextDataset downloads text data from the bittensor mountain dataset. 
    The class makes http requests to bittensor's IPFS backend server which contains the full dataset.
    By default, the GenesisTextDataset class will return a fully functioning pytorch dataloader.

    Examples:: 
            >>> dataset = bittensor.dataset(batch_size = 10, block_size=20)
            >>> # data.shape[batch_size, block_size]
            >>> data = next(dataset)
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            block_size: int = None,
            batch_size: int = None,
            num_workers: int = None,
            dataset_names: Union[list, str] = None,
            save_dataset: bool=None,
            no_tokenizer: bool=None,
            num_batches: int = None,
            _mock:bool=None,
            dataset_name: list = None, # For backwards compatibility
        ):
        r""" Create and init the GenesisTextDataset class, which handles dataloading from ipfs.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.dataset.config()
                block_size (:obj:`int`, `optional`):
                    Number of text items to pull for each example.
                batch_size (:obj:`int`, `optional`):
                    Batch size.
                num_workers (:obj:`int`, `optional`):
                    Number of workers for data loader.
                dataset_names (:obj:`list`,`str`, `optional`):
                    Which datasets to use (ArXiv, BookCorpus2, Books3, DMMathematics, EnronEmails, EuroParl, 
                    Gutenberg_PG, HackerNews, NIHExPorter, OpenSubtitles, PhilPapers, UbuntuIRC, YoutubeSubtitles)).
                save_dataset (:obj:`bool`, `optional`):
                    Save the downloaded dataset or not.
                no_tokenizer (:obj:`bool`, `optional`):
                    To return non-tokenized text (EXPERIMENTAL, DO NOT USE)
                num_batches (:obj:`int`, `optional`):
                    The number of batches of data to prepare for the dataloader.
                _mock (:obj:`bool`, `optional`):
                    For testing, if true the dataset if filled with fake text data.  
        """   
        if config == None: 
            config = dataset.config()
        config = copy.deepcopy( config )
        config.dataset.block_size = block_size if block_size != None else config.dataset.block_size
        config.dataset.batch_size = batch_size if batch_size != None else config.dataset.batch_size
        config.dataset.num_workers = num_workers if num_workers != None else config.dataset.num_workers
        config.dataset.dataset_names = dataset_names if dataset_names != None else config.dataset.dataset_names
        config.dataset.save_dataset = save_dataset if save_dataset != None else config.dataset.save_dataset
        config.dataset.no_tokenizer = no_tokenizer if no_tokenizer != None else config.dataset.no_tokenizer
        config.dataset.num_batches = num_batches if num_batches != None else config.dataset.num_batches
        config.dataset._mock = _mock if _mock != None else config.dataset._mock
        dataset.check_config( config )

        if dataset_name is not None:
            warnings.warn("dataset_name as a parameter is deprecated and will be removed in a future release. Use `dataset_names` instead.", DeprecationWarning)
            config.dataset.dataset_names = dataset_name

        if config.dataset._mock:
            return dataset_mock.MockGenesisTextDataset(
                block_size = config.dataset.block_size,
                batch_size = config.dataset.batch_size,
                num_workers = config.dataset.num_workers,
                dataset_names = config.dataset.dataset_names,
                data_dir = config.dataset.data_dir,
                save_dataset = config.dataset.save_dataset,
                max_datasets = config.dataset.max_datasets,
                no_tokenizer = config.dataset.no_tokenizer,
                num_batches = config.dataset.num_batches,
            )
        else:
            return dataset_impl.GenesisTextDataset(
                block_size = config.dataset.block_size,
                batch_size = config.dataset.batch_size,
                num_workers = config.dataset.num_workers,
                dataset_names = config.dataset.dataset_names,
                data_dir = config.dataset.data_dir,
                save_dataset = config.dataset.save_dataset,
                max_datasets = config.dataset.max_datasets,
                no_tokenizer = config.dataset.no_tokenizer,
                num_batches = config.dataset.num_batches,
            )

    @classmethod
    def mock(cls):
        return dataset( _mock = True, dataset_names = ['Books3'])

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser 
            Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        dataset.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'dataset.batch_size', type=int, help='Batch size.', default = bittensor.defaults.dataset.batch_size)
            parser.add_argument('--' + prefix_str + 'dataset.block_size', type=int, help='Number of text items to pull for each example..', default = bittensor.defaults.dataset.block_size)
            parser.add_argument('--' + prefix_str + 'dataset.num_workers',  type=int, help='Number of workers for data loader.', default = bittensor.defaults.dataset.num_workers)
            parser.add_argument('--' + prefix_str + 'dataset.dataset_names', type=str, required=False, nargs='*', action='store', help='Which datasets to use (ArXiv, BookCorpus2, Books3, DMMathematics, EnronEmails, EuroParl, Gutenberg_PG, HackerNews, NIHExPorter, OpenSubtitles, PhilPapers, UbuntuIRC, YoutubeSubtitles)).',
                                                                    default = bittensor.defaults.dataset.dataset_names)
            parser.add_argument('--' + prefix_str + 'dataset.data_dir', type=str, help='Where to save and load the data.', default = bittensor.defaults.dataset.data_dir)
            parser.add_argument('--' + prefix_str + 'dataset.save_dataset', action='store_true', help='Save the downloaded dataset or not.', default = bittensor.defaults.dataset.save_dataset)
            parser.add_argument('--' + prefix_str + 'dataset.max_datasets',  type=int, help='Number of datasets to load', default = bittensor.defaults.dataset.max_datasets)
            parser.add_argument('--' + prefix_str + 'dataset.no_tokenizer', action='store_true', help='To return non-tokenized text (EXPERIMENTAL, DO NOT USE)',default=False)
            parser.add_argument('--' + prefix_str + 'dataset.num_batches', type=int, help='The number of data to download each time(measured by the number of batches).', default=bittensor.defaults.dataset.num_batches)
            parser.add_argument('--' + prefix_str + 'dataset._mock', action='store_true', help='To turn on dataset mocking for testing purposes.', default=False)

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
        defaults.dataset = bittensor.Config()
        defaults.dataset.batch_size = os.getenv('BT_DATASET_BATCH_SIZE') if os.getenv('BT_DATASET_BATCH_SIZE') != None else 10
        defaults.dataset.block_size = os.getenv('BT_DATASET_BLOCK_SIZE') if os.getenv('BT_DATASET_BLOCK_SIZE') != None else 20
        defaults.dataset.num_workers = os.getenv('BT_DATASET_NUM_WORKERS') if os.getenv('BT_DATASET_NUM_WORKERS') != None else 0
        defaults.dataset.dataset_names = os.getenv('BT_DATASET_DATASET_NAME') if os.getenv('BT_DATASET_DATASET_NAME') != None else 'default'
        defaults.dataset.data_dir = os.getenv('BT_DATASET_DATADIR') if os.getenv('BT_DATASET_DATADIR') != None else '~/.bittensor/data/'
        defaults.dataset.save_dataset = os.getenv('BT_DATASET_SAVE_DATASET') if os.getenv('BT_DATASET_SAVE_DATASET') != None else False
        defaults.dataset.max_datasets = os.getenv('BT_DATASET_MAX_DATASETS') if os.getenv('BT_DATASET_MAX_DATASETS') != None else 3
        defaults.dataset.num_batches = os.getenv('BT_DATASET_NUM_BATCHES') if os.getenv('BT_DATASET_NUM_BATCHES') != None else 100

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config for batch size, block size, corpus size, num_workers and dataset
        """
        assert config.dataset.batch_size > 0, 'Batch size must be larger than 0'
        assert config.dataset.block_size > 0, 'Block size must be larger than 0'
        assert config.dataset.num_workers >= 0, 'num_workers must be equal to or larger than 0'
        assert isinstance(config.dataset.save_dataset, bool) , 'save_dataset must be True/False only'
