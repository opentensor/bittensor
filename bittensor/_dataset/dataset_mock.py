
""" Implementation for the mock dataset which returns dummy tokenized text.
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

from torch.utils.data.dataloader import DataLoader
import torch
from loguru import logger
import bittensor
from . import dataset_impl


logger = logger.opt(colors=True)

class MockGenesisTextDataset( dataset_impl.Dataset ):
    def __init__(
        self,
        block_size,
        batch_size,
        num_workers,
        dataset_names,
        data_dir,
        save_dataset,
        max_datasets,
        no_tokenizer,
        num_batches,
    ):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = bittensor.tokenizer( version = bittensor.__version__ )
        self.dataset_names = dataset_names
        self.data_dir = data_dir
        self.save_dataset = save_dataset
        self.datafile_size_bound = 262158
        self.max_datasets = max_datasets
        self.__infinite_dataset_iterator = None
        self.no_tokenizer = no_tokenizer

        # Retrieve a random slice of the genesis dataset
        self.data = []
        self.data_remained = []

    def close(self):
        pass

    def __del__(self):
        self.close()

    def construct_text_corpus(self, min_data_len = 0):         
        data_corpus = []
        total_dataset_len = 0
        i = 0
        while (total_dataset_len < min_data_len):
            text = "lorem ipsum data is not here this is super fake but maybe you could still learn from it?"
            text_list = text.split() 
            data_corpus.extend(text_list)
            total_dataset_len += len(text_list)
            i += 1
        return data_corpus

    def _fill_data(self, epoch_length:int = 100):
        data_size = epoch_length * self.batch_size * self.block_size
        
        # Make sure the data remained is at least as big as data_size 
        while len(self.data_remained) < (data_size) :
            self.data_remained += self.construct_text_corpus(min_data_len = data_size)

        self.data = self.data_remained[:data_size]
        del self.data_remained[:data_size]

    def dataloader(self, epoch_length = 100):
        """ Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): The epoch length of the miner. If this length is not set or if it is larger than the dataset,
            then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
            is returned. Defaults to None.

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        self._fill_data(epoch_length)
        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True)
    
    def __next__(self):
        """Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter(list(self.dataloader()))
        try:
            return next(self.__infinite_dataset_iterator)
        except StopIteration:
            self.__infinite_dataset_iterator = iter(list(self.dataloader()))
            return next(self.__infinite_dataset_iterator)

    def __len__(self):
        """Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        if (self.data == None) or (self.block_size == None) or (self.block_size == 0):
            return 0
        return round( len(self.data) / self.block_size )

    def __getitem__(self, idx):
        """ Returns a block of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                torch.tensor(dix)
        """
        start_idx = (idx * self.block_size) % len(self.data)
        end_idx = start_idx + self.block_size
        if self.no_tokenizer == False:
            tokenized_text = torch.tensor(self.tokenizer(" ".join(self.data[start_idx:end_idx]), truncation=True)['input_ids'], dtype=torch.long)
        elif self.no_tokenizer == True:
            tokenized_text = " ".join(self.data[start_idx:end_idx])

        return tokenized_text[:self.block_size]

