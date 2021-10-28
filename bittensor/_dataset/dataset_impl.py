""" Implementation for the dataset and GenesisTextDataset class, which handles dataloading from ipfs
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
import random

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests


from loguru import logger
import bittensor

logger = logger.opt(colors=True)


class Dataset():
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self):
        
        # Used to retrieve directory contentx
        self.file_get = 'http://ipfs.opentensor.ai:5001/api/v0/object/get'
        self.pin_get = 'http://ipfs.opentensor.ai:5001/api/v0/pin/ls'
        # Used when current corpus has been exhausted
        self.refresh_corpus = False

    @staticmethod
    def requests_retry_session(
            retries=5,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 504),
            session=None,
        ):
        """ Creates a retriable session for request calls. This enables
        automatic retries and back-off retries should any request calls fail.

        Args:
            retries (int, optional): Maximum number of retries. Defaults to 3.
            backoff_factor (float, optional): Factor by which to back off if a retry fails. Defaults to 0.3.
            status_forcelist (tuple, optional): A set of integer HTTP status codes that we should force a retry on. Defaults to (500, 502, 504).
            session ([type], optional): Session for which to set up the retries. Defaults to None.

        Returns:
            requests.Session(): A Requests Session object set up for retries and backoff.
        """

        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def retrieve_file(self, address: str, params = None, action: str = 'post'):
        """Connects to Pinata IPFS gateway and retrieves files.

        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        session.params.update(params)
        if action == 'get':
            response = Dataset.requests_retry_session(session=session).get(address)
        elif action == 'post':
            response = Dataset.requests_retry_session(session=session).post(address)

        return response

    def __len__(self):
        """ Returns length of the dataset that the dataset is processing
        """

    def __getitem__(self, idx):
        """ Returns the next batch from the dataset.
        """

class GenesisTextDataset( Dataset ):
    """ One kind of dataset that caters for the data from ipfs 
    """
    def __init__(
        self,
        block_size,
        batch_size,
        max_corpus_size,
        num_workers,
        dataset_name,
        data_dir,
        save_dataset
    ):
        super().__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_corpus_size = max_corpus_size
        self.num_workers = num_workers
        self.tokenizer = bittensor.tokenizer( version = bittensor.__version__ )
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.save_dataset = save_dataset
        self.__infinite_dataset_iterator = None

        # Retrieve a random slice of the genesis dataset
        self.data = []

        # Used to refresh corpus if we've exhausted the whole dataset
        self.refresh_corpus = True

        if not os.path.isdir(os.path.expanduser(data_dir)):
            os.makedirs(os.path.expanduser(data_dir))

        self.directories = []
        self.init_directories()

    def init_directories(self):
        r""" Init function for getting directories
        Where a directory could be leading to a data file or a directory file 
        """
        # --- Getting dataset hashes from pin/ls.
        dataset_hashes = [] 
        response = self.retrieve_file(self.pin_get, (('type', 'recursive'),), action = 'post')
        if response.status_code != 200:
            dataset_hashes= [
                'QmSQ6AnnWQUy4bETQSAgkgCkJ1AQePSeKvbaFejizj5HP3',
                'QmVbNzncoJK8WwyAoWxLndk4999iyyYZbCKpEvUxrFXp1N',
                'QmYg67pZwPsX3qH31tEc1qexrPc88zUkZG4AqsNDZo5FEX'
            ]
        else:
            for hash, v in response.json()['Keys'].items():
                dataset_hashes.append(hash) 
        
        # --- Getting directories from the dataset hashes.
        # --- directories: List[ Map {Name: str, Hash: str, Size: int} ]
        directories = []
        for dataset_hash in dataset_hashes:
            response = self.retrieve_file(self.file_get, (('arg', dataset_hash),))
            if response.status_code != 200:
                logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(dir_hash))
            
            else:
                directory = response.json()
                if directory and 'Links' in directory.keys(): 
                    directories += directory['Links']

        if len(directories) == 0:
            directories = None
        
        self.directories = directories

    def extract_sub_directory(self, directory):
        """
        Check response from retrieve_directory(), 
        if the response is fine then return the links,
        else return None

        Returns:
            list of dict: links 
        """
        print('extract_sub_directory ', directory)
        # --- If the size of directory is small, it is leads to data file.
        if directory['Size'] <= 262158:
            return directory

        # --- Else, the directory leads to more directories, send request to break the directory.
        else:
            response = self.retrieve_file(self.file_get, (('arg', directory['Hash']),))
            
            # --- Return none if the request failed.
            if response.status_code != 200:
                logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(directory))
                return []
            
            else:
                # --- Pick a random sub_directory, run recursion until we have found a data file
                sub_directories = response.json()
                if sub_directories and 'Links' in sub_directories.keys():
                    random_sub_directory = random.choice(sub_directories['Links'])
                    
                    if random_sub_directory['Name'] == '':
                        random_sub_directory['Name'] = directory['Name']
                    
                    return self.extract_sub_directory(random_sub_directory)
                else:
                    logger.warning("Directory seems empty, ignoring directory:".ljust(20) + "<blue>{}</blue>". format(dir_hash))
        return []

    def get_text(self, file):
        """
        Load the data from disk if it is already in the the data_dir,
        else download it from  IPFS using retrieve_text_file and save it

        Returns:
            str: the data as string
        """
        text = None
        file_name = file['Name']
        file_hash = file['Hash']
        full_path = os.path.expanduser(os.path.join(self.data_dir, file_name))

        # Load text from path
        if os.path.exists(full_path):
            try:
                with open(full_path, mode='r') as f:
                    text = f.read()
                logger.success("Loaded:".ljust(20) + "<blue>{}</blue>".format(file_name))
            except Exception:
                logger.warning("Load failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        # Download text
        if text == None:
            response = self.retrieve_file(self.file_get, (('arg', file_hash),))

            if response.status_code != 200:
                logger.warning("Failed to retrieve file, ignoring file:".ljust(20) + "<blue>{}</blue>".format(file_name))
            else:
                text = response.text
                logger.success("Downloaded:".ljust(20) + "<blue>{}</blue>".format(file_name))
                
                # Saving text
                if self.save_dataset:
                    try:
                        with open(full_path, mode = 'w+') as f:
                            f.write(text)
                            logger.success("Saved:".ljust(20) + "<blue>{}</blue>".format(file_name))
                    except Exception:
                        logger.warning("Save failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        return text

    def construct_text_corpus(self):
        """Connects to Pinata IPFS gateway and retrieves the directory of genesis datasets.

        Returns:
            string: Contents of the text file.
        """
        try:
            logger.success("Retrieving a dataset files from the IPFS gateway...")

            # Retrieves the directory for the given dataset

            data_corpus = []

            # Generate a random order of the directories
            directory_order = list(range(len(self.directories)))
            random.shuffle(directory_order)

            # Pick a random dataset file and return its contents
            if self.directories:
                # Let's construct a dataset!
                total_dataset_size = 0
                i = 0

                # Make sure the file we chose satisfies our maximum file size requirement
                while total_dataset_size <= self.max_corpus_size:
                    # Get random data file
                    random_dataset_dir = self.extract_sub_directory(self.directories[directory_order[i]])
                    
                    if random_dataset_dir == None:
                        pass

                    text = self.get_text(random_dataset_dir)
                    print('get_text ', len(text))
                    if text != None:
                        data_corpus.extend(text.split())
                        total_dataset_size += int(random_dataset_dir['Size'])
                    i += 1

                return data_corpus

            logger.error("It appears the directory is empty... Restart your miner to try again.")
            return None
        except Exception as e:
            logger.error("Ran into exception when trying to retrieve dataset from IPFS: {}".format(e))

        return None

    def dataloader(self, epoch_length=None):
        """ Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): The epoch length of the miner. If this length is not set or if it is larger than the dataset,
            then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
            is returned. Defaults to None.

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        scale = 1
        # If we've exhausted the dataset, retrieve another corpus.
        if self.refresh_corpus or len(self) < (epoch_length * self.batch_size) * scale:
            self.data = self.construct_text_corpus()
            self.refresh_corpus = False

        # If epoch_length is set then we just need a slice of
        # the dataset we downloaded of length epoch_length.
        if epoch_length:

            # Set up upper bound of indices to fit the batch size we want.
            idx_bound = epoch_length * self.batch_size * scale
            if idx_bound < len(self):
                # Collect enough random indices to batch together using batch_size into epoch_length batches
                random_start = random.randint(0, len(self) - round(idx_bound ))
                indices = list(range(random_start, random_start + idx_bound))

                subset = Subset(self, indices)

                # Clear out these indices from our current corpus
                try:
                    del self.data[random_start: random_start + idx_bound]
                except Exception:
                    # There is too little data left over for us to delete according to our epoch_length,
                    # let's get more data!
                    self.refresh_corpus = True
            else:
                self.refresh_corpus = True
                return DataLoader(self,
                            shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=True)


            # Set up dataloader
            return DataLoader(subset,
                            shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=True)

        # If epoch_length is not set or it is higher than the total size of the dataset,
        #  then just shuffle dataset and return the whole thing.
        self.refresh_corpus = True
        return DataLoader(self,
                            shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=True)

    def __next__(self):
        """Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter([input for input in self.dataloader(1000000)])
        try:
            return next(self.__infinite_dataset_iterator)
        except StopIteration:
            self.__infinite_dataset_iterator = iter([input for input in self.dataloader(1000000)])
            return next(self.__infinite_dataset_iterator)

    def __len__(self):
        """Returns length of dataset minus the block size

        Returns:
            int: length of dataset minus block size
        """
        return max(len(self.data) - self.block_size, 0)

    def __getitem__(self, idx):
        """ Returns a batch of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                torch.tensor(dix)
        """
        start_idx = (idx*self.block_size)%len(self)
        end_idx = start_idx + self.block_size

        tokenized_text = torch.tensor(self.tokenizer(" ".join(self.data[start_idx:end_idx]), padding=True, truncation=True)['input_ids'], dtype=torch.long)

        return tokenized_text[:self.block_size]
