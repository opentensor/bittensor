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

import bittensor
import torch
import random
import os

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from loguru import logger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from loguru import logger
logger = logger.opt(colors=True)

class Dataloader():
    def __init__(self):
        # IPFS hash of the genesis dataset
        # TODO (shibshib): Find a proper way to set this as config instead of hardcoding it.
        # More dataset hashes can be added as we add directories for other modalities.
        self.genesis_text_dataset_hash = "QmXwfPoh2QFYqC6cYcW8kzyd9ruFfhnUi2kVBkdhawjUzj"
        self.wikitext_text_dataset_hash = 'QmRjFNn3XpYMycVzTE4YcVcxk45vNZcTAjKmtEqPLKwAWd'
        self.test_text_dataset_hash = 'QmRhWSMPQzTiWcdGYy8vpRMxSxCAKDJBaXvmum4fjkF7cJ'
        self.validation_text_dataset_hash = 'QmQnE8wBmxKgNteFkZ1RAdZFit16iSeHwX6zSpYfwFmAuG'

        # Used to retrieve directory contentx
        self.dag_get = 'https://gateway.pinata.cloud/api/v0/dag/get'
        # Used to retrieve file contents
        self.file_cat = 'https://gateway.pinata.cloud/api/v0/cat'

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

    def retrieve_directory(self, dir_hash: str):
        """Connects to Pinata IPFS gateway and retrieves the directory of
        genesis datasets.

        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        params = (('arg', dir_hash),)
        session.params.update(params)
        directory = None

        response = Dataloader.requests_retry_session(session=session).post(self.dag_get)

        if response.status_code == 200:
            directory = response.json()

        return directory

    def __len__(self):
        """ Returns length of the dataset that the dataloader is processing
        """
        pass

    def __getitem__(self, idx):
        """returns the next batch from the dataset.
        """
        pass

class GenesisTextDataloader( Dataloader ):

    def __init__(
        self,
        block_size,
        batch_size,
        max_corpus_size,
        num_workers,
        dataset,
        data_dir
    ):
        super(GenesisTextDataloader, self).__init__()
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_corpus_size = max_corpus_size
        self.num_workers = num_workers
        self.tokenizer = bittensor.tokenizer( version = bittensor.__version__ )
        self.dataset = dataset
        self.data_dir = data_dir

        # Retrieve a random slice of the genesis dataset
        self.data = []

        # Used to refresh corpus if we've exhausted the whole dataset
        self.refresh_corpus = True

        if not os.path.isdir(os.path.expanduser(data_dir)):
            os.makedirs(os.path.expanduser(data_dir))

    def get_text(self, file_hash: str, file_name: str):
        """
        Load the data from disk if it is already in the the data_dir,
        else download it from  IPFS using retrieve_text_file and save it

        Returns:
            str: the data as string
        """
        text = ''
        full_path = os.path.expanduser(os.path.join(self.data_dir, file_name))

        # Load text from path
        if (os.path.exists(full_path)):
            try:
                with open(full_path, mode='r') as f:
                    text = f.read()
                logger.success("Loaded:".ljust(20) + "<blue>{}</blue>".format(file_name))
            except Exception as e:
                logger.warning("Load failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        # Download text
        if (text == ''):
            text = self.retrieve_text_file(file_hash).text
            logger.success("Downloaded:".ljust(20) + "<blue>{}</blue>".format(file_name))

            # Saving text
            try:
                with open(full_path, mode = 'w+') as f:
                    f.write(text)
                    logger.success("Saved:".ljust(20) + "<blue>{}</blue>".format(file_name))
            except Exception as e:
                    logger.warning("Save failed:".ljust(20) + "<blue>{}</blue>".format(file_name))

        return text

    def retrieve_text_file(self, file_hash: str):
        """Connects to Pinata IPFS gateway and retrieves the contents of
        a genesis text file.

        Returns:
            str: The contents of the file.
        """
        session = requests.Session()
        params = (('arg', file_hash),)
        session.params.update(params)
        directory = None

        response = Dataloader.requests_retry_session(session=session).post(self.file_cat)
        if response.status_code == 200:
            directory = response

        return directory

    def construct_text_corpus(self):
        """Connects to Pinata IPFS gateway and retrieves the directory of genesis datasets.

        Returns:
            string: Contents of the text file.
        """
        try:
            logger.success("Retrieving a dataset files from the IPFS gateway...")

            # Retrieves the directory for the given dataset
            if self.dataset == 'train':
                directory_links = []
                for hash in [self.genesis_text_dataset_hash, self.wikitext_text_dataset_hash]:
                    dir = self.retrieve_directory(hash)
                    if dir and 'links' in dir.keys():
                        directory_links.extend(dir['links'])

            elif self.dataset == 'test':
                if dir and 'links' in dir.keys():
                    directory_links = self.retrieve_directory(self.test_text_dataset_hash)['links']

            elif self.dataset == 'validation':
                if dir and 'links' in dir.keys():
                    directory_links = self.retrieve_directory(self.validation_text_dataset_hash)['links']

            data_corpus = []

            # Generate a random order of the directories
            directory_order = list(range(len(directory_links)))
            random.shuffle(directory_order)

            # Pick a random dataset file and return its contents
            if directory_links:
                # Let's construct a dataset!
                random_dataset_file = directory_links[directory_order[0]]
                file_name = random_dataset_file['Name']
                total_dataset_size = 0

                # Make sure the file we chose satisfies our maximum file size requirement
                i = 1
                while total_dataset_size <= self.max_corpus_size:
                    # Find file hash
                    random_dataset_file_hash = random_dataset_file['Cid']['/']
                    text = self.get_text(random_dataset_file_hash, file_name)
                    data_corpus.extend(text.split())
                    total_dataset_size += int(random_dataset_file['Size'])

                    # Retrieve next file descriptor
                    random_dataset_file = directory_links[directory_order[i]]
                    file_name = random_dataset_file['Name']
                    i += 1

                return data_corpus

            logger.error("It appears the directory is empty... Restart your miner to try again.")
            return None
        except Exception as ex:
            logger.error("Ran into exception when trying to retrieve dataset from IPFS: {}".format(ex))

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
        scale = 2.1
        # If we've exhausted the dataset, retrieve another corpus.
        if self.refresh_corpus or len(self) < (epoch_length * self.batch_size) * scale:
            self.data = self.construct_text_corpus()
            self.refresh_corpus = False

        # If epoch_length is set then we just need a slice of
        # the dataset we downloaded of length epoch_length.
        if epoch_length and epoch_length < len(self.data):

            # Set up upper bound of indices to fit the batch size we want.
            idx_bound = epoch_length * self.batch_size
            if idx_bound * scale< len(self):
                # Collect enough random indices to batch together using batch_size into epoch_length batches
                random_start = random.randint(0, len(self) - round(idx_bound * 2.1))
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

    def __len__(self):
        """Returns length of dataset minus the block size

        Returns:
            int: length of dataset minus block size
        """
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """ Returns a batch of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                x
        """
        chunk = self.data[idx:idx + self.block_size]

        dix = []
        block_num=0
        while block_num < self.block_size and len(chunk) > block_num:
            tokenized = self.tokenizer(chunk[block_num], padding=True, truncation=True)['input_ids']
            for t in tokenized:
                if block_num < self.block_size:
                    dix.append(t)
                    block_num += 1

        if len(dix) == 0:
            return None

        x = torch.tensor(dix, dtype=torch.long)
        return x
