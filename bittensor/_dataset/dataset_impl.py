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

import concurrent
import json
import os
import random
import time
import warnings
from multiprocessing import cpu_count
from typing import Union

import requests
import torch
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from torch.utils.data.dataloader import DataLoader

import bittensor

from .thread_queue import ThreadQueue

logger = logger.opt(colors=True)


class Dataset:
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self):
        
        # Used to retrieve directory contentx
        self.dataset_dir = 'http://global.ipfs.opentensor.ai/api/v0/cat' 
        self.text_dir = 'http://global.ipfs.opentensor.ai/api/v0/object/get'
        self.mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
        # Used when current corpus has been exhausted
        self.refresh_corpus = False
        

    @staticmethod
    def requests_retry_session(
            retries=1,
            backoff_factor=0.5,
            status_forcelist=(104, 500, 502, 504),
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

    def get_ipfs_directory(self, address: str, file_meta: dict, action: str = 'post', timeout : int = 180):
        r"""Connects to IPFS gateway and retrieves directory.
        Args:
            address: (:type:`str`, required):
                The target address of the request. 
            params: (:type:`tuple`, optional):
                The arguments of the request. eg. (('arg', dataset_hash),)
            action: (:type:`str`, optional):
                POST or GET.
            timeout: (:type:`int`, optional):
                Timeout for getting the server's response. 
        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        session.params.update((('arg', file_meta['Hash']), ))
        
        try:
            if action == 'get':
                response = self.requests_retry_session(session=session).get(address, timeout=timeout)
            elif action == 'post':
                response = self.requests_retry_session(session=session).post(address, timeout=timeout)

        except Exception as E:
            logger.error(f"Failed to get from IPFS {file_meta['Name']} {E}")
            return None

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
        self.IPFS_fails = 0
        self.backup_dataset_cap_size = 5e7 # set 50MB limit per folder
        self.IPFS_fails_max = 10
        self.num_batches = num_batches

        # Ensure dataset_names is formatted correctly
        if isinstance(self.dataset_names, str):
            self.dataset_names = [self.dataset_names]

        allowed_datasets = bittensor.__datasets__ + ["default"]
        for dataset_name in self.dataset_names:
            if dataset_name not in allowed_datasets:
                self.dataset_names.remove(dataset_name)
                warnings.warn(f"Requested dataset {dataset_name} not in allowed datasets: {allowed_datasets}")

        # Retrieve a random slice of the genesis dataset
        self.data = []
        self.data_reserved = []

        # Used to refresh corpus if we've exhausted the whole dataset
        self.refresh_corpus = True

        self.build_hash_table()

        os.makedirs(os.path.expanduser(data_dir), exist_ok=True)
            
        self.data_queue = ThreadQueue(
            producer_target = self.reserve_multiple_data,
            producer_arg = (self.num_batches, ),
            buffer_size = 1
        )

    def __del__(self):
        self.close()

    def close(self):
        self.data_queue.close()

    def get_folder_size(self, folder):
        r""" Get the size (in byte) of a folder inside the data_dir.
        Args:
            folder (str):
                The name of the folder
        
        Returns:
            total_size (int):
                The memory size of the folder (in byte). 
        """
        total_size = 0
        full_path = os.path.expanduser(os.path.join(self.data_dir, folder))
        for dirpath, dirnames, filenames in os.walk(full_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size

    def load_hash(self, file_meta):
        r""" Load a hash from disk.
        Args:
            file_meta (dict of str: int):
                Specify the details of the dataset in the format of {'Name': , 'Hash':}.

        Returns:
            text (str): 
                The text in the file.                
        """

        full_path = os.path.expanduser(os.path.join(self.data_dir, file_meta['Folder'], file_meta['Hash']))
        if os.path.exists(full_path):
            try:
                with open(full_path, mode='r') as f:
                    text = f.read()

                logger.success("Loaded from disk:".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
            except Exception:
                logger.success("Could not load from disk:".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
                pass

            return text
        
        return None

    def save_hash(self, file_meta, text):
        r""" Save a hash to disk.
        Args:
            file_meta (dict of str: int):
                Specify the details of the dataset in the format of {'Name': , 'Hash':}.
            text (str): 
                The string to save to the file.
        
        Returns:
            text (str):
                The text in the file.                
        """
        folder_path = os.path.expanduser(os.path.join(self.data_dir, file_meta['Folder']))
        full_path = os.path.expanduser(os.path.join(self.data_dir, file_meta['Folder'], file_meta['Hash']))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        try:
            with open(full_path, mode = 'w+') as f:
                f.write(text)
                logger.success("Saved:".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
            return True
        
        except Exception as E:
            logger.warning("Save failed:".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
            return False

    def get_text(self, file_meta):
        r""" Either load a file from disk or download it from IPFS
        Args:
            file_meta (dict of str: int):
                Specify the details of the file in the format of {'Name': , 'Hash':}.

        Return:
            text (str):
                The text that we get from the file (from disk or IPFS).     
        """
        text = None
        response = self.get_ipfs_directory(self.text_dir, file_meta)
        if (response != None) and (response.status_code == 200):
            try:
                text = json.loads(response.text)['Data']
            except json.decoder.JSONDecodeError:
                text = response.text

            self.IPFS_fails = 0
            
            if self.save_dataset and self.dataset_hashes[file_meta['Folder']]['Size'] < self.backup_dataset_cap_size:
                self.save_hash( file_meta, text )
                self.dataset_hashes[file_meta['Folder']]['Size'] += file_meta['Size']
            
        else:
            logger.warning("Failed to get text".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
            self.IPFS_fails += 1
            
        return text 

    def get_dataset(self , file_meta):
        r""" Either load a dataset, which is a list of hashes, from disk or download it from IPFS
        Args:
            file_meta (dict of str: int):
                Specify the details of the dataset in the format of {'Name': , 'Hash':}.

        Return:
            hashes (list):
                The hashes from the dataset downloaded from disk or IPFS.     
        """
        # --- Load text from path
        logger.success( f"Getting dataset: {file_meta['Name']}" )
        
        hashes = self.load_hash(file_meta)

        if hashes != None:
            hashes = json.loads(hashes)

        # --- If couldnt load from path, download text.
        else:
            response = self.get_ipfs_directory(self.dataset_dir, file_meta)
            if (response != None) and (response.status_code == 200):
                self.IPFS_fails = 0
                hashes = response.json()

                # --- Save text if the save_dataset flag is on.
                if self.save_dataset :
                    self.save_hash(file_meta, json.dumps(response.json()) )
                    
            else:
                self.IPFS_fails += 1
                logger.warning("Failed to get dataset".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))
                return None

        if hashes == None:
            return None
        else:
            for h in hashes:
                h['Folder'] = file_meta['Name']
            return hashes 

    def get_hashes_from_dataset(self):
        r""" Getting directories .
        Where a directory could be leading to a data file or a directory file.

        Returns:
            directories (:type:`list`, `required`)
                A list of directory.
                    directory: Map{ Name: str, Hash: str, Size: int }: 
                        A random directory that lead to a datafile.
        """
        def get_hashes(dataset_meta):
            if self.IPFS_fails > self.IPFS_fails_max:
                sub_directories = json.loads(self.load_hash(dataset_meta))
                for sub_directory in sub_directories:
                    sub_directory['Folder'] = dataset_meta['Name']
            else:
                sub_directories = self.get_dataset(dataset_meta)

            if sub_directories != None:
                return sub_directories

            return []
        
        directories = []
        self.IPFS_fails = 0
        
        if self.dataset_names == ['default']:
            i = 0
            dataset_hashes = list(self.dataset_hashes.values())
            random.shuffle(dataset_hashes)
            
            for dataset_hash in dataset_hashes: 
                dataset_meta = {'Folder': 'mountain', 'Name': dataset_hash['Name'], 'Hash': dataset_hash['Hash']}
                directories += get_hashes(dataset_meta)
                i += 1
                if i >= self.max_datasets:
                    break
                    
        else:
            for key in self.dataset_names:
                if key in self.dataset_hashes.keys():
                    dataset_meta = {'Folder': 'mountain','Name': key, 'Hash': self.dataset_hashes[key]['Hash'] }  
                    directories += get_hashes(dataset_meta)

                else:
                    logger.error('Incorrect dataset name:'.ljust(20) + " <red>{}</red>.".format(key)+' Must be one of the following {}'.format(bittensor.__datasets__))

        if len(directories) == 0:
            logger.error('Could not get any directory from IPFS or local.')
            directories = None
          
        return directories

    def get_root_text_hash(self, file_meta):
        r"""
        With recursion, from the given directory, get a directory that leads to a datafile.

        Args:
            directory: Map{ Name: str, Hash: str, Size: int }: 
                The original directory to look up a datafile for.

        Returns:
            directory: Map{ Name: str, Hash: str, Size: int }: 
                A random directory that lead to a datafile.
        """
        # --- If the size of directory is small, it is leads to data file, return the data file.
        if file_meta['Size'] <= self.datafile_size_bound:
            return file_meta

        # --- Else, the directory leads to more directories, return a random data file within the directories.
        else:
            response = self.get_ipfs_directory(self.text_dir, file_meta)
            # --- Return none if the request failed.
            if (response == None) or (response.status_code != 200):
                logger.warning("Failed to retrieve directory, ignoring directory:".ljust(20) + "<blue>{}</blue>".format(file_meta))
                return None
            
            # --- Pick a random sub_directory, run recursion until we have found a data file
            else:
                sub_directories = response.json()
                if sub_directories and 'Links' in sub_directories.keys() and len(sub_directories['Links']) >= 1:
                    random_sub_directory = random.choice(sub_directories['Links'])

                    # --- Fill the name of the random_sub_directory if it is empty. 
                    if random_sub_directory['Name'] == '':
                        random_sub_directory['Name'] = file_meta['Name']
                        random_sub_directory['Folder'] = file_meta['Folder']

                    
                    return self.get_root_text_hash(random_sub_directory)
                else:
                    logger.warning("Directory seems empty, ignoring directory:".ljust(20) + "<blue>{}</blue>". format(file_meta))
        return None

    def get_text_from_local(self, min_data_len):

        folders = os.listdir( os.path.expanduser (self.data_dir))
        if self.dataset_names == ['default']:
            folders_avail = folders
            random.shuffle(folders_avail)
            folders_avail = folders_avail[:self.max_datasets]
        else:
            folders_avail = []
            for dataset_name in self.dataset_names:
                if dataset_name in folders:
                    folders_avail.append(dataset_name)
            random.shuffle(folders_avail)

        files = [] 
        for folder in folders_avail:
            file_names = os.listdir(os.path.expanduser(os.path.join(self.data_dir, folder)))
            sub_files = [{'Name': file_name,'Folder': folder, 'Hash': file_name} for file_name in file_names]
            files += sub_files

        random.shuffle(files)
        data_corpus = []
        total_dataset_len = 0

        for text_file in files:
            # --- Get text from the datafile directory
            text = self.load_hash(text_file)

            if text != None:
                text_list = text.split() 
                data_corpus.extend(text_list)
                total_dataset_len += len(text_list)
            
            if (total_dataset_len > min_data_len) :
                break

        return data_corpus

    def construct_text_corpus(self, min_data_len = 0):
        """ Main function for generating the text data.
        1. Get directories from a random dataset_hash (dataset_hash is the result from calling pin/ls).
        2. Pick a random directory and get the directory that would lead to a datafile.    
        3. Get text from the directory.
        4. Repeat 2,3 until we have reached the min data length

        Returns:
            text: str: 
                Contents of the text data.
        """
        self.IPFS_fails = 0
        data_corpus = []
        try:
            # --- Get directories from a random dataset_hash
            directories = list(self.get_hashes_from_dataset())

            # --- Generate a random order of the directories
            random.shuffle(directories)

            # --- Pick random directories and get their text contents.
            if directories:
                total_dataset_size = 0
                total_dataset_len = 0
                i = 0

                # --- Dont stop until the corpus size and the minimum data_length was reached.
                n_workers = cpu_count() if self.num_workers == 0 else self.num_workers
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    while (total_dataset_len < min_data_len) and (self.IPFS_fails <= self.IPFS_fails_max):
                        future_map = {}
                        for idx, call_arg in enumerate(directories[:n_workers]):
                            future = executor.submit(self.get_text, call_arg)
                            future_map[future] = call_arg
                        
                        for i, future in enumerate(concurrent.futures.as_completed(future_map)):
                            text = future.result()
                            if text is not None:
                                text_list = text.split()
                                data_corpus.extend(text_list)
                                total_dataset_len += len(text_list)
                        
                        logger.success("Loaded from IPFS".ljust(20) + f"<yellow>{ round(total_dataset_len / min_data_len * 100) }%</yellow>  " + "<blue>{}</blue>".format([file_meta['Name'] for file_meta in directories[:n_workers]]))
                        directories = directories[n_workers:]

            else:
                logger.error("It appears the directory is empty... Restart your miner to try again.")

        except Exception as e:
            logger.error("Ran into exception when trying to retrieve dataset from IPFS: {}".format(e))


        if len(data_corpus) == 0:
            logger.error("Fail to construct any text from IPFS, getting from local instead.")
            data_corpus = self.get_text_from_local(min_data_len)

        return data_corpus

    def reserve_multiple_data(self, epoch_length = 100, multiples = 2):
        r""" Make sure the reserved data meet the multiple, 
        If not, then keep constructing text corpus.
        Arg:
            epoch_length (int, optional): 
                A dataloader for a subset of the dataset of epoch_length is returned.
            
            multiples (int, optional):
                The number of dataloader that the data_reserved should be able to create.

        Return: 
            success (bool):
                If we have got the data ready.
        """
        logger.success(f"Reserving data with multiples: {multiples}")
        data_size = epoch_length * self.batch_size * self.block_size
        
        while len(self.data_reserved) < data_size * multiples :
            self.data_reserved += self.construct_text_corpus(min_data_len = data_size)

        logger.success(f"Dataset download completed, {multiples} copy of data reserved")
        return True

    def set_data_size(self, batch_size, block_size):
        r""" Update the size of data (batch_size, block_size) that we need.

        Args: 
            batch_size(int, required):
                The batch_size of data that should be produced by dataloader.

            block_size(int, required):
                The block_size of data that should be produced by dataloader. 
        """
        def check_valid(size):
            r""" Check if the size is a valid positive intiget, if not, return False.
            """
            if size <= 0 or (not isinstance(size, int)):
                return False
            else:
                return True

        old_batch_size = self.batch_size
        old_block_size = self.block_size
        
        if check_valid(batch_size):
            self.batch_size = batch_size
        
        if check_valid(block_size):
            self.block_size = block_size

        # empty the queue
        while not self.data_queue.queue.empty():
            self.data_queue.queue.get()

        # empty the dataset_iterator with the old sizing
        self.__infinite_dataset_iterator = iter([])

        logger.success(f"Updated data size: batch_size: {old_batch_size} --> {self.batch_size}, block_size: {old_block_size} --> {self.block_size}")

    def dataloader(self, epoch_length = 100):
        r""" Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): 
                A dataloader for a subset of the dataset of epoch_length is returned.

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        logger.success(f"Getting a new Dataloader")
        data_size = epoch_length * self.batch_size * self.block_size
        if len(self.data_reserved) < data_size:
            self.reserve_multiple_data(self.num_batches, 1)

        self.data = self.data_reserved[:data_size]

        del self.data_reserved[:data_size]

        # Datalaoder calls self._getitem_ functions until the self.data uses up, and group the result by batch size
        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True)
    
    def set_dataset_iterator(self):
        r""" Get a new dataset that is ready from the queue. The result would be updated to self.__infinite_dataset_iterator__ . 
        """
        success = False 
        while not success:
            if not self.data_queue.queue.empty() :
                ready = self.data_queue.queue.get() # the queue stores a bool ready signal
                dataset = self.dataloader(self.num_batches)
                if dataset:
                    self.__infinite_dataset_iterator = iter([input for input in dataset])
                    success = True
            else:
                time.sleep(2)

        return

    def __next__(self):
        """Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.set_dataset_iterator()

        try:
            return next(self.__infinite_dataset_iterator)
        
        except StopIteration:
            self.set_dataset_iterator()
            return next(self.__infinite_dataset_iterator)

    def __len__(self):
        """Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        if (self.data == None) or (self.block_size == None) or (self.block_size == 0):
            return 0
        return round( len(self.data) / self.block_size )

    def __getitem__(self, idx: int) -> Union[str, torch.tensor]:
        """ Returns a block of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                torch.tensor(dix)
        """
        start_idx = (idx * self.block_size) % len(self.data)
        end_idx = start_idx + self.block_size
        text = " ".join(self.data[start_idx:end_idx])

        if self.no_tokenizer is True:
            return text
        else:
            tokens = self.tokenizer(text, padding=True, truncation=True)["input_ids"]
            return torch.tensor(tokens, dtype=torch.long)[:self.block_size]

    def build_hash_table(self):
        self.IPFS_fails = 0
        self.dataset_hashes = {}
        response = None

        mountain_meta = {'Name': 'mountain', 'Folder': 'meta_data', 'Hash': self.mountain_hash}
        
        while response == None:
            self.IPFS_fails += 1
            response = self.get_ipfs_directory(self.text_dir, mountain_meta)
            
            if response:
                dataset_hashes = response.json()['Links']
                if self.save_dataset:
                    self.save_hash(mountain_meta, json.dumps(dataset_hashes) )
            
            if self.IPFS_fails > self.IPFS_fails_max and response == None:
                dataset_hashes = json.loads(self.load_hash(mountain_meta))
                break

        for i in dataset_hashes:
            name = i['Name'][:-4]
            dataset_meta = {'Name': name, 'Hash': i['Hash'], 'Size': self.get_folder_size(name) }
            self.dataset_hashes[name] = dataset_meta
