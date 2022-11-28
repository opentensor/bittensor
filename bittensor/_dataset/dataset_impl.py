"""
 Implementation of Dataset using Asyncio and IPFS
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import asyncio
import aiohttp
import bittensor
from copy import deepcopy
import json
from loguru import logger
import random
import os
import torch
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Union, Dict, List, Any
from .utils import sync_wrapper

logger = logger.opt(colors=True)

class GenesisTextDataset:
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    ipfs_url = bittensor.__ipfs_url__
    dataset_dir = bittensor.__dataset_dir__
    text_dir = bittensor.__text_dir__
    mountain_hash = bittensor.__mountain_hash__

    def __init__(
            self, 
            batch_size: int, 
            sequence_length: int,
            block_size_bytes: int ,
            max_hash_size:int,
            num_workers: int,
            datasets: Union[List[str], str], 
            no_tokenizer: bool ,
            data_dir: str ,
            save_dataset : bool ,
            load_dataset : bool ,
            buffer_size:int,
            buffer_calls_per_update: int,
            num_batches: int ,
            max_datasets: int ,
            max_directories: int,
            loop: Optional['asyncio.loop'] = None ):
        """
        The genesis dataset that represents interfacing with the mountain stored in IPFS hosted by Bittensor's IPFS nodes.
        Args:
            batch_size (int, required):
                The size of the batch.
            sequence_length (int, required):
                The length of the seqeunce (tokenwise).
            block_size_bytes (int, required): 
                The  size of the text data block which is used to create. 
                multiple samples.
            max_hash_size (int, required): 
                The maximum size of the block that represents the hash.
            num_workers (int, required):
                Number of workers for pytorch Dataset.
            datasets (Optional[Union[List[str], str]]):
                List of dataset names to include from the pile.
            no_tokenizer (bool, required):
                Do not inlcude tokenizer.
            data_dir (str, required):
                Directory for saving assets.
            save_dataset (bool, required):
                Save the dataset text hashes.
            load_dataset (bool, required):
                Load the dataset text hashes.
            num_batches (int, required):
                Number of generated batches in epoch.
            max_datasets (int, required):
                Max number of datasets.
            buffer_size (int, required):
                Size of blocks to buffer while training.
            buffer_calls_per_update (int, required):
                Calls per block when caching.
            loop ('asyncio.loop',  optional):
                Asyncio loop for class, defaults to default event loop.
        """
        self.__infinite_dataset_iterator = None
        self.batch_size = batch_size
        self.block_size_bytes = block_size_bytes
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.datasets = datasets
        self.max_directories = max_directories
        self.max_hash_size = max_hash_size
        self.set_event_loop(loop=loop)
        self.max_datasets = max_datasets
    
        if self.datasets == 'default' or self.datasets == None:
            self.datasets = self.available_datasets

        self.datasets = self.datasets[:self.max_datasets]
        self.no_tokenizer = no_tokenizer
        self.buffer_calls_per_update = buffer_calls_per_update
        self.sample_cat_tasks = []
        self.sample_buffer = []

        # Tokenizer 
        self.tokenizer =  bittensor.tokenizer()
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_idx = self.tokenizer(self.pad_token)['input_ids'][0]

        self.data_dir =  data_dir
        self.save_dataset = save_dataset
        self.load_dataset = load_dataset

        # set the buffer
        self.set_buffer(buffer_size=buffer_size)

        # TODO: currently the number of batches is inert as this loop runs forever
        self.num_batches = num_batches
        self.sample_count = 0
        self.batch_count = 0

        # Build the text corpus by fetching the hashes of the textfiles (Current Heirarchy)
        self.construct_text_corpus(datasets=self.datasets, load=self.load_dataset, save=self.save_dataset)

    def construct_text_corpus(self, datasets:Optional[List[str]] = None, save:bool = False, load:bool = False) -> None :
        """ Building all of the datasets specified by getting each of their 
            text hashes from IPFS or local
        Args:
            datasets (List[str], optional):
                List of dataset names to include from the pile.
            save (bool, required):
                Save the dataset hashes locally.
            load (bool, required):
                Load the dataset hashes locally.
        """

        datasets = datasets if datasets else self.datasets
        all_text_file_metas = []
        dataset_hash_map = {}
        tasks = []
        self.dataset_size_map = {d:0 for d in datasets}
        
        # Gather dataset hashes async as their state is independent.
        for dataset in datasets:
            tasks += [self.async_build_single_dataset(dataset=dataset, save=save, load=load)]

        # Get the hashes asynchronously for each dataset.
        dataset_hashes = asyncio.run(asyncio.gather(*tasks))

        # Create a hash map of dataset -> text hashes.
        for k,v in zip(datasets, dataset_hashes):
            if len(v) > 0:
                dataset_hash_map[k] = v
        
        self.dataset_hash_map = dataset_hash_map

        # Flatten the hashes to a list of hashes.
        for k,file_meta_list in dataset_hash_map.items():
            all_text_file_metas += [fm  for fm in file_meta_list if fm['Size'] >= self.block_size_bytes]
        
        # Ensure the hash list is not empty.
        assert len(all_text_file_metas) > 0
        self.all_text_file_metas = all_text_file_metas

    async def async_save_json(self, path:str, obj:Union[dict, list], include_root:Optional[bool]=True) -> str:
        """ 
        Async save of json for storing text hashes

        Args:
            path (str, required):
                Path of json to save. If include_root is true, then it is suffixed with {self.data_dir}.
            obj (bool, required):
                The object to save locally
            include_root (bool, optional):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
            
        Returns: 
            path (str)
                Path of the saved JSON.
        """
        
        if include_root:
            path = os.path.join(self.data_dir, path)

        dir_path = os.path.dirname(path)

        # Ensure the json is the prefix.
        if path[-len('.json'):] != '.json':
            path += '.json'

        # Ensure the directory exists, make otherwise.
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        assert os.access( dir_path , os.W_OK ), f'dir_path:{dir_path} is not writable'
        with open(path, 'w') as outfile:
            json.dump(obj, outfile)

        return path

    save_json = sync_wrapper(async_save_json)

    async def async_load_json(self, path:str, include_root:Optional[bool] = True, default:Optional[Union[list, dict]] = {}) -> Union[list, dict]:

        """ 
        Async save of json for storing text hashes
        Args:
            path (str, required):
                Path of the loaded json
            include_root (bool, optional):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
            default (dict, optional):
                If there is a file not found, what is the default value.
        Returns: 
            obj (dict)
                Object of the saved JSON as a dictionary.
        """
        
        # Include the root if true (self.data_dir).
        if include_root:
            if self.data_dir != path[:len(self.data_dir)]:
                path = os.path.join(self.data_dir, path)

        # Ensure extension.
        dir_path = os.path.dirname(path)
        if os.path.splitext(path)[-1] != '.json':
            path += '.json'

        # Ensure dictionary.
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        # Load default if file does not exist.
        try:
            with open(path, 'r') as f:
                obj = json.load(f)
        except FileNotFoundError:
            obj = default
        except json.JSONDecodeError:
            obj = default
        if isinstance(obj, str):
            obj = json.loads(obj)

        return obj

    load_json = sync_wrapper(async_load_json)

    async def async_build_single_dataset(self, dataset:str , save:Optional[bool]=False, load:Optional[bool]=True) -> List[dict] :
        """ Building a single dataset by fetching its text file metas ({Hash:str, Name:str, Size: int})
        Args:
            dataset (List[str], required):
                The name of the dataset.
            load (bool, optional):
                Load the dataset hashes locally
            save (bool, optional):
                Save the dataset hahses locally.

        Returns: 
            text_file_metas (List[Dict): 
                List of text file metas with the format of {Hash:String, Size:String, Name:String}.
        """
        # Hash to the meta file to avoid duplication in case we load two of the same file_meta.
        hash2file_meta = {}

        # If load is true, load the hashes, otherwise, fetch them from ipfs.
        if load:
            loaded_file_metas =  self.load_json(path=f'{dataset}/file_metas', default=[])
            for file_meta in loaded_file_metas:
                hash2file_meta[file_meta['Hash']] = file_meta
            
            text_file_metas = list(hash2file_meta.values())
                        
        else:
            # Get the folder_hashes from the dataset.
            folder_hashes = (await self.get_folder_hashes(self.dataset2hash[dataset]))[:self.max_directories]

            # For each folder, get the text hashes.
            tasks = []
            for f in folder_hashes:
                tasks.append(self.get_folder_text_hashes(f, dataset=dataset))

            # Some hashes are incomplete, ensure they have Size and Hash Field.
            for folder_text_file_metas in await asyncio.gather(*tasks):
                for file_meta in folder_text_file_metas:
                    if 'Size' in file_meta and 'Hash' in file_meta:
                        hash2file_meta[file_meta['Hash']] = file_meta   
                    
            text_file_metas = list(hash2file_meta.values())

            # If save is true, then save the hashes into the {self.data_dir}/{dataset}/file_metas.
            if save:
                self.save_json(path=f'{dataset}/file_metas', obj=text_file_metas)
        
        # Calculate the size.
        self.dataset_size_map[dataset]  = sum([fm['Size'] for fm in text_file_metas])

        return text_file_metas

    def set_data_size(self, batch_size:Optional[int] = None, block_size:Optional[int] = None, sequence_length:Optional[int] = None,  block_size_bytes:Optional[int]= None, buffer_size:Optional[int]=None) -> None:
        r""" 
        Update the size of data (batch_size, sequence_length, block_size_bytes) that we need.

        Args: 
            batch_size (int, optional):
                The batch_size of data that should be produced by dataloader.

            sequence_length (int, optional):
                The number of tokens for each sample.

            block_size_bytes (int, optional):
                The block_size_bytes of data in bytes that should be produced by dataloader. 

            buffer_size(int, optional):
                The size of the buffer. 
        """

        def check_valid(size:int):
            r""" 
            Check if the size is a valid positive integer, if not, return False.
            """
            if (not isinstance(size, int)) or size <= 0:
                return False
            else:
                return True
        
        if check_valid(batch_size):
            self.batch_size = batch_size
            self.__infinite_dataset_iterator = None

        if check_valid(sequence_length):
            self.sequence_length = sequence_length

        if check_valid(block_size):
            logger.warning('The block size represents the seqeunce length and will be depracted')
            self.sequence_length = sequence_length
    
        if check_valid(block_size_bytes):
            self.block_size_bytes = block_size_bytes

        if check_valid(buffer_size):
            self.set_buffer(buffer_size= buffer_size)

    def set_buffer(self, buffer_size:int) -> None:
        """
        Set the buffer and ensure it is valid.

        Args:
            buffer_size (int, required):
                The size of the sample buffer.
        """
        if not hasattr(self, 'sample_buffer'):
            self.sample_buffer = []

        self.buffer_size = buffer_size 

        # If the buffer is smaller than the current buffer, trim it to match the new size.
        if len(self.sample_buffer) > self.buffer_size:
            self.sample_buffer = self.sample_buffer[:self.buffer_size]
            
    async def async_generate_sample(self)-> List[str]:
        '''
        Checks the sample buffer, and builds it if it is empty

        Returns:
            self.sample_buffer (List[str]): 
                The sample buffer.
        '''
        # See if there is free space, if so, add jobs to fill the free space with samples.
        buffer_free_space = self.buffer_size - len(self.sample_buffer) 
        
        if buffer_free_space > 0  :
            
            # Sample the file_metas randomly.
            sample_cat_params_list = random.sample(self.all_text_file_metas, buffer_free_space)

            # Build the asyncio jobs.
            self.sample_cat_tasks += [self.cat(cid=sample_cat_params['Hash'], offset=0, length=self.max_hash_size) for sample_cat_params in sample_cat_params_list]
            
            # This currently synchronytes on all of the self.sample_cat_tasks, completing when they all are finished.
            finished_tasks, running_tasks  = await asyncio.wait(self.sample_cat_tasks) 
            self.sample_cat_tasks = list(running_tasks)
            finished_tasks = list(finished_tasks)

            # Add the finished task results into the buffer.
            for finished_task in finished_tasks:
                sample = finished_task.result()
                self.sample_buffer += [finished_task.result()]

        # Randomly sample the text file from the buffer.
        random_idx = random.randint(0,len(self.sample_buffer)-1)
        raw_chunk = self.sample_buffer[random_idx]

        # Increment the counters.
        self.sample_count += 1
        self.batch_count += self.sample_count //  self.batch_size


        if self.block_size_bytes < len(raw_chunk):
            start_idx = random.randint(0, len(raw_chunk) - self.block_size_bytes)
        else:
            start_idx = 0
        
        end_idx = start_idx + self.block_size_bytes
        sample = raw_chunk[start_idx:end_idx]

        # If the batch count exceeds the calls per update, pop the first elemetn out of the buffer.
        if (self.batch_count) >= self.buffer_calls_per_update:
            self.sample_count = 0 
            self.sample_buffer.pop(0)
        
        return sample

    def __getitem__(self, idx: Optional[int] = None) -> Union[List[str], torch.tensor]:
        '''
        Sample from the sample_buffer via self.async_generate_sample. This fetches a random block of text
        with a size of self.block_size_bytes in bytes.
        Args:
            idx (int):
                Sample index of dataset.
            
        Returns:
            output (Union[str, torch.tensor])
        '''
        # Random sampel idx if None.
        if idx == None:
            idx = random.randint(0, self.__len__())

        # only sample if the buffer is less than the buffer_size

        raw_text_bytes = asyncio.run(self.async_generate_sample())

        # Decode the bytes into a string.
        try:
            raw_text = raw_text_bytes.decode()
        except UnicodeDecodeError as e:
            raw_text = str(raw_text_bytes[2:-1])

        # If there is no tokenizer specified return text with the seqeunce length being the number of " " split elements.

        if self.no_tokenizer:
            raw_text =raw_text.split()
            output = raw_text[:self.sequence_length]
            remainder = self.sequence_length - len(output)

            if remainder > 0:
                # left side padding
                output = [self.pad_token]*remainder + output 

            output = ' '.join(output)
        else:
            output = self.tokenizer(raw_text, max_length=self.sequence_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
            output = output.to(torch.long).squeeze(0) #  [1,seq_len] -> [seq_len]

        return output
    
    async def get_dataset_hashes(self)-> List[dict]:
        '''
        Get the hashes representing the root of each dataset
        
        Returns
            response (dict):
            
        '''
        mountain_meta = {'Name': 'mountain', 'Folder': 'meta_data', 'Hash': self.mountain_hash}
        response = await self.api_post( 'object/get',  params={'arg': mountain_meta['Hash']}, return_json= True)
        response = response.get('Links', None)
        return response

    async def get_folder_hashes(self, file_meta:dict) -> List[str]:
        '''
        Get the folder hashes from the dataset.

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
        Returns folder_hashes (List[str])
        
        '''

        links = (await self.get_links(file_meta))
        
        # Build the tasks to fetch the links of the folder.
        unfinished = [asyncio.create_task(self.api_post('object/get', params={'arg':link['Hash']}, return_json=True)) for link in links]
        folder_hashes = []
        just_links = []

        # Gather results until all tasks are finished.
        while len(unfinished)>0:
            finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            for res in await asyncio.gather(*finished):
                folder_hashes.extend(res.get('Links'))
        
        # Sometimes, the folder_hashes are empty with some datasets.
        # This means the root links are the folder links.
        # TODO (for constructing text corpus): We need the root links to be 1 level for more consistancy.
        if len(folder_hashes) == 0:
            folder_hashes = links

        return folder_hashes

    async def cat(self, cid:str, offset:int = 0, length:int = None)->bytes:
        '''
        Cat endpoint.
        Args:
            cid (str):
                CID of the object.
            offset (int):
                The offset in bytes.
            length  (int):
                The length in bytes.
            
        Returns:
            response (bytes):
                The response from the cat call.
                
        '''
        params = dict(arg=cid, offset=offset)
        params['length'] = length
        headers = {}
        response = await self.api_post('cat', params=params, headers=headers, chunk_size=10000000, num_chunks=1)
        return response

    async def get_folder_text_hashes(
                                    self, 
                                    file_meta:dict, 
                                    dataset:str, 
                                    max_chunks:int = 1, 
                                    chunk_size:int = 100000000) -> List[Dict[str, Union[str, int]]]:
        """
        Get text hashes from a folder

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
            dataset (str):
                The name of the dataset for self.dataset_hash_map.
            max_chunks (int): 
                Max number of chunks to call when fetching file metas.
        
        Returns 
            text_file_metas (List[Dict[str, Union[str, int]]):
                List of the text file_metas of the folder.
        """
        text_file_metas = []
        
        for chunk_i in range(max_chunks):
            data = await self.cat(file_meta['Hash'], offset=chunk_i*chunk_size ,length=chunk_size)
            hashes = ['['+h + '}]'for h in data.decode().split('},')]
            for i in range(len(hashes)-1):
                try:
                    decoded_hash = json.loads(hashes[i+1][1:-1])
                    decoded_hash_size_bytes = decoded_hash.get('Size', 0)
                    if decoded_hash_size_bytes > 0:
                        self.dataset_size_map[dataset] += decoded_hash_size_bytes
                        text_file_metas.append(decoded_hash)
                except json.JSONDecodeError:
                    pass
                
                hashes[i] ='{'+ hashes[i+1] + '}'

        return text_file_metas

    async def get_links(self, file_meta:dict) -> List[dict]:
        '''
        Get Links from file_meta

        Args
            file_meta (dict, required): 
                Dictionary containing hash and name of root link.
        '''
        response = await self.api_post( 'object/get',  params={'arg': file_meta['Hash']}, return_json= True)
        response_links = response.get('Links', [])
        return response_links

    async def api_post(
                    self, 
                    endpoint:str, 
                    params:Optional[Dict[str, Any]] = {}, 
                    headers:Optional[Dict[str, Any]] = {}, 
                    return_json:Optional[bool] = False,  
                    content_type:Optional[str] = None, 
                    chunk_size:Optional[int] = 1024, 
                    num_chunks:Optional[int] = None, 
                    sock_connect:Optional[int]=20, 
                    sock_read:Optional[int]=20) -> Union[Dict, 'aiohttp.Response', bytes]:
        '''
        Async api post to ipfs server.

        Args:
            endpoint (str):
                Endpoint path with such that path is "self.ipfs_url/{endpoint}".
            params (Dict[str, Any], optional):
                Params for api request.
            headers (Dict[str, Any], optional): 
                Headers for api request.
            return_json (bool, optional): 
                Return repsonse as json.
            content_type (str, optional):
                Content type of request.
            chunk_size (int, optional):
                Chunk size of streaming endpoint.
            num_chunks (int, optional):
                Number of chunks to stream.
            sock_connect (int, optional):
                The timeout for connecting to a socket.
            sock_read (int, optional):
                The timeout for reading a socket.
        Returns:
            return_result (Union[Dict, 'aiohttp.Response', bytes]):
                The result of the response. 
                    - Dictionary if return_json = True. 
                    - Bytes if num_chunks > 0
                    - aiohttp.Response if num_chunks == 0 and return_json == False
        '''
        url = os.path.join(self.ipfs_url, endpoint)
        return_result = None
        timeout = aiohttp.ClientTimeout(sock_connect=sock_connect, sock_read=sock_read)
        
        async with aiohttp.ClientSession( timeout=timeout) as session:
            async with session.post(url,params=params,headers=headers) as res:
                # Return a Json of the response.
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                # If num_chunks is not None, iterate through the chunks of chunk_size.
                if num_chunks:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    @property
    def available_datasets(self) -> List[str]:
        '''
        List of available datasets.

        Retuns:
            List of available datasets.
        '''
        return list(self.dataset2hash.keys())

    @property
    def dataset_hashes(self) -> List[str]:
        '''
        Return the dataset hashes:

        Returns
            self._dataset_hashes (List[str]):
                A list of the dataset hashes
        '''
        # This avoids us from having to call this multiple times from IPFS.
        if not hasattr(self, '_dataset_hashes'):
            self._dataset_hashes = asyncio.run(self.get_dataset_hashes())
        return self._dataset_hashes

    @property
    def dataset2hash(self) -> Dict:
        '''
        Dictionary to hash
        '''
        return {v['Name'].replace('.txt', '') :v for v in self.dataset_hashes}
    
    @property
    def dataset_size(self) -> int:
        '''
        The size of the dataset in bytes.
        '''
        return sum(list(self.dataset_size_map.values()))


    def dataloader(self, epoch_length:Optional[int] = 100) -> DataLoader:
        """ 
        Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): 
                
                The epoch length of the miner. If this length is not set or if it is larger than the dataset,
                then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
                is returned. 

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """
        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True)
    
    def __next__(self) -> Dict[str, torch.tensor]:
        """
        Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter(self.dataloader())

        try:
            return next(self.__infinite_dataset_iterator)
        except StopIteration:
            self.__infinite_dataset_iterator = iter(list(self.dataloader()))
            return next(self.__infinite_dataset_iterator)

    def set_event_loop(self, loop = None) -> 'asyncio.loop':
        '''
        Sets the event loop.

        Args:
            loop (asyncio.loop, optional):
                The asyncio loop you want to set to self.loop
        
        Returns:
            self.loop (asyncio.loop)
        '''
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop

    def __len__(self) -> int:
        """
        Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        return self.dataset_size // self.block_size_bytes

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        # Cancel sample tasks.
        if len(self.sample_cat_tasks)> 0:
            for t in self.sample_cat_tasks:
                t.cancel()

        
