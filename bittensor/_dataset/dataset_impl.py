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
import asyncio
import json
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
import os
import random
import threading
from copy import deepcopy
from queue import Queue
from typing import *
import aiohttp
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

import bittensor
from .utils import CustomThread, ThreadManager
from munch import Munch
logger = logger.opt(colors=True)
import streamlit as st



class GenesisTextDataset:
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    ipfs_url = bittensor.__ipfs_url__
    dataset_dir = bittensor.__dataset_dir__
    text_dir = bittensor.__text_dir__
    mountain_hash = bittensor.__mountain_hash__

    def __init__(
            self, 
            batch_size: int=8, 
            sequence_length: int=256,
            block_size: int = 10000,
            max_hash_size= 1000000,
            num_workers: int = 1,
            datasets: Optional[Union[List[str], str]]=['ArXiv'], 
            loop:'asyncio.loop'=None, 
            no_tokenizer: bool = False,
            data_dir: str =  os.path.expanduser('~/./bittensor/data'),
            save_dataset : bool = False,
            load_dataset : bool = True,
            buffer_size:int=100,
            num_batches: int = 100,
            max_datasets: int = 2,
            max_directories: int=10,
            buffer_calls_per_update: int=100):
        """
        Args:
            batch_size (int):
                The size of the batch.
            sequence_length (int):
                The length of the seqeunce (tokenwise).
            block_size (int): 
                The  size of the text data block which is used to create 
                multiple samples.
            max_hash_size (int): 
                The maximum size of the block that represents the hash
            num_workers (int):
                Number of workers for pytorch Dataset.
            datasets (Optional[Union[List[str], str]]):
                List of dataset names to include from the pile.
            loop ('asyncio.loop'):
                Asyncio loop for class, defaults to default event loop.
            no_tokenizer (bool):
                Do not inlcude tokenizer.
            data_dir (str):
                Directory for saving assets.
            save_dataset (bool):
                Save the dataset text hashes.
            load_dataset (bool):
                Load the dataset text hashes.
            buffer_size (int):
                Size of the queue for asynchronously putting data blocks.
            num_batches (int):
                Number of generated batches in epoch.
            max_datasets (int):
                Max number of datasets.
            buffer_size (int):
                size of blocks to buffer while training
            buffer_calls_per_update (int):
                calls per block when caching
        """
        self.__infinite_dataset_iterator = None
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.datasets = datasets
        self.max_directories = max_directories
        self.max_hash_size = max_hash_size
        self.set_event_loop(loop=loop)
        self.max_datasets = max_datasets
    
        assert len(self.datasets) > 0
        if self.datasets == 'default' or self.datasets == None:
            self.datasets = self.available_datasets
        self.datasets = self.datasets[:self.max_datasets]
        self.no_tokenizer = no_tokenizer
        self.buffer_calls_per_update = buffer_calls_per_update
        self.sample_cat_tasks = []
        self.sample_buffer = []
        self.tokenizer =  bittensor.tokenizer()
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_idx = self.tokenizer(self.pad_token)['input_ids'][0]
        self.data_dir =  data_dir
        self.save_dataset = save_dataset
        self.load_dataset = load_dataset

        self.set_buffer(buffer_size=buffer_size)
        self.num_batches = num_batches
        self.sample_count = 0
        self.batch_count = 0
        self.build_datasets(datasets=self.datasets, load=self.load_dataset, save=self.save_dataset)


    def build_datasets(self, datasets:List[str]=None, save:bool=False, load:bool=False, loop:'asyncio.loop'=None, out_queue:Queue=None) -> None :
        """ Building all of the datasets specified by getting each of their 
            text hashes from IPFS or local
        Args:
            datasets (List[str]):
                Axon to serve.s
            save (bool):
                Save the dataset hashes locally.
            load (bool):
                Load the dataset hashes locally
            loop (asyncio.Loop):
                Asyncio loop 
        """
        if datasets == None:
            datasets = self.datasets

        all_text_file_metas = []
        dataset_hash_map = {}


        tasks = []

        self.dataset_size_map = {d:0 for d in datasets}
        
        # Gather dataset hashes async as their state is independent.
        for dataset in datasets:
            tasks += [self.async_build_single_dataset(dataset=dataset, save=save, load=load, loop=loop, out_queue=out_queue)]

        dataset_hashes = asyncio.run(asyncio.gather(*tasks))

        # Create a hash map of dataset -> text hashes.
        for k,v in zip(datasets, dataset_hashes):
            if len(v) > 0:
                dataset_hash_map[k] = v
                
        self.dataset_hash_map = dataset_hash_map
        for  k,file_meta_list in dataset_hash_map.items():
            all_text_file_metas += v
        self.all_text_file_metas = all_text_file_metas

    construct_text_corpus = build_datasets

    @staticmethod
    def build_sample_cat_params(all_text_file_metas:List[dict], block_size:int)-> List[str]:
        '''
        Builds sample cat params from all_text_file_metas using block_size.

        Args:
            all_text_file_metas (List[dict]):
                List of file metas containing {Hash:str, Size:int in bytes}.
            block_size (int):
                Size of the block.
            
        Returns 
            sample_cat_params_list (List[dict]):
                List of params for cat representing a block of text from an api call to ipfs cat.
        '''
        assert len(all_text_file_metas)>0
        sample_cat_params_list = [] 

        for file_meta in all_text_file_metas:
            for offset in range(0, file_meta['Size'], block_size):
                if file_meta['Size'] - offset >= block_size:
                    params = { 'cid': file_meta['Hash'],'offset': offset, 'length': block_size}
                    sample_cat_params_list.append(params)
        
        return sample_cat_params_list

    async def async_save_json(self, 
                              path:str,
                              obj:Union[dict, list],
                              include_root:bool=True) -> str:
        """ 
        Async save of json for storing text hashes

        Args:
            path (List[str]):
                Axon to serve.
            obj (bool):
                The object to save locally
            include_root (bool):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
            
        Returns: 
            path (str)
                Path of the saved JSON
        """
        
        if include_root:
            path = os.path.join(self.data_dir, path)

        dir_path = os.path.dirname(path)

        # ensure the json is the prefix
        if path[-len('.json'):] != '.json':
            path += '.json'

        # ensure the directory exists, make otherwise
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        assert os.access( dir_path , os.W_OK ), f'dir_path:{dir_path} is not writable'
        with open(path, 'w') as outfile:
            json.dump(obj, outfile)

        return path

    def save_json(self,loop:'asyncio.loop'=None, *args,**kwargs) -> str:
        '''
        Sync verson of async_save_json
        Args
            loop (asyncio.loop):
                The asyncio loop to be past, otherwise self.loop

        Returns 
            output (dict) 

        '''
        output = asyncio.run(self.async_save_json(*args,**kwargs))
        return output

    async def async_load_json(self, path:str,include_root:bool=True, default:Union[list, dict]={}) -> Union[list, dict]:

        """ 
        Async save of json for storing text hashes
        Args:
            path (str):
                Path of the loaded json
            include_root (bool):
                Include self.data_dir as the prefix.
                    - if True, ths meants shortens the batch and 
                    specializes it to be with respect to the dataset's 
                    root path which is in ./bittensor/dataset
        Returns: 
            obj (str)
                Object of the saved JSON.
        """
        
        if include_root:
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

    def load_json(self, loop:'asyncio.loop'=None, *args,**kwargs) -> Union[list, dict]:
        '''
        Sync verson of async_save_json

        Args
            loop (asyncio.loop):
                The asyncio loop to be past, otherwise self.loop
        Returns 
            output (dict, list) 
                The output of the loaded JSON.
        '''
        output =  asyncio.run(self.async_load_json(*args,**kwargs))
        return output

    async def async_build_single_dataset(self, dataset:str = None, save:bool=False, load:bool=True, loop: 'asyncio.loop' =None, out_queue:Queue=None) -> List[dict] :
        """ Building a single dataset by fetching its text file metas ({Hash:str, Name:str, Size: int})
        Args:
            dataset (List[str]):
                The name of the dataset
            num_samples (int):
                The number of samples the user want so get from the dataset
            load (bool):
                Load the dataset hashes locally
            save (bool):
                Save the dataset hahses locally.
            loop (asyncio.Loop):
                Asyncio loop 

        Returns: 
            text_file_metas (List[Dict): 
                List of text file metas with the format of {Hash:String, Size:String, Name:String}
        """


        hash2file_meta = {}
        if load:
            loaded_file_metas =  self.load_json(path=f'{dataset}/file_metas', default=[], loop=loop)
            for file_meta in loaded_file_metas:
                hash2file_meta[file_meta['Hash']] = file_meta
            
            text_file_metas = list(hash2file_meta.values())
                        

        else:
                
            folder_hashes = (await self.get_folder_hashes(self.dataset2hash[dataset]))[:self.max_directories]

            tasks = []
            for f in folder_hashes:
                tasks.append(self.get_folder_text_hashes(f, dataset=dataset))

            for folder_text_file_metas in await asyncio.gather(*tasks):
                for file_meta in folder_text_file_metas:
                    if 'Size' in file_meta and 'Hash' in file_meta:
                        if out_queue != None:
                            out_queue.put(file_meta)
                        hash2file_meta[file_meta['Hash']] = file_meta   
            

                    
            text_file_metas = list(hash2file_meta.values())

  

            if save:
                self.save_json(path=f'{dataset}/file_metas', obj=text_file_metas, loop=loop)
        self.dataset_size_map[dataset]  = sum([fm['Size'] for fm in text_file_metas])

        return text_file_metas


    def set_data_size(self, batch_size:int, block_size:int, buffer_size:int) -> None:
        r""" Update the size of data (batch_size, block_size) that we need.

        Args: 
            batch_size(int, required):
                The batch_size of data that should be produced by dataloader.

            block_size(int, required):
                The block_size of data that should be produced by dataloader. 

            buffer_size(int, required):
                The size of the buffer. 
        """


        def check_valid(size):
            r""" Check if the size is a valid positive intiget, if not, return False.
            """
            if size <= 0 or (not isinstance(size, int)):
                return False
            else:
                return True
        
        if check_valid(batch_size):
            self.batch_size = batch_size
            self.__infinite_dataset_iterator = None
        
        if check_valid(block_size):
            self.block_size = block_size

        if check_valid(buffer_size):
            self.set_buffer(buffer_size= buffer_size)


    def set_buffer(self, buffer_size=10) -> list:
        '''
        Set the buffer and ensure it is valid.
        '''
        if not hasattr(self, 'sample_buffer'):
            self.sample_buffer = []

        self.buffer_size = buffer_size 
        if len(self.sample_buffer) > self.buffer_size:
            self.sample_buffer = self.sample_buffer[:self.buffer_size]




    async def async_generate_sample(self, local_buffer_fraction=0.2, calls_per_update=100, sample_cat_tasks:list=None, sample_buffer = None, out_queue=None, in_queue=None)-> List[str]:
        '''
        Checks the sample buffer, and builds it if it is empty

        Returns:
            self.sample_buffer (List[str]): 
                The sample buffer.
        '''

        buffer_free_space = self.buffer_size - len(self.sample_buffer) 
        if buffer_free_space > 0  :
            sample_cat_params_list = random.sample(self.all_text_file_metas, buffer_free_space)
            self.sample_cat_tasks += [self.cat(cid=sample_cat_params['Hash'], offset=0, length=self.max_hash_size) for sample_cat_params in sample_cat_params_list]
            finished_tasks, running_tasks  = await asyncio.wait(self.sample_cat_tasks) 
            self.sample_cat_tasks = list(running_tasks)
            finished_tasks = list(finished_tasks)

            for finished_task in finished_tasks:
                sample = finished_task.result()
                self.sample_buffer += [finished_task.result()]
            

        random_idx = random.randint(0,len(self.sample_buffer)-1)
        raw_chunk = self.sample_buffer[random_idx]
        self.sample_count += 1

        if self.block_size < len(raw_chunk):
            start_idx = random.randint(0, len(raw_chunk) - self.block_size)
        else:
            start_idx = 0
        end_idx = start_idx + self.block_size
        sample = raw_chunk[start_idx:end_idx]

        if out_queue != None:
            out_queue.put(sample)

        if (self.sample_count //  self.batch_size) >= self.buffer_calls_per_update:
            self.sample_count = 0 
            self.sample_buffer.pop(0)

        return sample

    def __getitem__(self, idx: Optional[int]= None) -> Union[List[str], torch.tensor]:
        '''
        Sample from queue or lazy loading. 
        This involves sampling large text files that are then bufferd, generating
        multiple samples per text file. When a threshold number of samples are generated
        from the blocks, a FIFO priority is used to churn old blocks for new blocks, to avoid
        data staleness.

        Args:
            idx (int):
                Sample index of dataset
            
        Returns:
            output (Union[str, torch.tensor])


        '''
        # Random sampel idx if None.
        if idx == None:
            idx = random.randint(0, self.__len__())

        # only sample if the buffer is less than the buffer_size

        raw_text_bytes = asyncio.run(self.async_generate_sample())
        


        try:
            raw_text = raw_text_bytes.decode()
        except UnicodeDecodeError as e:
            raw_text = str(raw_text_bytes[2:-1])
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

    async def get_folder_hashes(self, 
                                file_meta:dict) -> List[str]:
        '''
        Get the folder hashes from the dataset.

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
        Returns folder_hashes (List[str])
        
        '''

        links = (await self.get_links(file_meta))
        unfinished = [asyncio.create_task(self.api_post('object/get', params={'arg':link['Hash']}, return_json=True)) for link in links]
        folder_hashes = []
        just_links = []
        while len(unfinished)>0:
            finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            for res in await asyncio.gather(*finished):
                folder_hashes.extend(res.get('Links'))
            
        if len(folder_hashes) == 0:
            folder_hashes = links

        return folder_hashes

    async def cat(self, cid:str, offset:int=0, length:int=None)->bytes:
        '''
        Cat endpoint.
        Args:
            cid (str):
                cid of the object.
            offset (int):
                The offset in bytes.
            length  (int):
                The length in bytes.
            
        Returns:
            response (bytes):
                
        '''
        params = dict(arg=cid, offset=offset)
        params['length'] = self.block_size
        headers = {}
        response = await self.api_post('cat', params=params, headers=headers, chunk_size=length, num_chunks=1)
        return response
    async def get_folder_text_hashes(self, file_meta:dict, dataset:str, max_chunks:int=1, chunk_size:int = 100000000) -> List[str]:
        """
        Get text hashes from a folder

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
            dataset (str):
                The name of the dataset for self.dataset_hash_map
            max_chunks (int): 
                Max number of chunks to call when fetching file metas.
        
        Returns List[str]

        """
  
        decoded_hashes = []
        # we need to  set the 
        

        for chunk_i in range(max_chunks):

            data = await self.cat(file_meta['Hash'], offset=chunk_i*chunk_size ,length=chunk_size)

            hashes = ['['+h + '}]'for h in data.decode().split('},')]
            for i in range(len(hashes)-1):
                try:
                    decoded_hash = json.loads(hashes[i+1][1:-1])
                    decoded_hash_size_bytes = decoded_hash.get('Size', 0)
                    if decoded_hash_size_bytes > 0:
                        self.dataset_size_map[dataset] += decoded_hash_size_bytes
                        decoded_hashes.append(decoded_hash)
                except json.JSONDecodeError:
                    pass
                
                

                hashes[i] ='{'+ hashes[i+1] + '}'

        
        return decoded_hashes

    total = 0 
    async def get_text(self, file_meta, offset:int = 0, length:int= 1024, loop=None, queue=None) -> str:
        
        """
        Get text hashes from a folder

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
            num_hashes:
                The maximum number of hashes before stopping.
        
        Returns List[str]

        """
        
        if loop == None:
            loop = self.loop
        
        if isinstance(file_meta, str): 
            file_meta  = {'Hash': file_meta}

        assert isinstance(file_meta, dict )
        
        headers = {}
        # we need to  set the 
        content_type = None
        url = f'{self.ipfs_url}/cat'
        data = await self.cat(cid=file_meta['Hash'],  offset= offset, length=length)
        if isinstance(queue, Queue): 
            queue.put(str(data))
        else:
            return str(data)

    async def get_links(self, file_meta:dict, **kwargs) -> List[dict]:
        '''
        Get Links from file_meta

        Args
            file_meta (dict): 
                Dictionary containing hash and name of root link

        Returns (List[dict])

        '''
        response = await self.api_post( 'object/get',  params={'arg': file_meta['Hash']}, return_json= True)
        response_links = response.get('Links', [])
        return response_links

    async def api_post(self, 
                      endpoint:str, 
                      return_json:bool = False, 
                      content_type:str=None, 
                      chunk_size:int=1024, 
                      num_chunks:int=None, 
                      **kwargs) -> 'aiohttp.Response':
        
        '''
        async api post

        Args:
            url (str):
                url of endpoint.
            return_json (bool): 
                Return repsonse as json.
            content_type (str):
                Content type of request.
            chunk_size (int):
                Chunk size of streaming endpoint.
            num_chunks (int):
                Number of chunks to stream.
        Returns (aiohttp.Response)
        '''
        url = os.path.join(self.ipfs_url, endpoint)
        headers = kwargs.pop('headers', {}) 
        params = kwargs.pop('params', kwargs)
        return_result = None

        # we need to  set the 
        timeout = aiohttp.ClientTimeout(sock_connect=10, sock_read=10)
        async with aiohttp.ClientSession( timeout=timeout) as session:
            async with session.post(url,params=params,headers=headers) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                # if num_chunks != None
                if num_chunks:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    async def api_get(self, 
                      endpoint:str,
                      headers:str={},
                      params:str={},
                     return_json:bool = True,
                     content_type:str=None, 
                     chunk_size:int=1024, 
                     num_chunks:int=None,
                     **kwargs) -> 'aiohttp.Response':
        '''
        async api post

        Args:
            url (str):
                url of endpoint.
            return_json (bool): 
                Return repsonse as json.
            content_type (str):
                Content type of request.
            chunk_size (int):
                Chunk size of streaming endpoint.
            num_chunks (int):
                Number of chunks to stream.
        Returns (aiohttp.Response)
        '''
        url = os.path.join(self.ipfs_url, endpoint)
        return_result = None
        async with aiohttp.ClientSession(loop=self.loop) as session:
            async with session.get(url,params=params,headers=headers) as res:
                if return_json: 
                    return_result = await res.json(content_type=content_type)
                else:
                    return_result = res

                if chunk_size:
                    return_result = b''
                    async for data in res.content.iter_chunked(chunk_size):
                        return_result += data
                        num_chunks-= 1
                        if num_chunks == 0:
                            break
        return return_result

    ##############
    #   ASYNCIO
    ##############
    @staticmethod
    def reset_event_loop(set_loop:bool=True) -> 'asyncio.loop':
        '''
        Reset the event loop

        Args:
            set_loop (bool):
                Set event loop if true.

        Returns (asyncio.loop)
        '''
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop

    def set_event_loop(self, loop:'asyncio.loop'=None)-> 'asynco.loop':
        '''
        Set the event loop.

        Args:
            loop (asyncio.loop):
                Event loop.

        Returns (asyncio.loop)
        '''
        
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop
        
    @property
    def dataset2size(self) -> Dict:
        '''
        dataset to the number of hashes in the dataset
        '''
        return {k:v['Size'] for k,v in self.dataset2hash.items()}
    @property
    def available_datasets(self) -> List[str]:
        '''
        list of available datasets
        '''

        return list(self.dataset2hash.keys())
    @property
    def dataset2hash(self) -> Dict:
        '''
        Dictionary to hash
        '''
        return {v['Name'].replace('.txt', '') :v for v in self.dataset_hashes}
    
    @property
    def dataset_size(self):
        return sum(list(self.dataset_size_map.values()))

    @property
    def dataset_hashes(self) -> List[str]:
        '''
        Return the dataset hashes
        '''
        if not hasattr(self, '_dataset_hashes'):
            self._dataset_hashes = asyncio.run(self.get_dataset_hashes())
        return self._dataset_hashes

    def dataloader(self, epoch_length = 100) -> DataLoader:
        """ Creates a torch dataloader out of a subclass of this class.

        Args:
            epoch_length (int, optional): The epoch length of the miner. If this length is not set or if it is larger than the dataset,
            then a dataloader for the entire dataset is returned. Otherwise, a dataloader for a subset of the dataset of epoch_length
            is returned. Defaults to None.

        Returns:
            torch.utils.data.dataloader.DataLoader: Pytorch dataloader.
        """

        return DataLoader(self,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=True)
    
    def __next__(self) -> Dict[str, torch.tensor]:
        """Returns the next element from the dataset. 
        """
        if self.__infinite_dataset_iterator == None:
            self.__infinite_dataset_iterator = iter(self.dataloader())

        try:
            return next(self.__infinite_dataset_iterator)
        except StopIteration:
            self.__infinite_dataset_iterator = iter(list(self.dataloader()))
            return next(self.__infinite_dataset_iterator)

    def __len__(self) -> int:
        """
        Returns number of samples (blocks) of dataset

        Returns:
            length: int
        """
        return self.dataset_size // self.block_size


    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        del self.tokenizer

