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
from queue import Queue
from typing import *
import aiohttp
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader
import bittensor
logger = logger.opt(colors=True)

def chunk(sequence:list, 
        chunk_size:Optional[int]=None, 
        append_remainder:bool=False,
        distribute_remainder:bool=False,
        num_chunk: Optional[int]= None):

    if chunk_size is None:
        assert (type(num_chunks) == int)
        chunk_size = len(sequence) // num_chunks

    if chunk_size >= len(sequence):
        return [sequence]
    remainder_chunk_len = len(sequence) % chunk_size
    remainder_chunk = sequence[:remainder_chunk_len]
    sequence = sequence[remainder_chunk_len:]
    sequence_chunks = [sequence[j:j + chunk_size] for j in range(0, len(sequence), chunk_size)]

    if append_remainder:
        # append the remainder to the sequence
        sequence_chunks.append(remainder_chunk)
    else:
        if distribute_remainder:
            # distributes teh remainder round robin to each of the chunks
            for i, remainder_val in enumerate(remainder_chunk):
                chunk_idx = i % len(sequence_chunks)
                sequence_chunks[chunk_idx].append(remainder_val)

    return sequence_chunks



class ThreadManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_threads:int=None):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: 
                The maximum number of threads that can be used to
                execute the given calls.
        """
        self.max_threads = max_threads
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = []
        self._shutdown_lock = threading.Lock()
        self._shutdown = False


    def submit(self, fn, args:Optional[list]=[],kwargs:Optional[dict]={}) -> Any:
        '''
        Submit a function with args and kwargs on a seperate thread.

        Args
            fn (Callable):
                Function to place on the thread.
            args (list):
                Arguments to a function.
            kwargs (dict):
                Key word arguments to a function.
        '''
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            
            thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            self._threads.append(thread)

        return thread


    @property
    def threads(self):
        '''List threads.'''
        return self._threads

    def __del__(self):
        self.shutdown()

    def shutdown(self, wait=True):
        '''Shutdown threads'''
        if wait:
            for t in self._threads:
                # forces thread to stop
                t.raise_exception()
                t.join()

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
            num_workers: int = 1,
            datasets: Optional[Union[List[str], str]]=['ArXiv'], 
            loop:'asyncio.loop'=None, 
            no_tokenizer: bool = False,
            data_dir: str =  os.path.expanduser('~/./bittensor/data'),
            save_dataset : bool = False,
            load_dataset : bool = True,
            run_generator:bool=False,
            buffer_size:int=100,
            num_batches: int = 100,
            max_datasets: int = 2,
            num_folders: int = 10,
            max_blocks_per_dataset: int = 10,
            max_directories: int=10,
            cache_size: int = 10, 
            cache_calls_per_block: int=100):

        """
        Args:
            batch_size (int):
                The size of the batch.
            sequence_length (int):
                The length of the seqeunce (tokenwise).
            block_size (int): 
                The  size of the text data block which is used to create 
                multiple samples.
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
            run_generator (bool):
                Run the background loop for fecthing data blocks.
            buffer_size (int):
                Size of the queue for asynchronously putting data blocks.
            num_batches (int):
                Number of generated batches in epoch.
            max_datasets (int):
                Max number of datasets.
            cache_size (int):
                size of blocks to cache while training
            cache_calls_per_block (int):
                calls per block when caching

        """
        self.__infinite_dataset_iterator = None
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.datasets = datasets
        self.max_blocks_per_dataset = max_blocks_per_dataset
        self.num_folders = num_folders
        self.set_event_loop(loop=loop)
        # if datasets is None then refer to all of the availabe datasets 
        self.max_datasets = max_datasets
        assert len(self.datasets) > 0
        if self.datasets == 'default' or self.datasets == None:
            self.datasets = self.available_datasets
        self.datasets = self.datasets[:self.max_datasets]
        self.no_tokenizer = no_tokenizer


        self.tokenizer =  bittensor.tokenizer()
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_idx = self.tokenizer(self.pad_token)['input_ids'][0]
        self.data_dir =  data_dir
        self.save_dataset = save_dataset
        self.load_dataset = load_dataset
        self.run_generator= run_generator
        self.buffer_size = buffer_size
        self.num_batches = num_batches

        self.sample_count = 0
        self.batch_count = 0
       
       
       # we need to build the dataset or load existing text file hashes
        # notice the heirarchy of ipfs hashes is DATASET -> FOLDER -> TEXT HASH, 
        # we want to flatten each dataset FOLDER -> TEXT HASH into FOLDER*TEXT
        self.build_datasets(datasets=self.datasets, load=self.load_dataset, save=self.save_dataset)
        self.set_cache(cache_size=cache_size)
        # this runs the a thread that has its own asyncio loop. 
        # The loop is passed into nested async functions to use loo.run_until_complete function
        
        sample_cat_params_chunks = chunk(self.sample_cat_params_list, self.cache_size)
        if self.run_generator:
            self.data_queue = Queue(self.cache_size)

            # the thread manager is used for running a background thread
            self.thread_manager = ThreadManager()
            # start the genrator
            self.thread_manager.submit(fn=self.sample_generator, kwargs=dict(queue=self.data_queue, loop=asyncio.new_event_loop()))

    def sample_generator(self, 
                         queue:Queue, 
                         loop:'asyncio.loop'=None, 
                         return_json:bool=False) -> None:

        """ Sample generator on seperate thread with its own asyncio loop for generating
            background samples while the user fetches them in the foreground.
        Args:
            queue (Queue):
                Queue for feeding the samples through for __getitem__ to fetch.
            batch_size (int):
                Batch size of the samples.
            sequence_length (int):
                Sequence Length of the samples.
            loop:'asyncio.loop'=None, 
                        return_json:bool=False
        """
        
        # this is for starting a new thread
        # the loop needs to be set within the new thread
        if loop != None:
            asyncio.set_event_loop(loop)

        # run through each chunk, then tokenize it,
        batch_count = 0

        sample_cat_params_chunks = chunk(self.sample_cat_params_list, self.cache_size)
        
        sample_count = 0
        batch_count = 0
        
        tasks = []
        for sample_cat_params in self.sample_cat_params_list:

            finished_tasks = []
            while len(tasks) >= self.cache_size:
                tmp_finished_tasks, tasks = asyncio.run(asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED))
                finished_tasks = finished_tasks + list(tmp_finished_tasks)
                tasks = list(tasks)

                if self.run_generator == False:
                    break
            
            
            tasks += [self.cat(**sample_cat_params)]

            for finished_task in finished_tasks:
                sample = asyncio.run(finished_task)
                queue.put(sample)
        

    def stop_generator(self):
        self.run_generator = False

    def build_datasets(self, datasets:List[str]=None, save:bool=False, load:bool=False, loop:'asyncio.loop'=None) -> None :
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
            tasks += [self.async_build_single_dataset(dataset=dataset, save=save, load=load, loop=loop)]

        dataset_hashes = self.async_run(asyncio.gather(*tasks), loop=loop)

        # Create a hash map of dataset -> text hashes.
        for k,v in zip(datasets, dataset_hashes):
            if len(v) > 0:
                dataset_hash_map[k] = v
                
        self.dataset_hash_map = dataset_hash_map
        for  k,file_meta_list in dataset_hash_map.items():
            all_text_file_metas += v
        self.all_text_file_metas = all_text_file_metas
        self.sample_cat_params_list = self.build_sample_cat_params(all_text_file_metas=self.all_text_file_metas, block_size=self.block_size)

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
        output = self.async_run(self.async_save_json(*args,**kwargs),loop=loop)
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
        output =  self.async_run(job=self.async_load_json(*args,**kwargs), loop=loop)
        return output

    async def async_build_single_dataset(self, dataset:str = None, num_folders = None, save:bool=False, load:bool=True, loop: 'asyncio.loop' =None) -> List[dict] :
        """ Building a single dataset by fetching its text file metas ({Hash:str, Name:str, Size: int})
        Args:
            dataset (List[str]):
                The name of the dataset
            num_folders (bool):
                The numbe of folders in the dataset.
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
        num_folders = num_folders if num_folders else self.num_folders
            
        folder_hashes = (await self.get_folder_hashes(self.dataset2hash[dataset]))[:num_folders]

        # In some cases, the dataset is the folder hash. This will throw an error otherwise
        if len(folder_hashes) == 0:
            folder_hashes = [self.dataset2hash[dataset]]
        random.shuffle(folder_hashes)

        hash2file_meta = {}
        if load:
            loaded_file_metas =  self.load_json(path=f'{dataset}/file_metas', default=[], loop=loop)
            for file_meta in loaded_file_metas:
                hash2file_meta[file_meta['Hash']] = file_meta

        tasks = []
        for f in folder_hashes:
            self.total = 0
            tasks.append(self.get_text_file_metas(f, dataset=dataset))
        
        loaded_file_metas = []
        for folder_text_file_metas in await asyncio.gather(*tasks):
            for file_meta in folder_text_file_metas:
                if 'Size' in file_meta and 'Hash' in file_meta:
                    loaded_file_metas.append(file_meta)

            for file_meta in loaded_file_metas:
                # sometimes the file_meta is empty.
                hash2file_meta[file_meta['Hash']] = file_meta   

        text_file_metas = list(hash2file_meta.values())


        if save:
            self.save_json(path=f'{dataset}/file_metas', obj=text_file_metas, loop=loop)
        
        return text_file_metas

    def set_data_size(self, batch_size, block_size) -> None:
        r""" Update the size of data (batch_size, block_size) that we need.

        Args: 
            batch_size(int, required):
                The batch_size of data that should be produced by dataloader.

            block_size(int, required):
                The block_size of data that should be produced by dataloader. 
        """

        assert self.run_generator == False, 'Please ensure that run_generator is false'

        def check_valid(size):
            r""" Check if the size is a valid positive intiget, if not, return False.
            """
            if size <= 0 or (not isinstance(size, int)):
                return False
            else:
                return True
        
        if check_valid(batch_size):
            self.batch_size = batch_size
        
        if check_valid(block_size):
            self.block_size = block_size
            # rebuild the cat params list
            self.sample_cat_params_list = self.build_sample_cat_params(all_text_file_metas=self.all_text_file_metas, block_size=self.block_size)

        # empty the queue
        if self.run_generator:
            while not self.data_queue.empty():
                self.data_queue.get()
        else:
            # empty sample cache
            self.set_cache(cache_size=self.cache_size)
            self.sample_cache = []
        # empty the dataset_iterator with the old sizing
        self.__infinite_dataset_iterator = None

    def set_cache(self, cache_size=10) -> list:
        '''
        Set the cache and ensure it is valid.
        '''
        self.sample_cache = [] # Cache list for raw text.
        self.cache_size = min(cache_size, len(self) )  # The maximum size of the cache (FIFO).
        self.cache_size = (self.cache_size // self.batch_size) * self.batch_size
        return self.sample_cache

    def resolve_sample_cache(self)-> List[str]:
        '''
        Checks the sample cache, and builds it if it is empty

        Returns:
            self.sample_cache (List[str]): 
                The sample cache.
        '''
        if len(self.sample_cache) == 0:
            sample_cat_params_list = random.sample(self.sample_cat_params_list, self.cache_size)
            cache_sample_tasks = [self.cat(**sample_cat_params) for sample_cat_params in sample_cat_params_list]
            self.sample_cache = [sample_bytes for sample_bytes in self.async_run(asyncio.gather(*cache_sample_tasks))]


        return self.sample_cache

    def __getitem__(self, idx: Optional[int]= None, filler_token:str='FILLER_TEXT') -> Union[List[str], torch.tensor]:
        '''
        Sample from queue or lazy loading. 
        This involves sampling large text files that are then cached, generating
        multiple samples per text file. When a threshold number of samples are generated
        from the blocks, a FIFO priority is used to churn old blocks for new blocks, to avoid
        data staleness.

        Args:
            idx (int):
                Sample index of dataset
            filler_token (str):
                Filler token to pad raw text 
            
        Returns:
            output (Union[str, torch.tensor])


        '''
        # Random sampel idx if None.
        if idx == None:
            idx = random.randint(0, self.__len__())

        # only sample if the cache is less than the cache_size
        self.sample_count += 1
        self.batch_count =  self.sample_count // self.batch_size
        if self.run_generator :
            raw_text_bytes = self.data_queue.get()
        else:
            self.resolve_sample_cache()
            raw_text_bytes = self.sample_cache.pop(idx % len(self.sample_cache))
        
        try:
            raw_text = raw_text_bytes.decode()
        except UnicodeDecodeError as e:
            raw_text = str(raw_text_bytes[2:-1])
        if self.no_tokenizer:
            raw_text =raw_text.split()
            output = raw_text[:self.sequence_length]
            remainder = self.sequence_length - len(output)
            if remainder > 0:
                output = output + [self.pad_token]*remainder
            output = ' '.join(output)
        else:
            
            tokenized_text =  self.tokenizer(raw_text, padding=True)['input_ids']
            output = torch.tensor(tokenized_text)[:self.sequence_length]           
            # Append the remainder with 0 (TODO use stop token)
            seqeunce_length_remainder =  self.sequence_length - output.shape[0]
            if seqeunce_length_remainder>0:
                output = torch.nn.functional.pad(input=output, pad=(0,seqeunce_length_remainder), mode='constant', value=self.pad_token_idx  ) 

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
                                file_meta:dict,
                                num_folders:int = 5) -> List[str]:
        '''
        Get the folder hashes from the dataset.

        Args:
            file_meta (dict):
                File meta contianing the hash and name of the link.
            num_folders (int):
                The number of folders to load at once
        Returns folder_hashes (List[str])
        
        '''
        links = (await self.get_links(file_meta))[:100]

        unfinished = [self.loop.create_task(self.api_post('object/get', params={'arg':link['Hash']}, return_json=True)) for link in links]
        folder_hashes = []
        while len(unfinished)>0:
            finished, unfinished = await asyncio.wait(unfinished, return_when=asyncio.FIRST_COMPLETED)
            for res in await asyncio.gather(*finished):
                folder_hashes.extend(res.get('Links'))
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
    async def get_text_file_metas(self, file_meta:dict, dataset:str, max_chunks:int=100) -> List[str]:
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
        chunk_size = 2048

        for chunk_i in range(max_chunks):

            data = await self.cat(file_meta['Hash'], offset=chunk_i*chunk_size ,length=chunk_size)

            hashes = ['['+h + '}]'for h in data.decode().split('},')]
            for i in range(len(hashes)-1):
                if self.dataset_size_map[dataset] >= self.max_bytes_per_dataset:
                    return decoded_hashes
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
         
    def async_run(self, job:'asyncio.coroutine', loop: 'asyncio.loop'=None) -> Any: 
        '''
        Set the event loop.

        Args:
            job (asyncio.Task)
            loop (asyncio.loop):
                Event loop.

        '''
        
        if loop == None:
            loop = self.loop
        return loop.run_until_complete(job)

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
            self._dataset_hashes = self.async_run(self.get_dataset_hashes())
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
        return len(self.sample_cat_params_list)


    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        '''
        Close queue and thread manager.
        '''
        if self.run_generator:
            self.stop_generator()
        if hasattr(self, 'data_queue'):
            del self.data_queue
        if hasattr(self, 'thread_manager'):
            del self.thread_manager
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
    @property
    def max_bytes_per_dataset(self):
        return self.max_blocks_per_dataset*self.block_size 