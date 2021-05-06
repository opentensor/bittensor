import bittensor
import torch
import random
import requests

from loguru import logger
from bittensor.dataloaders.dataloader import BittensorDataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

from loguru import logger
logger = logger.opt(colors=True)

class GenesisTextDataloader(BittensorDataLoader):
    
    def __init__(
            self,
            batch_size, 
            block_size,
            config = None
        ):
        super(GenesisTextDataloader, self).__init__()
        
        assert batch_size > 0, 'Batch size must be larger than 0'
        assert block_size > 0, 'Block size must be larger than 0'
        
        if config == None:
            config = BittensorDataLoader.default_config()

        self.config = config
        self.block_size = block_size
        self.tokenizer = bittensor.__tokenizer__()
        self.batch_size = batch_size
        
        # Retrieve a random slice of the genesis dataset
        self.data = self.construct_text_corpus()

        # Used to refresh corpus if we've exhausted the whole dataset
        self.refresh_corpus = False
        
    
    def retrieve_text_file(self, file_hash: str):
        """Connects to Infura IPFS gateway and retrieves the contents of 
        a genesis text file.

        Returns:
            str: The contents of the file.
        """
        session = requests.Session()
        params = (('arg', file_hash),)
        session.params.update(params)
        directory = None

        response = BittensorDataLoader.requests_retry_session(session=session).post(self.file_cat)

        if response.status_code == 200:
            directory = response
        
        return directory       

    def construct_text_corpus(self):
        """Connects to Infura IPFS gateway and retrieves the directory of genesis datasets.
        
        Returns:
            string: Contents of the text file. 
        """
        try:
<<<<<<< HEAD
            logger.info("Retrieving a dataset files from the IPFS gateway...")
=======
            logger.info("Retrieving a dataset file from the IPFS gateway...")
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
            directory = self.retrieve_directory(self.genesis_text_dataset_hash)
            data_corpus = []

            # Pick a random dataset file and return its contents
            if directory and 'links' in directory.keys():
                # Let's construct a dataset!
                random_dataset_file = random.choice(directory['links'])
                filename = random_dataset_file['Name']
                total_dataset_size = int(random_dataset_file['Size'])

                # Make sure the file we chose satisfies our maximum file size requirement
                while total_dataset_size <= self.config.dataloader.max_corpus_size:

                    # Find file hash
                    random_dataset_file_hash = random_dataset_file['Cid']['/']

                    # Retrieve file contents
                    file_contents = self.retrieve_text_file(random_dataset_file_hash)
<<<<<<< HEAD
                    logger.info("<green>Added:</green> <cyan>{}</cyan>".format(filename))
=======
                    logger.info("Adding {} to the training corpus...".format(filename))
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
                    data_corpus.extend(file_contents.text.split())

                    # Retrieve next file descriptor
                    random_dataset_file = random.choice(directory['links'])
                    filename = random_dataset_file['Name']
                    total_dataset_size += int(random_dataset_file['Size'])
                
<<<<<<< HEAD
                logger.info('')
=======
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
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

        # If we've exhausted the dataset, retrieve another corpus.
<<<<<<< HEAD
        if self.refresh_corpus:
=======
        if self.refresh_corpus or len(self.data) < self.block_size:
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
            self.data = self.construct_text_corpus()
            self.refresh_corpus = False

        # If epoch_length is set then we just need a slice of 
        # the dataset we downloaded of length epoch_length. 
<<<<<<< HEAD
        if epoch_length and epoch_length < len(self):
            
            # Set up upper bound of indices to fit the batch size we want. 
            idx_bound = epoch_length * self.batch_size

            # Collect enough random indices to batch together using batch_size into epoch_length batches
            random_start = random.randint(0, len(self) - idx_bound)
            indices = list(range(random_start, random_start + idx_bound))
            
            subset = Subset(self, indices)
            
            # Clear out these indices from our current corpus
            try:
                del self.data[random_start: random_start + idx_bound]
            except Exception:
                # There is too little data left over for us to delete according to our epoch_length, 
                # let's get more data!
                self.refresh_corpus = True
=======
        if epoch_length and epoch_length < len(self.data):
            
            # Set up upper bound of indices to fit the batch size we want. 
            idx_bound = epoch_length * self.batch_size 
            if idx_bound < len(self):
                # Collect enough random indices to batch together using batch_size into epoch_length batches
                random_start = random.randint(0, len(self) - idx_bound)
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
                            num_workers=self.config.dataloader.num_workers,
                            collate_fn=self.collate_fn)
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681


            # Set up dataloader
            return DataLoader(subset,
                            batch_size=self.batch_size,
<<<<<<< HEAD
                            num_workers=self.config.dataloader.num_workers)
=======
                            num_workers=self.config.dataloader.num_workers,
                            collate_fn=self.collate_fn)
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
        
        # If epoch_length is not set or it is higher than the total size of the dataset,
        #  then just shuffle dataset and return the whole thing.
        self.refresh_corpus = True
        return DataLoader(self,
                            shuffle=True,
                            batch_size=self.batch_size,
<<<<<<< HEAD
                            num_workers=self.config.dataloader.num_workers)
=======
                            num_workers=self.config.dataloader.num_workers,
                            collate_fn=self.collate_fn)
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681

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
<<<<<<< HEAD

=======
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
        chunk = self.data[idx:idx + self.block_size]

        dix = []
        block_num=0
<<<<<<< HEAD
        while block_num < self.block_size:
=======
        while block_num < self.block_size and len(chunk) > block_num:
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
            tokenized = self.tokenizer(chunk[block_num], padding=True, truncation=True)['input_ids']
            for t in tokenized:
                if block_num < self.block_size:
                    dix.append(t)
                    block_num += 1

<<<<<<< HEAD

        x = torch.tensor(dix, dtype=torch.long)
        return x    
=======
        if len(dix) == 0:
            return None

        x = torch.tensor(dix, dtype=torch.long)
        return x    

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
>>>>>>> 993a5268483758f6d64654eff3275012ae4f7681
