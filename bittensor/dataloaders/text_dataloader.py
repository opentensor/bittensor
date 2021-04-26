import bittensor
import torch
import random
import requests

from loguru import logger
from bittensor.dataloaders.dataloader import BittensorDataLoader

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
        self.filename, self.data = self.retrieve_random_text_data()

        
    
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

    def retrieve_random_text_data(self):
        """Connects to Infura IPFS gateway and retrieves the directory of genesis datasets.
        
        Returns:
            string: Contents of the text file. 
        """
        try:
            logger.info("Retrieving a dataset file from the IPFS gateway...")
            directory = self.retrieve_directory(self.genesis_text_dataset_hash)

            # Pick a random dataset file and return its contents
            if directory and 'links' in directory.keys():
                random_dataset_file = random.choice(directory['links'])
                filename = random_dataset_file['Name']
                # Make sure the file we chose satisfies our maximum file size requirement
                if random_dataset_file['Size'] <= self.config.dataloader.max_file_size:
                    random_dataset_file_hash = random_dataset_file['Cid']['/']
                    file_contents = self.retrieve_text_file(random_dataset_file_hash)
                    logger.info("Retrieved {} as training data...".format(filename))
                    return filename, file_contents.text.split()

            logger.error("It appears the directory is empty... Restart your miner to try again.")
            return None
        except Exception as ex:
            logger.error("Ran into exception when trying to retrieve dataset from IPFS: {}".format(ex))

        return None

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
        while block_num < self.block_size:
            tokenized = self.tokenizer(chunk[block_num], padding=True, truncation=True)['input_ids']
            for t in tokenized:
                if block_num < self.block_size:
                    dix.append(t)
                    block_num += 1


        x = torch.tensor(dix, dtype=torch.long)
        return x    
