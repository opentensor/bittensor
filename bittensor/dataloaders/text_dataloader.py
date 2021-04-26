import bittensor
import torch

from bittensor.dataloaders.dataloader import BittensorDataLoader

class GenesisTextDataloader(BittensorDataLoader):
    
    def __init__(
            self,
            batch_size, 
            block_size,
            config = None
        ):
        super(GenesisTextDataloader, self).__init__()
        if config == None:
            config = BittensorDataLoader.default_config()

        self.config = config
        self.block_size = block_size
        self.tokenizer = bittensor.__tokenizer__()
        self.batch_size = batch_size

        # Retrieve a random slice of the genesis dataset
        self.filename, self.data = self.retrieve_random_text_data()

        
    def __len__(self):
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






            





