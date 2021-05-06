import argparse
import bittensor
import requests
from munch import Munch

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry



class BittensorDataLoader():
    def __init__(self):
        # IPFS hash of the genesis dataset
        # TODO (shibshib): Find a proper way to set this as config instead of hardcoding it.
        # More dataset hashes can be added as we add directories for other modalities.
        self.genesis_text_dataset_hash = "QmXwfPoh2QFYqC6cYcW8kzyd9ruFfhnUi2kVBkdhawjUzj"

        # Used to retrieve directory contentx
        self.dag_get = 'https://ipfs.infura.io:5001/api/v0/dag/get'
        # Used to retrieve file contents
        self.file_cat = 'https://ipfs.infura.io:5001/api/v0/cat'

        # Used when current corpus has been exhausted
        self.refresh_corpus = False
        
    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        BittensorDataLoader.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """ Add model params
        """
        parser.add_argument('--dataloader.max_corpus_size', default=1e+6, type=int, 
                                help='Maximum amount of data to download from IPFS into memory for training.')
        parser.add_argument('--dataloader.num_workers', default=1, type=int, 
                                help='Number of workers for data loader.')

    
    @staticmethod   
    def check_config(config: Munch):
        pass

    @staticmethod
    def requests_retry_session(
            retries=3,
            backoff_factor=0.3,
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
        """Connects to Infura IPFS gateway and retrieves the directory of 
        genesis datasets.

        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        params = (('arg', dir_hash),)
        session.params.update(params)
        directory = None

        response = BittensorDataLoader.requests_retry_session(session=session).post(self.dag_get)

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