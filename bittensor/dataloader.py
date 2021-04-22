import requests
import json
import argparse
import bittensor
from munch import Munch

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class GenesisDataloader():
    
    def __init__(
            self,
            config: 'Munch' = None,
        ):
        # IPFS hash of the genesis dataset
        # TODO (shibshib): Find a proper way to set this as config instead of hardcoding it.
        self.genesis_dataset_hash = "QmXwfPoh2QFYqC6cYcW8kzyd9ruFfhnUi2kVBkdhawjUzj"
        self.request_params = (('arg', self.genesis_dataset_hash),)

        if config == None:
            config = GenesisDataloader.default_config()
        self.config = config
        
    
    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        GenesisDataloader.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """ Add model params
        """
        parser.add_argument('--dataloader.max_file_size', default=1e+9, type=int, 
                                help='Maximum text file size (in bytes) to load into memory.')
    
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
        
    def retrieve_dataset_files(self):
        session = requests.Session()
        session.params.update(self.params)
        directory = None

        response = GenesisDataloader.requests_retry_session(session=session).post('https://ipfs.infura.io:5001/api/v0/dag/get')

        if response.status_code == 200:
            directory = response.json()
            return directory
