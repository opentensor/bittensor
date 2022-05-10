
from socket import timeout
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests

class Ipfs():
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self):
        
        # Used to retrieve directory contentx
        self.cat = 'http://global.ipfs.opentensor.ai/api/v0/cat' 
        self.node_get = 'http://global.ipfs.opentensor.ai/api/v0/object/get'
        self.ipns_resolve = 'http://global.ipfs.opentensor.ai/api/v0/name/resolve'

        self.mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
        self.latest_neurons_ipns = "k51qzi5uqu5di1eoe0o91g32tbfsgikva6mvz0jw0414zhxzhiakana67shoh7"
        self.historical_neurons_ipns = "k51qzi5uqu5dhf5yxm3kqw9hyrv28q492p3t32s23059z911a23l30ai6ziceh"
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

    def retrieve_directory(self, address: str, params = None, action: str = 'post', timeout: int = 180):
        r"""Connects to Pinata IPFS gateway and retrieves directory.

        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        session.params.update(params)
        if action == 'get':
            response = Ipfs.requests_retry_session(session=session).get(address, timeout=timeout)
        elif action == 'post':
            response = Ipfs.requests_retry_session(session=session).post(address, timeout=timeout)
        return response