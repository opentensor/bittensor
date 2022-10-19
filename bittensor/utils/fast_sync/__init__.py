from dataclasses import dataclass
import enum
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import List
from tqdm import tqdm

import bittensor

RAOPERTAO: int = bittensor.__rao_per_tao__
U64MAX: int = 18446744073709551615
U32MAX: int = 4294967295

class FastSyncException(Exception):
    """"Exception raised during fast sync of neurons"""
    pass

class FastSyncOSNotSupportedException(FastSyncException):
    """"Exception raised when the OS is not supported by fast sync"""
    pass

class FastSyncNotFoundException(FastSyncException):
    """"Exception raised when the fast sync binary is not found"""
    pass

class FastSyncFormatException(FastSyncException):
    """"Exception raised when the downloaded metagraph file is not formatted correctly"""
    pass

class FastSyncFileException(FastSyncException):
    """"Exception raised when the metagraph file cannot be read"""
    pass

class FastSyncRuntimeException(FastSyncException):
    """"Exception raised when the fast sync binary fails to run"""
    pass

class OS_NAME(enum.Enum):
    """Enum for OS_NAME"""
    LINUX = "linux"
    MAC = "macos"
    WINDOWS = "windows"

@dataclass
class NeuronData:
    """
    Dataclass for NeuronData
    From JSON of the form
    {
        "hotkey": str,
        "coldkey": str,
        "uid": int,
        "active": int,
        "ip": str,
        "ip_type": int,
        "port": int,
        "stake": str(int),
        "rank": str(int),
        "emission": str(int),
        "incentive": str(int),
        "consensus": str(int),
        "trust": str(int),
        "dividends": str(int),
        "modality": int,
        "last_update": str(int),
        "version": int,
        "priority": str(int),
        "weights": [
            [int, int],
        ],
        "bonds": [
            [int, str(int)],
        ],
    }
    """
    hotkey: str
    coldkey: str
    uid: int
    active: int
    ip: str
    ip_type: int
    port: int
    stake: int
    rank: int
    emission: int
    incentive: int
    consensus: int
    trust: int
    dividends: int
    modality: int
    last_update: int
    version: int
    priority: int
    weights: List[List[int]]
    bonds: List[List[int]]

class FastSync:
    endpoint_url: str

    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url

    @classmethod
    def get_platform(cls) -> str:
        return sys.platform

    @classmethod
    def get_os(cls) -> OS_NAME:
        """Returns the OS enum for the current OS"""
        platform = cls.get_platform()
        if platform == "linux" or platform == "linux2":
            return OS_NAME.LINUX
        elif platform == "darwin":
            return OS_NAME.MAC
        elif platform == "win32":
            return OS_NAME.WINDOWS
        else:
            raise Exception("Not sure what OS this is")

    @classmethod
    def verify_fast_sync_support(cls) -> None:
        """
        Verifies that the current system is supported by fast sync

        Raises:
            FastSyncOSNotSupportedException: If the current OS is not supported
            FastSyncNotFoundException: If the fast sync binary is not found
        """
        cls.verify_os_support()
        cls.verify_binary_exists()

    @classmethod
    def verify_os_support(cls) -> None:
        """
        Verifies that the current OS is supported by fast sync

        Raises:
            FastSyncOSNotSupportedException: If the current OS is not supported
        """

        try:
            OS = cls.get_os()
        except Exception as e:
            raise FastSyncOSNotSupportedException("OS not supported by fast sync: {}".format(e))
        
        if OS != OS.LINUX and OS != OS.MAC:
            raise FastSyncOSNotSupportedException("OS not supported for fast sync")
    
    @classmethod
    def verify_binary_exists(cls) -> None:
        """
        Verifies that the fast sync binary exists

        Raises:
            FastSyncNotFoundException: If the fast sync binary is not found
        """
        path_to_bin = cls.get_path_to_fast_sync()
        if not os.path.exists(path_to_bin) or not os.path.isfile(path_to_bin):
            raise FastSyncNotFoundException("Could not find fast sync binary at {}.".format(path_to_bin))

    @classmethod
    def get_path_to_fast_sync(cls) -> str:
        """Returns the path to the fast sync binary"""
        os_name: OS_NAME = cls.get_os()
        path_to_bin = os.path.join(os.path.dirname(__file__), f"../../../bin/subtensor-node-api-{os_name.value}")
        return path_to_bin

    def sync_neurons(self, block_hash: str) -> None:
        """Runs the fast sync binary to sync all neurons at a given block hash"""
        FastSync.verify_fast_sync_support()
        path_to_bin = FastSync.get_path_to_fast_sync()
        bittensor.__console__.print("Using subtensor-node-api for neuron retrieval...")
        # will write to ~/.bittensor/metagraph.json by default
        try:
            subprocess.run([path_to_bin, "sync_and_save", "-u", self.endpoint_url, '-b', block_hash], check=True, stdout=subprocess.PIPE)
        except subprocess.SubprocessError as e:
            raise FastSyncRuntimeException("Error running fast sync binary: {}".format(e))

    def get_blockAtRegistration_for_all_and_save(self, block_hash: str) -> None:
        """Runs the fast sync binary to get blockAtRegistration for all neurons at a given block hash"""
        FastSync.verify_fast_sync_support()
        path_to_bin = FastSync.get_path_to_fast_sync()
        bittensor.__console__.print("Using subtensor-node-api for blockAtRegistration storage retrieval...")
        # will write to ~/.bittensor/blockAtRegistration_all.json by default
        try:
            subprocess.run([path_to_bin, "get_block_at_registration_for_all_and_save", "-u", self.endpoint_url, '-b', block_hash], check=True, stdout=subprocess.PIPE)
        except subprocess.SubprocessError as e:
            raise FastSyncRuntimeException("Error running fast sync binary: {}".format(e))

    @classmethod
    def load_blockAtRegistration_for_all(cls, json_file_location: str = '~/.bittensor/blockAtRegistration_all.json') -> List[int]:
        """
        Loads neurons from the blockAtRegistration JSON file

        Args:
            json_file_location (str, optional): The location of the blockAtRegistration JSON file. Defaults to '~/.bittensor/blockAtRegistration_all.json'.
        
        Raises:
            FastSyncFileException: If the JSON file could not be read
            FastSyncFormatException: If the JSON file is not in the correct format
        
        Returns:
            List[int]
                a list of the blockAtRegistration numbers
        """
        try:
            with open(os.path.join(os.path.expanduser(json_file_location))) as f:
                file_data = f.read()
            return cls._load_neurons_from_blockAtRegistration_all_file_data(file_data)
        except FileNotFoundError:
            raise FastSyncFileException('{} not found. Try calling fast_sync_neurons() first.', json_file_location)
        except OSError:
            raise FastSyncFileException('Could not read {}', json_file_location)

    @classmethod
    def _load_neurons_from_blockAtRegistration_all_file_data(cls, file_data: str) -> List[int]:
        """
        Loads neurons from the blockAtRegistration_all JSON file data
        
        Raises: FastSyncFormatException if the file is not in the correct format

        Returns: List[int]
            a list of the blockAtRegistration numbers
        """
        try:
            data = json.loads(file_data)
        except json.JSONDecodeError:
            raise FastSyncFormatException('Could not parse blockAtRegistration JSON file data as json')

        # all the large ints are strings
        if not isinstance(data, list):
            raise FastSyncFormatException('Expected a JSON array at the top level')
        
        try:
            # validate the blockAtRegistration data
            blockAtRegistration_all: List[int] = [
                int(blockAtRegistration) for blockAtRegistration in tqdm(data)
            ]
        except Exception as e:
            raise FastSyncFormatException('Could not parse blockAtRegistration JSON file data: {}'.format(e))
            
        return blockAtRegistration_all

    @classmethod
    def _load_neurons_from_metragraph_file_data(cls, file_data: str) -> List[SimpleNamespace]:
        """
        Loads neurons from the metagraph file data
        See: https://github.com/opentensor/subtensor-node-api#neuron-structure
        
        Raises: FastSyncFormatException if the file is not in the correct format

        Returns: List[SimpleNamespace]
            a list of the Neurons
        """
        try:
            data = json.loads(file_data)
        except json.JSONDecodeError:
            raise FastSyncFormatException('Could not parse metagraph file data as json')

        # all the large ints are strings
        if not isinstance(data, list):
            raise FastSyncFormatException('Expected a JSON array at the top level')
        
        neurons: List[SimpleNamespace] = []
        try:
            
            for neuron_data in tqdm(data):
                # validate the neuron data
                neuron = SimpleNamespace( **neuron_data )
                # hotkey and coldkey are strings
                # uid is an int
                # active is an int
                # ip is a string
                # ip_type is an int
                # port is an int
                neuron.stake = int(neuron.stake) / RAOPERTAO
                neuron.rank = int(neuron.rank) / U64MAX
                neuron.emission = int(neuron.emission) / RAOPERTAO
                neuron.incentive = int(neuron.incentive) / U64MAX
                neuron.consensus = int(neuron.consensus) / U64MAX
                neuron.trust = int(neuron.trust) / U64MAX
                neuron.dividends = int(neuron.dividends) / U64MAX
                # modality is an int
                neuron.last_update = int(neuron.last_update)
                # version is an int
                neuron.priority = int(neuron.priority)
                # weights are already ints
                neuron.bonds = [[bond[0], int(bond[1])] for bond in neuron.bonds]

                neuron.is_null = False
                neurons.append( neuron )

        except Exception as e:
            raise FastSyncFormatException('Could not parse metagraph file data: {}'.format(e))
            
        return neurons

    @classmethod
    def load_neurons(cls, metagraph_location: str = '~/.bittensor/metagraph.json') -> List[SimpleNamespace]:
        """
        Loads neurons from the metagraph file

        Args:
            metagraph_location (str, optional): The location of the metagraph file. Defaults to '~/.bittensor/metagraph.json'.
        
        Raises:
            FastSyncFileException: If the metagraph file could not be read
            FastSyncFormatException: If the metagraph file is not in the correct format
        
        Returns:
            List[SimpleNamespace]
                a list of the Neurons
        """
        try:
            with open(os.path.join(os.path.expanduser(metagraph_location))) as f:
                file_data = f.read()
            return cls._load_neurons_from_metragraph_file_data(file_data)
        except FileNotFoundError:
            raise FastSyncFileException('{} not found. Try calling fast_sync_neurons() first.', metagraph_location)
        except OSError:
            raise FastSyncFileException('Could not read {}', metagraph_location)
