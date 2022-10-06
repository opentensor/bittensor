import enum
import json
import os
from platform import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import List

import bittensor


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

class OS_NAME(enum.Enum):
    """Enum for OS_NAME"""
    LINUX = "linux"
    MAC = "macos"
    WINDOWS = "windows"

class FastSync():
    endpoint_url: str

    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url

    @property
    def platform(self) -> str:
        return sys.platform

    @classmethod
    def get_os(cls) -> OS_NAME:
        """Returns the OS enum for the current OS"""
        platform = cls.platform
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
        except Exception:
            raise FastSyncOSNotSupportedException("OS not supported for fast sync")
        
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
        path_to_bin = os.path.join(os.path.dirname(__file__), f"../../bin/subtensor-node-api-{os_name.value}")
        return path_to_bin

    def sync_neurons(self, block_hash: str) -> None:
        """Runs the fast sync binary to sync all neurons at a given block hash"""
        FastSync.verify_fast_sync_support()
        path_to_bin = FastSync.get_path_to_fast_sync()
        bittensor.__console__.print("Using subtensor-node-api for neuron retrieval...")
        # will write to ~/.bittensor/metagraph.json by default
        subprocess.run([path_to_bin, "sync_and_save", "-u", self.endpoint_url, '-b', block_hash], check=True, stdout=subprocess.PIPE)
    
    @staticmethod
    def _load_neurons_from_metragraph_file_data(file_data: str) -> List[SimpleNamespace]:
        """
        Loads neurons from the metagraph file data
        
        Raises: FastSyncFormatException if the file is not in the correct format

        Returns: List[SimpleNamespace]
            a list of the Neurons
        """
        try:
            data = json.loads(file_data)
        except json.JSONDecodeError:
            raise FastSyncFormatException('Could not parse metagraph file data as json')

        # all the large ints are strings
        RAOPERTAO = bittensor.__rao_per_tao__
        U64MAX = 18446744073709551615
        """
        We expect a JSON array of:
        {
            "hotkey": str,
            "coldkey": str,
            "uid": int,
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
        neurons: SimpleNamespace = []
        try:
            for neuron_data in data:
                neuron = SimpleNamespace( **neuron_data )
                # hotkey and coldkey are strings
                # uid is an int
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
                neuron.bonds = [ [bond[0], int(bond[1])] for bond in neuron.bonds ]

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
