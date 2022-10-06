import enum
from io import TextIOWrapper
import json
import os
import subprocess
from sys import platform
from types import SimpleNamespace
from typing import List

import bittensor


class FastSyncException(Exception):
    """"Exception raised during fast sync of neurons"""
    pass

class OSNotSupported(FastSyncException):
    """"Exception raised when the OS is not supported"""
    pass

class FastSyncNotFoundException(FastSyncException):
    """"Exception raised when the fast sync binary is not found"""
    pass

class OS_NAME(enum.Enum):
    """Enum for OS_NAME"""
    LINUX = "linux"
    MAC = "macos"
    WINDOWS = "windows"

def get_os() -> OS_NAME:
    """Returns the OS enum for the current OS"""
    if platform == "linux" or platform == "linux2":
        return OS_NAME.LINUX
    elif platform == "darwin":
        return OS_NAME.MAC
    elif platform == "win32":
        return OS_NAME.WINDOWS
    else:
        raise Exception("Not sure what OS this is")

class FastSync():
    endpoint_url: str

    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url

    @staticmethod
    def verify_fast_sync_support() -> None:
        try:
            OS = get_os()
        except Exception:
            raise OSNotSupported("OS not supported for fast sync")
        
        if OS != OS.LINUX and OS != OS.MAC:
            raise OSNotSupported("OS not supported for fast sync")

        # verify that the binary exists
        path_to_bin = FastSync.get_path_to_fast_sync()
        if not os.path.exists(path_to_bin) or not os.path.isfile(path_to_bin):
            raise FastSyncNotFoundException("Could not find fast sync binary at {}.".format(path_to_bin))

    @staticmethod
    def get_path_to_fast_sync() -> str:
        """Returns the path to the fast sync binary"""
        os_name: OS_NAME = get_os()
        path_to_bin = os.path.join(os.path.dirname(__file__), f"../../bin/subtensor-node-api-{os_name.value}")
        return path_to_bin

    def fast_sync_neurons(self, block_hash: str) -> None:
        """Runs the fast sync binary to sync all neurons at a given block hash"""
        FastSync.verify_fast_sync_support()
        path_to_bin = FastSync.get_path_to_fast_sync()
        bittensor.__console__.print("Using subtensor-node-api for neuron retrieval...")
        # will write to ~/.bittensor/metagraph.json by default
        subprocess.run([path_to_bin, "sync_and_save", "-u", self.endpoint_url, '-b', block_hash], check=True, stdout=subprocess.PIPE)
    
    @staticmethod
    def _load_neurons_from_metragraph_file(file: TextIOWrapper) -> List[SimpleNamespace]:
        """Loads neurons from a metagraph file"""
        data = json.load(file)

        # all the large ints are strings
        RAOPERTAO = bittensor.__rao_per_tao__
        U64MAX = 18446744073709551615
        """
        We expect a JSON array of:
        {
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
            "last_update": int,
            "weights": [
                [int, int],
            ],
            "bonds": [
                [int, str(int)],
            ],
        }
        """
        neurons: SimpleNamespace = []
        for neuron_data in data:
            neuron = SimpleNamespace( **neuron_data )
            neuron.stake = int(neuron.stake) / RAOPERTAO
            neuron.rank = int(neuron.rank) / U64MAX
            neuron.trust = int(neuron.trust) / U64MAX
            neuron.consensus = int(neuron.consensus) / U64MAX
            neuron.incentive = int(neuron.incentive) / U64MAX
            neuron.dividends = int(neuron.dividends) / U64MAX
            neuron.emission = int(neuron.emission) / RAOPERTAO
            neuron.last_update = int(neuron.last_update)
            neuron.priority = int(neuron.priority)
            neuron.bonds = [ [bond[0], int(bond[1])] for bond in neuron.bonds ]
            # weights are already ints
            neuron.is_null = False
            neurons.append( neuron )

        return neurons

    def load_neurons(self, metagraph_location: str = '~/.bittensor/metagraph.json') -> List[SimpleNamespace]:
        try:
            with open(os.path.join(os.path.expanduser(metagraph_location))) as f:
                return self._load_neurons_from_metragraph_file(f)
        except FileNotFoundError:
            raise FastSyncException('{} not found. Try calling fast_sync_neurons() first.', metagraph_location)
    
        
    
