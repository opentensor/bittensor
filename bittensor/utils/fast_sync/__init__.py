from dataclasses import dataclass
import enum
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import List
import weakref

import bittensor
from bittensor.utils import is_valid_ss58_address

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
        try:
            subprocess.run([path_to_bin, "sync_and_save", "-u", self.endpoint_url, '-b', block_hash], check=True, stdout=subprocess.PIPE)
        except subprocess.SubprocessError as e:
            raise FastSyncRuntimeException("Error running fast sync binary: {}".format(e))
    
    @staticmethod
    def validate_neuron_data_and_return(neuron_data: object) -> NeuronData:
        """
        Validates the neuron data and returns a NeuronData object

        Args:
            neuron_data (object): The neuron data object to validate
        
        Returns:
            NeuronData: The validated neuron data object

        Raises:
            FastSyncFormatException: If the neuron data is not formatted correctly for any fields
        
        """
        if not isinstance(neuron_data, dict):
            raise FastSyncFormatException("Neuron data must be a dict")

        hotkey = neuron_data.get("hotkey")
        if not isinstance(hotkey, str):
            raise FastSyncFormatException("Neuron data must have a hotkey field of type str")
        
        if not is_valid_ss58_address(hotkey):
            raise FastSyncFormatException("Neuron data must have a valid hotkey")

        coldkey = neuron_data.get("coldkey")
        if not isinstance(coldkey, str):
            raise FastSyncFormatException("Neuron data must have a coldkey field of type str")

        if not is_valid_ss58_address(coldkey):
            raise FastSyncFormatException("Neuron data must have a valid coldkey")

        uid = neuron_data.get("uid")
        if not isinstance(uid, int):
            raise FastSyncFormatException("Neuron data must have a uid field of type int")

        ip = neuron_data.get("ip")
        if not isinstance(ip, str):
            raise FastSyncFormatException("Neuron data must have an ip field of type str")

        port = neuron_data.get("port")
        if not isinstance(port, int):
            raise FastSyncFormatException("Neuron data must have a port field of type int")

        stake = neuron_data.get("stake")
        if not isinstance(stake, str):
            raise FastSyncFormatException("Neuron data must have a stake field of type str")
        try:
            stake = int(stake)
            if stake < 0:
                raise FastSyncFormatException("Neuron data must have a stake field >= 0")
            neuron_data["stake"] = stake
        except ValueError:
            raise FastSyncFormatException("Neuron data stake field must be a valid int")

        rank = neuron_data.get("rank")
        if not isinstance(rank, str):
            raise FastSyncFormatException("Neuron data must have a rank field of type str")
        try:
            rank = int(rank)
            if rank < 0:
                raise FastSyncFormatException("Neuron data must have a rank field >= 0")
            if rank > U64MAX:
                raise FastSyncFormatException("Neuron data must have a rank field <= U64MAX")
            neuron_data["rank"] = rank
        except ValueError:
            raise FastSyncFormatException("Neuron data rank field must be a valid int")

        emission = neuron_data.get("emission")
        if not isinstance(emission, str):
            raise FastSyncFormatException("Neuron data must have an emission field of type str")
        try:
            emission = int(emission)
            if emission < 0:
                raise FastSyncFormatException("Neuron data must have an emission field >= 0")
            neuron_data["emission"] = emission
        except ValueError:
            raise FastSyncFormatException("Neuron data emission field must be a valid int")

        incentive = neuron_data.get("incentive")
        if not isinstance(incentive, str):
            raise FastSyncFormatException("Neuron data must have an incentive field of type str")
        try:
            incentive = int(incentive)
            if incentive < 0:
                raise FastSyncFormatException("Neuron data must have an incentive field >= 0")
            if incentive > U64MAX:
                raise FastSyncFormatException("Neuron data must have an incentive field <= U64MAX")
            neuron_data["incentive"] = incentive
        except ValueError:
            raise FastSyncFormatException("Neuron data incentive field must be a valid int")

        consensus = neuron_data.get("consensus")
        if not isinstance(consensus, str):
            raise FastSyncFormatException("Neuron data must have a consensus field of type str")
        try:
            consensus = int(consensus)
            if consensus < 0:
                raise FastSyncFormatException("Neuron data must have a consensus field >= 0")
            if consensus > U64MAX:
                raise FastSyncFormatException("Neuron data must have a consensus field <= U64MAX")
            neuron_data["consensus"] = consensus
        except ValueError:
            raise FastSyncFormatException("Neuron data consensus field must be a valid int")
        
        trust = neuron_data.get("trust")
        if not isinstance(trust, str):
            raise FastSyncFormatException("Neuron data must have a trust field of type str")
        try:
            trust = int(trust)
            if trust < 0:
                raise FastSyncFormatException("Neuron data must have a trust field >= 0")
            if trust > U64MAX:
                raise FastSyncFormatException("Neuron data must have a trust field <= U64MAX")
            neuron_data["trust"] = trust
        except ValueError:
            raise FastSyncFormatException("Neuron data trust field must be a valid int")

        dividends = neuron_data.get("dividends")
        if not isinstance(dividends, str):
            raise FastSyncFormatException("Neuron data must have a dividends field of type str")
        try:
            dividends = int(dividends)
            if dividends < 0:
                raise FastSyncFormatException("Neuron data must have a dividends field >= 0")
            if dividends > U64MAX:
                raise FastSyncFormatException("Neuron data must have a dividends field <= U64MAX")
            neuron_data["dividends"] = dividends
        except ValueError:
            raise FastSyncFormatException("Neuron data dividends field must be a valid int")

        modality = neuron_data.get("modality")
        if not isinstance(modality, int):
            raise FastSyncFormatException("Neuron data must have a modality field of type int")

        last_update = neuron_data.get("last_update")
        if not isinstance(last_update, str):
            raise FastSyncFormatException("Neuron data must have a last_update field of type str")
        try:    
            last_update = int(last_update)
            if last_update < 0:
                raise FastSyncFormatException("Neuron data must have a last_update field >= 0")
            neuron_data["last_update"] = last_update
        except ValueError:
            raise FastSyncFormatException("Neuron data last_update field must be a valid int")

        version = neuron_data.get("version")
        if not isinstance(version, int):
            raise FastSyncFormatException("Neuron data must have a version field of type int")

        priority = neuron_data.get("priority")
        if not isinstance(priority, str):
            raise FastSyncFormatException("Neuron data must have a priority field of type str")
        try:
            priority = int(priority)
            if priority < 0:
                raise FastSyncFormatException("Neuron data must have a priority field >= 0")
            neuron_data["priority"] = priority
        except ValueError:
            raise FastSyncFormatException("Neuron data priority field must be a valid int")
        
        weights = neuron_data.get("weights")
        if not isinstance(weights, list):
            raise FastSyncFormatException("Neuron data must have a weights field of type list")
        if not all(isinstance(w0, int) and isinstance(w1, int) for w0, w1 in weights):
            raise FastSyncFormatException("Neuron data weights field must be a list of int pairs")
        
        bonds = neuron_data.get("bonds")
        if not isinstance(bonds, list):
            raise FastSyncFormatException("Neuron data must have a bonds field of type list")
        if not all(isinstance(uid, int) and isinstance(b, str)  for uid, b in bonds):
            raise FastSyncFormatException("Neuron data bonds field must be a list of int, str(int) pairs")
        try:
            for bond, i in zip(bonds, range(len(bonds))):
                bonds[i] = (bond[0], int(bond[1]))

                if bonds[i][0] < 0:
                    raise FastSyncFormatException("Neuron data bonds field must have all uids >= 0")
                if bonds[i][1] < 0:
                    raise FastSyncFormatException("Neuron data bonds field must have a bond >= 0")
                neuron_data["bonds"] = bonds
        except ValueError:
            raise FastSyncFormatException("Neuron data bonds field must be a list of int, str(int) pairs")

        return NeuronData(
            **neuron_data
        )


    @classmethod
    def _load_neurons_from_metragraph_file_data(cls, file_data: str) -> List[SimpleNamespace]:
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
        if not isinstance(data, list):
            raise FastSyncFormatException('Expected a JSON array at the top level')
        
        neurons: SimpleNamespace = []
        try:
            for neuron_data, i in zip(data, range(len(data))):
                # validate the neuron data
                neuron_data_validated: NeuronData = cls.validate_neuron_data(neuron_data)
                neuron = SimpleNamespace( **neuron_data_validated )
                # hotkey and coldkey are strings
                # uid is an int
                # ip is a string
                # ip_type is an int
                # port is an int
                neuron.stake = neuron.stake / RAOPERTAO
                neuron.rank = neuron.rank / U64MAX
                neuron.emission = neuron.emission / RAOPERTAO
                neuron.incentive = neuron.incentive / U64MAX
                neuron.consensus = neuron.consensus / U64MAX
                neuron.trust = neuron.trust / U64MAX
                neuron.dividends = neuron.dividends / U64MAX
                # modality is an int
                # last_update is an int from the validate function
                # version is an int
                # priority is an int from the validate function
                # weights are already ints
                # bonds are already ints from the validate function

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
