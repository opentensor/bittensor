import copy
import os
import pickle
import typing
from abc import ABC, abstractmethod
from os import listdir
from os.path import join
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from bittensor.utils.btlogging import logging
from bittensor.utils.registration import torch, use_torch
from bittensor.utils.weight_utils import (
    convert_weight_uids_and_vals_to_tensor,
    convert_bond_uids_and_vals_to_tensor,
    convert_root_weight_uids_and_vals_to_tensor,
)
from . import settings
from .chain_data import AxonInfo

# For annotation purposes
if typing.TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor


METAGRAPH_STATE_DICT_NDARRAY_KEYS = [
    "version",
    "n",
    "block",
    "stake",
    "total_stake",
    "ranks",
    "trust",
    "consensus",
    "validator_trust",
    "incentive",
    "emission",
    "dividends",
    "active",
    "last_update",
    "validator_permit",
    "uids",
]
"""List of keys for the metagraph state dictionary used in NDArray serialization.

This list defines the set of keys expected in the metagraph's state dictionary when serializing and deserializing NumPy ndarray objects. Each key corresponds to a specific attribute or metric associated with the nodes in the metagraph.

- **version** (`str`): The version identifier of the metagraph state.
- **n** (`int`): The total number of nodes in the metagraph.
- **block** (`int`): The current block number in the blockchain or ledger.
- **stake** (`ndarray`): An array representing the stake of each node.
- **total_stake** (`float`): The sum of all individual stakes in the metagraph.
- **ranks** (`ndarray`): An array of rank scores assigned to each node.
- **trust** (`ndarray`): An array of trust scores for the nodes.
- **consensus** (`ndarray`): An array indicating consensus levels among nodes.
- **validator_trust** (`ndarray`): Trust scores specific to validator nodes.
- **incentive** (`ndarray`): Incentive values allocated to nodes.
- **emission** (`float`): The rate of emission for new tokens or units.
- **dividends** (`ndarray`): Dividend amounts distributed to nodes.
- **active** (`ndarray`): Boolean array indicating active (`True`) or inactive (`False`) nodes.
- **last_update** (`int`): Timestamp of the last state update.
- **validator_permit** (`ndarray`): Boolean array indicating nodes permitted to validate.
- **uids** (`ndarray`): Unique identifiers for each node in the metagraph.
"""


def get_save_dir(
    network: str, netuid: int, root_dir: Optional[list[str]] = None
) -> str:
    """
    Returns a directory path given ``network`` and ``netuid`` inputs.

    Args:
        network (str): Network name.
        netuid (int): Network UID.
        root_dir: list to the file path for the root directory of your metagraph saves (i.e. ['/', 'tmp', 'metagraphs'],
            defaults to ["~", ".bittensor", "metagraphs"]

    Returns:
        str: Directory path.
    """
    _root_dir = root_dir or ["~", ".bittensor", "metagraphs"]
    return os.path.expanduser(
        os.path.join(
            *_root_dir,
            f"network-{str(network)}",
            f"netuid-{str(netuid)}",
        )
    )


def latest_block_path(dir_path: str) -> str:
    """
    Get the latest block path from the provided directory path.

    Args:
        dir_path (str): Directory path.

    Returns:
        str: Latest block path.
    """
    latest_block = -1
    latest_file_full_path = None
    for filename in listdir(dir_path):
        full_path_filename = os.path.expanduser(join(dir_path, filename))
        try:
            block_number = int(filename.split("-")[1].split(".")[0])
            if block_number > latest_block:
                latest_block = block_number
                latest_file_full_path = full_path_filename
        except Exception:
            pass
    if not latest_file_full_path:
        raise ValueError(f"Metagraph not found at: {dir_path}")
    else:
        return latest_file_full_path


def determine_chain_endpoint_and_network(network: str) -> tuple[str, str]:
    """
    Determine the chain endpoint and network name from the passed arg

    Args:
        network: The network name (e.g. 'finney', 'test') or
            chain endpoint (e.g. wss://entrypoint-finney.opentensor.ai:443)

    Returns:
        (network name, chain endpoint)
    """
    pathless_network = network[:-1] if network.endswith("/") else network
    if pathless_network in settings.NETWORK_MAP:
        return pathless_network, settings.NETWORK_MAP[pathless_network]
    elif pathless_network in settings.REVERSE_NETWORK_MAP:
        return settings.REVERSE_NETWORK_MAP[pathless_network], pathless_network
    else:
        return "unknown", network


class MetagraphMixin(ABC):
    """
    The metagraph class is a core component of the Bittensor network, representing the neural graph that forms the backbone of the decentralized machine learning system.

    The metagraph is a dynamic representation of the network's state, capturing the interconnectedness and attributes of neurons (participants) in the Bittensor ecosystem. This class is not just a static structure but a live reflection of the network, constantly updated and synchronized with the state of the blockchain.

    In Bittensor, neurons are akin to nodes in a distributed system, each contributing computational resources and participating in the network's collective intelligence. The metagraph tracks various attributes of these neurons, such as stake, trust, and consensus, which are crucial for the network's incentive mechanisms and the Yuma Consensus algorithm as outlined in the `NeurIPS paper <https://bittensor.com/pdfs/academia/NeurIPS_DAO_Workshop_2022_3_3.pdf>`_. These attributes
    govern how neurons interact, how they are incentivized, and their roles within the network's
    decision-making processes.

    Args:
        netuid (int): A unique identifier that distinguishes between different instances or versions of the Bittensor network.
        network (str): The name of the network, signifying specific configurations or iterations within the Bittensor ecosystem.
        version (NDArray): The version number of the network, integral for tracking network updates.
        n (NDArray): The total number of neurons in the network, reflecting its size and complexity.
        block (NDArray): The current block number in the blockchain, crucial for synchronizing with the network's latest state.
        stake: Represents the cryptocurrency staked by neurons, impacting their influence and earnings within the network.
        total_stake: The cumulative stake across all neurons.
        ranks: Neuron rankings as per the Yuma Consensus algorithm, influencing their incentive distribution and network authority.
        trust: Scores indicating the reliability of neurons, mainly miners, within the network's operational context.
        consensus: Scores reflecting each neuron's alignment with the network's collective decisions.
        validator_trust: Trust scores for validator neurons, crucial for network security and validation.
        incentive: Rewards allocated to neurons, particularly miners, for their network contributions.
        emission: The rate at which rewards are distributed to neurons.
        dividends: Rewards received primarily by validators as part of the incentive mechanism.
        active: Status indicating whether a neuron is actively participating in the network.
        last_update: Timestamp of the latest update to a neuron's data.
        validator_permit: Indicates if a neuron is authorized to act as a validator.
        weights: Inter-neuronal weights set by each neuron, influencing network dynamics.
        bonds: Represents speculative investments by neurons in others, part of the reward mechanism.
        uids: Unique identifiers for each neuron, essential for network operations.
        axons (List): Details about each neuron's axon, critical for facilitating network communication.

    The metagraph plays a pivotal role in Bittensor's decentralized AI operations, influencing everything from data propagation to reward distribution. It embodies the principles of decentralized governance
    and collaborative intelligence, ensuring that the network remains adaptive, secure, and efficient.

    Example:
        Initializing the metagraph to represent the current state of the Bittensor network::

            from bittensor.core.metagraph import Metagraph
            metagraph = Metagraph(netuid=config.netuid, network=subtensor.network, sync=False)

        Synchronizing the metagraph with the network to reflect the latest state and neuron data::

            metagraph.sync(subtensor=subtensor)

        Accessing metagraph properties to inform network interactions and decisions::

            total_stake = metagraph.S
            neuron_ranks = metagraph.R
            neuron_incentives = metagraph.I
            axons = metagraph.axons
            neurons = metagraph.neurons

        Maintaining a local copy of hotkeys for querying and interacting with network entities::

            hotkeys = deepcopy(metagraph.hotkeys)
    """

    netuid: int
    network: str
    version: Union["torch.nn.Parameter", tuple[NDArray]]
    n: Union["torch.nn.Parameter", NDArray]
    block: Union["torch.nn.Parameter", NDArray]
    stake: Union["torch.nn.Parameter", NDArray]
    total_stake: Union["torch.nn.Parameter", NDArray]
    ranks: Union["torch.nn.Parameter", NDArray]
    trust: Union["torch.nn.Parameter", NDArray]
    consensus: Union["torch.nn.Parameter", NDArray]
    validator_trust: Union["torch.nn.Parameter", NDArray]
    incentive: Union["torch.nn.Parameter", NDArray]
    emission: Union["torch.nn.Parameter", NDArray]
    dividends: Union["torch.nn.Parameter", NDArray]
    active: Union["torch.nn.Parameter", NDArray]
    last_update: Union["torch.nn.Parameter", NDArray]
    validator_permit: Union["torch.nn.Parameter", NDArray]
    weights: Union["torch.nn.Parameter", NDArray]
    bonds: Union["torch.nn.Parameter", NDArray]
    uids: Union["torch.nn.Parameter", NDArray]
    axons: list[AxonInfo]
    chain_endpoint: Optional[str]
    subtensor: Optional["Subtensor"]

    @property
    def S(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Represents the stake of each neuron in the Bittensor network. Stake is an important concept in the
        Bittensor ecosystem, signifying the amount of network weight (or “stake”) each neuron holds,
        represented on a digital ledger. The stake influences a neuron's ability to contribute to and benefit
        from the network, playing a crucial role in the distribution of incentives and decision-making processes.

        Returns:
            NDArray: A tensor representing the stake of each neuron in the network. Higher values signify a greater stake held by the respective neuron.
        """
        return self.total_stake

    @property
    def R(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Contains the ranks of neurons in the Bittensor network. Ranks are determined by the network based
        on each neuron's performance and contributions. Higher ranks typically indicate a greater level of
        contribution or performance by a neuron. These ranks are crucial in determining the distribution of
        incentives within the network, with higher-ranked neurons receiving more incentive.

        Returns:
            NDArray: A tensor where each element represents the rank of a neuron. Higher values indicate higher ranks within the network.
        """
        return self.ranks

    @property
    def I(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Incentive values of neurons represent the rewards they receive for their contributions to the network.
        The Bittensor network employs an incentive mechanism that rewards neurons based on their
        informational value, stake, and consensus with other peers. This ensures that the most valuable and
        trusted contributions are incentivized.

        Returns:
            NDArray: A tensor of incentive values, indicating the rewards or benefits accrued by each neuron based on their contributions and network consensus.
        """
        return self.incentive

    @property
    def E(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Denotes the emission values of neurons in the Bittensor network. Emissions refer to the distribution or
        release of rewards (often in the form of cryptocurrency) to neurons, typically based on their stake and
        performance. This mechanism is central to the network's incentive model, ensuring that active and
        contributing neurons are appropriately rewarded.

        Returns:
            NDArray: A tensor where each element represents the emission value for a neuron, indicating the amount of reward distributed to that neuron.
        """
        return self.emission

    @property
    def C(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Represents the consensus values of neurons in the Bittensor network. Consensus is a measure of how
        much a neuron's contributions are trusted and agreed upon by the majority of the network. It is
        calculated based on a staked weighted trust system, where the network leverages the collective
        judgment of all participating peers. Higher consensus values indicate that a neuron's contributions
        are more widely trusted and valued across the network.

        Returns:
            NDArray: A tensor of consensus values, where each element reflects the level of trust and agreement a neuron has achieved within the network.

        """
        return self.consensus

    @property
    def T(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Represents the trust values assigned to each neuron in the Bittensor network. Trust is a key metric that
        reflects the reliability and reputation of a neuron based on its past behavior and contributions. It is
        an essential aspect of the network's functioning, influencing decision-making processes and interactions
        between neurons.

        The trust matrix is inferred from the network's inter-peer weights, indicating the level of trust each neuron
        has in others. A higher value in the trust matrix suggests a stronger trust relationship between neurons.

        Returns:
            NDArray: A tensor of trust values, where each element represents the trust level of a neuron. Higher values denote a higher level of trust within the network.
        """
        return self.trust

    @property
    def Tv(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Contains the validator trust values of neurons in the Bittensor network. Validator trust is specifically
        associated with neurons that act as validators within the network. This specialized form of trust reflects
        the validators' reliability and integrity in their role, which is crucial for maintaining the network's
        stability and security.

        Validator trust values are particularly important for the network's consensus and validation processes,
        determining the validators' influence and responsibilities in these critical functions.

        Returns:
            NDArray: A tensor of validator trust values, specifically applicable to neurons serving as validators, where higher values denote greater trustworthiness in their validation roles.
        """
        return self.validator_trust

    @property
    def D(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Represents the dividends received by neurons in the Bittensor network. Dividends are a form of reward or
        distribution, typically given to neurons based on their stake, performance, and contribution to the network.
        They are an integral part of the network's incentive structure, encouraging active and beneficial participation.

        Returns:
            NDArray: A tensor of dividend values, where each element indicates the dividends received by a neuron, reflecting their share of network rewards.
        """
        return self.dividends

    @property
    def B(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Bonds in the Bittensor network represent a speculative reward mechanism where neurons can accumulate
        bonds in other neurons. Bonds are akin to investments or stakes in other neurons, reflecting a belief in
        their future value or performance. This mechanism encourages correct weighting and collaboration
        among neurons while providing an additional layer of incentive.

        Returns:
            NDArray: A tensor representing the bonds held by each neuron, where each value signifies the proportion of bonds owned by one neuron in another.
        """
        return self.bonds

    @property
    def W(self) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Represents the weights assigned to each neuron in the Bittensor network. In the context of Bittensor,
        weights are crucial for determining the influence and interaction between neurons. Each neuron is responsible
        for setting its weights, which are then recorded on a digital ledger. These weights are reflective of the
        neuron's assessment or judgment of other neurons in the network.

        The weight matrix :math:`W = [w_{ij}]` is a key component of the network's architecture, where the :math:`i^{th}` row is set by
        neuron :math:`i` and represents its weights towards other neurons. These weights influence the ranking and incentive
        mechanisms within the network. Higher weights from a neuron towards another can imply greater trust or value
        placed on that neuron's contributions.

        Returns:
            NDArray: A tensor of inter-peer weights, where each element :math:`w_{ij}` represents the weight assigned by neuron :math:`i` to neuron :math:`j`. This matrix is fundamental to the network's functioning, influencing the distribution of incentives and the inter-neuronal dynamics.
        """
        return self.weights

    @property
    def hotkeys(self) -> list[str]:
        """
        Represents a list of ``hotkeys`` for each neuron in the Bittensor network.

        Hotkeys are unique identifiers used by neurons for active participation in the network, such as sending and receiving information or
        transactions. They are akin to public keys in cryptographic systems and are essential for identifying and authenticating neurons within the network's operations.

        Returns:
            List[str]: A list of hotkeys, with each string representing the hotkey of a corresponding neuron.

            These keys are crucial for the network's security and integrity, ensuring proper identification and authorization of network participants.

        Note:
            While the `NeurIPS paper <https://bittensor.com/pdfs/academia/NeurIPS_DAO_Workshop_2022_3_3.pdf>`_ may not explicitly detail the concept of hotkeys, they are a fundamental  of decentralized networks for secure and authenticated interactions.
        """
        return [axon.hotkey for axon in self.axons]

    @property
    def coldkeys(self) -> list[str]:
        """
        Contains a list of ``coldkeys`` for each neuron in the Bittensor network.

        Coldkeys are similar to hotkeys but are typically used for more secure, offline activities such as storing assets or offline signing of transactions. They are an important aspect of a neuron's security, providing an additional layer of protection for sensitive operations and assets.

        Returns:
            List[str]: A list of coldkeys, each string representing the coldkey of a neuron. These keys play a vital role in the secure management of assets and sensitive operations within the network.

        Note:
            The concept of coldkeys, while not explicitly covered in the NeurIPS paper, is a standard practice in
            blockchain and decentralized networks for enhanced security and asset protection.
        """
        return [axon.coldkey for axon in self.axons]

    @property
    def addresses(self) -> list[str]:
        """
        Provides a list of IP addresses for each neuron in the Bittensor network. These addresses are used for
        network communication, allowing neurons to connect, interact, and exchange information with each other.
        IP addresses are fundamental for the network's peer-to-peer communication infrastructure.

        Returns:
            List[str]: A list of IP addresses, with each string representing the address of a neuron. These addresses enable the decentralized, distributed nature of the network, facilitating direct communication and data exchange among neurons.

        Note:
            While IP addresses are a basic aspect of network communication, specific details about their use in
            the Bittensor network may not be covered in the `NeurIPS paper <https://bittensor.com/pdfs/academia/NeurIPS_DAO_Workshop_2022_3_3.pdf>`_. They are, however, integral to the
            functioning of any distributed network.
        """
        return [axon.ip_str() for axon in self.axons]

    @abstractmethod
    def __init__(
        self,
        netuid: int,
        network: str = settings.DEFAULT_NETWORK,
        lite: bool = True,
        sync: bool = True,
        subtensor: "Subtensor" = None,
    ):
        """
        Initializes a new instance of the metagraph object, setting up the basic structure and parameters based on the provided arguments.
        This method is the entry point for creating a metagraph object,
        which is a central component in representing the state of the Bittensor network.

        Args:
            netuid (int): The unique identifier for the network, distinguishing this instance of the metagraph within potentially multiple network configurations.
            network (str): The name of the network, which can indicate specific configurations or versions of the Bittensor network.
            lite (bool): A flag indicating whether to use a lite version of the metagraph. The lite version may contain less detailed information but can be quicker to initialize and sync.
            sync (bool): A flag indicating whether to synchronize the metagraph with the network upon initialization. Synchronization involves updating the metagraph's parameters to reflect the current state of the network.

        Example:
            Initializing a metagraph object for the Bittensor network with a specific network UID::

                metagraph = metagraph(netuid=123, network="finney", lite=True, sync=True)

        """

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the metagraph object. This representation includes key identifiers and attributes of the metagraph, making it easier to quickly understand
        the state and configuration of the metagraph in a simple format.

        Returns:
            str: A string that succinctly represents the metagraph, including its network UID, the total number of neurons (n), the current block number, and the network's name. This format is particularly useful for logging, debugging, and displaying the metagraph in a concise manner.

        Example:
            When printing the metagraph object or using it in a string context, this method is automatically invoked::

                print(metagraph)  # Output: "metagraph(netuid:1, n:100, block:500, network:finney)"
        """
        return f"metagraph(netuid:{self.netuid}, n:{self.n.item()}, block:{self.block.item()}, network:{self.network})"

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the metagraph object, intended for unambiguous understanding and debugging purposes. This method simply calls the :func:`__str__` method, ensuring
        consistency between the informal and formal string representations of the metagraph.

        Returns:
            str: The same string representation as provided by the :func:`__str__` method, detailing the metagraph's key attributes including network UID, number of neurons, block number, and network name.

        Example:
            The :func:`__repr__` output can be used in debugging to get a clear and concise description of the metagraph::

                metagraph_repr = repr(metagraph)
                print(metagraph_repr)  # Output mirrors that of __str__
        """
        return self.__str__()

    def metadata(self) -> dict:
        """
        Retrieves the metadata of the metagraph, providing key information about the current state of the
        Bittensor network. This metadata includes details such as the network's unique identifier (``netuid``),
        the total number of neurons (``n``), the current block number, the network's name, and the version of
        the Bittensor network.

        Returns:
            dict: A dictionary containing essential metadata about the metagraph, including:

            - ``netuid``: The unique identifier for the network.
            - ``n``: The total number of neurons in the network.
            - ``block``: The current block number in the network's blockchain.
            - ``network``: The name of the Bittensor network.
            - ``version``: The version number of the Bittensor software.

        Note:
            This metadata is crucial for understanding the current state and configuration of the network, as well as for tracking its evolution over time.
        """
        return {
            "netuid": self.netuid,
            "n": self.n.item(),
            "block": self.block.item(),
            "network": self.network,
            "version": settings.__version__,
        }

    def state_dict(self):
        return {
            "netuid": self.netuid,
            "network": self.network,
            "version": self.version,
            "n": self.n,
            "block": self.block,
            "stake": self.stake,
            "total_stake": self.total_stake,
            "ranks": self.ranks,
            "trust": self.trust,
            "consensus": self.consensus,
            "validator_trust": self.validator_trust,
            "incentive": self.incentive,
            "emission": self.emission,
            "dividends": self.dividends,
            "active": self.active,
            "last_update": self.last_update,
            "validator_permit": self.validator_permit,
            "weights": self.weights,
            "bonds": self.bonds,
            "uids": self.uids,
            "axons": self.axons,
            "neurons": self.neurons,
        }

    def sync(
        self,
        block: Optional[int] = None,
        lite: bool = True,
        subtensor: Optional["Subtensor"] = None,
    ):
        """
        Synchronizes the metagraph with the Bittensor network's current state. It updates the metagraph's attributes to reflect the latest data from the network, ensuring the metagraph represents the most current state of the network.

        Args:
            block (Optional[int]): A specific block number to synchronize with. If None, the metagraph syncs with the latest block. This allows for historical analysis or specific state examination of the network.
            lite (bool): If True, a lite version of the metagraph is used for quicker synchronization. This is beneficial when full detail is not necessary, allowing for reduced computational and time overhead.
            subtensor (Optional[bittensor.core.subtensor.Subtensor]): An instance of the subtensor class from Bittensor, providing an interface to the underlying blockchain data. If provided, this instance is used for data retrieval during synchronization.

        Example:
            Sync the metagraph with the latest block from the subtensor, using the lite version for efficiency::

                from bittensor.core.subtensor import Subtensor

                subtensor = Subtensor()
                metagraph.sync(subtensor=subtensor)

            Sync with a specific block number for detailed analysis::

                from bittensor.core.subtensor import Subtensor

                subtensor = Subtensor()
                metagraph.sync(block=12345, lite=False, subtensor=subtensor)

        NOTE:
            If attempting to access data beyond the previous 300 blocks, you **must** use the ``archive`` network for subtensor. Light nodes are configured only to store the previous 300 blocks if connecting to finney or test networks.

            For example::

                from bittensor.core.subtensor import Subtensor

                subtensor = Subtensor(network='archive')
                current_block = subtensor.get_current_block()
                history_block = current_block - 1200

                metagraph.sync(block=history_block, lite=False, subtensor=subtensor)
        """

        # Initialize subtensor
        subtensor = self._initialize_subtensor(subtensor)

        if (
            subtensor.chain_endpoint != settings.ARCHIVE_ENTRYPOINT
            or subtensor.network != "archive"
        ):
            cur_block = subtensor.get_current_block()
            if block and block < (cur_block - 300):
                logging.warning(
                    "Attempting to sync longer than 300 blocks ago on a non-archive node. Please use the 'archive' "
                    "network for subtensor and retry."
                )

        # Assign neurons based on 'lite' flag
        self._assign_neurons(block, lite, subtensor)

        # Set attributes for metagraph
        self._set_metagraph_attributes(block, subtensor)

        # If not a 'lite' version, compute and set weights and bonds for each neuron
        if not lite:
            self._set_weights_and_bonds(subtensor=subtensor)

    def _initialize_subtensor(self, subtensor: "Subtensor"):
        """
        Initializes the subtensor to be used for syncing the metagraph.

        This method ensures that a subtensor instance is available and properly set up for data retrieval during the synchronization process.

        If no subtensor is provided, this method is responsible for creating a new instance of the subtensor, configured according to the current network settings.

        Args:
            subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance provided for initialization. If ``None``, a new subtensor instance is created using the current network configuration.

        Returns:
            subtensor (bittensor.core.subtensor.Subtensor): The initialized subtensor instance, ready to be used for syncing the metagraph.

        Internal Usage:
            Used internally during the sync process to ensure a valid subtensor instance is available::

                subtensor = self._initialize_subtensor(subtensor)
        """
        if subtensor and subtensor != self.subtensor:
            self.subtensor = subtensor
        if not subtensor and self.subtensor:
            subtensor = self.subtensor
        if not subtensor:
            # TODO: Check and test the initialization of the new subtensor
            # Lazy import due to circular import (subtensor -> metagraph, metagraph -> subtensor)
            from bittensor.core.subtensor import Subtensor

            subtensor = Subtensor(network=self.chain_endpoint)
            self.subtensor = subtensor
        return subtensor

    def _assign_neurons(self, block: int, lite: bool, subtensor: "Subtensor"):
        """
        Assigns neurons to the metagraph based on the provided block number and the lite flag.

        This method is responsible for fetching and setting the neuron data in the metagraph, which includes neuron attributes like UID, stake, trust, and other relevant information.

        Args:
            block (int): The block number for which the neuron data needs to be fetched. If ``None``, the latest block data is used.
            lite (bool): A boolean flag indicating whether to use a lite version of the neuron data. The lite version typically includes essential information and is quicker to fetch and process.
            subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for fetching neuron data from the network.

        Internal Usage:
            Used internally during the sync process to fetch and set neuron data::

                from bittensor.core.subtensor import Subtensor

                block = 12345
                lite = False
                subtensor = Subtensor()
                self._assign_neurons(block, lite, subtensor)
        """
        if lite:
            self.neurons = subtensor.neurons_lite(block=block, netuid=self.netuid)
        else:
            self.neurons = subtensor.neurons(block=block, netuid=self.netuid)
        self.lite = lite

    @staticmethod
    def _create_tensor(data, dtype) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Creates a numpy array with the given data and data type. This method is a utility function used internally to encapsulate data into a np.array, making it compatible with the metagraph's numpy model structure.

        Args:
            data: The data to be included in the tensor. This could be any numeric data, like stakes, ranks, etc.
            dtype: The data type for the tensor, typically a numpy data type like ``np.float32`` or ``np.int64``.

        Returns:
            A tensor parameter encapsulating the provided data.

        Internal Usage:
            Used internally to create tensor parameters for various metagraph attributes::

                self.stake = self._create_tensor(neuron_stakes, dtype=np.float32)
        """
        # TODO: Check and test the creation of tensor
        return (
            torch.nn.Parameter(torch.tensor(data, dtype=dtype), requires_grad=False)
            if use_torch()
            else np.array(data, dtype=dtype)
        )

    def _set_weights_and_bonds(self, subtensor: "Optional[Subtensor]" = None):
        """
        Computes and sets the weights and bonds for each neuron in the metagraph. This method is responsible for processing the raw weight and bond data obtained from the network and converting it into a structured format suitable for the metagraph model.

        Args:
            subtensor: The subtensor instance used for fetching weights and bonds data. If ``None``, the weights and bonds are not updated.

        Internal Usage:
            Used internally during the sync process to update the weights and bonds of the neurons::

                self._set_weights_and_bonds(subtensor=subtensor)
        """
        # TODO: Check and test the computation of weights and bonds
        if self.netuid == 0:
            self.weights = self._process_root_weights(
                [neuron.weights for neuron in self.neurons],
                "weights",
                subtensor,
            )
        else:
            self.weights = self._process_weights_or_bonds(
                [neuron.weights for neuron in self.neurons], "weights"
            )
            self.bonds = self._process_weights_or_bonds(
                [neuron.bonds for neuron in self.neurons], "bonds"
            )

    def _process_weights_or_bonds(
        self, data, attribute: str
    ) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Processes the raw weights or bonds data and converts it into a structured tensor format. This method handles the transformation of neuron connection data (``weights`` or ``bonds``) from a list or other unstructured format into a tensor that can be utilized within the metagraph model.

        Args:
            data: The raw weights or bonds data to be processed. This data typically comes from the subtensor.
            attribute: A string indicating whether the data is ``weights`` or ``bonds``, which determines the specific processing steps to be applied.

        Returns:
            A tensor parameter encapsulating the processed weights or bonds data.

        Internal Usage:
            Used internally to process and set weights or bonds for the neurons::

                self.weights = self._process_weights_or_bonds(raw_weights_data, "weights")
        """
        data_array = []
        for item in data:
            if len(item) == 0:
                if use_torch():
                    data_array.append(torch.zeros(len(self.neurons)))
                else:
                    data_array.append(np.zeros(len(self.neurons), dtype=np.float32))
            else:
                uids, values = zip(*item)
                # TODO: Validate and test the conversion of uids and values to tensor
                if attribute == "weights":
                    data_array.append(
                        convert_weight_uids_and_vals_to_tensor(
                            len(self.neurons),
                            list(uids),
                            list(values),
                        )
                    )
                else:
                    data_array.append(
                        convert_bond_uids_and_vals_to_tensor(
                            len(self.neurons), list(uids), list(values)
                        ).astype(np.float32)
                    )
        tensor_param: Union["torch.nn.Parameter", NDArray] = (
            (
                torch.nn.Parameter(torch.stack(data_array), requires_grad=False)
                if len(data_array)
                else torch.nn.Parameter()
            )
            if use_torch()
            else (
                np.stack(data_array)
                if len(data_array)
                else np.array([], dtype=np.float32)
            )
        )
        if len(data_array) == 0:
            logging.warning(
                f"Empty {attribute}_array on metagraph.sync(). The '{attribute}' tensor is empty."
            )
        return tensor_param

    @abstractmethod
    def _set_metagraph_attributes(self, block, subtensor):
        pass

    def _process_root_weights(
        self, data: list, attribute: str, subtensor: "Subtensor"
    ) -> Union[NDArray, "torch.nn.Parameter"]:
        """
        Specifically processes the root weights data for the metagraph. This method is similar to :func:`_process_weights_or_bonds` but is tailored for processing root weights, which have a different structure and significance in the network.

        Args:
            data (list): The raw root weights data to be processed.
            attribute (str): A string indicating the attribute type, here it's typically ``weights``.
            subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for additional data and context needed in processing.

        Returns:
            A tensor parameter encapsulating the processed root weights data.

        Internal Usage:
            Used internally to process and set root weights for the metagraph::

                self.root_weights = self._process_root_weights(raw_root_weights_data, "weights", subtensor)
        """
        data_array = []
        n_subnets = subtensor.get_total_subnets() or 0
        subnets = subtensor.get_subnets()
        for item in data:
            if len(item) == 0:
                if use_torch():
                    data_array.append(torch.zeros(n_subnets))
                else:
                    data_array.append(np.zeros(n_subnets, dtype=np.float32))
            else:
                uids, values = zip(*item)
                # TODO: Validate and test the conversion of uids and values to tensor
                data_array.append(
                    convert_root_weight_uids_and_vals_to_tensor(
                        n_subnets, list(uids), list(values), subnets
                    )
                )

        tensor_param: Union[NDArray, "torch.nn.Parameter"] = (
            (
                torch.nn.Parameter(torch.stack(data_array), requires_grad=False)
                if len(data_array)
                else torch.nn.Parameter()
            )
            if use_torch()
            else (
                np.stack(data_array)
                if len(data_array)
                else np.array([], dtype=np.float32)
            )
        )
        if len(data_array) == 0:
            logging.warning(
                f"Empty {attribute}_array on metagraph.sync(). The '{attribute}' tensor is empty."
            )
        return tensor_param

    def save(self, root_dir: Optional[list[str]]) -> "Metagraph":
        """
        Saves the current state of the metagraph to a file on disk. This function is crucial for persisting the current state of the network's metagraph, which can later be reloaded or analyzed. The save operation includes all neuron attributes and parameters, ensuring a complete snapshot of the metagraph's state.

        Args:
            root_dir: list to the file path for the root directory of your metagraph saves
                (i.e. ['/', 'tmp', 'metagraphs'], defaults to ["~", ".bittensor", "metagraphs"]

        Returns:
            metagraph (bittensor.core.metagraph.Metagraph): The metagraph instance after saving its state.

        Example:
            Save the current state of the metagraph to the default directory::

                metagraph.save()

            The saved state can later be loaded to restore or analyze the metagraph's state at this point.

            If using the default save path::

                metagraph.load()

            If using a custom save path::

                metagraph.load_from_path(dir_path)
        """
        save_directory = get_save_dir(self.network, self.netuid, root_dir=root_dir)
        os.makedirs(save_directory, exist_ok=True)
        if use_torch():
            graph_filename = f"{save_directory}/block-{self.block.item()}.pt"
            state_dict = self.state_dict()
            state_dict["axons"] = self.axons
            state_dict["neurons"] = self.neurons
            torch.save(state_dict, graph_filename)
            torch.load(graph_filename)  # verifies that the file can be loaded correctly
        else:
            graph_filename = f"{save_directory}/block-{self.block.item()}.pt"
            state_dict = self.state_dict()
            with open(graph_filename, "wb") as graph_file:
                pickle.dump(state_dict, graph_file)
        return self

    def load(self, root_dir: Optional[list[str]]) -> None:
        """
        Loads the state of the metagraph from the default save directory. This method is instrumental for restoring the metagraph to its last saved state. It automatically identifies the save directory based on the ``network`` and ``netuid`` properties of the metagraph, locates the latest block file in that directory, and loads all metagraph parameters from it.

        This functionality is particularly beneficial when continuity in the state of the metagraph is necessary
        across different runtime sessions, or after a restart of the system. It ensures that the metagraph reflects
        the exact state it was in at the last save point, maintaining consistency in the network's representation.

        The method delegates to ``load_from_path``, supplying it with the directory path constructed from the metagraph's current ``network`` and ``netuid`` properties. This abstraction simplifies the process of loading the metagraph's state for the user, requiring no direct path specifications.

        Args:
            root_dir: list to the file path for the root directory of your metagraph saves
                (i.e. ['/', 'tmp', 'metagraphs'], defaults to ["~", ".bittensor", "metagraphs"]

        Returns:
            metagraph (bittensor.core.metagraph.Metagraph): The metagraph instance after loading its state from the default directory.

        Example:
            Load the metagraph state from the last saved snapshot in the default directory::

                metagraph.load()

            After this operation, the metagraph's parameters and neuron data are restored to their state at the time of the last save in the default directory.

        Note:
            The default save directory is determined based on the metagraph's ``network`` and ``netuid`` attributes. It is important to ensure that these attributes are set correctly and that the default save directory contains the appropriate state files for the metagraph.
        """
        self.load_from_path(get_save_dir(self.network, self.netuid, root_dir=root_dir))

    @abstractmethod
    def load_from_path(self, dir_path: str) -> "Metagraph":
        """
        Loads the state of the metagraph from a specified directory path. This method is crucial for restoring the metagraph to a specific state based on saved data. It locates the latest block file in the given
        directory and loads all metagraph parameters from it. This is particularly useful for analyses that require historical states of the network or for restoring previous states of the metagraph in different
        execution environments.

        The method first identifies the latest block file in the specified directory, then loads the metagraph state including neuron attributes and parameters from this file. This ensures that the metagraph is accurately reconstituted to reflect the network state at the time of the saved block.

        Args:
            dir_path (str): The directory path where the metagraph's state files are stored. This path should contain one or more saved state files, typically named in a format that includes the block number.

        Returns:
            metagraph (bittensor.core.metagraph.Metagraph): The metagraph instance after loading its state from the specified directory path.

        Example:
            Load the metagraph state from a specific directory::

                dir_path = "/path/to/saved/metagraph/states"
                metagraph.load_from_path(dir_path)

            The metagraph is now restored to the state it was in at the time of the latest saved block in the specified directory.

        Note:
            This method assumes that the state files in the specified directory are correctly formatted and
            contain valid data for the metagraph. It is essential to ensure that the directory path and the
            state files within it are accurate and consistent with the expected metagraph structure.
        """

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance

        for key, value in self.__dict__.items():
            if key == "subtensor":
                setattr(new_instance, key, None)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))

        return new_instance

    def __copy__(self):
        cls = self.__class__
        new_instance = cls.__new__(cls)

        for key, value in self.__dict__.items():
            if key == "subtensor":
                setattr(new_instance, key, None)
            else:
                setattr(new_instance, key, value)
        return new_instance


BaseClass: Union["torch.nn.Module", object] = torch.nn.Module if use_torch() else object
"""
Base class that extends :class:`torch.nn.Module` if PyTorch is used; otherwise, it defaults to object.
"""


class TorchMetaGraph(MetagraphMixin, BaseClass):
    def __init__(
        self,
        netuid: int,
        network: str = settings.DEFAULT_NETWORK,
        lite: bool = True,
        sync: bool = True,
        subtensor: "Subtensor" = None,
    ):
        """
        Initializes a new instance of the metagraph object, setting up the basic structure and parameters based on the provided arguments.
        This class requires Torch to be installed.
        This method is the entry point for creating a metagraph object, which is a central component in representing the state of the Bittensor network.

        Args:
            netuid (int): The unique identifier for the network, distinguishing this instance of the metagraph within potentially multiple network configurations.
            network (str): The name of the network, which can indicate specific configurations or versions of the Bittensor network.
            lite (bool): A flag indicating whether to use a lite version of the metagraph. The lite version may contain less detailed information but can be quicker to initialize and sync.
            sync (bool): A flag indicating whether to synchronize the metagraph with the network upon initialization. Synchronization involves updating the metagraph's parameters to reflect the current state of the network.

        Example:
            Initializing a metagraph object for the Bittensor network with a specific network UID::

                from bittensor.core.metagraph import Metagraph

                metagraph = Metagraph(netuid=123, network="finney", lite=True, sync=True)
        """
        torch.nn.Module.__init__(self)
        MetagraphMixin.__init__(self, netuid, network, lite, sync, subtensor)
        self.netuid = netuid
        self.network, self.chain_endpoint = determine_chain_endpoint_and_network(
            network
        )
        self.version = torch.nn.Parameter(
            torch.tensor([settings.version_as_int], dtype=torch.int64),
            requires_grad=False,
        )
        self.n: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.block: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.stake = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.total_stake: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.ranks: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.trust: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.consensus: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.validator_trust: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.incentive: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.emission: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.dividends: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.active = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.last_update = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.validator_permit = torch.nn.Parameter(
            torch.tensor([], dtype=torch.bool), requires_grad=False
        )
        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.bonds: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.uids = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.axons: list[AxonInfo] = []
        self.subtensor = subtensor
        if sync:
            self.sync(block=None, lite=lite, subtensor=subtensor)

    def _set_metagraph_attributes(self, block: int, subtensor: "Subtensor"):
        """
        Sets various attributes of the metagraph based on the latest network data fetched from the subtensor.

        This method updates parameters like the number of neurons, block number, stakes, trusts, ranks, and other neuron-specific information.

        Args:
            block (int): The block number for which the metagraph attributes need to be set. If ``None``, the latest block data is used.
            subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for fetching the latest network data.

        Internal Usage:
            Used internally during the sync process to update the metagraph's attributes::

                from bittensor.core.subtensor import Subtensor

                subtensor = Subtensor()
                block = subtensor.get_current_block()

                self._set_metagraph_attributes(block, subtensor)
        """
        self.n = self._create_tensor(len(self.neurons), dtype=torch.int64)
        self.version = self._create_tensor([settings.version_as_int], dtype=torch.int64)
        self.block = self._create_tensor(
            block if block else subtensor.block, dtype=torch.int64
        )
        self.uids = self._create_tensor(
            [neuron.uid for neuron in self.neurons], dtype=torch.int64
        )
        self.trust = self._create_tensor(
            [neuron.trust for neuron in self.neurons], dtype=torch.float32
        )
        self.consensus = self._create_tensor(
            [neuron.consensus for neuron in self.neurons], dtype=torch.float32
        )
        self.incentive = self._create_tensor(
            [neuron.incentive for neuron in self.neurons], dtype=torch.float32
        )
        self.dividends = self._create_tensor(
            [neuron.dividends for neuron in self.neurons], dtype=torch.float32
        )
        self.ranks = self._create_tensor(
            [neuron.rank for neuron in self.neurons], dtype=torch.float32
        )
        self.emission = self._create_tensor(
            [neuron.emission for neuron in self.neurons], dtype=torch.float32
        )
        self.active = self._create_tensor(
            [neuron.active for neuron in self.neurons], dtype=torch.int64
        )
        self.last_update = self._create_tensor(
            [neuron.last_update for neuron in self.neurons], dtype=torch.int64
        )
        self.validator_permit = self._create_tensor(
            [neuron.validator_permit for neuron in self.neurons], dtype=torch.bool
        )
        self.validator_trust = self._create_tensor(
            [neuron.validator_trust for neuron in self.neurons], dtype=torch.float32
        )
        self.total_stake = self._create_tensor(
            [neuron.total_stake.tao for neuron in self.neurons], dtype=torch.float32
        )
        self.stake = self._create_tensor(
            [neuron.stake for neuron in self.neurons], dtype=torch.float32
        )
        self.axons = [n.axon_info for n in self.neurons]

    def load_from_path(self, dir_path: str) -> "Metagraph":
        """
        Loads the metagraph state from a specified directory path.

        Args:
            dir_path (str): The directory path where the state file is located.

        Returns:
            metagraph (bittensor.core.metagraph.Metagraph): The current metagraph instance with the loaded state.

        Example::

            from bittensor.core.metagraph import Metagraph

            netuid = 1
            metagraph = Metagraph(netuid=netuid)

            metagraph.load_from_path("/path/to/dir")

        """

        graph_file = latest_block_path(dir_path)
        state_dict = torch.load(graph_file)
        self.n = torch.nn.Parameter(state_dict["n"], requires_grad=False)
        self.block = torch.nn.Parameter(state_dict["block"], requires_grad=False)
        self.uids = torch.nn.Parameter(state_dict["uids"], requires_grad=False)
        self.stake = torch.nn.Parameter(state_dict["stake"], requires_grad=False)
        self.total_stake = torch.nn.Parameter(
            state_dict["total_stake"], requires_grad=False
        )
        self.ranks = torch.nn.Parameter(state_dict["ranks"], requires_grad=False)
        self.trust = torch.nn.Parameter(state_dict["trust"], requires_grad=False)
        self.consensus = torch.nn.Parameter(
            state_dict["consensus"], requires_grad=False
        )
        self.validator_trust = torch.nn.Parameter(
            state_dict["validator_trust"], requires_grad=False
        )
        self.incentive = torch.nn.Parameter(
            state_dict["incentive"], requires_grad=False
        )
        self.emission = torch.nn.Parameter(state_dict["emission"], requires_grad=False)
        self.dividends = torch.nn.Parameter(
            state_dict["dividends"], requires_grad=False
        )
        self.active = torch.nn.Parameter(state_dict["active"], requires_grad=False)
        self.last_update = torch.nn.Parameter(
            state_dict["last_update"], requires_grad=False
        )
        self.validator_permit = torch.nn.Parameter(
            state_dict["validator_permit"], requires_grad=False
        )
        self.uids = torch.nn.Parameter(state_dict["uids"], requires_grad=False)
        self.axons = state_dict["axons"]
        self.neurons = state_dict["neurons"]
        if "weights" in state_dict:
            self.weights = torch.nn.Parameter(
                state_dict["weights"], requires_grad=False
            )
        if "bonds" in state_dict:
            self.bonds = torch.nn.Parameter(state_dict["bonds"], requires_grad=False)
        return self


class NonTorchMetagraph(MetagraphMixin):
    def __init__(
        self,
        netuid: int,
        network: str = settings.DEFAULT_NETWORK,
        lite: bool = True,
        sync: bool = True,
        subtensor: "Subtensor" = None,
    ):
        """
        Initializes a new instance of the metagraph object, setting up the basic structure and parameters based on the provided arguments.
        This class doesn't require installed Torch.
        This method is the entry point for creating a metagraph object, which is a central component in representing the state of the Bittensor network.

        Args:
            netuid (int): The unique identifier for the network, distinguishing this instance of the metagraph within potentially multiple network configurations.
            network (str): The name of the network, which can indicate specific configurations or versions of the Bittensor network.
            lite (bool): A flag indicating whether to use a lite version of the metagraph. The lite version may contain less detailed information but can be quicker to initialize and sync.
            sync (bool): A flag indicating whether to synchronize the metagraph with the network upon initialization. Synchronization involves updating the metagraph's parameters to reflect the current state of the network.

        Example:
            Initializing a metagraph object for the Bittensor network with a specific network UID::

                from bittensor.core.metagraph import Metagraph

                metagraph = Metagraph(netuid=123, network="finney", lite=True, sync=True)
        """
        # super(metagraph, self).__init__()
        MetagraphMixin.__init__(self, netuid, network, lite, sync, subtensor)

        self.netuid = netuid
        self.network, self.chain_endpoint = determine_chain_endpoint_and_network(
            network
        )
        self.version = (np.array([settings.version_as_int], dtype=np.int64),)
        self.n = np.array([0], dtype=np.int64)
        self.block = np.array([0], dtype=np.int64)
        self.stake = np.array([], dtype=np.float32)
        self.total_stake = np.array([], dtype=np.float32)
        self.ranks = np.array([], dtype=np.float32)
        self.trust = np.array([], dtype=np.float32)
        self.consensus = np.array([], dtype=np.float32)
        self.validator_trust = np.array([], dtype=np.float32)
        self.incentive = np.array([], dtype=np.float32)
        self.emission = np.array([], dtype=np.float32)
        self.dividends = np.array([], dtype=np.float32)
        self.active = np.array([], dtype=np.int64)
        self.last_update = np.array([], dtype=np.int64)
        self.validator_permit = np.array([], dtype=bool)
        self.weights = np.array([], dtype=np.float32)
        self.bonds = np.array([], dtype=np.int64)
        self.uids = np.array([], dtype=np.int64)
        self.axons: list[AxonInfo] = []
        self.subtensor = subtensor
        if sync:
            self.sync(block=None, lite=lite, subtensor=subtensor)

    def _set_metagraph_attributes(self, block: int, subtensor: "Subtensor"):
        """
        Sets various attributes of the metagraph based on the latest network data fetched from the subtensor.

        This method updates parameters like the number of neurons, block number, stakes, trusts, ranks, and other neuron-specific information.

        Args:
            block (int): The block number for which the metagraph attributes need to be set. If ``None``, the latest block data is used.
            subtensor (bittensor.core.subtensor.Subtensor): The subtensor instance used for fetching the latest network data.

        Internal Usage:
            Used internally during the sync process to update the metagraph's attributes::

                self._set_metagraph_attributes(block, subtensor)
        """
        # TODO: Check and test the setting of each attribute
        self.n = self._create_tensor(len(self.neurons), dtype=np.int64)
        self.version = self._create_tensor([settings.version_as_int], dtype=np.int64)
        self.block = self._create_tensor(
            block if block else subtensor.block, dtype=np.int64
        )
        self.uids = self._create_tensor(
            [neuron.uid for neuron in self.neurons], dtype=np.int64
        )
        self.trust = self._create_tensor(
            [neuron.trust for neuron in self.neurons], dtype=np.float32
        )
        self.consensus = self._create_tensor(
            [neuron.consensus for neuron in self.neurons], dtype=np.float32
        )
        self.incentive = self._create_tensor(
            [neuron.incentive for neuron in self.neurons], dtype=np.float32
        )
        self.dividends = self._create_tensor(
            [neuron.dividends for neuron in self.neurons], dtype=np.float32
        )
        self.ranks = self._create_tensor(
            [neuron.rank for neuron in self.neurons], dtype=np.float32
        )
        self.emission = self._create_tensor(
            [neuron.emission for neuron in self.neurons], dtype=np.float32
        )
        self.active = self._create_tensor(
            [neuron.active for neuron in self.neurons], dtype=np.int64
        )
        self.last_update = self._create_tensor(
            [neuron.last_update for neuron in self.neurons], dtype=np.int64
        )
        self.validator_permit = self._create_tensor(
            [neuron.validator_permit for neuron in self.neurons], dtype=bool
        )
        self.validator_trust = self._create_tensor(
            [neuron.validator_trust for neuron in self.neurons], dtype=np.float32
        )
        self.total_stake = self._create_tensor(
            [neuron.total_stake.tao for neuron in self.neurons], dtype=np.float32
        )
        self.stake = self._create_tensor(
            [neuron.stake for neuron in self.neurons], dtype=np.float32
        )
        self.axons = [n.axon_info for n in self.neurons]

    def load_from_path(self, dir_path: str) -> "Metagraph":
        """
        Loads the state of the Metagraph from a specified directory path.

        Args:
            dir_path (str): The directory path where the metagraph's state file is located.

        Returns:
            metagraph (:func:`bittensor.core.metagraph.Metagraph`): An instance of the Metagraph with the state loaded from the file.

        Raises:
            pickle.UnpicklingError: If there is an error unpickling the state file.
            RuntimeError: If there is an error loading the state file using PyTorch.
            ImportError: If there is an error importing PyTorch.
        """
        graph_filename = latest_block_path(dir_path)
        try:
            with open(graph_filename, "rb") as graph_file:
                state_dict = pickle.load(graph_file)
        except pickle.UnpicklingError:
            logging.info(
                "Unable to load file. Attempting to restore metagraph using torch."
            )
            logging.warning(
                ":warning: This functionality exists to load metagraph state from legacy saves, but will not be supported in the future."
            )
            try:
                import torch as real_torch

                state_dict = real_torch.load(graph_filename)
                for key in METAGRAPH_STATE_DICT_NDARRAY_KEYS:
                    state_dict[key] = state_dict[key].detach().numpy()
                del real_torch
            except (RuntimeError, ImportError):
                logging.error("Unable to load file. It may be corrupted.")
                raise

        self.n = state_dict["n"]
        self.block = state_dict["block"]
        self.uids = state_dict["uids"]
        self.stake = state_dict["stake"]
        self.total_stake = state_dict["total_stake"]
        self.ranks = state_dict["ranks"]
        self.trust = state_dict["trust"]
        self.consensus = state_dict["consensus"]
        self.validator_trust = state_dict["validator_trust"]
        self.incentive = state_dict["incentive"]
        self.emission = state_dict["emission"]
        self.dividends = state_dict["dividends"]
        self.active = state_dict["active"]
        self.last_update = state_dict["last_update"]
        self.validator_permit = state_dict["validator_permit"]
        self.axons = state_dict["axons"]
        self.neurons = state_dict["neurons"]
        if "weights" in state_dict:
            self.weights = state_dict["weights"]
        if "bonds" in state_dict:
            self.bonds = state_dict["bonds"]
        return self


Metagraph = TorchMetaGraph if use_torch() else NonTorchMetagraph
"""Metagraph class that uses :class:`TorchMetaGraph` if PyTorch is available; otherwise, it falls back to :class:`NonTorchMetagraph`.

- **With PyTorch**: When `use_torch()` returns `True`, `Metagraph` is set to :class:`TorchMetaGraph`, which utilizes PyTorch functionalities.
- **Without PyTorch**: When `use_torch()` returns `False`, `Metagraph` is set to :class:`NonTorchMetagraph`, which does not rely on PyTorch.
"""
