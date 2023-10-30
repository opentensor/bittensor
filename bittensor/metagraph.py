# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import bittensor
from os import listdir
from os.path import join
from typing import List, Optional


def get_save_dir(network: str, netuid: int) -> str:
    """
    Return directory path from network and netuid.

    Args:
        network (str): Network name.
        netuid (int): Network UID.

    Returns:
        str: Directory path.
    """
    return os.path.expanduser(
        f"~/.bittensor/metagraphs/network-{str(network)}/netuid-{str(netuid)}/"
    )


def latest_block_path(dir_path: str) -> int:
    """
    Get the latest block path from the directory.

    Args:
        dir_path (str): Directory path.

    Returns:
        int: Latest block path.
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
        except Exception as e:
            pass
    if not latest_file_full_path:
        raise ValueError(f"Metagraph not found at: {dir_path}")
    else:
        return latest_file_full_path


class metagraph(torch.nn.Module):
    """
    Metagraph class representing the neural network graph.

    Attributes:
        netuid (int): Network UID.
        network (str): Network name.
        version (torch.nn.Parameter): Version of the network.
        n (torch.nn.Parameter): Number of neurons in the graph.
        block (torch.nn.Parameter): Current block number.
        stake (torch.nn.Parameter): Stake of the neurons.
        total_stake (torch.nn.Parameter): Total stake of the neurons.
        ranks (torch.nn.Parameter): Ranks of the neurons.
        trust (torch.nn.Parameter): Trust values of the neurons.
        consensus (torch.nn.Parameter): Consensus values of the neurons.
        validator_trust (torch.nn.Parameter): Validator trust values of the neurons.
        incentive (torch.nn.Parameter): Incentive values of the neurons.
        emission (torch.nn.Parameter): Emission values of the neurons.
        dividends (torch.nn.Parameter): Dividends of the neurons.
        active (torch.nn.Parameter): Activation state of the neurons.
        last_update (torch.nn.Parameter): Last update time of the neurons.
        validator_permit (torch.nn.Parameter): Validator permit state of the neurons.
        weights (torch.nn.Parameter): Weights of the neurons.
        bonds (torch.nn.Parameter): Bonds of the neurons.
        uids (torch.nn.Parameter): UID values of the neurons.
        axons (List): List of axon information for the neurons.
    """

    @property
    def S(self) -> torch.FloatTensor:
        """
        Total stake of the neurons.

        Returns:
            torch.FloatTensor: Total stake.
        """
        return self.total_stake

    @property
    def R(self) -> torch.FloatTensor:
        """
        Ranks of the neurons.

        Returns:
            torch.FloatTensor: Ranks.
        """
        return self.ranks

    @property
    def I(self) -> torch.FloatTensor:
        """
        Incentive values of the neurons.

        Returns:
            torch.FloatTensor: Incentive values.
        """
        return self.incentive

    @property
    def E(self) -> torch.FloatTensor:
        """
        Emission values of the neurons.

        Returns:
            torch.FloatTensor: Emission values.
        """
        return self.emission

    @property
    def C(self) -> torch.FloatTensor:
        """
        Consensus values of the neurons.

        Returns:
            torch.FloatTensor: Consensus values.
        """
        return self.consensus

    @property
    def T(self) -> torch.FloatTensor:
        """
        Trust values of the neurons.

        Returns:
            torch.FloatTensor: Trust values.
        """
        return self.trust

    @property
    def Tv(self) -> torch.FloatTensor:
        """
        Validator trust values of the neurons.

        Returns:
            torch.FloatTensor: Validator trust values.
        """
        return self.validator_trust

    @property
    def D(self) -> torch.FloatTensor:
        """
        Dividends of the neurons.

        Returns:
            torch.FloatTensor: Dividends.
        """
        return self.dividends

    @property
    def B(self) -> torch.FloatTensor:
        """
        Bonds of the neurons.

        Returns:
            torch.FloatTensor: Bonds.
        """
        return self.bonds

    @property
    def W(self) -> torch.FloatTensor:
        """
        Weights of the neurons.

        Returns:
            torch.FloatTensor: Weights.
        """
        return self.weights

    @property
    def hotkeys(self) -> List[str]:
        """
        List of hotkeys for the neurons.

        Returns:
            List[str]: List of hotkeys.
        """
        return [axon.hotkey for axon in self.axons]

    @property
    def coldkeys(self) -> List[str]:
        """
        List of coldkeys for the neurons.

        Returns:
            List[str]: List of coldkeys.
        """
        return [axon.coldkey for axon in self.axons]

    @property
    def addresses(self) -> List[str]:
        """
        List of IP addresses for the neurons.

        Returns:
            List[str]: List of IP addresses.
        """
        return [axon.ip_str() for axon in self.axons]

    def __str__(self) -> str:
        """
        String representation of the metagraph.

        Returns:
            str: String representation.
        """
        return "metagraph(netuid:{}, n:{}, block:{}, network:{})".format(
            self.netuid, self.n.item(), self.block.item(), self.network
        )

    def __repr__(self) -> str:
        """
        String representation of the metagraph.

        Returns:
            str: String representation.
        """
        return self.__str__()

    def metadata(self) -> dict:
        """
        Get the metadata of the metagraph.

        Returns:
            dict: Metadata dictionary.
        """
        return {
            "netuid": self.netuid,
            "n": self.n.item(),
            "block": self.block.item(),
            "network": self.network,
            "version": bittensor.__version__,
        }

    def __init__(
        self, netuid: int, network: str = "finney", lite: bool = True, sync: bool = True
    ) -> "metagraph":
        """
        Initialize the metagraph object.

        Args:
            netuid (int): Network UID.
            network (str): Network name.
            lite (bool): Whether to use lite version of the metagraph.
            sync (bool): Whether to synchronize the metagraph.
        """
        super(metagraph, self).__init__()
        self.netuid = netuid
        self.network = network
        self.version = torch.nn.Parameter(
            torch.tensor([bittensor.__version_as_int__], dtype=torch.int64),
            requires_grad=False,
        )
        self.n = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.block = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.int64), requires_grad=False
        )
        self.stake = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.total_stake = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.ranks = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.trust = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.consensus = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.validator_trust = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.incentive = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.emission = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.dividends = torch.nn.Parameter(
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
        self.weights = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float32), requires_grad=False
        )
        self.bonds = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.uids = torch.nn.Parameter(
            torch.tensor([], dtype=torch.int64), requires_grad=False
        )
        self.axons = []
        if sync:
            self.sync(block=None, lite=lite)

    def sync(
        self,
        block: Optional[int] = None,
        lite: bool = True,
        subtensor: Optional["bittensor.subtensor"] = None,
    ) -> "metagraph":
        """
        Initiates the synchronization process of the metagraph.

        Args:
            block (int, optional): Block number to sync. If None, the current block is used.
            lite (bool): Whether to use lite version of the metagraph.
            subtensor (bittensor.subtensor, optional): Subtensor object to use for syncing.

        Returns:
            metagraph: Updated metagraph object.
        """
        # Initialize subtensor
        subtensor = self._initialize_subtensor(subtensor)

        # Assign neurons based on 'lite' flag
        self._assign_neurons(block, lite, subtensor)

        # Set attributes for metagraph
        self._set_metagraph_attributes(block, subtensor)

        # If not a 'lite' version, compute and set weights and bonds for each neuron
        if not lite:
            self._set_weights_and_bonds(subtensor=subtensor)

    def _initialize_subtensor(self, subtensor):
        """
        Initializes the subtensor to be used for syncing.

        Args:
            subtensor: The subtensor to initialize. If None, a new subtensor is created.

        Returns:
            subtensor: The initialized subtensor.
        """
        if not subtensor:
            # TODO: Check and test the initialization of the new subtensor
            subtensor = bittensor.subtensor(network=self.network)
        return subtensor

    def _assign_neurons(self, block, lite, subtensor):
        """
        Assigns neurons to the metagraph based on the 'lite' flag.

        Args:
            block: The block number for which the neurons need to be assigned.
            lite: Flag to decide the type of neurons to be assigned.
            subtensor: The subtensor to use for syncing.

        Returns:
            None.
        """
        # TODO: Check and test the conditions for assigning neurons
        if lite:
            self.neurons = subtensor.neurons_lite(block=block, netuid=self.netuid)
        else:
            self.neurons = subtensor.neurons(block=block, netuid=self.netuid)
        self.lite = lite

    def _set_metagraph_attributes(self, block, subtensor):
        """
        Sets attributes for the metagraph.

        Args:
            block: The block number for which the attributes need to be set.
            subtensor: The subtensor to use for syncing.

        Returns:
            None.
        """
        # TODO: Check and test the setting of each attribute
        self.n = self._create_tensor(len(self.neurons), dtype=torch.int64)
        self.version = self._create_tensor(
            [bittensor.__version_as_int__], dtype=torch.int64
        )
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

    def _create_tensor(self, data, dtype) -> torch.nn.Parameter:
        """
        Creates a tensor parameter with the given data and dtype.

        Args:
            data: The data to be included in the tensor.
            dtype: The datatype for the tensor.

        Returns:
            A tensor parameter.
        """
        # TODO: Check and test the creation of tensor
        return torch.nn.Parameter(torch.tensor(data, dtype=dtype), requires_grad=False)

    def _set_weights_and_bonds(self, subtensor: bittensor.subtensor = None):
        """
        Computes and sets weights and bonds for each neuron.

        Returns:
            None.
        """
        # TODO: Check and test the computation of weights and bonds
        if self.netuid == 0:
            self.weights = self._process_root_weights(
                [neuron.weights for neuron in self.neurons], "weights", subtensor
            )
        else:
            self.weights = self._process_weights_or_bonds(
                [neuron.weights for neuron in self.neurons], "weights"
            )
            self.bonds = self._process_weights_or_bonds(
                [neuron.bonds for neuron in self.neurons], "bonds"
            )

    def _process_weights_or_bonds(self, data, attribute: str) -> torch.nn.Parameter:
        """
        Processes weights or bonds based on the given attribute.

        Args:
            data: The weights or bonds data to be processed.
            attribute: The attribute to decide the type of processing ('weights' or 'bonds').

        Returns:
            The processed tensor parameter.
        """
        data_array = []
        for item in data:
            if len(item) == 0:
                data_array.append(torch.zeros(len(self.neurons)))
            else:
                uids, values = zip(*item)
                # TODO: Validate and test the conversion of uids and values to tensor
                if attribute == "weights":
                    data_array.append(
                        bittensor.utils.weight_utils.convert_weight_uids_and_vals_to_tensor(
                            len(self.neurons), uids, values
                        )
                    )
                else:
                    data_array.append(
                        bittensor.utils.weight_utils.convert_bond_uids_and_vals_to_tensor(
                            len(self.neurons), uids, values
                        )
                    )
        tensor_param = (
            torch.nn.Parameter(torch.stack(data_array), requires_grad=False)
            if len(data_array)
            else torch.nn.Parameter()
        )
        if len(data_array) == 0:
            bittensor.logging.warning(
                f"Empty {attribute}_array on metagraph.sync(). The '{attribute}' tensor is empty."
            )
        return tensor_param

    def _process_root_weights(
        self, data, attribute: str, subtensor: bittensor.subtensor
    ) -> torch.nn.Parameter:
        """
        Processes root weights based on the given attribute.

        Args:
            data: The weights or bonds data to be processed.
            attribute: The attribute to decide the type of processing ('weights' or 'bonds').

        Returns:
            The processed tensor parameter.
        """
        data_array = []
        n_subnets = subtensor.get_total_subnets()
        subnets = subtensor.get_subnets()
        for item in data:
            if len(item) == 0:
                data_array.append(torch.zeros(n_subnets))
            else:
                uids, values = zip(*item)
                # TODO: Validate and test the conversion of uids and values to tensor
                data_array.append(
                    bittensor.utils.weight_utils.convert_root_weight_uids_and_vals_to_tensor(
                        n_subnets, uids, values, subnets
                    )
                )

        tensor_param = (
            torch.nn.Parameter(torch.stack(data_array), requires_grad=False)
            if len(data_array)
            else torch.nn.Parameter()
        )
        if len(data_array) == 0:
            bittensor.logging.warning(
                f"Empty {attribute}_array on metagraph.sync(). The '{attribute}' tensor is empty."
            )
        return tensor_param

    def save(self) -> "metagraph":
        """
        Save the state of the metagraph object.

        Returns:
            metagraph: Updated metagraph object.
        """
        save_directory = get_save_dir(self.network, self.netuid)
        os.makedirs(save_directory, exist_ok=True)
        graph_file = save_directory + f"/block-{self.block.item()}.pt"
        state_dict = self.state_dict()
        state_dict["axons"] = self.axons
        torch.save(state_dict, graph_file)
        state_dict = torch.load(graph_file)
        return self

    def load(self) -> "metagraph":
        """
        Load the state of the metagraph object.

        Returns:
            metagraph: Updated metagraph object.
        """
        self.load_from_path(get_save_dir(self.network, self.netuid))

    def load_from_path(self, dir_path: str) -> "metagraph":
        """
        Load the state of the metagraph object from the specified path.

        Args:
            dir_path (str): Directory path.

        Returns:
            metagraph: Updated metagraph object.
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
        if "weights" in state_dict:
            self.weights = torch.nn.Parameter(
                state_dict["weights"], requires_grad=False
            )
        if "bonds" in state_dict:
            self.bonds = torch.nn.Parameter(state_dict["bonds"], requires_grad=False)
        return self
