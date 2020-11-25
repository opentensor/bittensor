
import asyncio
import copy
import argparse
import bittensor
import math
import netaddr
import numpy
import time
import threading
import torch
import traceback

from threading import Thread, Lock

from munch import Munch
from loguru import logger
from bittensor import bittensor_pb2
# from substrateinterface import SubstrateInterface, Keypair
from bittensor.subtensor import WSClient, Keypair
from typing import List

custom_type_registry = {
    "runtime_id": 2, 
    "types": {
            "NeuronMetadata": {
                    "type": "struct", 
                    "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"]]
                }
        }
}

def int_to_ip(int_val):
    return str(netaddr.IPAddress(int_val))
 
def ip_to_int(str_val):
    return int(netaddr.IPAddress(str_val))

class Metagraph():
 
    def __init__(self, config, keypair):
        r"""Initializes a new Metagraph subtensor interface.
        Args:
            config (bittensor.Config):
                An bittensor config object.
            keypair (substrateinterface.Keypair):
                An bittensor keys object.
        """
        self._config = config
        self.__keypair = keypair
        self.subtensor_client = WSClient(self._config.metagraph.chain_endpoint, self.__keypair)

        self._current_block = 0
        self._last_poll = -math.inf
        self._running = False

        # Map from neuron pubkey -> neuron index
        self._pubkey_index_map = {}
        self._index_for_key = {}

        # Number of neurons in graph.
        self._n = 0

        # List of bittensor_pb2.Neurons ordered by index
        self._neurons_list = []

        # Unique integer key for neurons
        self._next_unique_key = 0
        self._keys_list = []
   
        # List of List of weight_keys ordered by index
        self._weight_keys = []
        self._weight_vals = []

        # List of stake values ordered by index
        self._stake_list = []

        # List of emit values ordered by index
        self._emit_list = []

        # List of last poll ordered by index
        self._poll_list = []

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--metagraph.chain_endpoint', default='206.189.254.5:12345', type=str, 
                            help='chain endpoint.')

        parser.add_argument('--metagraph.polls_every_sec', default=25, type=int, 
                            help='Second until the next chain poll.')

        parser.add_argument('--metagraph.re_poll_neuron_every_blocks', default=20, type=int, 
                            help='Re poll info from neurons every n blocks.')

        parser.add_argument('--metagraph.stale_emit_limit', default=1000, 
                            help='Filter neurons with block time since emission greater than this value.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.metagraph.polls_every_sec > 5 and config.metagraph.polls_every_sec < 1000, 'metagraph.polls_every_sec must be in range [5, 1000]'
        assert config.metagraph.re_poll_neuron_every_blocks > 5 and config.metagraph.re_poll_neuron_every_blocks < 1000, 'metagraph.re_poll_neuron_every_blocks must be in range [5, 1000]'
        assert config.metagraph.stale_emit_limit > 1 and config.metagraph.re_poll_neuron_every_blocks < math.inf, 'metagraph.stale_emit_limit must be in range [1, inf]'
        return config

    def n (self) -> int:
        """ Returns the number of neurons in the network.

        Returns:
            n: (int): neuron count.
        """
        return self._n

    def block (self) -> int:
        """ Returns the current chain block number.

        Returns:
            block: current chain block number.
        """
        return self._current_block

    def neurons(self) -> List[bittensor_pb2.Neuron]:
        """ Returns information for each active neuron in the network.

       Returns:
            neurons: (List[bittensor_pb2.Neuron]): neuron info ordered by index.
        """
        return copy.deepcopy(self._neurons_list)

    def neurons_for_indices(self, indices: torch.Tensor) -> List[bittensor_pb2.Neuron]:
        """ Returns the information for neurons at passed indices.

        Args:
            indices: (torch.LongTensor)
                indexs to pull from neuron list.

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): neuron info ordered by index.
        """

        return [copy.deepcopy(self._neurons_list[int(idx)]) for idx in indices.tolist()]

    def indices(self) -> List[bittensor_pb2.Neuron]:
        """ Returns the indices into neuron set.

        Returns:
            indices: (List[bittensor_pb2.Neuron]): index array range(n)
        """
        return torch.Tensor(range(self._n))

    def neurons_for_keys(self, keys: torch.Tensor) -> torch.LongTensor:
        """ Returns neuron information for each key passed key.

        Returns:
            keys (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): unique keys for neurons.
        """
        return [copy.deepcopy(self._neurons_list[self._index_for_key[int(key)]]) for key in keys.tolist()]

    def keys(self) -> torch.LongTensor:
        """ Returns a torch tensor of unique integer keys for neurons.

        Returns:
            keys (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): unique keys for neurons.
        """
        return torch.Tensor(copy.deepcopy(self._keys_list))

    def emit (self) -> torch.LongTensor:
        """ Returns the last block emit time of each active neuron in the network.

        Returns:
            emit: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron emit block ordered by index.
        """
        return torch.Tensor(copy.deepcopy(self._emit_list))

    def poll (self) -> torch.LongTensor:
        """ Returns the metagraph poll block of each active neuron in the network.

        Returns:
            poll: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron poll block ordered by index.
        """
        return torch.Tensor(copy.deepcopy(self._poll_list))

    def stake (self) -> torch.LongTensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            stake: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron stake ordered by index.
        """
        return torch.Tensor(copy.deepcopy(self._stake_list))

    def weights (self) -> torch.FloatTensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(self.n, self.n)`): neuron stake ordered by index.
        """
        # TODO(const): cache this value.
        weights_numpy = numpy.zeros( (self._n, self._n))
        for index_i, (keys, vals) in enumerate(list(zip(self._weight_keys, self._weight_vals))):
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in self._pubkey_index_map:
                    index_j = self._pubkey_index_map[k]
                    weights_numpy[index_i, index_j] = float(val) / float(val_sum)
        return torch.Tensor(weights_numpy)

    async def pollchain(self):
        logger.info("***** Doing a chain poll *****")
        self._current_block = await self.subtensor_client.get_current_block()

        # Pull the last emit data from all nodes.
        emits = await self.subtensor_client.get_last_emit_data()

        for (key, val) in emits:
            # Filter on stale.
            if (self._current_block - val) > self._config.metagraph.stale_emit_limit:
                continue

            # Filter on recent poll.
            last_poll = self._poll_list[self._pubkey_index_map[key]] if key in self._pubkey_index_map else -math.inf
            if (self._current_block - last_poll) < self._config.metagraph.re_poll_neuron_every_blocks:
                continue

            # Poll.
            await self._pollpubkey(key)

        await asyncio.sleep(self._config.metagraph.polls_every_sec)
        await self.pollchain()

    async def _pollpubkey(self, pubkey):
        """ Polls info from the chain for a specific pubkey.

        Function call updates or appends new information to the stake vectors. If the neuron pubkey
        does not exist in the active set we assign an new index in the state vectors otherwise pull
        info from the local pubkey -> index mapping.

        """
        self._current_block = await self.subtensor_client.get_current_block()
        if pubkey in self._pubkey_index_map:
            index = self._pubkey_index_map[pubkey]
            append = False
        else:
            index = self._n
            self._n += 1
            key = self._next_unique_key
            self._next_unique_key += 1
            self._pubkey_index_map[pubkey] = index
            append = True

        try:
            stake = await self.subtensor_client.get_stake(pubkey)
            emit = await self.subtensor_client.get_last_emit_data(pubkey)
            info = await self.subtensor_client.neurons(pubkey)
            w_keys = await self.subtensor_client.weight_keys(pubkey)
            w_vals = await self.subtensor_client.weight_vals(pubkey)

            #logger.info("Stake: {}", stake)
            #logger.info("Emit: {}", emit)
            #logger.info("Neurons: {}", info)
            #logger.info("Weight keys: {}", w_keys)
            #logger.info("Weight vals: {}", w_vals)

            ipstr = int_to_ip(info['ip'])
            port = int(info['port'])
            neuron = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=pubkey,
                address=ipstr,
                port=port
            )

            if not append:
                self._neurons_list[index] = neuron
                self._stake_list[index] = int(stake)
                self._emit_list[index] = int(emit)
                self._weight_keys[index] = list(w_keys)
                self._weight_vals[index] = list(w_vals)
                self._poll_list[index] = self._current_block
            else:
                self._neurons_list.append(neuron)
                self._stake_list.append(int(stake))
                self._emit_list.append(int(emit))
                self._weight_keys.append(list(w_keys))
                self._weight_vals.append(list(w_vals))
                self._poll_list.append(self._current_block)
                self._keys_list.append( key )
                self._index_for_key[key] = index

            # Record number of peers on tblogger
            bittensor.session.tbwriter.write_network_data("# Peers", len(self._neurons_list))

        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            traceback.print_exc()

    def _build_torch_tensors(self):
        """ Builds torch objects from python polled state.

        """
        # Set torch tensors from weights.
        self._stake_torch = torch.Tensor(self._stake_list)
        self._emit_torch = torch.Tensor(self._emit_list)
        self._poll_torch = torch.Tensor(self._poll_list)
        self._keys_torch = torch.Tensor(self._keys_list)
        
        # Fill weights
        weights_numpy = numpy.zeros( (self._n, self._n))
        for index_i, (keys, vals) in enumerate(list(zip(self._weight_keys, self._weight_vals))):
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in self._pubkey_index_map:
                    index_j = self._pubkey_index_map[k]
                    weights_numpy[index_i, index_j] = float(val) / float(val_sum)
        self._weights_torch = torch.Tensor(weights_numpy)

    async def connect(self) -> bool:
        self.subtensor_client.connect()
        connected = await self.subtensor_client.is_connected()
        return connected

    async def subscribe (self, timeout) -> bool:
        await self.subtensor_client.subscribe(self._config.axon.remote_ip, self._config.axon.port)

        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            neurons = await self.subtensor_client.neurons()
            for n in neurons:
                if n[0] == self.__keypair.public_key:
                    return True
        return False
            
    async def unsubscribe (self, timeout):
        logger.info('Unsubscribe from chain endpoint')
        await self.subtensor_client.unsubscribe()

